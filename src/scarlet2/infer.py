"""Inference methods"""

import functools
import operator
from pprint import pformat

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS

from .scene import Scene
from .validation_utils import (
    ValidationError,
    ValidationInfo,
    ValidationMethodCollector,
    ValidationResult,
    ValidationWarning,
    print_validation_results,
)


# helper class to turn observation likelihood(s) into numpyro distribution
class _ObsDistribution(dist.Distribution):
    support = constraints.real_vector

    def __init__(self, obs, model, validate_args=None):
        self.obs = obs
        self.model = model
        event_shape = jnp.shape(model)
        super().__init__(
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def mean(self):
        return self.obs.render(self.model)

    @dist.util.validate_sample
    def log_prob(self, value):
        # numpyro needs sampling distribution of data (=value), not likelihood function of parameters
        return self.obs._log_likelihood(self.model, value)


@functools.lru_cache(maxsize=16)
def _eqx_module_class(name: str, fields: tuple):
    # Cached so repeated fit() calls with the same parameter set reuse the same class.
    # Class identity drives JAX's pytree treedef, so a fresh class would invalidate the
    # JIT cache for any function taking this module as an arg.
    annotations = {k: jax.Array for k in fields}
    return type(name, (eqx.Module,), {"__annotations__": annotations})


def _dict_to_eqx_module(name: str, data: dict):
    cls = _eqx_module_class(name, tuple(data.keys()))
    return cls(**data)


# -----------------------------------------------------------------------------
# Pair-similarity regularizer
# -----------------------------------------------------------------------------
#
# Penalizes the source-pair similarity that is the geometric signature of
# parasitic flux: when source A acquires a B-shaped bump, the cosine
# similarity between A's and B's morphologies rises. The penalty is
#
#     R = sum_{A != B} sigma_AB * rho_AB
#
# where rho_AB is the cosine similarity of the band-summed morphologies and
# sigma_AB is the cosine similarity of the SEDs. The SED factor scales the
# penalty by the strength of the spectral degeneracy: maximal where the
# data likelihood cannot tell A and B apart, fading where SEDs differ.
# -----------------------------------------------------------------------------


class PairSimilarity(eqx.Module):
    """Configuration for the pair-similarity regularizer used by :func:`fit`.

    Penalizes the morphology-cosine x SED-cosine similarity summed over
    source pairs. Targets the parasitic-flux failure mode: when source A
    absorbs flux in the shape of a neighbour B, the morphology cosine
    rises, and the penalty pushes that B-shaped component back down.
    The SED-cosine factor scales the penalty by the degree of spectral
    degeneracy between the pair.

    Parameters
    ----------
    weight : float
        Relative coefficient of the regularizer: the target ratio of the
        penalty to the initial NLL. ``fit()`` evaluates the NLL and the
        unweighted penalty once at initialization and rescales internally
        so that ``R = weight * |NLL_init|`` at the start of optimization.
        ``0.0`` is a no-op (bit-identical to the unregularized loss).
        Empirically a value around ``0.01`` (penalty ~1% of the NLL)
        works well.
    eps : float, optional
        Small constant added to cosine denominators for numerical stability.

    Notes
    -----
    Implemented as an ``eqx.Module`` with all-static fields so it threads
    through ``eqx.filter_jit`` without being treated as a differentiable
    leaf and without forcing recompilation across identical configs.
    """

    weight: float = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-12)


def _cosine_matrix(X, eps):
    """Cosine similarity between rows of X. Shape (N, D) -> (N, N)."""
    norms = jnp.sqrt(jnp.sum(X * X, axis=1, keepdims=True))
    Xn = X / (norms + eps)
    return Xn @ Xn.T


def _pair_similarity_from_stack(per_source, cfg: PairSimilarity):
    """Compute R from a stack of per-source models in a common frame.

    ``per_source`` has shape ``(K, C, H, W)`` for multi-band scenes or
    ``(K, H, W)`` for single-band. Pure tensor math so it can be exercised
    in tests without a Scene. Does not apply ``cfg.weight``; the caller
    multiplies.
    """
    per_source = jnp.maximum(per_source, 0.0)

    if per_source.ndim == 4:
        M = per_source.sum(axis=1)  # (K, H, W)
        f = per_source.sum(axis=(2, 3))  # (K, C)
    else:
        M = per_source
        # Degenerate single-band case: SED cosine is identically 1, so
        # only morphology contributes.
        f = jnp.ones((per_source.shape[0], 1))

    M_flat = M.reshape(M.shape[0], -1)
    rho = _cosine_matrix(M_flat, eps=cfg.eps)
    sigma = _cosine_matrix(f, eps=cfg.eps)

    # rho, sigma are matmuls of non-negative matrices and thus non-negative
    # element-wise; the maxima are belt-and-braces against any roundoff dust.
    pair_term = jnp.maximum(sigma, 0.0) * jnp.maximum(rho, 0.0)
    # Sum the strict upper triangle directly. We avoid `(sum - trace) / 2`:
    # in float32 those are two reductions of comparable magnitude (~K) summed
    # in different orders, so when off-diagonals are tiny the diagonals do
    # not cancel exactly and the result can flip slightly negative.
    return jnp.sum(jnp.triu(pair_term, k=1))


def _evaluate_per_source(scene_obj):
    """Stack of per-source models in the scene frame: shape ``(K, C, H, W)``."""
    return jnp.stack([scene_obj.evaluate_source(s) for s in scene_obj.sources], axis=0)


def _pair_similarity_penalty(scene_obj, cfg: PairSimilarity, per_source=None):
    """Compute the pair-similarity regularizer for a rendered Scene.

    If ``per_source`` (the ``(K, C, H, W)`` stack) is already available from the
    caller, pass it in to avoid re-evaluating each source.
    """
    if cfg.weight == 0.0:
        return jnp.array(0.0)

    if per_source is None:
        per_source = _evaluate_per_source(scene_obj)
    return cfg.weight * _pair_similarity_from_stack(per_source, cfg)


def sample(scene, observations, *args, seed=0, num_warmup=100, num_samples=200, progress_bar=True, **kwargs):
    """Sample `parameters` of every source in `scene` to get posteriors given `observations`.

    This method runs the HMC NUTS sampler from `numpyro` to get parameter
    posteriors. It uses the likelihood of `observations` as well as the `prior`
    attribute set for every :py:class:`~scarlet2.Parameter` in `parameters`.

    Parameters
    ----------
    scene : :py:class:`~scarlet2.Scene`
        The model of the scene.
    observations: :py:class:`~scarlet2.Observation` or list
        The observations to fit the models to.
    *args: list, optional
        Additional arguments passed. Only used for backwards (v0.3) compatibility.
    seed: int, optional
        RNG seed for the sampler
    num_warmup: int, optional
        Number of samples during HMC warm-up
    num_samples: int, optional
        Number of samples to create from tuned HMC
    progress_bar: bool, optional
        Whether to show a progress bar
    **kwargs: dict, optional
        Additional keyword arguments passed to the `numpyro.infer.NUTS` sampler.

    Notes
    -----
    Requires `numpyro`

    Returns
    -------
    numpyro.infer.mcmc.MCMC
    """
    # making sure we can iterate
    if not isinstance(observations, (list, tuple)):
        observations = (observations,)
    obs_params = {}
    for obs in observations:
        obs.check_set_renderer(scene.frame)
        obs_params.update(obs.parameters)

    # scene and observations can have parameters: combine them into one model
    parameters = scene.parameters | obs_params
    if len(parameters) == 0:
        msg = "Scene and Observation(s) must have at least one parameter. Found none."
        raise AttributeError(msg)

    # find all non-fixed parameters and their priors
    priors = {name: p.prior for name, (idx, p) in parameters.items()}
    has_none = any(prior is None for prior in priors.values())
    if has_none:
        msg = f"All parameters need to have priors set. Got:\n{pformat(priors)}"
        raise AttributeError(msg)

    values = scene.get()
    for obs in observations:
        values |= obs.get()
    init_values = values.copy()

    # construct eqx.Module containing all parameter arrays as attributes
    values = _dict_to_eqx_module("ParamModel", values)

    # define the pyro model, where every parameter value becomes a sample,
    # and the observations sample from their likelihood given the rendered model
    def pyro_model(values):
        samples = {name: numpyro.sample(name, param.prior) for name, (node, param) in parameters.items()}
        scene_ = scene.set(samples)
        pred = scene_()  # create scene once for all observations
        # evaluate likelihood of multiple observations
        for i, obs_ in enumerate(observations):
            numpyro.sample(f"obs.{i}", _ObsDistribution(obs_.set(values), pred), obs=obs_.data)

    # if not told otherwise: use init from current value of model
    init_strategy = kwargs.pop("init_strategy", None)
    if init_strategy is None:
        from functools import partial

        from numpyro.infer.initialization import init_to_value

        init_strategy = partial(init_to_value, values=init_values)

    nuts_kernel = NUTS(pyro_model, init_strategy=init_strategy, **kwargs)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=progress_bar)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, values)
    return mcmc


def fit(
    scene,
    observations,
    *args,
    schedule=None,
    max_iter=100,
    e_rel=1e-4,
    progress_bar=True,
    callback=None,
    pair_similarity=None,
    **kwargs,
):
    """Fit model `parameters` of every source in `scene` to match `observations`.

    Computes the best-fit parameters of all components in every source by
    first-order gradient descent with the Yogi optimizer from `optax`.

    Parameters
    ----------
    scene : :py:class:`~scarlet2.Scene`
        The model of the scene.
    observations: :py:class:`~scarlet2.Observation` or list
        The observations to fit the model to.
    *args: list, optional
        Additional arguments passed. Only used for backwards (v0.3) compatibility.
    schedule: callable, optional
        A function that maps optimizer step count to value. See :py:class:`optax.Schedule` for details.
    max_iter: int, optional
        Maximum number of optimizer iterations
    e_rel: float, optional
        Upper limit for the relative change in the norm of any parameter to
        terminate the optimization early.
    progress_bar: bool, optional
        Whether to show a progress bar
    callback: callable, optional
        Function to be called on the current state of the optimized scene.
        Signature `callback(scene, convergence, loss) -> None`, where
        `convergence` is a tree of the same structure as `scene`, and `loss`
        is the current value of the log_posterior.
    pair_similarity : :py:class:`~scarlet2.PairSimilarity`, optional
        If given, adds a pair-similarity regularizer to the loss. ``None``
        (default) disables it (bit-identical to the un-regularized version).
        Penalizes the cosine similarity of source-pair morphologies, scaled
        by the cosine similarity of their SEDs. Targets parasitic flux
        directly via the morphology cosine. See
        :py:class:`~scarlet2.PairSimilarity` for the available options.
    **kwargs: dict, optional
        Additional keyword arguments passed to the `optax.scale_by_yogi` optimizer.

    Notes
    -----
    Requires `optax`. The returned scene carries the *best* parameters seen during
    optimization (lowest loss), not the last iteration's, since Yogi/Adam can be
    non-monotonic. Diagnostic info (loss history, best loss, iteration count) is
    attached as :py:attr:`Scene.fit_info`.

    Returns
    -------
    Scene, list(Observation)
        The scene and observation(s) with best-fit parameters. ``scene.fit_info``
        contains ``{"loss", "pair_similarity", "best_loss", "n_iter"}``.
        The ``pair_similarity`` array report zero when the regularizer is
        disabled.
    """
    try:
        from tqdm.auto import trange
    except ImportError as err:
        raise ImportError("scarlet2.Scene.fit() requires optax and numpyro.") from err

    # making sure we can iterate
    if not isinstance(observations, (list, tuple)):
        observations = (observations,)
    obs_params = {}
    for obs in observations:
        obs.check_set_renderer(scene.frame)
        obs_params.update(obs.parameters)

    # scene and observations can have parameters: combine them into one model
    parameters = scene.parameters | obs_params
    if len(parameters) == 0:
        msg = "Scene and Observation(s) must have at least one parameter. Found none."
        raise AttributeError(msg)

    # canonicalize: strip any fit_info from a previously-fit scene so the JIT cache key
    # is identical for fresh and re-fitted inputs (fit_info is a static eqx field)
    scene = scene.replace("fit_info", None)

    values = scene.get()
    for obs in observations:
        values |= obs.get()

    # construct eqx.Module containing all parameter arrays as attributes
    values = _dict_to_eqx_module("ParamModel", values)
    # same tree structure but for param specs so that we can use them below
    treedef = jt.structure(values)
    params = tuple(param for name, (node, param) in parameters.items())
    params = jt.unflatten(treedef, params)
    steps = jt.map(lambda param: param.stepsize, params)

    # build (or fetch from cache) the optax optimizer; identity-stable across fit() calls
    # with matching kwargs and schedule, so JAX's JIT cache stays warm
    schedule_fn = schedule if callable(schedule) else _unit_schedule
    optim = _build_optim(tuple(sorted(kwargs.items())), schedule_fn)

    # transform to unconstrained parameters
    values = _constraint_replace(values, params, inv=True)
    opt_state = optim.init(values)

    # default: regularizer off (weight=0.0 short-circuits to a no-op)
    pair_cfg = pair_similarity if pair_similarity is not None else PairSimilarity(weight=0.0)

    # Calibrate pair-similarity: user-facing `weight` is relative to the initial NLL.
    # We compute |NLL_init| and R_init (unweighted) once, then replace the cfg with
    # one carrying the absolute multiplier. Skip when weight=0 (no-op) or R_init==0.
    if pair_cfg.weight != 0.0:
        values_init = _constraint_replace(values, params)
        scene_init = scene.set(values_init)
        nll_init = -sum(obs.set(values_init).log_likelihood(scene_init()) for obs in observations)
        r_init = _pair_similarity_from_stack(_evaluate_per_source(scene_init), pair_cfg)
        scale = jnp.where(r_init > 0, jnp.abs(nll_init) / (r_init + pair_cfg.eps), 0.0)
        pair_cfg = PairSimilarity(weight=float(pair_cfg.weight * scale), eps=pair_cfg.eps)

    # initialize best-fit tracker (loss is minimized, so start at +inf).
    # Use an explicit dtype so the array is strongly-typed: jnp.where()'s output
    # in iter 1 is strongly-typed, and JAX caches weak vs. strong separately —
    # passing a weak inf would force a second compile on iter 2.
    best_loss = jnp.array(jnp.inf, dtype=jnp.float32)
    best_values = values
    losses = []
    pair_history = []

    with trange(max_iter, disable=not progress_bar) as t:
        for step in t:  # noqa: B007
            # optimizer step
            values, loss, pair, opt_state, convergence, best_loss, best_values = _make_step(
                values,
                params,
                scene,
                observations,
                optim,
                opt_state,
                steps,
                best_loss,
                best_values,
                pair_cfg,
            )
            losses.append(loss)
            pair_history.append(pair)

            # compute max change across all non-fixed parameters for convergence test
            max_change = jax.tree_util.tree_reduce(lambda a, b: max(a, b), convergence)

            # report current iteration results to callback
            if callback is not None:
                values_ = _constraint_replace(values, params)
                scene_ = scene.set(values_)
                callback(scene_, convergence, loss)

            # Log the loss and max_change in the tqdm progress bar
            t.set_postfix(loss=f"{loss:08.2f}", max_change=f"{max_change:1.6f}")

            # test convergence
            if max_change < e_rel:
                break

    # transform best-fit values back to constrained variables and replace in scene
    best_values = _constraint_replace(best_values, params)
    # scene_ is a copy, but its registry_key still points to scene.parameters, can thus be reused
    scene_ = scene.set(best_values)
    obs_ = tuple(obs.set(best_values) for obs in observations)

    # attach diagnostic info to the returned scene
    fit_info = {
        "loss": jnp.stack(losses),
        "pair_similarity": jnp.stack(pair_history),
        "best_loss": best_loss,
        "n_iter": step + 1,
    }
    scene_ = scene_.replace("fit_info", fit_info)

    # (re)-import `VALIDATION_SWITCH` at runtime to avoid using a static/old value
    from .validation_utils import VALIDATION_SWITCH

    if VALIDATION_SWITCH:
        from .validation import check_fit

        for obs in observations:
            validation_results = check_fit(scene_, obs)
            print_validation_results(f"Fit validation results for observation {obs.name}", validation_results)

    return scene_, obs_


def _constraint_replace(values, params, inv=False):
    # replace any parameter with constraint into unconstrained ones by calling its constraint bijector
    def transform(value, param):
        if param.constraint is not None:
            func = param.constraint_transform if inv is False else param.constraint_transform.inv
            return func(value)
        else:
            return value

    return jt.map(transform, values, params)


def _unit_schedule(_):
    # module-level so optax.scale_by_schedule(_unit_schedule) has stable identity across calls
    return 1.0


@functools.lru_cache(maxsize=8)
def _build_optim(kwargs_items, schedule):
    """Construct (and cache) the optax optimizer.

    Cached on hashable args so repeated `fit()` calls with the same configuration
    reuse the same optimizer instance. JAX's JIT cache keys static args by Python
    identity, so a fresh `optax.chain(...)` from each fit() call would otherwise
    miss the cache and recompile `_make_step` every time. See
    https://github.com/pmelchior/scarlet2/issues/120.
    """
    import optax

    return optax.chain(
        optax.scale_by_yogi(**dict(kwargs_items)),
        optax.scale_by_schedule(schedule),
    )


# update step for optax optimizer
@eqx.filter_jit
def _make_step(
    values, params, scene, observations, optim, opt_state, steps, best_loss, best_values, pair_cfg
):
    loss_fn = lambda values: _loss_fn(values, params, scene, observations, pair_cfg)
    (loss, (pair,)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(values)
    updates, opt_state = optim.update(grads, opt_state, values)
    # apply per-parameter stepsizes; minus sign because we want gradient descent
    updates = jt.map(
        lambda u, s, p: None if u is None else (-s(p) * u if callable(s) else -s * u),
        updates,
        steps,
        values,
        is_leaf=lambda x: x is None,
    )
    values_ = eqx.apply_updates(values, updates)

    # for convergence criterion: compute norms of parameters and updates
    norm = lambda x, dx: 0 if dx is None else jnp.linalg.norm(dx) / jnp.linalg.norm(x)
    convergence = jt.map(lambda x, dx: norm(x, dx), values, updates)

    # track best-fit: loss is at the *input* `values` (pre-update), so on improvement
    # we keep the input values, not the post-update ones
    is_better = loss < best_loss
    best_loss = jnp.where(is_better, loss, best_loss)
    best_values = jt.map(lambda b, v: jnp.where(is_better, v, b), best_values, values)

    return values_, loss, pair, opt_state, convergence, best_loss, best_values


# loss function for the optimizer;
# returns the loss and a tuple of aux values (pair-similarity penalty) for tracking in fit_info
def _loss_fn(values, params, scene, observations, pair_cfg):
    # parameters now obey constraints
    # transformation happens in the grad path, so gradients are wrt to unconstrained variables
    # likelihood and prior grads transparently apply the Jacobians of these transformations
    values_ = _constraint_replace(values, params)
    scene_ = scene.set(values_)
    # If any regularizer is active, we need the per-source stack anyway —
    # build it once and reuse its sum as the model so we don't loop over
    # sources twice.
    needs_per_source = pair_cfg.weight != 0.0
    if needs_per_source:
        per_source = _evaluate_per_source(scene_)
        model = per_source.sum(axis=0)
    else:
        per_source = None
        model = scene_()
    log_like = sum(obs.set(values_).log_likelihood(model) for obs in observations)

    # add log prior for all parameters which define priors
    # Note: This calls priors separately even if they support batched execution
    # see https://github.com/pmelchior/scarlet2/issues/103 for a possible solution
    # however, testing after #103 got merged suggests that tree_reduce is faster than grouping
    log_prior = jt.reduce(
        operator.add,
        jt.map(
            lambda value, param: param.prior.log_prob(value) if param.prior is not None else 0,
            values_,
            params,
        ),
    )

    # pair-similarity regularizer (no-op when pair_cfg.weight == 0.0)
    pair = _pair_similarity_penalty(scene_, pair_cfg, per_source=per_source)

    # has_aux=True: return penalty values alongside the loss so they can be
    # tracked in fit_info without a second forward pass
    return -(log_like + log_prior) + pair, (pair,)


# Compute Hessian-vector products (HVPs) using JAX's jvp and grad.
# for regular functions f
def hvp(f, primals, tangents):
    """Calculate the Hessian-vector product of a function f"""
    return jax.jvp(jax.grad(f), primals, tangents)[1]


# for score functions
def hvp_grad(grad_f, primals, tangents):
    """Calculate the Hessian-vector product of a gradient function grad_f"""
    return jax.jvp(grad_f, primals, tangents)[1]


# diagonals of Hessian from HVPs
def hvp_rad(hvp, shape, key=None, max_iter=100):
    """Approximate the diagonal of the Hessian

    Uses the Hutchinson's method to approximate the diagonal of the Hessian by computing the
    Hessian-vector products with random Rademacher vectors.

    Parameters
    ----------
    hvp : callable
        A function that computes the Hessian-vector product.
    shape : tuple
        The shape of the input to the Hessian-vector product function.
    key : jax.random.PRNGKey, optional
        A random key for generating Rademacher vectors. If None, a new key will be generated.
    max_iter : int, optional
        The maximum number of iterations for the approximation. Default is 100.

    Returns
    -------
    jax.numpy.ndarray
        An array containing the approximate diagonal of the Hessian.
    """
    h = jnp.zeros(shape, dtype=jnp.float32)
    if key is None:
        key = jax.random.PRNGKey(0)

    # Fixed-length loop (no data-dependent early stop) so this is jittable. A fresh
    # Rademacher probe per iteration is essential: reusing one `key` would draw the
    # same vector every time and never average out the off-diagonal contamination.
    for _ in range(max_iter):
        key, subkey = jax.random.split(key)
        z = jax.random.rademacher(subkey, shape, dtype=jnp.float32)
        h += jnp.multiply(z, hvp(z))
    return h / max_iter


def uncertainty(
    scene,
    observations,
    pair_similarity=None,
    max_iter=100,
    return_hessian=False,
    return_samples=False,
    key=None,
):
    """Estimate the per-parameter curvature (diagonal Hessian) of a fitted model.

    Computes the diagonal of the Hessian of the negative log-posterior loss
    (:func:`_loss_fn`) at the current (assumed best-fit) parameter values, i.e.
    the local marginal precision of each parameter. Under a Laplace
    approximation the "1-sigma" uncertainty of a parameter is $1 / sqrt(H)$
    for the corresponding (positive) diagonal entry.

    If ``return_hessian`` is ``True``,  the raw Hessian curvature is returned
    instead so the caller can handle non-positive entries (a direction not at a
    minimum) explicitly rather than receiving NaNs from the square root.

    The Hessian diagonal is estimated stochastically with Hutchinson's method
    (:func:`hvp_rad`) applied to the Hessian-vector product of :func:`_loss_fn`
    (:func:`hvp`). Every non-fixed parameter of ``scene`` and ``observations``
    is flattened into a single vector, so the estimate is taken jointly across
    all parameters.

    The Hessian is evaluated in the unconstrained space and transformed via the exact
    change-of-variables ``H_phi = H_theta / (d phi / d theta)^2`` (valid at
    an extremum). Parameters without a constraint are returned unchanged.

    Parameters
    ----------
    scene : :py:class:`~scarlet2.Scene`
        The (fitted) model of the scene.
    observations : :py:class:`~scarlet2.Observation` or list
        The observations the model was fit to.
    pair_similarity : :py:class:`~scarlet2.PairSimilarity`, optional
        Same regularizer configuration as passed to :func:`fit`. Must match
        what was used during fitting so that the loss (and hence its Hessian)
        is identical. ``None`` (default) disables it.
    max_iter : int, optional
        Maximum number of Rademacher probes to estimate the Hessian diagonal.
    return_hessian : bool, optional
        If ``True``, return the raw Hessian diagonal rather than the uncertainty (1/sqrt(H)).
    return_samples: bool or int
        If not ``False``, return the stated number samples according to the Gaussian uncertainties.
        The dictionary has the same shape as flattened chains from :func:`sample`.
    key : jax.random.PRNGKey, optional
        A random key for generating Rademacher vectors. If None, a new key will be generated.

    Returns
    -------
    dict
        Maps every non-fixed parameter name to the diagonal uncertainty of the same
        shape as the parameter, expressed in the natural (constrained) parameter
        space.
    """
    # making sure we can iterate
    if not isinstance(observations, (list, tuple)):
        observations = (observations,)
    obs_params = {}
    for obs in observations:
        obs.check_set_renderer(scene.frame)
        obs_params.update(obs.parameters)

    # scene and observations can have parameters: combine them into one model
    parameters = scene.parameters | obs_params
    if len(parameters) == 0:
        msg = "Scene and Observation(s) must have at least one parameter. Found none."
        raise AttributeError(msg)

    # canonicalize like fit(): strip fit_info so the pytree/JIT cache key is stable
    scene = scene.replace("fit_info", None)

    values = scene.get()
    for obs in observations:
        values |= obs.get()
    value_names = tuple(values.keys())

    # construct eqx.Module containing all parameter arrays as attributes (as in fit)
    values = _dict_to_eqx_module("ParamModel", values)
    treedef = jt.structure(values)
    params = tuple(param for name, (node, param) in parameters.items())
    params = jt.unflatten(treedef, params)

    # default: regularizer off (weight=0.0 short-circuits to a no-op)
    pair_cfg = pair_similarity if pair_similarity is not None else PairSimilarity(weight=0.0)

    # _loss_fn differentiates wrt unconstrained variables, so evaluate the Hessian there
    values_u = _constraint_replace(values, params, inv=True)

    # scalar loss over a single flat vector spanning all parameters
    flat, unravel = ravel_pytree(values_u)
    loss_fn = lambda x: _loss_fn(unravel(x), params, scene, observations, pair_cfg)[0]

    # jit the Hessian-vector product so the ~max_iter Hutchinson probes reuse one compile
    if key is None:
        key = jax.random.PRNGKey(0)
    hvp_z = jax.jit(lambda z: hvp(loss_fn, (flat,), (z,)))
    diag = hvp_rad(hvp_z, flat.shape, key=key, max_iter=max_iter)

    # unconstrained-space diagonal Hessian, as a parameter tree
    diag_u = unravel(diag)

    # Transform the diagonal Hessian to the natural (constrained) space. At an
    # extremum the score dL/dtheta vanishes, so the change-of-variables keeps only
    # the quadratic term:
    #     H_phi = H_theta (d theta / d phi)^2 = H_theta / (d phi / d theta)^2.
    # The constraint bijectors are elementwise, so a jvp with unit tangents gives the
    # per-element derivative jac = d(constrained)/d(unconstrained). Parameters without
    # a constraint have jac == 1, so their curvature is returned unchanged. Keeping the
    # Hessian (rather than converting to 1/sqrt(H)) avoids NaNs when a near-boundary
    # constrained parameter has a small or slightly-negative unconstrained curvature.
    ones = jt.map(jnp.ones_like, values_u)
    _, jac = jax.jvp(lambda v: _constraint_replace(v, params), (values_u,), (ones,))
    hess = jt.map(lambda h, j: h / j**2, diag_u, jac)

    if return_hessian:
        return {name: getattr(hess, name) for name in value_names}

    # uncertainty = 1/sqrt(H) in the natural (constrained) space
    def sigma(h):
        # remove unreliable Hessian elements
        h_0 = h[h > 0]
        med = jnp.median(h_0)
        std = jnp.std(h_0)
        h = jnp.where((h > 0) & (jnp.abs(h - med) < 10 * std), h, med)
        return 1 / jnp.sqrt(h)

    errors = {name: sigma(getattr(hess, name)) for name in value_names}
    if return_samples is False:
        return errors
    normals = {name: dist.Normal(scene.get(name), scale=errors.get(name)).to_event(1) for name in errors}
    # dictionary: parameter name -> sampled values
    return {name: normals[name].sample(key, sample_shape=(return_samples,)) for name in errors}


class FitValidator(metaclass=ValidationMethodCollector):
    """A class containing all of the validation checks for a Scene objects after
    calling `.fit()`.

    Note that the metaclass is defined as `MethodCollector`, which collects all
    validation methods in this class into a single class attribute list called
    `validation_checks`. This allows for easy iteration over all checks."""

    def __init__(self, scene: Scene, observation):
        """Initialize the FitValidator.

        Parameters
        ----------
        scene : Scene
            The scene object to validate.
        observation : Observation
            The observation object containing the data to validate against.
        """
        self.scene = scene
        self.observation = observation

        # These are placeholders, waiting for the actual width of distribution to be
        # implemented - see issue https://github.com/pmelchior/scarlet2/issues/192
        self.chi2_tolerable_threshold = 1.5
        self.chi2_critical_threshold = 5.0

    def check_goodness_of_fit(self) -> ValidationResult:
        """Evaluate the goodness of the model fit to the data by calling the Observation
        class's `goodness_of_fit` method. Please see the docstring for that method
        for details.

        Returns
        -------
        ValidationResult
            A subclass of ValidationResult indicating the result of the check.
        """
        obs = self.observation

        chi2 = obs.goodness_of_fit(self.scene())
        context = {"chi2": chi2}

        ret_val: ValidationResult = ValidationInfo(
            "The model fit is good.", check=self.__class__.__name__, context=context
        )
        if self.chi2_tolerable_threshold <= chi2 < self.chi2_critical_threshold:
            ret_val = ValidationWarning(
                "The model fit is acceptable, but the goodness of fit is not optimal.",
                check=self.__class__.__name__,
                context=context,
            )
        elif chi2 >= self.chi2_critical_threshold or jnp.isnan(chi2):
            ret_val = ValidationError(
                "The model fit is poor.", check=self.__class__.__name__, context=context
            )

        return ret_val

    def check_chi_square_in_box_and_border(self) -> list[ValidationResult]:
        """Evaluate the weighted mean (weighted by the inverse variance weights)
        of the squared residuals for each source. Chi square is also computed for
        the perimeter outside the box of with `border_width`.

        Returns
        -------
        list[ValidationResult]
            A list of ValidationResult subclasses for each source. For each source
            there will be two results. One for inside the bounding box and one
            for the border. The ValidationResults will each be one of the following:
            - If the chi-square is above the critical threshold, a ValidationError.
            - If the chi-square is below the tolerable threshold, a ValidationInfo.
            - If the chi-square is between the two thresholds, a ValidationWarning.
        """
        obs = self.observation

        chi2_per_source = obs.eval_chi_square_in_box_and_border(self.scene)

        validation_results: list[ValidationResult] = []
        for i, chi2 in chi2_per_source.items():
            chi2_inside = chi2["in"]
            chi2_outside = chi2["out"]

            if chi2_inside < self.chi2_tolerable_threshold:
                validation_results.append(
                    ValidationInfo(
                        f"The chi-square in the box for source {i} is good.",
                        check=self.__class__.__name__,
                        context={"chi2_in": chi2_inside, "source": i},
                    )
                )
            elif self.chi2_tolerable_threshold <= chi2_inside < self.chi2_critical_threshold:
                validation_results.append(
                    ValidationWarning(
                        f"The chi-square in the box for source {i} is acceptable, but not optimal.",
                        check=self.__class__.__name__,
                        context={"chi2_in": chi2_inside, "source": i},
                    )
                )
            elif chi2_inside >= self.chi2_critical_threshold:
                validation_results.append(
                    ValidationError(
                        f"The chi-square in the box for source {i} is poor.",
                        check=self.__class__.__name__,
                        context={"chi2_in": chi2_inside, "source": i},
                    )
                )

            if chi2_outside < self.chi2_tolerable_threshold:
                validation_results.append(
                    ValidationInfo(
                        f"The chi-square in the border for source {i} is good.",
                        check=self.__class__.__name__,
                        context={"chi2_border": chi2_outside, "source": i},
                    )
                )
            elif self.chi2_tolerable_threshold <= chi2_outside < self.chi2_critical_threshold:
                validation_results.append(
                    ValidationWarning(
                        f"The chi-square in the border for source {i} is acceptable, but not optimal.",
                        check=self.__class__.__name__,
                        context={"chi2_border": chi2_outside, "source": i},
                    )
                )
            elif chi2_outside >= self.chi2_critical_threshold:
                validation_results.append(
                    ValidationError(
                        f"The chi-square in the border for source {i} is poor.",
                        check=self.__class__.__name__,
                        context={"chi2_border": chi2_outside, "source": i},
                    )
                )

        return validation_results
