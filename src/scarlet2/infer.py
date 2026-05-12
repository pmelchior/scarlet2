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
# Responsibility regularizer
# -----------------------------------------------------------------------------
#
# Penalizes "parasitic flux", i.e. configurations where one source acquires
# flux in pixels that another source dominates, while leaving genuine overlaps
# essentially untouched.
#
# For each source k with rendered model S_k(x) >= 0, define
#    gamma_k(x) = S_k(x) / (sum_j S_j(x) + eps)        # responsibility
#    w_k(x)     = S_k(x) / sum_y S_k(y)                # source's flux distribution
#
# The regularizer is the cross-entropy of w_k against gamma_k,
# weighted flux-fractionally and summed over sources:
#
#    R = sum_k sum_x w_k(x) * (-log gamma_k(x))
#
# Properties:
#   - Scale-invariant in the global flux normalization.
#   - Asymmetric: penalizes w_k being large where gamma_k is small (parasitic),
#     but not gamma_k being large where w_k is small.
#   - Genuine equal overlaps contribute a flat ~K * log(K) baseline.
# -----------------------------------------------------------------------------


_RESPONSIBILITY_MODES = ("band_summed", "per_band", "joint")


class Responsibility(eqx.Module):
    """Configuration for the responsibility regularizer used by :func:`fit`.

    Penalizes "parasitic flux", i.e. configurations where one source acquires
    flux in pixels that another source dominates, while leaving genuine
    overlaps essentially untouched. Pass an instance of this class to
    ``fit(..., responsibility=Responsibility(weight=...))`` to enable.

    Parameters
    ----------
    weight : float
        Coefficient ``lambda`` of the regularizer added to the loss. ``0.0``
        is a no-op (bit-identical to the unregularized loss).
        As a guide, ``lambda * K log (K) ~= Loss / 100``, where ``K`` is the number
        of components in the scene and ``Loss`` is the unregularized loss.
    eps : float, optional
        Small constant added to denominators for numerical stability. Set to
        a small fraction of the noise RMS in your model units.
    support_threshold : float or None, optional
        If set, restrict each source's contribution to pixels where it has
        flux above this threshold (soft sigmoid mask). Recommended: a few
        times the noise RMS in the model frame. ``None`` (default) applies
        the regularizer over all pixels with nonzero flux.
    support_tau : float, optional
        Smoothing scale of the soft support mask. Only used when
        ``support_threshold`` is not ``None``.
    mode : {"band_summed", "per_band", "joint"}, optional
        How the channel axis of multi-band scenes is treated:

        - ``"band_summed"`` (default): sum per-source models over the channel
          axis first, computing 2D responsibilities. Right answer for
          factorized sources (morphology x spectrum).
        - ``"per_band"``: normalize per-source weights separately per channel
          and sum cross-entropies across channels. Right for sources with
          spatially varying spectra (color gradients, AGN+host, transient
          on host).
        - ``"joint"``: treat ``(c, x)`` as one big index. Kept for
          reproducibility of older runs.

    Notes
    -----
    Implemented as an ``eqx.Module`` with all-static fields so it threads
    through ``eqx.filter_jit`` without being treated as a differentiable
    leaf and without forcing recompilation across identical configs.
    """

    weight: float = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-12)
    support_threshold: float | None = eqx.field(static=True, default=None)
    support_tau: float = eqx.field(static=True, default=1.0)
    mode: str = eqx.field(static=True, default="band_summed")

    def __check_init__(self):
        if self.mode not in _RESPONSIBILITY_MODES:
            msg = f"Responsibility.mode must be one of {_RESPONSIBILITY_MODES}, got {self.mode!r}"
            raise ValueError(msg)


def _responsibility_from_stack(per_source, cfg: Responsibility):
    """Compute R from a stack of per-source models in a common frame.

    ``per_source`` has shape ``(K, ...)``: leading axis enumerates sources,
    remaining axes are spatial (and optionally spectral). Pure tensor math so
    it can be exercised in tests without constructing a Scene. Does not apply
    ``cfg.weight``; the caller multiplies.
    """
    # band_summed mode: collapse the channel axis (axis=1) of a (K, C, H, W)
    # stack into (K, H, W) so responsibilities are computed in 2D. 2D inputs
    # (K, H, W) are unaffected.
    if cfg.mode == "band_summed" and per_source.ndim == 4:
        per_source = jnp.sum(per_source, axis=1)

    # Clip to non-negative: morphologies are typically constrained >= 0, but
    # some parameterizations (e.g. gradient steps in unconstrained space before
    # the constraint bijector is applied) can transiently dip below zero.
    per_source = jnp.maximum(per_source, 0.0)

    total = jnp.sum(per_source, axis=0)
    gamma = per_source / (total + cfg.eps)

    # Per-source weight w_k. In "per_band" mode on a (K, C, H, W) stack, w is
    # normalized separately for each channel (axes 2..end), so cross-entropies
    # are summed independently per band. In other modes, w is normalized over
    # all non-leading axes (channel folded into spatial). For ndim<4, per_band
    # has no separate channel axis and falls back to the joint normalization.
    if cfg.mode == "per_band" and per_source.ndim >= 4:
        spatial_axes = tuple(range(2, per_source.ndim))
    else:
        spatial_axes = tuple(range(1, per_source.ndim))
    flux_per_source = jnp.sum(per_source, axis=spatial_axes, keepdims=True)
    w = per_source / (flux_per_source + cfg.eps)

    if cfg.support_threshold is not None:
        mask = jax.nn.sigmoid((per_source - cfg.support_threshold) / cfg.support_tau)
        w = w * mask
        w = w / (jnp.sum(w, axis=spatial_axes, keepdims=True) + cfg.eps)

    log_gamma = jnp.log(gamma + cfg.eps)
    return -jnp.sum(w * log_gamma)


def _responsibility_penalty(scene_obj, cfg: Responsibility):
    """Compute the responsibility regularizer for a rendered Scene.

    ``scene_obj`` is the parameter-substituted Scene (i.e. ``scene.set(values_)``);
    we need this rather than the rendered total because we evaluate each source
    in the scene frame separately to form the responsibility map.
    """
    if cfg.weight == 0.0:
        return jnp.array(0.0)

    # scene.evaluate_source places each source into a zero array of the full
    # scene bbox, so all entries live on the same (C, H, W) grid.
    per_source = jnp.stack([scene_obj.evaluate_source(s) for s in scene_obj.sources], axis=0)
    return cfg.weight * _responsibility_from_stack(per_source, cfg)


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


def _pair_similarity_penalty(scene_obj, cfg: PairSimilarity):
    """Compute the pair-similarity regularizer for a rendered Scene."""
    if cfg.weight == 0.0:
        return jnp.array(0.0)

    per_source = jnp.stack([scene_obj.evaluate_source(s) for s in scene_obj.sources], axis=0)
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
    responsibility=None,
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
    responsibility : :py:class:`~scarlet2.Responsibility`, optional
        If given, adds a responsibility regularizer to the loss. ``None``
        (default) disables the regularizer entirely (the loss is bit-identical
        to the un-regularized version). The regularizer penalizes parasitic
        flux, i.e. configurations where one source acquires flux in pixels
        that another source dominates. See :py:class:`~scarlet2.Responsibility`
        for the available options.
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
        contains ``{"losses", "responsibility", "pair_similarity", "best_loss", "n_iter"}``.
        The ``responsibility`` and ``pair_similarity`` arrays trace each
        regularizer's per-iteration contribution (zero when the regularizer is
        disabled).
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

    # default: regularizers off (weight=0.0 short-circuits to a no-op)
    resp_cfg = responsibility if responsibility is not None else Responsibility(weight=0.0)
    pair_cfg = pair_similarity if pair_similarity is not None else PairSimilarity(weight=0.0)

    # Calibrate pair-similarity: user-facing `weight` is relative to the initial NLL.
    # We compute |NLL_init| and R_init (unweighted) once, then replace the cfg with
    # one carrying the absolute multiplier. Skip when weight=0 (no-op) or R_init==0.
    if pair_cfg.weight != 0.0:
        values_init = _constraint_replace(values, params)
        scene_init = scene.set(values_init)
        nll_init = -sum(obs.set(values_init).log_likelihood(scene_init()) for obs in observations)
        r_init = _pair_similarity_from_stack(
            jnp.stack([scene_init.evaluate_source(s) for s in scene_init.sources], axis=0),
            pair_cfg,
        )
        scale = jnp.where(r_init > 0, jnp.abs(nll_init) / (r_init + pair_cfg.eps), 0.0)
        pair_cfg = PairSimilarity(weight=float(pair_cfg.weight * scale), eps=pair_cfg.eps)

    # initialize best-fit tracker (loss is minimized, so start at +inf).
    # Use an explicit dtype so the array is strongly-typed: jnp.where()'s output
    # in iter 1 is strongly-typed, and JAX caches weak vs. strong separately —
    # passing a weak inf would force a second compile on iter 2.
    best_loss = jnp.array(jnp.inf, dtype=jnp.float32)
    best_values = values
    losses = []
    resp_history = []
    pair_history = []

    with trange(max_iter, disable=not progress_bar) as t:
        for step in t:  # noqa: B007
            # optimizer step
            values, loss, resp, pair, opt_state, convergence, best_loss, best_values = _make_step(
                values,
                params,
                scene,
                observations,
                optim,
                opt_state,
                steps,
                best_loss,
                best_values,
                resp_cfg,
                pair_cfg,
            )
            losses.append(loss)
            resp_history.append(resp)
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
        "losses": jnp.stack(losses),
        "responsibility": jnp.stack(resp_history),
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
    values, params, scene, observations, optim, opt_state, steps, best_loss, best_values, resp_cfg, pair_cfg
):
    def loss_fn(values):
        # parameters now obey constraints
        # transformation happens in the grad path, so gradients are wrt to unconstrained variables
        # likelihood and prior grads transparently apply the Jacobians of these transformations
        values_ = _constraint_replace(values, params)
        scene_ = scene.set(values_)
        log_like = sum(obs.set(values_).log_likelihood(scene_()) for obs in observations)

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

        # responsibility regularizer (no-op when resp_cfg.weight == 0.0)
        resp = _responsibility_penalty(scene_, resp_cfg)
        # pair-similarity regularizer (no-op when pair_cfg.weight == 0.0)
        pair = _pair_similarity_penalty(scene_, pair_cfg)

        # has_aux=True: return penalty values alongside the loss so they can be
        # tracked in fit_info without a second forward pass
        return -(log_like + log_prior) + resp + pair, (resp, pair)

    (loss, (resp, pair)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(values)
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

    return values_, loss, resp, pair, opt_state, convergence, best_loss, best_values


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
