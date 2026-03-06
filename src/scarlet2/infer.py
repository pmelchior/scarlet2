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


def _dict_to_eqx_module(name: str, data: dict):
    # Build field annotations dynamically
    annotations = {k: type(v) for k, v in data.items()}

    # Create a class that inherits from eqx.Module
    cls = type(name, (eqx.Module,), {"__annotations__": annotations})

    # Instantiate with the dict values
    return cls(**data)


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
    **kwargs,
):
    """Fit model `parameters` of every source in `scene` to match `observations`.

    Computes the best-fit parameters of all components in every source by
    first-order gradient descent with the Adam optimizer from `optax`.

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
    **kwargs: dict, optional
        Additional keyword arguments passed to the `optax.scale_by_adam` optimizer.

    Notes
    -----
    Requires `optax`

    Returns
    -------
    Scene, list(Observation)
        The scene and observation(s) with updated parameters
    """
    try:
        import optax
        import optax._src.base as base
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

    def scale_by_stepsize() -> base.GradientTransformation:
        # adapted from optax.scale_by_param_block_norm()
        def init_fn(params):
            del params
            return base.EmptyState()

        def update_fn(updates, state, params):
            if params is None:
                raise ValueError(base.NO_PARAMS_MSG)
            updates = jt.map(
                # minus because we want gradient descent
                lambda u, s, p: None if u is None else -s * u if not callable(s) else -s(p) * u,
                updates,
                steps,
                params,
                is_leaf=lambda x: x is None,
            )
            return updates, state

        return base.GradientTransformation(init_fn, update_fn)

    # run adam, followed by stepsize adjustments
    optim = optax.chain(
        optax.scale_by_adam(**kwargs),
        optax.scale_by_schedule(schedule if callable(schedule) else lambda x: 1),
        scale_by_stepsize(),
    )

    # transform to unconstrained parameters
    values = _constraint_replace(values, params, inv=True)
    opt_state = optim.init(values)

    with trange(max_iter, disable=not progress_bar) as t:
        for step in t:  # noqa: B007
            # optimizer step
            values, loss, opt_state, convergence = _make_step(
                values, params, scene, observations, optim, opt_state
            )

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

    # transform back to constrained variables and replace in scene
    values = _constraint_replace(values, params)
    # scene_ is a copy, but its registry_key still points to scene.parameters, can thus be reused
    scene_ = scene.set(values)
    obs_ = tuple(obs.set(values) for obs in observations)

    # (re)-import `VALIDATION_SWITCH` at runtime to avoid using a static/old value
    from .validation_utils import VALIDATION_SWITCH

    if VALIDATION_SWITCH:
        from .validation import check_fit

        for obs in observations:
            print(f"Running validation checks on the fit of the scene for observation {obs.name}.")
            validation_results = check_fit(scene_, obs)
            print_validation_results("Fit validation results", validation_results)

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


# update step for optax optimizer
@eqx.filter_jit
def _make_step(values, params, scene, observations, optim, opt_state):
    def loss_fn(values):
        # parameters now obey constraints
        # transformation happens in the grad path, so gradients are wrt to unconstrained variables
        # likelihood and prior grads transparently apply the Jacobians of these transformations
        values_ = _constraint_replace(values, params)
        scene_ = scene.set(values_)()
        log_like = sum(obs.set(values_).log_likelihood(scene_) for obs in observations)

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

        return -(log_like + log_prior)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(values)
    updates, opt_state = optim.update(grads, opt_state, values)
    values_ = eqx.apply_updates(values, updates)

    # for convergence criterion: compute norms of parameters and updates
    norm = lambda x, dx: 0 if dx is None else jnp.linalg.norm(dx) / jnp.linalg.norm(x)
    convergence = jt.map(lambda x, dx: norm(x, dx), values, updates)

    return values_, loss, opt_state, convergence


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
