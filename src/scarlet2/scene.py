from collections import defaultdict

import equinox as eqx
import jax
import jax.numpy as jnp

from . import Scenery
from .bbox import overlap_slices
from .frame import Frame
from .module import Module, Parameters
from .renderer import ChannelRenderer
from .validation_utils import (
    ValidationError,
    ValidationInfo,
    ValidationMethodCollector,
    ValidationResult,
    ValidationWarning,
    print_validation_results,
)


class Scene(Module):
    """Model of the celestial scene

    This class connects the main functionality of `scarlet2`: the fitting of an
    :py:class:`~scarlet2.Observation` (or several) by a :py:class:`~scarlet2.Source`
    model (or several). Model parameters can be optimized or samples with any method
    implemented in jax, but this class provides the :py:func:`fit` and
    :py:func:`sample` methods as built-in solutions.
    """

    frame: Frame
    """Portion of the sky represented by this model"""
    sources: list
    """List of :py:class:`~scarlet2.Source` comprised in this model"""

    def __init__(self, frame):
        """
        Parameters
        ----------
        frame: `Frame`
            Portion of the sky represented by this model

        Examples
        --------
        The class provides a context so that sources can be added to the same model frame:

        >>> with Scene(model_frame) as scene:
        >>>    Source(center, spectrum, morphology)

        This adds a single source to the list :py:attr:`~scarlet2.Scene.sources`
        of `scene`. The context provides a common definition of the model frame,
        so that, e.g., `center` can be given as :py:class:`astropy.coordinates.SkyCoord`
        and will automatically be converted to the pixel coordinate in the model frame.

        The constructed source does not go out of scope after the `with` context
        is closed, it is stored in the scene.

        See Also
        --------
        :py:class:`~scarlet2.Scenery`, :py:class:`~scarlet2.Source`
        """
        self.frame = frame
        self.sources = list()

    def __call__(self):
        """What to run when the scene is called"""
        model = jnp.zeros(self.frame.bbox.shape)
        for source in self.sources:
            model += self.evaluate_source(source)
        return model

    def evaluate_source(self, source):
        """Evaluate a single source in the frame of this scene.

        This method inserts the model of `source` into the proper location in `scene`.

        Parameters
        ----------
        source: :py:class:`~scarlet2.Source`
            The source to evaluate.

        Returns
        -------
        array
            Array of the dimension indicated by :py:attr:`shape`.
        """
        model_ = source()
        # cut out region from model, add single source model
        bbox, bbox_ = overlap_slices(self.frame.bbox, source.bbox, return_boxes=True)
        sub_model_ = jax.lax.dynamic_slice(model_, bbox_.start, bbox_.shape)

        # add model_ back in full model
        model = jnp.zeros(self.frame.bbox.shape)
        model = jax.lax.dynamic_update_slice(model, sub_model_, bbox.start)
        return model

    def __enter__(self):
        # context manager to register sources
        # purpose is to provide scene.frame to source inits that will need some of its information
        # also allows us to append the sources automatically to the scene
        Scenery.scene = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Scenery.scene = None

    def sample(
        self, observations, parameters, seed=0, num_warmup=100, num_samples=200, progress_bar=True, **kwargs
    ):
        """Sample `parameters` of every source in the scene to get posteriors given `observations`.

        This method runs the HMC NUTS sampler from `numpyro` to get parameter
        posteriors. It uses the likelihood of `observations` as well as the `prior`
        attribute set for every :py:class:`~scarlet2.Parameter` in `parameters`.

        Parameters
        ----------
        observations: :py:class:`~scarlet2.Observation` or list
            The observations to fit the models to.
        parameters: :py:class:`~scarlet2.Parameters`
            Parameters to sample. This method will ignore all parameters that are not in this list.
            Every parameter in the list needs to have the attribute `prior` set.
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
        # uses numpyro NUTS on all non-fixed parameters
        # requires that those have priors set
        try:
            import numpyro
            import numpyro.distributions as dist
            import numpyro.distributions.constraints as constraints
            from numpyro.infer import MCMC, NUTS
        except ImportError as err:
            raise ImportError("scarlet2.Scene.sample() requires numpyro.") from err

        # making sure we can iterate
        if not isinstance(observations, (list, tuple)):
            observations = (observations,)

        # helper class to turn observation likelihood(s) into numpyro distribution
        class ObsDistribution(dist.Distribution):
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

        # find all non-fixed parameters and their priors
        priors = {p.name: p.prior for p in parameters}
        has_none = any(prior is None for prior in priors.values())
        if has_none:
            from pprint import pformat

            msg = f"All parameters need to have priors set. Got:\n{pformat(priors)}"
            raise AttributeError(msg)

        # define the pyro model, where every parameter becomes a sample,
        # and the observations sample from their likelihood given the rendered model
        def pyro_model(model):
            samples = tuple(numpyro.sample(p.name, p.prior) for p in parameters)
            model_ = model.replace(parameters, samples)
            pred = model_()  # create prediction once for all observations
            # dealing with multiple observations
            for i, obs_ in enumerate(observations):
                numpyro.sample(f"obs.{i}", ObsDistribution(obs_, pred), obs=obs_.data)

        from numpyro.infer import MCMC, NUTS

        # use init from current value of model
        try:
            init_strategy = kwargs.pop("init_strategy")
        except KeyError:
            from functools import partial

            from numpyro.infer.initialization import init_to_value

            values = {p.name: p.node for p in parameters}
            init_strategy = partial(init_to_value, values=values)

        nuts_kernel = NUTS(pyro_model, init_strategy=init_strategy, **kwargs)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=progress_bar)
        rng_key = jax.random.PRNGKey(seed)
        mcmc.run(rng_key, self)
        return mcmc

    def fit(
        self,
        observations,
        parameters,
        schedule=None,
        max_iter=100,
        e_rel=1e-4,
        progress_bar=True,
        callback=None,
        **kwargs,
    ):
        """Fit model `parameters` of every source in the scene to match `observations`.

        Computes the best-fit parameters of all components in every source by
        first-order gradient descent with the Adam optimizer from `optax`.

        Parameters
        ----------
        observations: :py:class:`~scarlet2.Observation` or list
            The observations to fit the model to.
        parameters: :py:class:`~scarlet2.Parameters`
            Parameters to optimize. This method will ignore all parameters that are not in this list.
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
        Scene
            The scene model with updated parameters
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
        assert isinstance(parameters, Parameters)

        # make a stepsize tree
        where = lambda model: model.get(parameters)
        replace = tuple(p.stepsize for p in parameters)
        steps = eqx.tree_at(where, self, replace=replace)

        def scale_by_stepsize() -> base.GradientTransformation:
            # adapted from optax.scale_by_param_block_norm()
            def init_fn(params):
                del params
                return base.EmptyState()

            def update_fn(updates, state, params):
                if params is None:
                    raise ValueError(base.NO_PARAMS_MSG)
                updates = jax.tree_util.tree_map(
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
        scene = _constraint_replace(self, parameters, inv=True)

        # get optimizer initialized with the optimization parameters
        filter_spec = self.get_filter_spec(parameters)
        opt_state = optim.init(scene) if filter_spec is None else optim.init(eqx.filter(scene, filter_spec))

        with trange(max_iter, disable=not progress_bar) as t:
            for step in t:  # noqa: B007
                # optimizer step
                scene, loss, opt_state, convergence = _make_step(
                    scene, observations, parameters, optim, opt_state, filter_spec=filter_spec
                )

                # compute max change across all non-fixed parameters for convergence test
                max_change = jax.tree_util.tree_reduce(lambda a, b: max(a, b), convergence)

                # report current iteration results to callback
                if callback is not None:
                    scene_ = _constraint_replace(scene, parameters)
                    callback(scene_, convergence, loss)

                # Log the loss and max_change in the tqdm progress bar
                t.set_postfix(loss=f"{loss:08.2f}", max_change=f"{max_change:1.6f}")

                # test convergence
                if max_change < e_rel:
                    break

        returned_scene = _constraint_replace(scene, parameters)  # transform back to constrained variables

        # (re)-import `VALIDATION_SWITCH` at runtime to avoid using a static/old value
        from .validation_utils import VALIDATION_SWITCH

        if VALIDATION_SWITCH:
            from .validation import check_fit

            for i, obs in enumerate(observations):
                print(f"Running validation checks on the fit of the scene for observation {i}.")
                validation_results = check_fit(returned_scene, obs)
                print_validation_results("Fit validation results", validation_results)

        return returned_scene

    def set_spectra_to_match(self, observations, parameters):
        """Sets the spectra of every source in the scene to match the observations

        Computes the best-fit amplitude of the rendered model of all components in every
        channel of every observation as a linear inverse problem.

        Parameters
        ----------
        observations: :py:class:`~scarlet2.Observation` or list
            The observations used to set the spectra.
        parameters: :py:class:`~scarlet2.Parameters`
            Parameters to adjust. This method will ignore all spectrum parameters that are not arrays
            or not included in this list.

        Returns
        -------
        None
        """

        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # extract multi-channel model for every source
        spectrum_parameters = []
        models = []
        for i, src in enumerate(self.sources):
            # search for spectrum in parameters: only works for standard arrays
            if isinstance(src.spectrum, jnp.ndarray):
                for p in parameters:
                    if p.node is src.spectrum:
                        spectrum_parameters.append(i)
                        # update source to have flat spectrum
                        src = eqx.tree_at(lambda src: src.spectrum, src, jnp.ones_like(p.node))
                        break

            # evaluate the model for any source so that fit includes it even if its spectrum is not updated
            model = self.evaluate_source(src)  # assumes all sources are single components

            # check for models with identical initializations, see scarlet repo issue #282
            # if duplicate: raise ValueError
            for model_indx in range(len(models)):
                if jnp.allclose(model, models[model_indx]):
                    message = f"Source {i} has a model identical to source {model_indx}.\n"
                    message += "This is likely not intended, and the second source should be deleted."
                    raise ValueError(message)
            models.append(model)

        models = jnp.array(models)
        num_models = len(models)

        for obs in observations:
            # independent channels, no mixing
            # solve the linear inverse problem of the amplitudes in every channel
            # given all the rendered morphologies
            # spectrum = (M^T Sigma^-1 M)^-1 M^T Sigma^-1 * im
            num_channels = obs.frame.C
            images = obs.data
            weights = obs.weights
            morphs = jnp.stack([obs.render(model) for model in models], axis=0)
            spectra = jnp.zeros((num_models, num_channels))
            for c in range(num_channels):
                im = images[c].reshape(-1)
                w = weights[c].reshape(-1)
                m = morphs[:, c, :, :].reshape(num_models, -1)
                mw = m * w[None, :]
                # check if all components have nonzero flux in c.
                # because of convolutions, flux can be outside the box,
                # so we need to compare weighted flux with unweighted flux,
                # which is the same (up to a constant) for constant weights.
                # so we check if *most* of the flux is from pixels with non-zero weight
                nonzero = jnp.sum(mw, axis=1) / jnp.sum(m, axis=1) / jnp.mean(w) > 0.1
                nonzero = jnp.flatnonzero(nonzero)
                if len(nonzero) == num_models:
                    covar = jnp.linalg.inv(mw @ m.T)
                    spectra = spectra.at[:, c].set(covar @ m @ (im * w))
                else:
                    covar = jnp.linalg.inv(mw[nonzero] @ m[nonzero].T)
                    spectra = spectra.at[nonzero, c].set(covar @ m[nonzero] @ (im * w))

            # update the parameters with the best-fit spectrum solution
            channel_map = ChannelRenderer(self.frame, obs.frame).channel_map
            if channel_map is not None:
                channel_map.get_slice()
            noise_bg = 1 / jnp.median(jnp.sqrt(obs.weights), axis=(-2, -1))
            for i in spectrum_parameters:
                src_ = self.sources[i]
                # faint galaxy can have erratic solution, bound from below by noise_bg
                v = src_.spectrum.at[channel_map].set(jnp.maximum(spectra[i], noise_bg))
                self.sources[i] = eqx.tree_at(lambda src: src.spectrum, src_, v)


def _constraint_replace(self, parameters, inv=False):
    # replace any parameter with constraint into unconstrained ones by calling its constraint bijector
    # return transformed pytree
    where_in = lambda model: model.get(parameters)
    param_values = where_in(self)
    if not inv:
        replace = tuple(
            p.constraint_transform(v) if p.constraint is not None else v
            for p, v in zip(parameters, param_values, strict=False)
        )
    else:
        replace = tuple(
            p.constraint_transform.inv(v) if p.constraint is not None else v
            for p, v in zip(parameters, param_values, strict=False)
        )

    return eqx.tree_at(where_in, self, replace=replace)


# update step for optax optimizer
@eqx.filter_jit
def _make_step(model, observations, parameters, optim, opt_state, filter_spec=None):
    from .nn import ScorePrior, pad_fwd

    def loss_fn(model):
        if any(param.constraint is not None for param in parameters):
            # parameters now obey constraints
            # transformation happens in the grad path, so gradients are wrt to unconstrained variables
            # likelihood and prior grads transparently apply the Jacobians of these transformations
            model = _constraint_replace(model, parameters)

        pred = model()
        log_like = sum(obs.log_likelihood(pred) for obs in observations)

        param_values = model.get(parameters)

        log_prior = 0

        # Gather parameters with the same ScorePrior for parallel evaluation
        grouped = defaultdict(list)
        for param, value in zip(parameters, param_values, strict=False):
            if isinstance(param.prior, ScorePrior):
                grouped[param.prior].append(pad_fwd(value, param.prior._model.shape)[0])
            elif param.prior is not None:
                log_prior += param.prior.log_prob(value)

        if len(grouped) > 0:
            log_prior += sum(
                sum(
                    jax.vmap(prior.log_prob)(jnp.stack(arr_list, axis=0))
                    for prior, arr_list in grouped.items()
                )
            )

        return -(log_like + log_prior)

    if filter_spec is None:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    else:

        @eqx.filter_value_and_grad
        def filtered_loss_fn(diff_model, static_model):
            model = eqx.combine(diff_model, static_model)
            return loss_fn(model)

        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = filtered_loss_fn(diff_model, static_model)

    updates, opt_state = optim.update(grads, opt_state, model)
    model_ = eqx.apply_updates(model, updates)

    # for convergence criterion: compute norms of parameters and updates
    norm = lambda x, dx: 0 if dx is None else jnp.linalg.norm(dx) / jnp.linalg.norm(x)
    convergence = jax.tree_util.tree_map(lambda x, dx: norm(x, dx), *(model, updates))

    return model_, loss, opt_state, convergence


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
        class's `goodness_of_fit` method.

        For a Gaussian noise model, the gof is defined as the averaged squared deviation of the model from the
        data, scaled by the variance of the data, aka mean chi squared
        :math:`\frac{1}{N}\\sum_i=1^N w_i (m_i - d_i)^2` with inverse variance weights :math:`w_i`.

        Up to a normalization, the gof is identical to :py:class:`~scarlet2.Observation.log-likelihood`.

        Parameters
        ----------
        model: array
            The (pre-rendered) predicted data cube, typically from evaluating :py:class:`~scarlet2.Scene`

        Returns
        -------
        float
        """
        obs = self.observation

        chi2 = obs.goodness_of_fit(self.scene())

        ret_val: ValidationResult = ValidationInfo("The model fit is good.", check=self.__class__.__name__)
        if self.chi2_tolerable_threshold <= chi2 < self.chi2_critical_threshold:
            ret_val = ValidationWarning(
                "The model fit is acceptable, but the goodness of fit is not optimal.",
                check=self.__class__.__name__,
                context={"chi2": chi2},
            )
        elif chi2 >= self.chi2_critical_threshold:
            ret_val = ValidationError(
                "The model fit is poor.", check=self.__class__.__name__, context={"chi2": chi2}
            )

        return ret_val

    def check_chi_square_in_box_and_border(self) -> list[ValidationResult]:
        """Evaluate the weighted mean (weighted by the inverse variance weights)
        of the squared residuals for each source. Chi square is also computed for
        the perimeter outside the box of with `border_width`.

        Returns
        -------
        None or ValidationError
            If the chi-square is above the tolerable threshold, returns a ValidationError.
            Otherwise, returns None.
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


class SceneValidator(metaclass=ValidationMethodCollector):
    """A class containing all of the validation checks for a Scene object.
    A convenience function exists that will run all the checks in this class and
    returns a list of validation results:

    :py:func:`~scarlet2.validation.check_scene`.
    """

    def __init__(self, scene: Scene, parameters: Parameters, observation):
        """Initialize the SceneValidator.

        Parameters
        ----------
        scene : Scene
            The scene object to validate.
        parameters : Parameters
            The parameters of the scene to validate.
        observation : Observation
            The observation object containing the data to validate against.
        """
        self.scene = scene
        self.parameters = parameters
        self.observation = observation

    def check_source_boxes_comparable_to_observation(self) -> list[ValidationResult]:
        """Check that the bounding boxes of all sources are comparable to the observation.

        This checks that the bounding boxes of all sources are contained within the
        observation's bounding box.

        Returns
        -------
        list[ValidationResult]
            A list of validation results, which can be either `ValidationInfo` or
            `ValidationWarning`.
        """
        validation_results: list[ValidationResult] = []
        for i, source in enumerate(self.scene.sources):
            source_and_observation_box = source.bbox & self.observation.frame.bbox
            if source_and_observation_box == source.bbox:
                validation_results.append(
                    ValidationInfo(
                        f"Source {i} has a bounding box inside the observation.",
                        check=self.__class__.__name__,
                        context={"bbox": source.bbox, "observation_bbox": self.observation.frame.bbox},
                    )
                )
            elif source_and_observation_box.spatial.get_extent() == [0, 0, 0, 0]:
                validation_results.append(
                    ValidationWarning(
                        f"Source {i} has a bounding box outside of the observation.",
                        check=self.__class__.__name__,
                        context={"bbox": source.bbox, "observation_bbox": self.observation.frame.bbox},
                    )
                )

            else:
                validation_results.append(
                    ValidationWarning(
                        f"Source {i} has a bounding box that is partially inside the observation.",
                        check=self.__class__.__name__,
                        context={"bbox": source.bbox, "observation_bbox": self.observation.frame.bbox},
                    )
                )

        return validation_results

    def check_parameters_have_necessary_fields(self) -> list[ValidationResult]:
        """Check that all parameters in the scene have the necessary fields set.

        This checks that all parameters in the scene have the `prior` or `stepsize`
        attributes set, but not both.

        Returns
        -------
        list[ValidationResult]
            A list of validation results, which can be either `ValidationInfo`
            or `ValidationError`.
        """
        validation_results: list[ValidationResult] = []
        for i in range(len(self.parameters)):
            param = self.parameters[i]
            prior_is_none = param.prior is None
            stepsize_is_none = param.stepsize is None
            if (prior_is_none and not stepsize_is_none) or (not prior_is_none and stepsize_is_none):
                validation_results.append(
                    ValidationInfo(
                        f"Parameter {param.name} has prior or stepsize, but not both set.",
                        check=self.__class__.__name__,
                        context={
                            "parameter.name": param.name,
                            "parameter.prior": param.prior,
                            "parameter.stepsize": param.stepsize,
                        },
                    )
                )
            if prior_is_none and stepsize_is_none:
                validation_results.append(
                    ValidationError(
                        f"Parameter {param.name} does not have prior or stepsize set.",
                        check=self.__class__.__name__,
                        context={
                            "parameter.name": param.name,
                            "parameter.prior": param.prior,
                            "parameter.stepsize": param.stepsize,
                        },
                    )
                )

        return validation_results
