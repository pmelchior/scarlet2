import equinox as eqx
import jax
import jax.numpy as jnp

from . import Scenery
from .bbox import overlap_slices
from .frame import Frame
from .module import Module, Parameters
from .renderer import ChannelRenderer
from .spectrum import ArraySpectrum


class Scene(Module):
    frame: Frame = eqx.field(static=True)
    sources: list

    def __init__(self, frame):
        self.frame = frame
        self.sources = list()

    def __call__(self):
        model = jnp.zeros(self.frame.bbox.shape)
        for source in self.sources:
            model += self._eval_src_in_frame(source)
        return model

    def _eval_src_in_frame(self, source):
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

    def sample(self, observations, parameters, seed=0, num_warmup=100, num_samples=200, progress_bar=True, **kwargs):
        # uses numpyro NUTS on all non-fixed parameters
        # requires that those have priors set
        try:
            import numpyro
            import numpyro.distributions as dist
            import numpyro.distributions.constraints as constraints
            from numpyro.infer import MCMC, NUTS
        except ImportError:
            raise ImportError("scarlet2.Scene.sample() requires numpyro.")

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
        nuts_kernel = NUTS(pyro_model, **kwargs)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=progress_bar)
        rng_key = jax.random.PRNGKey(seed)
        mcmc.run(rng_key, self)
        return mcmc

    def fit(self, observations, parameters, schedule=None, max_iter=100, e_rel=1e-4, progress_bar=True, callback=None,
            **kwargs):
        # optax fit with adam optimizer
        # Transforms constrained parameters into unconstrained ones
        # and filters out fixed parameters
        # TODO: check alternative optimizers
        try:
            import tqdm
            import optax
            import optax._src.base as base
            from numpyro.distributions.transforms import biject_to
        except ImportError:
            raise ImportError("scarlet2.Scene.fit() requires optax and numpyro.")

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
                    # lambda u, step, param: -step * u if not callable(step) else -step(param,niter) * u,
                    lambda u, s, p: -s * u if not callable(s) else -s(p) * u,
                    # minus because we want gradient descent
                    updates, steps, params)
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
        if filter_spec is None:
            opt_state = optim.init(scene)
        else:
            opt_state = optim.init(eqx.filter(scene, filter_spec))

        with tqdm.trange(max_iter, disable=not progress_bar) as t:
            for step in t:
                # optimizer step
                scene, loss, opt_state, convergence = _make_step(scene, observations, parameters, optim, opt_state,
                                                                 filter_spec=filter_spec)

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

        return _constraint_replace(scene, parameters)  # transform back to constrained variables

    def set_spectra_to_match(self, observations, parameters):
        """Sets the spectra of every source in the scene to match the observations.

        Computes the best-fit amplitude of the rendered model of all components in every
        channel of every observation as a linear inverse problem.

        Parameters
        ----------
        observations: `scarlet2.Observation` or list thereof
        """

        if not hasattr(observations, "__iter__"):
            observations = (observations,)

        # extract multi-channel model for every source
        spectrum_parameters = []
        models = []
        for i, src in enumerate(self.sources):
            if isinstance(src.spectrum, ArraySpectrum):
                # search for spectrum.data in parameters
                for p in parameters:
                    if p.node is src.spectrum.data:
                        spectrum_parameters.append(i)
                        # update source to have flat spectrum
                        src = eqx.tree_at(lambda src: src.spectrum.data, src, jnp.ones_like(p.node))
                        break

            # evaluate the model for any source so that fit includes it even if its spectrum is not updated
            model = self._eval_src_in_frame(src)  # assumes all sources are single components

            # check for models with identical initializations, see scarlet repo issue #282
            # if duplicate: raise ValueError
            for l in range(len(models)):
                if jnp.allclose(model, models[l]):
                    message = f"Source {i} has a model identical to source {l}.\n"
                    message += "This is likely not intended, and the second source should be deleted."
                    raise ValueError(message)
            models.append(model)

        models = jnp.array(models)
        K = len(models)

        for obs in observations:
            # independent channels, no mixing
            # solve the linear inverse problem of the amplitudes in every channel
            # given all the rendered morphologies
            # spectrum = (M^T Sigma^-1 M)^-1 M^T Sigma^-1 * im
            C = obs.frame.C
            images = obs.data
            weights = obs.weights
            morphs = jnp.stack([obs.render(model) for model in models], axis=0)
            spectra = jnp.zeros((K, C))
            for c in range(C):
                im = images[c].reshape(-1)
                w = weights[c].reshape(-1)
                m = morphs[:, c, :, :].reshape(K, -1)
                mw = m * w[None, :]
                # check if all components have nonzero flux in c.
                # because of convolutions, flux can be outside the box,
                # so we need to compare weighted flux with unweighted flux,
                # which is the same (up to a constant) for constant weights.
                # so we check if *most* of the flux is from pixels with non-zero weight
                nonzero = jnp.sum(mw, axis=1) / jnp.sum(m, axis=1) / jnp.mean(w) > 0.1
                nonzero = jnp.flatnonzero(nonzero)
                if len(nonzero) == K:
                    covar = jnp.linalg.inv(mw @ m.T)
                    spectra = spectra.at[:, c].set(covar @ m @ (im * w))
                else:
                    covar = jnp.linalg.inv(mw[nonzero] @ m[nonzero].T)
                    spectra = spectra.at[nonzero, c].set(covar @ m[nonzero] @ (im * w))

            # update the parameters with the best-fit spectrum solution
            channel_map = ChannelRenderer(self.frame, obs.frame).channel_map
            noise_bg = 1 / jnp.median(jnp.sqrt(obs.weights), axis=(-2, -1))
            for i in spectrum_parameters:
                src_ = self.sources[i]
                # faint galaxy can have erratic solution, bound from below by noise_bg
                v = src_.spectrum.data.at[channel_map].set(jnp.maximum(spectra[i], noise_bg))
                self.sources[i] = eqx.tree_at(lambda src: src.spectrum.data, src_, v)

def _constraint_replace(self, parameters, inv=False):
    # replace any parameter with constraint into unconstrained ones by calling its constraint bijector
    # return transformed pytree
    where_in = lambda model: model.get(parameters)
    param_values = where_in(self)
    if not inv:
        replace = tuple(
            p.constraint_transform(v) if p.constraint is not None else v for p, v in zip(parameters, param_values))
    else:
        replace = tuple(
            p.constraint_transform.inv(v) if p.constraint is not None else v for p, v in zip(parameters, param_values))

    return eqx.tree_at(where_in, self, replace=replace)

# update step for optax optimizer
@eqx.filter_jit
def _make_step(model, observations, parameters, optim, opt_state, filter_spec=None):
    def loss_fn(model):
        if any(param.constraint is not None for param in parameters):
            # parameters now obey constraints
            # transformation happens in the grad path, so gradients are wrt to unconstrained variables
            # likelihood and prior grads transparently apply the Jacobians of these transformations
            model = _constraint_replace(model, parameters)

        pred = model()
        log_like = sum(obs.log_likelihood(pred) for obs in observations)

        param_values = model.get(parameters)
        log_prior = sum(param.prior.log_prob(value)
                        for param, value in zip(parameters, param_values)
                        if param.prior is not None
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
