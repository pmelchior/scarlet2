import equinox as eqx
import jax
import jax.numpy as jnp

from .bbox import overlap_slices
from .frame import Frame
from .module import Module, rgetattr


class Scenery:
    # static store for context manager
    scene = None

class Scene(Module):
    frame: Frame = eqx.field(static=True)
    sources: list

    def __init__(self, frame):
        self.frame = frame
        self.sources = list()
        super().__post_init__()

    def __call__(self):
        model = jnp.zeros(self.frame.bbox.shape)
        for source in self.sources:
            model_ = source()

            # cut out region from model, add single source model
            bbox, bbox_ = overlap_slices(self.frame.bbox, source.bbox, return_boxes=True)
            sub_model = jax.lax.dynamic_slice(model, bbox.start, bbox.shape)
            sub_model_ = jax.lax.dynamic_slice(model_, bbox_.start, bbox_.shape)
            sub_model += sub_model_

            # add model_ back in full model
            model = jax.lax.dynamic_update_slice(model, sub_model, bbox.start)
        return model

    def __enter__(self):
        # context manager to register sources
        # purpose is to provide scene.frame to source inits that will need some of its information
        # also allows us to append the sources automatically to the scene
        Scenery.scene = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Scenery.scene = None

    def sample(self, observations, **kwargs):
        # uses numpyro NUTS on all non-fixed parameters
        # requires that those have priors set
        try:
            import numpyro
            import numpyro.distributions as dist
            import numpyro.distributions.constraints as constraints
            from numpyro.infer import MCMC, NUTS
        except ImportError:
            raise ImportError("scarlet2.Scene.sample() requires numpyro.")

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
        parameters = self.get_parameters(return_info=True)
        priors = {name: info["prior"] for name, (p, info) in parameters.items()}
        has_none = any(prior is None for prior in priors.values())
        if has_none:
            from pprint import pformat
            msg = f"All parameters need to have priors set. Got:\n{pformat(priors)}"
            raise AttributeError(msg)

        # define the pyro model, where every parameter becomes a sample,
        # and the observations sample from their likelihood given the rendered model
        def pyro_model(model, obs=None):
            names = tuple(priors.keys())
            samples = tuple(numpyro.sample(name, prior) for name, prior in priors.items())
            model = model.replace(names, samples)
            pred = model()  # create prediction once for all observations
            # dealing with multiple observations
            if not isinstance(observations, (list, tuple)):
                numpyro.sample('obs', ObsDistribution(obs, pred), obs=obs.data)
            else:
                for i, obs_ in enumerate(obs):
                    numpyro.sample(f'obs.{i}', ObsDistribution(obs_, pred), obs=obs_.data)

        from numpyro.infer import MCMC, NUTS
        nuts_kernel = NUTS(pyro_model, dense_mass=True)
        mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(rng_key, self, obs=observations)
        return mcmc

    def fit(self, observations, max_iter=100, e_rel=1e-4, progress_bar=True, callback=None, **kwargs):
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

        # dealing with multiple observations
        if not isinstance(observations, (list, tuple)):
            observations = (observations,)
        else:
            observations = observations

        # get step sizes for each parameter
        parameters = self.get_parameters(return_info=True)
        stepsizes = {name: info["stepsize"] for name, (p, info) in parameters.items()}
        # make a stepsize tree
        where = lambda model: tuple(rgetattr(model, n) for n in stepsizes.keys())
        replace = tuple(stepsizes.values())
        steps = eqx.tree_at(where, self, replace=replace)
        it = 0

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
                    lambda u, s, p: -s * u if not callable(s) else -s(p, it) * u,
                    # minus because we want gradient descent
                    updates, steps, params)
                return updates, state

            return base.GradientTransformation(init_fn, update_fn)

        # run adam, followed by stepsize adjustments
        optim = optax.chain(
            optax.scale_by_adam(**kwargs),
            scale_by_stepsize(),
        )

        # transform to unconstrained parameters
        constraint_fn = {name: biject_to(info["constraint"]) for name, (value, info) in parameters.items() if
                         info["constraint"] is not None}
        scene = _constraint_replace(self, constraint_fn, inv=True)

        # get optimizer initialized with the filtered (=non-fixed) parameters
        filter_spec = self.filter_spec
        if filter_spec is None:
            opt_state = optim.init(scene)
        else:
            opt_state = optim.init(eqx.filter(scene, filter_spec))

        with tqdm.trange(max_iter, disable=not progress_bar) as t:
            for step in t:
                # optimizer step
                scene_, loss, opt_state = _make_step(scene, observations, optim, opt_state, filter_spec=filter_spec,
                                                     constraint_fn=constraint_fn)
                # Log the loss in the tqdm progress bar
                t.set_postfix(loss=f"{loss:08.2f}")

                # report current iteration results to callback
                if callback is not None:
                    callback(scene_, loss)

                # terminate optimization if all parameter change less than e_rel
                if e_rel is not None:
                    crit = lambda x, x_: jnp.linalg.norm(x - x_) < e_rel * jnp.linalg.norm(x_)
                    converged = tuple(
                        crit(p, p_) for (p, p_) in zip(scene.parameters.values(), scene_.parameters.values()))
                    if all(converged):
                        break

                scene = scene_
                it += 1
        return _constraint_replace(scene_, constraint_fn)  # transform back to constrained variables


def _constraint_replace(self, constraint_fn, inv=False):
    # replace any parameter with constraints into unconstrained ones by calling constraint_fn
    # return transformed pytree
    parameters = self.get_parameters(return_info=True)
    names = tuple(name
                  for name, (value, info) in parameters.items()
                  if info["constraint"] is not None
                  )
    transform = lambda value, fn: fn(value)
    inv_transform = lambda value, fn: fn.inv(value)
    transform = (transform, inv_transform)
    values = tuple(transform[inv](value, constraint_fn[name])
                   for name, (value, info) in parameters.items()
                   if info["constraint"] is not None
                   )
    return self.replace(names, values)


# update step for optax optimizer
@eqx.filter_jit
def _make_step(model, observations, optim, opt_state, filter_spec=None, constraint_fn=None):

    def loss_fn(model):
        if constraint_fn is not None:
            # parameters now obey constraints
            model = _constraint_replace(model, constraint_fn)

        pred = model()
        parameters = model.get_parameters(return_info=True)
        log_like = sum(obs.log_likelihood(pred) for obs in observations)
        log_prior = sum(info["prior"].log_prob(p)
                        for name, (p, info) in parameters.items()
                        if info["prior"] is not None
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
    return model_, loss, opt_state
