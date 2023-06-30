import equinox as eqx
import jax
import jax.numpy as jnp

from .bbox import overlap_slices
from .frame import Frame
from .module import Module


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

    def fit(self, observations, max_iter=100, verbose=False):
        try:
            import optax
        except ImportError:
            raise ImportError("scarlet2.Scene.fit() requires optax. Please install optax.")

        optim, scene_, obs_, filter_spec, constraint_fn, opt_state = _init_optim(self, observations, learning_rate=1e-2)
        for step in range(max_iter):
            scene_, loss, opt_state = _make_step(scene_, obs_, optim, opt_state, filter_spec=filter_spec,
                                                 constraint_fn=constraint_fn)
            if verbose:
                print(f"step={step}, loss={loss.item()}")

        return _constraint_replace(scene_, constraint_fn)


# provide default fitter if optax is present
try:
    import optax
    from numpyro.distributions.transforms import biject_to


    def _constraint_replace(self, constraint_fn, inv=False):
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


    @eqx.filter_jit
    def _make_step(model, observations, optim, opt_state, filter_spec=None, constraint_fn=None):
        parameters = model.get_parameters(return_info=True)

        def loss_fn(model):
            if constraint_fn is not None:
                # parameters now obey constraints
                model = _constraint_replace(model, constraint_fn)

            pred = model()
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

        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, loss, opt_state


    def _init_optim(self, obs, optim=None, **kwargs):
        # dealing with multiple observations
        if not isinstance(obs, (list, tuple)):
            obs_ = (obs,)
        else:
            obs_ = obs

        # set up optimimzer
        # TODO: different step sizes for optimizer
        # see https://github.com/deepmind/optax/blob/7063ce8022ce6165926de369d6fefa207e318127/optax/_src/combine.py#L65
        if optim is None:
            optim = optax.adam(**kwargs)

        # transform to unconstrained parameters
        parameters = self.get_parameters(return_info=True)
        constraint_fn = {name: biject_to(info["constraint"]) for name, (value, info) in parameters.items() if
                         info["constraint"] is not None}
        model_ = _constraint_replace(self, constraint_fn, inv=True)

        # get optimizer set up with the filtered (and unconstrained) parameters
        filter_spec = self.filter_spec
        if filter_spec is None:
            opt_state = optim.init(model_)
        else:
            opt_state = optim.init(eqx.filter(model_, filter_spec))

        return optim, model_, obs_, filter_spec, constraint_fn, opt_state

except ImportError:
    pass
