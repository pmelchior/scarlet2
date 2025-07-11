# To be removed as part of issue #168
# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D103
# ruff: noqa: D106

"""Neural network priors"""
from functools import partial

try:
    import numpyro.distributions as dist
    import numpyro.distributions.constraints as constraints

except ImportError as err:
    raise ImportError("scarlet2.nn requires numpyro.") from err

import equinox as eqx
import jax.numpy as jnp
from jax import custom_vjp, vjp


def pad_fwd(x, model_shape):
    """Zero-pads the input image to the model size

    Parameters
    ----------
    x : jnp.array
        data to be padded
    model_shape : tuple
        shape of the prior model to be used

    Returns
    -------
    x : jnp.array
        data padded to same size as model_shape
    pad: tuple
        padding amount in every dimension
    """
    assert all(
        model_shape[d] >= x.shape[d] for d in range(x.ndim)
    ), "Model size must be larger than data size"
    if model_shape == x.shape:
        pad = 0
        return x, pad

    pad = tuple(
        # even padding
        (int(gap / 2), int(gap / 2))
        if (gap := model_shape[d] - x.shape[d]) % 2 == 0
        # uneven padding
        else (int(gap // 2), int(gap // 2) + 1)
        # over all dimensions
        for d in range(x.ndim)
    )
    # perform the zero-padding
    x = jnp.pad(x, pad, "constant", constant_values=0)
    return x, pad


# reverse pad back to original size
def pad_back(x, pad):
    """Removes the zero-padding from the input image

    Parameters
    ---------
    x : jnp.array
        padded data to same size as model_shape
    pad: tuple
        padding amount in every dimension

    Returns
    -------
    x : jnp.array
        data returned to it pre-pad shape
    """
    slices = tuple(slice(low, -hi) if hi > 0 else slice(low, None) for (low, hi) in pad)
    return x[slices]


# calculate score function (jacobian of log-probability)
def calc_grad(x, model):
    """Calculates the gradient of the log-prior
    using the ScoreNet model chosen

    Parameters
    ----------
    x:
        array of the data
    model:
        the model to calculate the score function

    Returns
    -------
    score_func : array of the score function
    """
    # cast to float32, expand to (batch, shape), and pad to match the shape of the score model
    x_, pad = pad_fwd(jnp.float32(x), model.shape)

    # run score model, expects (batch, shape)
    if jnp.ndim(x) == len(model.shape):
        x_ = jnp.expand_dims(x_, axis=0)
    score_func = model.func(x_)
    if jnp.ndim(x) == len(model.shape):
        score_func = jnp.squeeze(score_func, axis=0)

    # remove padding
    if pad != 0:
        score_func = pad_back(score_func, pad)
    return score_func


# jax gradient function to calculate jacobian
def vgrad(f, x):
    y, vjp_fn = vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]


# Here we define a custom vjp for the log_prob function
# such that for gradient calls in jax, the score prior
# is returned


@partial(custom_vjp, nondiff_argnums=(0,))
def _log_prob(model, x):
    return 0.0


def _log_prob_fwd(model, x):
    score_func = calc_grad(x, model)
    return 0.0, score_func  # cannot directly call log_prob in Class object


def _log_prob_bwd(model, res, g):
    score_func = res  # Get residuals computed in f_fwd
    return (g * score_func,)  # create the vector (g) jacobian (score_func) product


# register the custom vjp
_log_prob.defvjp(_log_prob_fwd, _log_prob_bwd)


class ScorePrior(dist.Distribution):
    class ScoreWrapper(eqx.Module):
        func: callable
        shape: tuple

    support = constraints.real_vector
    _model = ScoreWrapper

    def __init__(self, model, shape, *args, **kwargs):
        """Score-matching neural network to represent the prior distribution

        This class is used to calculate the gradient of the log-probability of the prior distribution.
        A custom vjp is created to return the score when calling `jax.grad()`.

        Parameters
        ----------
        model: callable
            Returns the score value given parameter: `model(x) -> score`
        shape: tuple
            Shape of the parameter the model can accept
        *args: tuple
            List of unnamed parameter for model, e.g. `model(x, *args) -> score`
        **kwargs: dict
            List of named parameter for model, e.g. `model(x, **kwargs) -> score`
        """
        # helper class that ensures the model function binds the args/kwargs and has a shape
        wrapper = ScorePrior.ScoreWrapper(partial(model.__call__, *args, **kwargs), shape)
        self._model = wrapper

        super().__init__(
            validate_args=None,
        )

    def __call__(self, x):
        return self._model.func(x)

    def sample(self, key, sample_shape=()):
        # TODO: add ability to draw samples from the prior, if desired
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def log_prob(self, x):
        return _log_prob(self._model, x)
