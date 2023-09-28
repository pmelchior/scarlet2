# -------------------------------------------------- #
# This class is used to calculate the gradient       #
# of the log-probability via calling the nn prior    #
# ScoreNet model. A custom vjp is created to return  #
# the nn prior when calling jax.grad()               #
# -------------------------------------------------- #

try:
    import numpyro.distributions as dist
    import numpyro.distributions.constraints as constraints
    from galaxygrad import HSC_ScoreNet32, HSC_ScoreNet64, ZTF_ScoreNet32, ZTF_ScoreNet64,HSC_LogNet32, HSC_LogNet64# (https://pypi.org/project/galaxygrad/0.0.19)

except ImportError:
    raise ImportError("scarlet2.nn requires numpyro and galaxygrad=0.0.19")

import jax.numpy as jnp
import jax.scipy as jsp
from jax import custom_vjp
from jax import vjp


def pad_fwd(x, trained_model):
    """Zero-pads the input image to the nearest 32 or 64 pixels"""
    data_size = x.shape[1]
    pad = True
    
    # select the HSC trained model
    if trained_model in ('hsc', 'None'):
        if data_size <= 32:
            pad_gap = 32 - data_size
            ScoreNet = HSC_ScoreNet32
        else:
            pad_gap = 64 - data_size
            ScoreNet = HSC_ScoreNet64

    
    # select the ZTF trained model
    elif trained_model == 'ztf':
        if data_size <= 32:
            pad_gap = 32 - data_size
            ScoreNet = ZTF_ScoreNet32
        else:
            pad_gap = 64 - data_size
            ScoreNet = ZTF_ScoreNet64
    # select the custom trained model
    else:
        ScoreNet = trained_model
    # dont pad if we dont need to
    if pad_gap == 0:
        pad = False
    # calculate how much to pad    
    if pad_gap % 2 == 0:
        pad_lo = pad_hi = int(pad_gap / 2)
    else:
        pad_lo = int(pad_gap // 2)
        pad_hi = int(pad_lo + 1)
    # perform the zero-padding
    if pad:
        if jnp.ndim(x) == 3:
            x[0] = jnp.pad(x[0], ((pad_lo, pad_hi), (pad_lo, pad_hi)), 'constant', constant_values=0) 
        else:
            x = jnp.pad(x, ((pad_lo, pad_hi), (pad_lo, pad_hi)), 'constant', constant_values=0)
    return x, ScoreNet, pad_lo, pad_hi , pad
    
# reverse pad back to original size
def pad_back(x, pad_lo, pad_hi):
    """Removes the zero-padding from the input image"""
    if jnp.ndim(x) > 2: 
        x[0] = x[pad_lo:-pad_hi, pad_lo:-pad_hi]
    else:
        x = x[pad_lo:-pad_hi, pad_lo:-pad_hi]
    return x

# calculate score function (jacobian of log-probability)
def calc_grad(x, trained_model):
    """Calculates the gradient of the log-probability using the ScoreNet model"""
    # perform padding if needed
    t = 0.0 # corresponds to noise free gradient
    x = jnp.float32(x) # cast to float32
    x, ScoreNet, pad_lo, pad_hi, pad = pad_fwd(x, trained_model)
    assert (x.shape[1] % 32) == 0, f"image size must be 32 or 64, got: {x.shape[1]}"
    if jnp.ndim(x) == 2:
        x = jnp.expand_dims(x, axis=0)
        nn_grad = ScoreNet(x, t=t)
        nn_grad = jnp.squeeze(nn_grad, axis=0)
    else:
        nn_grad = ScoreNet(x, t=t)
    # return to original size
    if pad: nn_grad = pad_back(nn_grad, pad_lo, pad_hi)
    return nn_grad


# ------------------------------------------------- #
#   transformation corrections in training space    #
# ------------------------------------------------- #
# jax gradient function
def vgrad(f, x):
    y, vjp_fn = vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]

# transform to training space
def transform(x):
    sigma_y = 0.10
    return jnp.log(x + 1) / sigma_y
# ------------------------------------------------- #


# inheritance from Distribution class
class NNPrior(dist.Distribution):
    support = constraints.real_vector

    """Prior distribution based on a neural network"""

    # construct custom vector-jacobian product

    def __init__(self, trained_model='None', shape=1, validate_args=None):
        # TODO: what's up with these globals?
        # NOTE: this is a hack to get around the fact that    
        # jax doesn't allow flexible arguments to be passed   
        # to the log_prob function                           
        global custom_model
        custom_model = trained_model

        # TODO: needs shape of the model images
        event_shape = jnp.shape(shape)
        super().__init__(
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        # TODO: add ability to draw samples from the prior, if desired
        # NOTE: this will have a fixed size, ie the training data array size
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError
    
    @custom_vjp
    def log_prob(x):
        return 0.0

    def log_prob_fwd(x):
        # Returns primal output and residuals to be used in backward pass by f_bwd
        x = transform(x) # transform to trainign space
        nn_grad = calc_grad(x, custom_model)
        nn_grad = vgrad(transform, x) * nn_grad # chain rule
        return 0.0, nn_grad  # cannot directly call log_prob in Class object

    def log_prob_bwd(res, g):
        nn_grad = res # Get residuals computed in f_fwd
        vjp = (g * nn_grad,) # create the vector (g) jacobian (nn_grad) product
        return vjp
    
    # register the custom vjp 
    log_prob.defvjp(log_prob_fwd, log_prob_bwd)
