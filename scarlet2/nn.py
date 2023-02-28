# -------------------------------------------------- #
# This class is used to calculate the gradient       #
# of the log-probability via calling the nn prior    #
# ScoreNet model. A custom vjp is created to return  #
# the nn prior when calling jax.grad()               #
# -------------------------------------------------- #

from galaxygrad import ScoreNet32, ScoreNet64 # (https://pypi.org/project/galaxygrad/0.0.4/) 
import jax.numpy as jnp
from jax import custom_vjp
from .distribution import Distribution # import base class

# TODO: Currently will fail on image sizes over 64x64, think of how
# I want to handle this, could train a higher res model?

# pad up for ScoreNet
def pad_fwd(x):
    """Zero-pads the input image to the nearest 32 or 64 pixels"""
    data_size = x.shape[1]
    pad = True
    if data_size <= 32:
        pad_gap = 32 - data_size
        ScoreNet = ScoreNet32
    else:
        pad_gap = 64 - data_size
        ScoreNet = ScoreNet64
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
            x[0] = jnp.pad(x[0], ((pad_lo, pad_hi), (pad_lo, pad_hi)), 'constant')
        else:
            x = jnp.pad(x, ((pad_lo, pad_hi), (pad_lo, pad_hi)), 'constant')
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
def calc_grad(x):
    # perform padding if needed
    x = jnp.float32(x) # cast to float32
    x, ScoreNet, pad_lo, pad_hi, pad = pad_fwd(x)
    assert (x.shape[1] % 32) == 0, f"image size must be 32 or 64, got: {x.shape[1]}"
    # Scorenet needs (n, 64, 64) or (n, 32, 32)
    if jnp.ndim(x) == 2:
        x = jnp.expand_dims(x, axis=0)
        nn_grad = ScoreNet(x)
        nn_grad = jnp.squeeze(nn_grad, axis=0)
    else:
        nn_grad = ScoreNet(x)
    # return to original size
    if pad: nn_grad = pad_back(nn_grad, pad_lo, pad_hi)
    return nn_grad

# inheritance from Distribution class
class NNPrior(Distribution):
    """Prior distribution based on a neural network"""
    # construct custom vector-jacobian product 
    
    # TODO: These may need to be moved outside of the class
    @custom_vjp
    def log_prob(x):
        return 0.0
    
    def log_prob_fwd(x):
    # Returns primal output and residuals to be used in backward pass by f_bwd
        nn_grad = calc_grad(x)
        return 0.0, nn_grad # cannot directly call log_prob in Class object
    
    def log_prob_bwd(res, g):
        nn_grad = res # Get residuals computed in f_fwd
        vjp = (g * nn_grad,) # create the vector (g) jacobian (nn_grad) product
        return vjp
        
    # register the custom vjp    
    log_prob.defvjp(log_prob_fwd, log_prob_bwd)
