# -------------------------------------------------- #
# This class is used to calculate the gradient       #
# of the log-probability via calling the nn prior    #
# ScoreNet model. A custom vjp is created to return  #
# the nn prior when calling jax.grad()               #
# -------------------------------------------------- #

from galaxygrad import ScoreNet32, ScoreNet64, ScoreNetZTF # (https://pypi.org/project/galaxygrad/0.0.4/) 
import jax.numpy as jnp
from jax import custom_vjp
import jax.scipy as jsp
from jax import jit
import equinox as eqx
from .distribution import Distribution # import base class


def pad_fwd(x, trained_model):
    """Zero-pads the input image to the nearest 32 or 64 pixels"""
    data_size = x.shape[1]
    pad = True
    
    # select the HSC trained model
    if trained_model == 'None':
        if data_size <= 32:
            pad_gap = 32 - data_size
            ScoreNet = ScoreNet32 
        else:
            pad_gap = 64 - data_size
            ScoreNet = ScoreNet64
    
    # select the ZTF trained model
    elif trained_model == 'ztf':
        ScoreNet = ScoreNetZTF
        
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
def calc_grad(x, trained_model):
    # perform padding if needed
    t = 0.0 # corresponds to noise free gradient
    x = jnp.float32(x) # cast to float32
    x, ScoreNet, pad_lo, pad_hi, pad = pad_fwd(x, trained_model)
    assert (x.shape[1] % 32) == 0, f"image size must be 32 or 64, got: {x.shape[1]}"
    # Scorenet needs (n, 64, 64) or (n, 32, 32)
    if jnp.ndim(x) == 2:
        x = jnp.expand_dims(x, axis=0)
        nn_grad = ScoreNet(x, t=t)
        nn_grad = jnp.squeeze(nn_grad, axis=0)
    else:
        nn_grad = ScoreNet(x, t=t)
    # return to original size
    if pad: nn_grad = pad_back(nn_grad, pad_lo, pad_hi)
    # gaussian filter to smooth the gradient (minimised artifacts)
    x = jnp.linspace(-4, 4, 9)  #jnp.linspace(-10,10,nn_grad.shape[0])#jnp.linspace(-4, 4, 9) # kernal dims
    scale = 1.2 #0.75
    window = jsp.stats.norm.pdf(x,loc=0, scale=scale) * jsp.stats.norm.pdf(x[:, None],loc=0, scale=scale) # Gaussian kernal
    smooth_grad = jsp.signal.convolve(nn_grad, window, mode='same')
    return smooth_grad
    # Testing out some things for smoother gradients
    #return nn_grad


# inheritance from Distribution class
class NNPrior(Distribution):
    """Prior distribution based on a neural network"""
    # construct custom vector-jacobian product 
    def __init__(self, trained_model='None'):
        global custom_model 
        custom_model = trained_model
        
    """
    Note, I cannot pass "self" to the functions below
    as the jax custom_vjp decorator cannot handle them
    hence the use of a global variable to pass the model
    """
    
    @custom_vjp
    def log_prob(x):
        return 0.0
    
    def log_prob_fwd(x):
    # Returns primal output and residuals to be used in backward pass by f_bwd
        nn_grad = calc_grad(x, custom_model)
        return 0.0, nn_grad # cannot directly call log_prob in Class object
    
    def log_prob_bwd(res, g):
        nn_grad = res # Get residuals computed in f_fwd
        vjp = (g * nn_grad,) # create the vector (g) jacobian (nn_grad) product
        return vjp
        
    # register the custom vjp    
    log_prob.defvjp(log_prob_fwd, log_prob_bwd)