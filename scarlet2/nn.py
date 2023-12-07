# -------------------------------------------------- #
# This class is used to calculate the gradient       #
# of the log-probability via calling the score prior #
# ScoreNet model. A custom vjp is created to return  #
# the score prior when calling jax.grad()            #
# -------------------------------------------------- #
try:
    import numpyro.distributions as dist
    import numpyro.distributions.constraints as constraints

except ImportError:
    raise ImportError("scarlet2.nn requires numpyro.")

import jax.numpy as jnp
from jax import custom_vjp
from jax import vjp



def pad_fwd(x, model_size=32):
    """Zero-pads the input image to the model size
    
    Parameters
    ----------
    x : array of the bounding box
    model_size : int
        size of the model to be used
        
    Returns
    -------
    x : padded array same size as model_size
    pad_lo : int
        amount of padding on the lower side
    pad_hi : int
        amount of padding on the higher side
    pad : bool
        whether or not padding was performed
    """
    data_size = x.shape[1]
    pad = True
    pad_gap = model_size - data_size
    assert pad_gap >= 0, "Model size must be larger than max box size"
    # dont pad if we dont need to
    if pad_gap == 0:
        pad = False
    # calculate how much to pad    
    elif pad_gap % 2 == 0:
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
    return x, pad_lo, pad_hi , pad
    
    
# reverse pad back to original size
def pad_back(x, pad_lo, pad_hi):
    """Removes the zero-padding from the input image
    Paremters
    ---------
    x : array of the bounding box
    pad_lo : int
        amount of padding on the lower side
    pad_hi : int
        amount of padding on the higher side
        
    Returns
    -------
    x : array of the bounding box original size
    """
    if jnp.ndim(x) > 2: 
        x[0] = x[pad_lo:-pad_hi, pad_lo:-pad_hi]
    else:
        x = x[pad_lo:-pad_hi, pad_lo:-pad_hi]
    return x


# calculate score function (jacobian of log-probability)
def calc_grad(x, model, model_size=32, t=0.02):
    """Calculates the gradient of the log-prior 
    using the ScoreNet model chosen
    
    Parameters
    ----------
    x : array of the data
    model : the model to calculate the score function
    model_size : int
        size of the model to be used
    t : float for the temperature to evaluate the score function
        
    Returns
    -------
    score_func : array of the score function
    """
    x = jnp.float32(x) # cast to float32
    x, pad_lo, pad_hi, pad = pad_fwd(x, model_size)
    assert (x.shape[1] % 32) == 0, f"image size must be 32 or 64, got: {x.shape[1]}"
    if jnp.ndim(x) == 2:
        x = jnp.expand_dims(x, axis=0)
        score_func = model(x,t=t)
        score_func = jnp.squeeze(score_func, axis=0)
    else:
        score_func = model(x,t=t)
    # return to original size
    if pad: 
        score_func = pad_back(score_func, pad_lo, pad_hi)
    return score_func


# jax gradient function to calculate jacobian
def vgrad(f, x):
    y, vjp_fn = vjp(f, x)
    return vjp_fn(jnp.ones(y.shape))[0]



# Here we define a custom vjp for the log_prob function
# such that for gradient calls in jax, the score prior
# is returned
from functools import partial
@partial(custom_vjp, nondiff_argnums=(0,1,2,3))
def _log_prob(model, transform, model_size, temperature, x):
    return 0
    
def log_prob_fwd(model, transform, model_size, temperature, x):
    x = transform(x)
    score_func = calc_grad(x, model, model_size, temperature)
    score_func = vgrad(transform, x) * score_func # chain rule
    return 0.0, score_func  # cannot directly call log_prob in Class object

def log_prob_bwd(model, transform, model_size, temperature, res, g):
    score_func = res # Get residuals computed in f_fwd
    return (g * score_func,) # create the vector (g) jacobian (score_func) product

# register the custom vjp 
_log_prob.defvjp(log_prob_fwd, log_prob_bwd)


# inheritance from Distribution class
class ScorePrior(dist.Distribution):
    support = constraints.real_vector
    """Prior distribution based on a neural network"""

    def __init__(self, model='None', transform='None', model_size=32, temperature=0.02, validate_args=None):
        self.model = model 
        self.model_size = model_size    
        self.temperature = temperature
        if transform == 'None':
            self.transform = lambda x: x
        else:
            self.transform = transform                

        event_shape = jnp.shape([self.model_size, self.model_size])
        super().__init__(
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        # TODO: add ability to draw samples from the prior, if desired
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError
    
    def log_prob(self, x):
        return _log_prob(self.model, self.transform, self.model_size, self.temperature, x)
