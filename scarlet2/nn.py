# -------------------------------------------------- #
# This class is used to calculate the gradient       #
# of the log-probability via calling the nn prior    #
#  ScoreNet model.                                   #
# -------------------------------------------------- #
from scorenet import ScoreNet32, ScoreNet64
import jax
import jax.numpy as jnp
from jax import custom_jvp
from distribution import Distribution # import base classe

# TODO: Currently will fail on image sizes over 64x64, think of how
# I want to handle this, could train a higher res model?

# pad up for ScoreNet
def pad_fwd(x):
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
    if jnp.ndim(x) > 2: 
        x[0] = x[pad_lo:-pad_hi, pad_lo:-pad_hi]
    else:
        x = x[pad_lo:-pad_hi, pad_lo:-pad_hi]
    return x

# calculate score function alone
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

    @custom_jvp 
    def log_prob(x):
        return 0.0
            
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.custom_jvp.html
    @log_prob.defjvp
    def log_prob_jvp(primals, tangents):
        x, = primals
        x_dot = tangents
        primal_out  = 0.0 #log_prob(x)
        tangent_out = calc_grad(x) 
        return primal_out, tangent_out