# ----------------------------------------------------- #
# This class is used to calculate the combined gradient #
# of the log-probability (SCARLET 1 grad) and the       #  
# score function via calling the nn prior ScoreNet      #
# Note: the score, by definition is the gradient of the #
# log-likelihood of the prior, so we are really         #
# combining two independed gradient finding methods for #
# log-likelihood, one via promixal gradient descent,    #
# and one directl by taking the approx grad of the data #
# ----------------------------------------------------- #
from scorenet import ScoreNet32, ScoreNet64
import jax
import jax.numpy as jnp
from distribution import Distribution # import base classe

# TODO: Currently will fail on image sizes over 64x64, think of how
# I want to handle this, could train a higher res model?

# inheritance from Distribution - to get log_prob
class NNPrior(Distribution):

    # pad up for ScoreNet
    def pad_fwd(self,x):
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
    def pad_back(self,x, pad_lo, pad_hi):
        if jnp.ndim(x) > 2: 
            x[0] = x[pad_lo:-pad_hi, pad_lo:-pad_hi]
        else:
            x = x[pad_lo:-pad_hi, pad_lo:-pad_hi]
        return x
    
    # calculate score function alone
    def calc_grad(self, x):
        # perform padding if needed
        x, ScoreNet, pad_lo, pad_hi, pad = self.pad_fwd(x)
        assert (x.shape[1] % 32) == 0, f"image size must be 32 or 64, got: {x.shape[1]}"
        # Scorenet needs (n, 64, 64) or (n, 32, 32)
        if jnp.ndim(x) == 2:
            x = jnp.expand_dims(x, axis=0)
            nn_grad = ScoreNet(x)
            nn_grad = jnp.squeeze(nn_grad, axis=0)
        else:
            nn_grad = ScoreNet(x)
        # return to original size
        if pad: nn_grad = self.pad_back(nn_grad, pad_lo, pad_hi)
        return nn_grad
    
    # calculate the total gradient
    # TODO: perhaps make w a function of how close we are to convergence
    # explicilty cast to float for jax.grad to work
    # TODO: perhaps normalise nn.grad and log_prob grad so same
    # in magnitude, hence making choosing values for w and m easier
    def total_grad(self, x):
        x = jnp.float32(x) # cast to float32 for jax.grad
        init_size = x.shape
        w1 = 0.0 ; w2 = 1.0
        sum_grad = w1 * jax.grad(self.log_prob)(x) + w2 * self.calc_grad(x)
        assert sum_grad.shape == init_size, f"grad shape: {sum_grad.shape} != init_size: {init_size}"
        return sum_grad