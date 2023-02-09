import jax.numpy as jnp

class Distribution:
    def log_prob(self, x):
        return jnp.sum(x)
        #return NotImplementedError
