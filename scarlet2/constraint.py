import jax.numpy as jnp

class Constraint:
    def check(self, x):
        return NotImplementedError

    def transform(self, x):
        return NotImplementedError

    def inverse(self, x):
        return NotImplementedError

class PositiveConstraint(Constraint):
    def check(self, x):
        return x > 0
    def transform(self, x):
        return jnp.exp(x)
    def inverse(self, x):
        return np.log(x)
