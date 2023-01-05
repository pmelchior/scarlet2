import equinox as eqx
import jax.numpy as jnp

class Constraint(eqx.Module):
    def check(self, x):
        return NotImplementedError
    def transform(self, x):
        return NotImplementedError
    def inverse(self, x):
        return NotImplementedError

class NoConstraint(Constraint):
    def check(self, x):
        return jnp.ones(x.shape, dtype=jnp.bool)
    def transform(self, x):
        return x
    def inverse(self, x):
        return x

class PositiveConstraint(Constraint):
    def check(self, x):
        return x > 0
    def transform(self, x):
        return jnp.exp(x)
    def inverse(self, x):
        return np.log(x)
