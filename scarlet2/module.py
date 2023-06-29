import functools

import equinox as eqx
import jax
from numpyro import distributions
from numpyro.distributions import constraints


# recursively finding attributes:
# from https://stackoverflow.com/a/31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class Parameter(eqx.Module):
    value: (jnp.ndarray, float, complex, bool, int)
    constraint: (None, constraints.Constraint) = None
    prior: (None, distributions.Distribution) = None
    stepsize: float = 1
    fixed: bool = False

    def log_prior(self):
        if self.prior is None:
            return 0
        return self.prior.log_prob(self.value)


class Module(eqx.Module):
    @property
    def parameters(self):
        # tree_flatten to terminate at Parameters
        is_leaf = lambda node: isinstance(node, Parameter)
        # but it traverses also other paths that don't end with Parameters, need to filter for those again
        # and remove fixed parameters
        return [p for p in jax.tree_util.tree_flatten(self, is_leaf)[0] if isinstance(p, Parameter)]
