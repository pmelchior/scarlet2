import equinox as eqx
import jax

from .constraint import Constraint
from .distribution import Distribution


class Parameter(eqx.Module):
    value: jax.numpy.ndarray
    constraint: (Constraint, None) = None
    prior: (Distribution, None) = None
    fixed: bool = False

    def __call__(self):
        if self.constraint is None:
            return self.value
        return self.constraint.transform(self.value)

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
        return [p for p in jax.tree_util.tree_flatten(self, is_leaf)[0] if
                isinstance(p, Parameter) and p.fixed is not True]
