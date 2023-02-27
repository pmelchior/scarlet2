import equinox as eqx
import jax

from .constraint import Constraint
from .distribution import Distribution


class Parameter(eqx.Module):
    value: jax.numpy.ndarray
    constraint: (Constraint, None) = None
    prior: (Distribution, None) = None
    fixed: bool = False
    
    # testing explicitly placing prior here
    

    def __call__(self):
        if self.constraint is None:
            return self.value
        return self.constraint.transform(self.value)

    def log_prior(self):
        #self.prior = nn
        if self.prior is None:
            return 0
        return self.prior.log_prob(self.value)


# flatten nested lists/tuples, from https://stackoverflow.com/a/64938679
def flatten(l):
    if isinstance(l, (tuple, list)):
        for x in l:
            yield from flatten(x)
    else:
        yield l


class Module(eqx.Module):
    @property
    def parameters(self):
        ps = tuple()
        # need explicit flattening because tree_flatten leaves lists in place
        for x in tuple(flatten(self.tree_flatten()[0])):
            if isinstance(x, Parameter):
                ps = ps + (x,)
            # recursively add parameters of submodules
            elif isinstance(x, Module):
                ps = ps + x.parameters
        return ps

    def log_prior(self):
        return sum(p.log_prior() for p in self.parameters)
