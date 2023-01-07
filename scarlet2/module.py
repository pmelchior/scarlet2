import distrax
import equinox as eqx
import jax

from .constraint import Constraint, NoConstraint
from .prior import NoPrior


class Parameter(eqx.Module):
    value: jax.numpy.ndarray
    constraint: Constraint = eqx.static_field() # needs to be static to avoid copy during filtering
    prior: distrax.Distribution = eqx.static_field()
    fixed: bool

    def __init__(self, value, constraint=None, prior=None, fixed=False):
        self.value = value
        if constraint is None:
            constraint = NoConstraint()
        self.constraint = constraint
        if prior is None:
            prior = NoPrior()
        self.prior = prior
        self.fixed = fixed

    def __call__(self):
        return self.constraint.transform(self.value)

    def log_prior(self):
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
        return sum(p.prior.log_prob(p.value) for p in self.parameters)
