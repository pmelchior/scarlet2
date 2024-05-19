import equinox as eqx
import jax
import jax.numpy as jnp
from varname import argname


class Module(eqx.Module):

    def __call__(self):
        raise NotImplementedError

    def get(self, parameter):
        if isinstance(parameter, (list, tuple, {}.keys().__class__)):
            return tuple(self.get(p) for p in parameter)
        assert isinstance(parameter, Parameter)
        return parameter << self

    def replace(self, parameters, values):
        """Replace attribute with given name by other value
        """
        if not isinstance(parameters, (list, tuple, {}.keys().__class__)):
            parameters = (parameters,)
            values = (values,)
        where = lambda model: model.get(parameters)
        return eqx.tree_at(where, self, replace=values)

    def get_filter_spec(self, parameters):
        """Get equinox filter_spec for all fields named in parameters
        """
        if not isinstance(parameters, (list, tuple, {}.keys().__class__)):
            parameters = (parameters,)

        filtered = jax.tree_util.tree_map(lambda _: False, self)
        where = lambda model: model.get(parameters)
        values = (True,) * len(parameters)
        filtered = eqx.tree_at(where, filtered, replace=values)
        if all(jax.tree_util.tree_leaves(filtered)):
            return None
        return filtered


class Parameter:
    # name: str
    # where_in: callable = eqx.field(repr=False)
    # constraint: (None, object)
    # constraint_transform: (None, callable) = eqx.field(repr=False, default=None)
    # prior: (None, object)
    # stepsize: float

    def __init__(self, node, constraint=None, prior=None, stepsize=1):
        # find name of variable node
        self.name = argname('node', vars_only=False)
        # find base dataclass
        name_root = self.name.split(".")[0]
        # create function that ingests root and returns node
        lambda_str = f"lambda {name_root}: {self.name}"
        self.where_in = eval(lambda_str)

        if constraint is not None:
            self.constraint = constraint
            try:
                from numpyro.distributions.transforms import biject_to
            except ImportError:
                raise ImportError("scarlet2.Parameter requires numpyro.")
            # transformation to unconstrained parameters
            self.constraint_transform = biject_to(constraint)

        self.prior = prior
        self.stepsize = stepsize

    def __lshift__(self, root):
        return self.where_in(root)


def relative_step(x, *args, factor=0.01, minimum=1e-6):
    """Step size set at `factor` times the norm of `x`

    As step size functions have the signature (array, int) -> float, *args captures, 
    and then ignores, the iteration counter.
    """
    return jnp.maximum(minimum, factor * jnp.linalg.norm(x))