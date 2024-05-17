import functools

import equinox as eqx
import jax
import jax.numpy as jnp


# recursively finding attributes:
# from https://stackoverflow.com/a/31174427
# with modification to unpack a list with an attribute counter
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if not isinstance(obj, (list, tuple)):
            return getattr(obj, attr, *args)
        return obj.__getitem__(int(attr))

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class Module(eqx.Module):

    def __call__(self):
        raise NotImplementedError

    def get(self, parameter):
        if isinstance(parameter, (list, tuple, {}.keys().__class__)):
            return tuple(self.get(p) for p in parameter)
        if isinstance(parameter, Parameter):
            name = parameter.name
        elif isinstance(parameter, str):
            name = parameter
        else:
            return None
        return rgetattr(self, name)

    def replace(self, parameters, values):
        """Replace attribute with given name by other value
        """
        if not isinstance(parameters, (list, tuple, {}.keys().__class__)):
            parameters = (parameters,)
            values = (values,)
        where = lambda model: tuple(model.get(p) for p in parameters)
        replace = values
        return eqx.tree_at(where, self, replace=replace)

    def get_filter_spec(self, parameters):
        """Get equinox filter_spec for all fields named in parameters
        """
        if not isinstance(parameters, (list, tuple, {}.keys().__class__)):
            parameters = (parameters,)

        filtered = jax.tree_util.tree_map(lambda _: False, self)
        where = lambda model: tuple(model.get(p) for p in parameters)
        replace = tuple(True for n in range(len(parameters)))
        filter = eqx.tree_at(where, filtered, replace=replace)
        if all(jax.tree_util.tree_leaves(filter)):
            return None
        return filter


class Parameter(Module):
    name: str
    constraint: (None, object) = None
    prior: (None, object) = None
    stepsize: float = 1


def relative_step(x, *args, factor=0.01, minimum=1e-6):
    """Step size set at `factor` times the norm of `x`

    As step size functions have the signature (array, int) -> float, *args captures, 
    and then ignores, the iteration counter.
    """
    return jnp.maximum(minimum, factor * jnp.linalg.norm(x))