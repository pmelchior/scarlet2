import functools

import equinox as eqx
import jax
import jax.numpy as jnp
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
    _param_info: dict = eqx.field(static=True, init=False, repr=False)

    def __post_init__(self):
        # if user specifies Parameter instead of ndarray:
        # extract value from Parameter instances, but save their metadata
        for name in self.__dataclass_fields__.keys():
            # not all fields need to have attributes set
            try:
                p = getattr(self, name)
            except AttributeError:
                continue
            self.set(name, p)

    def set(self, name, p):
        # check if _param_info exists
        if not hasattr(self, '_param_info'):
            setattr(self, '_param_info', dict())

        default_info = {
            "constraint": None,
            "prior": None,
            "stepsize": 1,
            "fixed": False
        }
        # unpack Parameter
        if isinstance(p, Parameter):
            self._param_info[name] = {
                "constraint": p.constraint,
                "prior": p.prior,
                "stepsize": p.stepsize,
                "fixed": p.fixed
            }
            # store value as Module attribute instead of Parameter
            setattr(self, name, p.value)
        elif eqx.is_array_like(p):
            # store default infos for all other array-like parameters
            self._param_info[name] = default_info
            setattr(self, name, p)

    def replace(self, name, val):
        # replace named attribute by other value
        # WARNING:
        # This function will create inconsistent _param_info for named parameter.
        # Use only for explicit optimizers/samplers calls that don't need/understand _param_info
        if not isinstance(name, (list, tuple)):
            name = (name,)
            val = (val,)
        where = lambda model: tuple(rgetattr(model, n) for n in name)
        replace = val
        return eqx.tree_at(where, self, replace=replace)

    @property
    def parameters(self):
        # Get all non-fixed parameters as dict: name->attribute
        names = self.__dataclass_fields__.keys()
        params = {}
        for n in names:
            a = getattr(self, n)
            if isinstance(a, jnp.ndarray):
                if not self._param_info[n]["fixed"]:
                    params[n] = a
            elif isinstance(a, Module):
                params_ = a.parameters
                for k, v in params_.items():
                    params[n + "." + k] = v
        return params

    @property
    def filter_spec(self):
        # Get equinox filter_spec for all non-fixed parameters
        filtered = jax.tree_util.tree_map(lambda _: False, self)
        params = self.parameters
        where = lambda model: tuple(rgetattr(model, n) for n in params.keys())
        replace = tuple(True for n in range(len(params)))
        filter = eqx.tree_at(where, filtered, replace=replace)
        if all(jax.tree_util.tree_leaves(filter)):
            return None
        return filter
