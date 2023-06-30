import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from numpyro import distributions
from numpyro.distributions import constraints


# recursively finding attributes:
# from https://stackoverflow.com/a/31174427
# with modification to unpack a list with an attribute counter
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if not isinstance(obj, (list, tuple)):
            return getattr(obj, attr, *args)
        return obj.__getitem__(int(attr))

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
        self._param_info = dict()
        # if user specifies Parameter instead of ndarray:
        # extract value from Parameter instances, but save their metadata
        for name in self.__dataclass_fields__.keys():
            # not all fields need to have attributes set
            try:
                p = getattr(self, name)
            except AttributeError:
                continue
            self.set(name, p)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def set(self, name, p):

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
            object.__setattr__(self, name, p.value)
        elif eqx.is_array_like(p):
            # store default infos for all other array-like parameters
            self._param_info[name] = default_info
            object.__setattr__(self, name, p)

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
        return self.get_parameters()

    def get_parameters(self, return_info=False):
        # Get all non-fixed parameters as dict: name->attribute
        names = self.__dataclass_fields__.keys()
        params = {}
        info = {}
        for name in names:
            a = getattr(self, name)
            if eqx.is_array_like(a):
                infodict = self._param_info[name]
                if not infodict["fixed"]:
                    params[name] = a
                    info[name] = infodict
            # recursively get all parameters from child models
            elif isinstance(a, Module):
                params_ = a.get_parameters(return_info=True)
                for k, (p, infodict) in params_.items():
                    name_ = f"{name}.{k}"
                    params[name_] = p
                    info[name_] = infodict
            elif isinstance(a, (list, tuple)):
                for i, a_ in enumerate(a):
                    params_ = a_.get_parameters(return_info=True)
                    for k, (p, infodict) in params_.items():
                        name_ = f"{name}.{i}.{k}"
                        params[name_] = p
                        info[name_] = infodict
        if return_info:
            params = {k: (params[k], info[k]) for k in params.keys()}
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
