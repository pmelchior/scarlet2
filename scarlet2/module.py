import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import varname

class Module(eqx.Module):

    def __call__(self):
        raise NotImplementedError

    def make_parameters(self):
        return Parameters(self)

    def get(self, parameters):
        assert isinstance(parameters, Parameters)
        return parameters.extract_from(self)

    def replace(self, parameters, values):
        """Replace attribute with given name by other value
        """
        where = lambda model: model.get(parameters)
        return eqx.tree_at(where, self, replace=values)

    def get_filter_spec(self, parameters):
        """Get equinox filter_spec for all fields named in parameters
        """
        filtered = jax.tree_util.tree_map(lambda _: False, self)
        where = lambda model: model.get(parameters)
        values = (True,) * len(parameters)
        filtered = eqx.tree_at(where, filtered, replace=values)
        if all(jax.tree_util.tree_leaves(filtered)):
            return None
        return filtered


class Parameter:

    def __init__(self, node, name=None, constraint=None, prior=None, stepsize=0):
        if name is None:
            self.name = varname.argname('node', vars_only=False)
        else:
            self.name = name
        self.node = node

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

    def __repr__(self):
        # equinox-like formatting
        mess = f"{self.__class__.__name__}(\n"
        for name in ["name", "node", "constraint", "prior", "stepsize"]:
            field = getattr(self, name)
            if name == "node" and isinstance(field, jax.Array):
                field = eqx._pretty_print._pformat_array(field, short_arrays=True)
            mess += f"  {name}={field},\n"
        mess += ")\n"
        return mess


class Parameters:
    def __init__(self, base):
        self.base = base
        self._base_leaves = jtu.tree_leaves(base)
        self._params = list()
        self._leave_idx = list()

    def __repr__(self):
        # equinox-like formatting
        mess = f"{self.__class__.__name__}(\n"
        mess += f"  base={self.base.__class__.__name__},\n"
        mess += f"  parameters=[\n"
        for p in self._params:
            mess_ = p.__repr__()
            for line in mess_.splitlines():
                mess += "    " + line + "\n"
        mess += "  ]"
        mess += ")\n"
        return mess

    def __iadd__(self, parameter):
        assert isinstance(parameter, Parameter)
        found = False
        for i, leaf in enumerate(self._base_leaves):
            if leaf is parameter.node:
                self._params.append(parameter)
                self._leave_idx.append(i)
                found = True
                break
        if not found:
            mess = f"Parameter '{parameter.name}' not in {self.base.__class__.__name__}!"
            raise RuntimeError(mess)
        return self

    def __isub__(self, name):
        for i, param in enumerate(self._params):
            if param.name == name:
                del self._params[i]
                del self._leave_idx[i]
                break
        return self

    def __getitem__(self, i):
        return self._params[i]

    def __len__(self):
        return len(self._params)

    def extract_from(self, root):
        # create function that ingests root and returns all nodes
        assert jtu.tree_structure(root) == jtu.tree_structure(self.base)
        root_leaves = jtu.tree_leaves(root)
        return tuple(root_leaves[idx] for idx in self._leave_idx)


def relative_step(x, *args, factor=0.01, minimum=1e-6):
    """Step size set at `factor` times the norm of `x`

    As step size functions have the signature (array, int) -> float, *args captures, 
    and then ignores, the iteration counter.
    """
    return jnp.maximum(minimum, factor * jnp.linalg.norm(x))
