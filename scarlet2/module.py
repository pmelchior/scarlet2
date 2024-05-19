import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


class Module(eqx.Module):

    def __call__(self):
        raise NotImplementedError

    def make_param(self, node, constraint=None, prior=None, stepsize=0):
        return Parameter(self, node, constraint=constraint, prior=prior, stepsize=stepsize)

    def get(self, parameter):
        if isinstance(parameter, (list, tuple, {}.keys().__class__)):
            return tuple(self.get(p) for p in parameter)
        assert isinstance(parameter, Parameter)
        return parameter.extract_from(self)

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

    def __init__(self, base, node, constraint=None, prior=None, stepsize=0):
        self.base = base
        self.node = node
        self.extract_from = None
        leaves = jtu.tree_leaves(base)
        for i, leaf in enumerate(leaves):
            if leaf is node:
                # create function that ingests base and returns node
                self.extract_from = lambda base: jtu.tree_leaves(base)[i]
                break
        if self.extract_from is None:
            raise RuntimeError(f"{node} not in {base}!")

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



def relative_step(x, *args, factor=0.01, minimum=1e-6):
    """Step size set at `factor` times the norm of `x`

    As step size functions have the signature (array, int) -> float, *args captures, 
    and then ignores, the iteration counter.
    """
    return jnp.maximum(minimum, factor * jnp.linalg.norm(x))