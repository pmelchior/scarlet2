import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import varname

from astropy.coordinates import SkyCoord
import astropy.units as u

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

        self.constraint = constraint
        
        self.prior = prior
        self.stepsize = stepsize
    
    def set_constraint(self):
        if self.constraint is not None:
            try:
                from numpyro.distributions.transforms import biject_to
            except ImportError:
                raise ImportError("scarlet2.Parameter requires numpyro.")
            # transformation to unconstrained parameters
            self.constraint_transform = biject_to(self.constraint)

            # check if parameter is valid under transform
            unconstrained = self.constraint_transform.inv(self.node)
            if not jnp.isfinite(unconstrained).all():
                raise ValueError(f"Parameter {self.name} has infeasible values for constraint {self.constraint}!")

    def __repr__(self):
        # equinox-like formatting
        chunks = []
        for name in ["name", "node", "constraint", "prior", "stepsize"]:
            field = getattr(self, name)
            if name == "node" and isinstance(field, jax.Array):
                field = eqx._pretty_print._pformat_array(field, short_arrays=True)
            chunks.append(f"  {name}={field}")
        inner = ",\n".join(chunks)
        mess = f"{self.__class__.__name__}(\n{inner}\n)"
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
        chunks = []
        for p in self._params:
            mess_ = p.__repr__()
            chunk = ""
            for line in mess_.splitlines(keepends=True):
                chunk += "    " + line
            chunks.append(chunk)

        mess += ",\n".join(chunks)
        mess += "\n  ]\n"
        mess += ")"
        return mess

    def __iadd__(self, parameter):
        assert isinstance(parameter, Parameter)
        found = False
        for i, leaf in enumerate(self._base_leaves):
            if leaf is parameter.node:
                parameter = self.to_pixels(parameter)
                parameter.set_constraint()
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
    

    def to_pixels(self, parameter):
        frame = self.base.frame
        used_sky_coords_prior = False

        for fieldname in ['node', 'constraint', 'prior', 'stepsize']:
            field = getattr(parameter, fieldname)
            if isinstance(field, u.Quantity):
                setattr(parameter, fieldname, frame.u_to_pixel(field))
            if isinstance(field, SkyCoord):
                    setattr(parameter, fieldname, frame.get_pixel(field))
            for name in dir(field):
                try:
                    attrib = getattr(field, name)
                    if isinstance(attrib, u.Quantity):
                        setattr(field, name, frame.u_to_pixel(attrib))
                    if isinstance(attrib, SkyCoord):
                        setattr(field, name, frame.get_pixel(attrib))
                        used_sky_coords_prior = (fieldname == 'prior')
                except:
                    pass

            if used_sky_coords_prior:
                try:
                    import numpyro.distributions as dist
                except ImportError:
                    raise ImportError("scarlet2.Parameter requires numpyro.")
                
                # converting SkyCoord to Array in numpyro distributions requires
                # to update batch and event shape 
                batch_shape = max([getattr(field, name).shape 
                                   for name in field.reparametrized_params])
                setattr(field, '_batch_shape', batch_shape)
                setattr(parameter, fieldname, dist.Independent(field, 1))
                
            used_sky_coords_prior = False

        return parameter


def relative_step(x, *args, factor=0.01, minimum=1e-6):
    """Step size set at `factor` times the norm of `x`

    As step size functions have the signature (array, int) -> float, *args captures, 
    and then ignores, the iteration counter.
    """
    return jnp.maximum(minimum, factor * jnp.linalg.norm(x))