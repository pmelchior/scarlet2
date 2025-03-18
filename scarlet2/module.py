import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import varname
from astropy.coordinates import SkyCoord


class Module(eqx.Module):
    """Scarlet2 base module

    Derives directly from :py:class:`equinox.Module`, i.e. from python dataclasses, and adds extra functionality to deal
    with optimizable parameters.
    """
    def __call__(self):
        raise NotImplementedError

    def make_parameters(self):
        """Construct :py:class:`Parameters` for this module
        """
        return Parameters(self)

    def get(self, parameters):
        """Get parameter arrays from this module

        Parameters
        ----------
        parameters: :py:class:`Parameters`
            List of parameters to search

        Returns
        -------
        list
            requested data arrays for `parameters`
        """
        assert isinstance(parameters, Parameters)
        return parameters.extract_from(self)

    def replace(self, parameters, values):
        """Replace parameter arrays by another value

        Parameters
        ----------
        parameters: :py:class:`Parameters`
            List of parameters to search
        values: list
            List of values to replace the current parameter arrays with
            Needs to be in the same shape and order as the current parameter values

        Returns
        -------
        scarlet2.Module
            Modified module. All other module components are unchanged.
        """
        where = lambda model: model.get(parameters)
        return eqx.tree_at(where, self, replace=values)

    def get_filter_spec(self, parameters):
        """Get equinox filter_spec for all fields named in parameters

        Parameters
        ----------
        parameters: :py:class:`Parameters`
            List of parameters to search

        Returns
        -------
        list
            requested data arrays for `parameters`
        """
        filtered = jax.tree_util.tree_map(lambda _: False, self)
        where = lambda model: model.get(parameters)
        values = (True,) * len(parameters)
        filtered = eqx.tree_at(where, filtered, replace=values)
        if all(jax.tree_util.tree_leaves(filtered)):
            return None
        return filtered


class Parameter:
    """Definition of optimizable parameters"""

    def __init__(self, node, name=None, constraint=None, prior=None, stepsize=0):
        """Initialize parameter definition

        Parameters
        ----------
        node: jnp.array
            Data portion of a member of :py:class:`~scarlet2.Module`
        name: str, optional
            Name to assign to this parameter
            If not set, uses :py:mod:`varname` to determine the name `node` has within its module.
        constraint: :py:class:`numpyro.distributions.constraints.Constraint`, optional
            Region over which the parameter value is valid. Contains a bijective transformation to reach this region.
            Cannot be used at the same time as `prior`.
        prior: :py:class:`numpyro.distributions.Distribution`, optional
            Distribution to determine the probability of a parameter value.
            This is used by the optimization in :py:meth:`scarlet2.Scene.fit` and :py:meth:`scarlet2.Scene.sample`.
        stepsize: (float, callable)
            Step size, or function to determine it (e.g. :py:func:`~scarlet2.relative_step`) for parameter updates.
            This is used by the optimization in :py:meth:`scarlet2.Scene.fit`.
        """
        if name is None:
            self.name = varname.argname('node', vars_only=False)
        else:
            self.name = name
        self.node = node

        if prior is not None and constraint is not None:
            raise AttributeError(f"Cannot set prior and constraint on the same parameter {self.name}!")

        self.constraint = constraint
        self.prior = prior
        self.stepsize = stepsize

    def apply_constraint(self):
        """Transform the value of the parameter to the unconstrained region"""

        # TODO: What is this thing doing???
        # It doesn't modify in place and it does not return
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
    """Collection of optimizable parameters

    This class acts like a standard python list of :py:class:`~scarlet2.Parameter` instances.
    It supports `len()`, item access, item addition, etc.

    Attributes
    ----------
    base: :py:class:`~scarlet2.Module`
        Module the parameters refer to
    """

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
        """Add parameter to collection

        Parameters
        ----------
        parameter: :py:class:`~scarlet2.Parameter`
            Parameter to be added
        """
        assert isinstance(parameter, Parameter)
        found = False
        for i, leaf in enumerate(self._base_leaves):
            if leaf is parameter.node:
                parameter = self.to_pixels(parameter)
                parameter.apply_constraint()
                self._params.append(parameter)
                self._leave_idx.append(i)
                found = True
                break
        if not found:
            mess = f"Parameter '{parameter.name}' not in {self.base.__class__.__name__}!"
            raise RuntimeError(mess)
        return self

    def __isub__(self, name):
        """Remove parameter from collection

        Parameters
        ----------
        parameter: :py:class:`~scarlet2.Parameter`
            Parameter to be removed. Silently ignores if `parameter` is not in the collection.
        """
        for i, param in enumerate(self._params):
            if param.name == name:
                del self._params[i]
                del self._leave_idx[i]
                break
        return self

    def __getitem__(self, i):
        """Access item in collection

        Parameters
        ----------
        i: (int,slice)

        Returns
        -------
        :py:class:`~scarlet2.Parameter`
            If `i` is a slice, returns a subset of the collection.
        """
        return self._params[i]

    def __len__(self):
        """Length of the collection"""
        return len(self._params)

    def to_pixels(self, parameter):
        """Convert parameter to pixel coordinates of the model frame

        scarlet2 models are optimized in pixel coordinates (defined by the model frame of :py:class:`~scarlet2.Scene`.
        Therefore parameters (or their priors, stepsize, etc) that are defined in :py:mod:`astropy.units` or
        :py:class:`astropy.SkyCoord` need to be transformed to pixel coordinates.

        See details in issue :issue:`51`.

        Parameters
        ----------
        parameter: :py:class:`~scarlet2.Parameter`
            Parameter to transform from sky to pixel coordinates.
        """
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

    This step size is useful for `Parameter` instances whose uncertainty is relative, not absolute,
    e.g. for :py:class:`~scarlet2.ArraySpectrum`.

    Parameters
    ----------
    x: jnp.array
        Array to compute step size for
    *args: list
        Additional arguments
    factor: float
        Scale norm by this number
    minimum: float
        Minimum return value to prevent zero step sizes

    Returns
    -------
    float
        factor*norm(x), or `minimum`, whichever is larger.
    """
    return jnp.maximum(minimum, factor * jnp.linalg.norm(x))