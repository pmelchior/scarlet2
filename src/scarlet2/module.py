import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import varname
from astropy.coordinates import SkyCoord

from . import Parameterization
from .validation_utils import (
    ValidationError,
    ValidationMethodCollector,
    ValidationResult,
    print_validation_results,
)


class Module(eqx.Module):
    """Scarlet2 base module

    Derives directly from :py:class:`equinox.Module`, i.e. from python dataclasses,
    and adds extra functionality to deal with optimizable parameters.
    """

    def __call__(self):
        """Evaluate the model"""
        raise NotImplementedError

    def set_parameters(self, parameters):
        """Define parameters for this module

        Parameters
        ----------
        parameters: :py:class:`Parameters`
            Optimization parameters

        Returns
        -------
        self
        """
        # Monkey patching parameters onto base model:
        # allows to move the parameters with the model without eqx replicating them during fit/sample
        object.__setattr__(self, "parameters", parameters)
        return self

    def get_parameters(self, ptree):
        """Get parameter arrays from this module

        Returns
        -------
        tuple
            requested data arrays for `parameters`
        """
        if not hasattr(self, "parameters"):
            return tuple()

        get_node = lambda node, p: node if isinstance(p, Parameter) else None
        leaves = jtu.tree_leaves(jtu.tree_map(get_node, self, ptree))
        return tuple(l for l in leaves if l is not None)

    def replace_parameters(self, values):
        """Replace parameter arrays by another value

        Parameters
        ----------
        values: list
            List of values to replace the current parameter arrays with
            Needs to be in the same shape and order as the current parameter values

        Returns
        -------
        :py:class:`Module`
            Modified module. All other module components are unchanged.
        """
        where = lambda model: model.get_parameters()  # noqa: E731
        return eqx.tree_at(where, self, replace=values)

    def get_filter_spec(self):
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
        if not hasattr(self, "parameters"):
            return None

        filtered = jax.tree_util.tree_map(lambda _: False, self)
        where = lambda model: model.get_parameters()  # noqa: E731
        values = (True,) * len(self.parameters)
        filtered = eqx.tree_at(where, filtered, replace=values)
        return filtered


class Parameter:
    """Class representing a single optimizable parameter"""

    def __init__(self, node, name=None, constraint=None, prior=None, stepsize=None):
        """Definition of optimizable parameter

        Parameters
        ----------
        node: array
            Data portion of a member of :py:class:`~scarlet2.Module`
        name: str, optional
            Name to assign to this parameter
            If not set, uses :py:mod:`varname` to determine the name `node` has
            within its module.
        constraint: :py:class:`numpyro.distributions.constraints.Constraint`, optional
            Region over which the parameter value is valid. Contains a bijective
            transformation to reach this region. Cannot be used at the same time
            as `prior`.
        prior: :py:class:`numpyro.distributions.distribution.Distribution`, optional
            Distribution to determine the probability of a parameter value.
            This is used by the optimization in :py:meth:`scarlet2.Scene.fit`
            and :py:meth:`scarlet2.Scene.sample`.
        stepsize: (float, callable)
            Step size, or function to determine it
            (e.g. :py:func:`~scarlet2.relative_step`) for parameter updates.
            This is used by the optimization in :py:meth:`scarlet2.Scene.fit`.

        See Also
        --------
        :py:class:`~scarlet2.Parameters`,
        """
        if name is None:
            self.name = varname.argname("node", vars_only=False)
        else:
            self.name = name
        self.constraint = constraint
        if self.constraint is not None:
            try:
                from numpyro.distributions.transforms import biject_to
            except ImportError as err:
                raise ImportError("scarlet2.Parameter requires numpyro.") from err
            # transformation to unconstrained parameters
            self.constraint_transform = biject_to(self.constraint)

        self.prior = prior
        self.stepsize = stepsize
        self.node = node

        # add this source to the active scene
        try:
            base = Parameterization.parameters.base
            # go to pixel frame even if specified in sky coords
            if hasattr(base, "frame"):
                self.to_pixels(base.frame)
            Parameterization.parameters.__iadd__(self)
        except AttributeError as err:
            msg = "A Parameter instance should only be created within the context of Parameters\n"
            msg += "Use 'with Parameters(scene) as p: Parameter(...)'"
            raise RuntimeError(msg) from err

    def to_pixels(self, frame):
        """Convert parameter to pixel coordinates of the model frame

        scarlet2 models are optimized in pixel coordinates (defined by the model
        frame of :py:class:`~scarlet2.Scene`. Therefore parameters (or their priors,
        stepsize, etc) that are defined in :py:mod:`astropy.units` or
        :py:class:`astropy.SkyCoord` need to be transformed to pixel coordinates.

        See details in issue :issue:`51`.

        Parameters
        ----------
        frame: :py:class:`~scarlet2.Frame`
            Frame to define sky coordinates
        """
        used_sky_coords_prior = False

        for fieldname in ["node", "constraint", "prior", "stepsize"]:
            field = getattr(self, fieldname)
            if isinstance(field, u.Quantity):
                setattr(self, fieldname, frame.u_to_pixel(field))
            if isinstance(field, SkyCoord):
                setattr(self, fieldname, frame.get_pixel(field))
            for name in dir(field):
                try:
                    attrib = getattr(field, name)
                    if isinstance(attrib, u.Quantity):
                        setattr(field, name, frame.u_to_pixel(attrib))
                    if isinstance(attrib, SkyCoord):
                        setattr(field, name, frame.get_pixel(attrib))
                        used_sky_coords_prior = fieldname == "prior"
                except Exception:
                    # jax throws exceptions for deprecated attributes, so we ignore exceptions silently
                    pass

            if used_sky_coords_prior:
                try:
                    import numpyro.distributions as dist
                except ImportError as err:
                    raise ImportError("scarlet2.Parameter requires numpyro.") from err

                # converting SkyCoord to Array in numpyro distributions requires
                # to update batch and event shape
                batch_shape = max([getattr(field, name).shape for name in field.reparametrized_params])
                field._batch_shape = batch_shape
                setattr(self, fieldname, dist.Independent(field, 1))

            used_sky_coords_prior = False

    def __repr__(self):
        # equinox-like formatting
        chunks = []
        for name in ["name", "node", "constraint", "prior", "stepsize"]:
            field = getattr(self, name)
            field = eqx.tree_pformat(field)
            chunks.append(f"  {name}={field}")
        inner = ",\n".join(chunks)
        mess = f"{self.__class__.__name__}(\n{inner}\n)"
        return mess


class Parameters:
    """Collection class that contains parameters"""

    def __init__(self, base):
        """Collection of optimizable parameters

        This class acts like a standard python list of :py:class:`~scarlet2.Parameter` instances.
        It supports `len()`, item access, item addition, etc.

        Attributes
        ----------
        base: :py:class:`~scarlet2.Module` or tuple of modules
            Module(s) the parameters refer to

        Examples
        --------
        >>> with Scene(model_frame) as scene:
        >>>     Source(center1, spectrum1, morph1)
        >>>     Source(center2, spectrum2, morph2)
        >>>
        >>> with Parameters(scene):
        >>>     Parameter(scene.sources[0].spectrum.data,
        >>>               name=f"spectrum:0",
        >>>               constraint=constraints.positive,
        >>>               stepsize=relative_step)
        >>> maxiter = 200
        >>> scene_ = scene.fit(observation, max_iter=maxiter)

        This defines a scene with two sources, initialized with their respective
        `center`, `spectrum`, and `morphology` parameters. It then fits `observation`
        by adjusting only the spectrum array of the first source for 200 steps.

        See Also
        --------
        :py:class:`~scarlet2.Parameter`, :py:class:`~scarlet2.Scene`, :py:func:`~scarlet2.relative_step`
        """
        assert isinstance(base, Module)
        self.base = base
        self._base_leaves = jtu.tree_leaves(base)
        self._params = list()
        self._leave_idx = list()
        # monkey patching parameters onto base
        self.base.set_parameters(self)

    def as_tree(self):
        pnodes = tuple(p.node for p in self._params)
        entry_in = lambda entry, _list: any(_ is entry for _ in _list)
        first_in = lambda entry, _list: list(_ is entry for _ in _list).index(True)

        def node_transform(node):
            if entry_in(node, pnodes):
                i = first_in(node, pnodes)
                return self._params[i]
            else:
                return None

        return jtu.tree_map(node_transform, self.base)

    def __enter__(self):
        # context manager to register sources
        # purpose is to provide scene.frame to source inits that will need some of its information
        # also allows us to append the sources automatically to the scene
        Parameterization.parameters = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Parameterization.parameters = None

        # (re)-import `VALIDATION_SWITCH` at runtime to avoid using a static/old value
        from .validation_utils import VALIDATION_SWITCH

        if VALIDATION_SWITCH:
            from .validation import check_parameters

            validation_results = check_parameters(self)
            print_validation_results("Parameters validation results", validation_results)

    def __repr__(self):
        # equinox-like formatting
        mess = f"{self.__class__.__name__}(\n"
        mess += f"  base={self.base.__class__.__name__},\n"
        mess += "  parameters=[\n"
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
        name: str
            Name of the parameter to be removed. Silently ignores if named parameter
            is not in the collection.
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
            Index or slice to access the collection.

        Returns
        -------
        :py:class:`~scarlet2.Parameter`
            If `i` is a slice, returns a subset of the collection.
        """
        return self._params[i]

    def __len__(self):
        """Length of the collection"""
        return len(self._params)


def relative_step(x, *args, factor=0.01, minimum=1e-6):
    """Step size set at `factor` times the norm of `x`

    This step size is useful for `Parameter` instances whose uncertainty is relative, not absolute,
    e.g. for :py:class:`~scarlet2.ArraySpectrum`.

    Parameters
    ----------
    x: array
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


class ParameterValidator(metaclass=ValidationMethodCollector):
    """A class containing all of the validation checks for a Scene object.
    A convenience function exists that will run all the checks in this class and
    returns a list of validation results:

    :py:func:`~scarlet2.validation.check_scene`.
    """

    def __init__(self, parameter: Parameter):
        """Initialize the SceneValidator.

        Parameters
        ----------
        parameter : Parameter
            The parameter to validate.
        """
        self.parameter = parameter

    def check_constrained_parameter_is_feasible(self) -> list[ValidationResult]:
        """Check that a constrained parameter has a feasible value.

        Returns
        -------
        list[ValidationResult]
            A list of validation results, which can be either `ValidationInfo`
            or `ValidationError`.
        """
        validation_results: list[ValidationResult] = []
        param = self.parameter
        constraint_is_none = param.constraint is None
        if not constraint_is_none:
            is_feasible = param.constraint.check(param.node)
        if param.constraint is not None and not is_feasible.all():
            validation_results.append(
                ValidationError(
                    f"Parameter {param.name} value is infeasible.",
                    check=self.__class__.__name__,
                    context={
                        "name": param.name,
                        "constraint": param.constraint,
                        "feasible": is_feasible,
                    },
                )
            )
        return validation_results

    def check_parameter_has_necessary_fields(self) -> list[ValidationResult]:
        """Check that all parameter ave the necessary fields set.

        This checks that all parameters in the scene have the `prior` or `stepsize`
        attributes set, but not both.

        Returns
        -------
        list[ValidationResult]
            A list of validation results, which can be either `ValidationInfo`
            or `ValidationError`.
        """
        validation_results: list[ValidationResult] = []
        param = self.parameter
        if param.prior is None and param.stepsize is None:
            validation_results.append(
                ValidationError(
                    f"Parameter {param.name} does not have prior or stepsize set.",
                    check=self.__class__.__name__,
                    context={
                        "name": param.name,
                        "prior": param.prior,
                        "stepsize": param.stepsize,
                    },
                )
            )
        if param.prior is not None and param.constraint is not None:
            validation_results.append(
                ValidationError(
                    f"Parameter {param.name} has both prior or constraint set. Choose one.",
                    check=self.__class__.__name__,
                    context={
                        "name": param.name,
                        "prior": param.prior,
                        "constraint": param.constraint,
                    },
                )
            )

        return validation_results
