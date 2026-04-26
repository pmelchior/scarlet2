import re

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import varname
from astropy.coordinates import SkyCoord
from numpyro.distributions.transforms import biject_to

from . import Parameterization, parameter_registry
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

    registry_key: str = eqx.field(init=False, default="", repr=False)

    def __call__(self):
        """Evaluate the model"""
        raise NotImplementedError

    def replace(self, name, value):
        """Replace member attribuge `name` with `value`

        Parameters
        ----------
        name: str
            Name of member to replace
        value: any
            Value to replace member with

        Returns
        -------
        Module
            The modified module.
        """
        return eqx.tree_at(lambda x: getattr(x, name), self, replace=value)

    @property
    def parameters(self):
        """Parameters defined for this module

        Returns
        -------
        dict
            name: (node, param) mapping for all parameters
        """
        return parameter_registry.get(self.registry_key, dict())

    def get(self, name=None):
        """Get parameter(s) from this module

        Parameters
        ----------
        name: str, optional
            Name of parameter. If not set, returns all parameters.

        Returns
        -------
        dict
            requested data arrays for `parameters`
        """
        leaves = jt.leaves(self)
        if name is None:
            return {name: leaves[idx] for name, (idx, param) in self.parameters.items()}
        else:
            if name in self.parameters:
                idx, param = self.parameters[name]
                return leaves[idx]
            else:
                return None

    def set(self, values):
        """Set parameter(s) from this module with `values`

        Parameters
        ----------
        values: dict[str,jnp.array]
            values to replace parameters with, identified by their `name`

        Returns
        -------
        Module:
            new module with parameter(s) replaced by `values`
        """
        # .get_parameters produces values as dict, but infer.fit requires a values dataclass
        values_ = values if isinstance(values, dict) else values.__dict__

        # get idx for all names values that are also in params
        params = self.parameters  # name: (idx, param)
        if len(params) == 0:
            return self

        def get_pair(name):
            idx = params[name][0]
            value = values[name] if isinstance(values, dict) else getattr(values, name)
            return idx, value

        found_leaves = dict([get_pair(name) for name in values_ if name in params])

        def get_leaves(model):
            leaves = jt.leaves(model)
            return tuple(leaves[i] for i in found_leaves)

        where = lambda model: get_leaves(model)
        return eqx.tree_at(where, self, replace=found_leaves.values())


def _to_pixels(frame, field):
    """Convert `field` to pixel coordinates of `frame`

    scarlet2 models are optimized in pixel coordinates (defined by the model
    frame of :py:class:`~scarlet2.Scene`. Therefore parameters (or their priors,
    stepsize, etc) that are defined in :py:mod:`astropy.units` or
    :py:class:`astropy.SkyCoord` need to be transformed to pixel coordinates.

    See details in issue :issue:`51`.

    Parameters
    ----------
    frame: :py:class:`~scarlet2.Frame`
        Frame to define sky coordinates
    field: any
        Array or attribute to be converted to pixel coordinates

    Returns
    -------
    field: any
        Array or attribute converted to pixel coordinates
    """
    # field or stepsize
    if isinstance(field, u.Quantity):
        return frame.u_to_pixel(field)
    elif isinstance(field, SkyCoord):
        return frame.get_pixel(field)
    else:
        # Module, numpyro dist or constraint
        for name in dir(field):
            try:
                attrib = getattr(field, name)
                if isinstance(attrib, u.Quantity):
                    setattr(field, name, frame.u_to_pixel(attrib))
                if isinstance(attrib, SkyCoord):
                    setattr(field, name, frame.get_pixel(attrib))
            except Exception:
                # jax throws exceptions for deprecated attributes, so we ignore exceptions silently
                pass
    return field


def _sanitize_attr_name(name: str) -> str:
    """Replace disallowed characters for Python class attributes with '_'."""
    # Replace any character that isn't alphanumeric or underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Prefix with '_' if starts with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


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
        # get the active parameter dict
        try:
            parameters = Parameterization.parameters
            base = parameters.base
        except AttributeError as err:
            msg = "A Parameter instance should only be created within the context of Parameters\n"
            msg += "Use 'with Parameters(scene): Parameter(...)'"
            raise RuntimeError(msg) from err

        if name is None:
            name = varname.argname("node", vars_only=False)
        name = _sanitize_attr_name(name)

        # go to pixel frame if parameter attributes are specified in sky coords: for scene and observation
        if hasattr(base, "frame"):
            constraint = _to_pixels(base.frame, constraint) if constraint is not None else None
            prior = _to_pixels(base.frame, prior) if prior is not None else None
            stepsize = _to_pixels(base.frame, stepsize) if stepsize is not None else None
        self.constraint = constraint
        self.prior = prior
        self.stepsize = stepsize

        # define constraint bijector functions
        if self.constraint is not None:
            self.constraint_transform = biject_to(self.constraint)

        # add this source to the active scene
        parameters.__iadd__(name, node, self)

    def __repr__(self):
        # equinox-like formatting
        chunks = []
        for name in ["constraint", "prior", "stepsize"]:
            field = getattr(self, name)
            field = eqx.tree_pformat(field)
            chunks.append(f"  {name}={field}")
        inner = ",\n".join(chunks)
        mess = f"{self.__class__.__name__}(\n{inner}\n)"
        return mess


class Parameters(dict):
    """Collection class that contains parameters"""

    def __init__(self, base):
        """Collection of optimizable parameters

        This class creates a Pytree with the same shape as `base` and with :py:class:`~scarlet2.Parameter`
        instances and the nodes corresponding to the optimized parameters in `base`.

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

        This example defines a scene with two sources, initialized with their respective
        `center`, `spectrum`, and `morphology` parameters. It then fits `observation`
        by adjusting only the spectrum array of the first source for 200 steps.

        See Also
        --------
        :py:class:`~scarlet2.Parameter`, :py:class:`~scarlet2.Scene`, :py:func:`~scarlet2.relative_step`
        """
        self.base = base
        # monkey patch key into base for parameter lookup
        key = hex(id(self.base))
        object.__setattr__(self.base, "registry_key", key)

        # if key is already in registry: delete to get the new Parameters
        if key in parameter_registry:
            del parameter_registry[key]

    def __enter__(self):
        # context manager to register Parameter instances
        Parameterization.parameters = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # save with base as key, and the remaining dict as value
        key = self.base.registry_key
        parameter_registry[key] = self
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
        for name, (_node, param) in self.items():
            mess += f"  {name}:"
            mess_ = param.__repr__()
            for line in mess_.splitlines(keepends=True):
                mess += "  " + line
            mess += ",\n"
        mess += ")\n"
        return mess

    def __iadd__(self, name, node, parameter):
        """Add parameter to collection

        Parameters
        ----------
        name: str
            Parameter name
        node: jnp.ndarray
            Parameter array in the base model
        parameter: :py:class:`~scarlet2.Parameter`
            Parameter specification to be added
        """
        # find index of node in leaves of base
        # Note: this lookup would be broken if someone modifies base after the parameters are define
        # The context manager of Scene therefore resets the registry_key of base for an empty parameter list
        leaves = jt.leaves(self.base)
        idx = None
        for i, leaf in enumerate(leaves):
            if leaf is node:
                idx = i
                break
        if idx is None:
            raise RuntimeError(f"Parameter {node} not found in {self.base}")

        self[name] = (idx, parameter)
        return self

    def __isub__(self, name):
        """Remove parameter from collection

        Parameters
        ----------
        name: str
            Parameter name in the base model
        """
        self._params.pop(name, None)
        return self


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

    def __init__(self, name: str, parameter: Parameter, node: jax.Array) -> None:
        """Initialize the SceneValidator.

        Parameters
        ----------
        name: str
            Parameter name
        parameter : Parameter
            The parameter information to validate.
        node : jax.Array
            The parameters array to validate.

        """
        self.name = name
        self.parameter = parameter
        self.node = node

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
                    f"Parameter {self.name} does not have prior or stepsize set.",
                    check=self.__class__.__name__,
                    context={
                        "name": self.name,
                        "prior": param.prior,
                        "stepsize": param.stepsize,
                    },
                )
            )

        return validation_results

    def check_constrained_parameter_is_feasible(self) -> list[ValidationResult]:
        """Check that a constrained parameter has a feasible value.

        Returns
        -------
        list[ValidationResult]
            A list of validation results, which can be either `ValidationInfo`
            or `ValidationError`.
        """
        validation_results: list[ValidationResult] = []
        name, node, param = self.name, self.node, self.parameter
        constraint_is_none = param.constraint is None
        if not constraint_is_none:
            is_feasible = param.constraint.check(node)
        if param.constraint is not None and not is_feasible.all():
            validation_results.append(
                ValidationError(
                    f"Parameter {name} value is infeasible.",
                    check=self.__class__.__name__,
                    context={
                        "name": name,
                        "constraint": param.constraint,
                        "infeasible_at": jnp.argwhere(~is_feasible),
                    },
                )
            )
        return validation_results
