import re

import astropy.units as u
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import varname
from astropy.coordinates import SkyCoord

from . import Parameterization, parameter_registry
from .validation_utils import (
    ValidationError,
    ValidationMethodCollector,
    ValidationResult,
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
        leaves = jtu.tree_leaves(self)
        if name is None:
            return {name: leaves[idx] for name, (idx, param) in self.parameters.items()}
        else:
            if name in self.parameters:
                idx, param = self.parameters[name]
                return {name: leaves[idx]}
            else:
                return {}

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

        def get_pair(name):
            idx = params[name][0]
            value = values[name] if isinstance(values, dict) else getattr(values, name)
            return idx, value

        found_leaves = dict([get_pair(name) for name in values_ if name in params])

        def get_leaves(model):
            leaves = jtu.tree_leaves(model)
            return tuple(leaves[i] for i in found_leaves)

        where = lambda model: get_leaves(model)
        return eqx.tree_at(where, self, replace=found_leaves.values())


def _to_pixels(frame, field):
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
    field: any
        Attribute of parameter to be converted to pixel coordinates

    Returns
    -------
    field: any
        Attribute converted to pixel coordinates
    """
    # field or stepsize
    if isinstance(field, u.Quantity):
        return frame.u_to_pixel(field)
    elif isinstance(field, SkyCoord):
        return frame.get_pixel(field)
    else:
        # numpyro dist or constraint
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

        # TODO: is this needed for distributions that use SkyCoord arguments?
        # Doesn't play nice with ScorePrior
        #
        # if isinstance(field, dist.Distribution):
        #     # converting SkyCoord to Array in numpyro distributions requires
        #     # to update batch and event shape
        #     batch_shape = max([getattr(field, name).shape for name in field.reparametrized_params])
        #     field._batch_shape = batch_shape
        #     return dist.Independent(field, 1)

    return field


def sanitize_attr_name(name: str) -> str:
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
        if name is None:
            name = varname.argname("node", vars_only=False)
        name = sanitize_attr_name(name)

        # add this source to the active scene
        try:
            parameters = Parameterization.parameters

            # TODO: go to pixel frame even if specified in sky coords
            # if hasattr(base, "frame"):
            #     node_ = _to_pixels(base.frame, node)
            #     # node needs to be updated in base!
            #     replace_node = lambda n: node_ if n is node else n
            #     base = jtu.tree_map(replace_node, base)
            #     base.set_parameters(parameters)
            #
            #     constraint = _to_pixels(base.frame, constraint) if constraint is not None else None
            #     # prior = _to_pixels(base.frame, prior) if prior is not None else None
            #     stepsize = _to_pixels(base.frame, stepsize) if stepsize is not None else None

            self.constraint = constraint
            self.prior = prior
            self.stepsize = stepsize

            # define constraint bijector functions
            if self.constraint is not None:
                try:
                    from numpyro.distributions.transforms import biject_to
                except ImportError as err:
                    raise ImportError("scarlet2.Parameter requires numpyro.") from err
                self.constraint_transform = biject_to(self.constraint)

            # add parameter to parameter tree and update parameters.tree
            parameters.__iadd__(name, node, self)

        except AttributeError as err:
            msg = "A Parameter instance should only be created within the context of Parameters\n"
            msg += "Use 'with Parameters(scene): Parameter(...)'"
            raise RuntimeError(msg) from err

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

    def __enter__(self):
        # context manager to register sources
        # purpose is to provide scene.frame to source inits that will need some of its information
        # also allows us to append the sources automatically to the scene

        # monkey patch key into base for base.parameter lookup
        key = hex(id(self.base))
        object.__setattr__(self.base, "registry_key", key)

        # put this instance on global context
        Parameterization.parameters = self

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # save with base as key, and the remaining dict as value
        key = self.base.registry_key
        if key not in parameter_registry:
            parameter_registry[key] = super().copy()
        else:
            parameter_registry[key].update(super().copy())

        Parameterization.parameters = None

        # (re)-import `VALIDATION_SWITCH` at runtime to avoid using a static/old value

        # TODO: reenable
        # if VALIDATION_SWITCH:
        #     from .validation import check_parameters
        #
        #     validation_results = check_parameters(self)
        #     print_validation_results("Parameters validation results", validation_results)

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
        # TODO: what do we do when somebody is modifying base (like adding a source to scene)?
        leaves = jtu.tree_leaves(self.base)
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

    def __init__(self, parameter: Parameter) -> None:
        """Initialize the SceneValidator.

        Parameters
        ----------
        parameter : Parameter
            The parameter to validate.
        parameters : Parameters
            The parameters collection for `parameter`.

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
        node, param = self.parameter
        constraint_is_none = param.constraint is None
        if not constraint_is_none:
            is_feasible = param.constraint.check(node)
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
        node, param = self.parameter
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
