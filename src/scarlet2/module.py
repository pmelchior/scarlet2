import logging

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
    ValidationInfo,
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

    def make_parameters(self):
        """Construct :py:class:`Parameters` for this module"""
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
        :py:class:`Module`
            Modified module. All other module components are unchanged.
        """
        where = lambda model: model.get(parameters)  # noqa: E731
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
        where = lambda model: model.get(parameters)  # noqa: E731
        values = (True,) * len(parameters)
        filtered = eqx.tree_at(where, filtered, replace=values)
        if all(jax.tree_util.tree_leaves(filtered)):
            return None
        return filtered


class Parameter:
    """Class representing a single optimizable parameter"""

    def __init__(self, node, name=None, constraint=None, prior=None, stepsize=0):
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
        self.node = node

        if prior is not None and constraint is not None:
            raise AttributeError(f"Cannot set prior and constraint on the same parameter {self.name}!")

        self.constraint = constraint
        self.prior = prior
        self.stepsize = stepsize

        # add this source to the active scene
        try:
            Parameterization.parameters.__iadd__(self)
        except AttributeError:
            # to be backwards compatible: only emit a warning, don't raise
            msg = "A Parameter instance should only be created within the context of Parameters\n"
            msg += "Use 'with Parameters(scene) as p: Parameter(...)'"
            logging.warn(msg)
            pass

    def apply_constraint(self):
        """Transform the value of the parameter to the unconstrained region"""

        # TODO: What is this thing doing???
        # It doesn't modify in place and it does not return
        if self.constraint is not None:
            try:
                from numpyro.distributions.transforms import biject_to
            except ImportError as err:
                raise ImportError("scarlet2.Parameter requires numpyro.") from err
            # transformation to unconstrained parameters
            self.constraint_transform = biject_to(self.constraint)

            # check if parameter is valid under transform
            unconstrained = self.constraint_transform.inv(self.node)
            if not jnp.isfinite(unconstrained).all():
                raise ValueError(
                    f"Parameter {self.name} has infeasible values for constraint {self.constraint}!"
                )

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
        base: :py:class:`~scarlet2.Module`
            Module the parameters refer to

        Examples
        --------
        >>> with Scene(model_frame) as scene:
        >>>     Source(center1, spectrum1, morph1)
        >>>     Source(center2, spectrum2, morph2)
        >>>
        >>> with Parameters(scene) as parameters:
        >>>     Parameter(scene.sources[0].spectrum.data,
        >>>               name=f"spectrum:0",
        >>>               constraint=constraints.positive,
        >>>               stepsize=relative_step)
        >>> maxiter = 200
        >>> scene_ = scene.fit(observation, parameters, max_iter=maxiter)

        This defines a scene with two sources, initialized with their respective
        `center`, `spectrum`, and `morphology` parameters. It then fits `observation`
        by adjusting only the spectrum array of the first source for 200 steps.

        See Also
        --------
        :py:class:`~scarlet2.Parameter`, :py:class:`~scarlet2.Scene`, :py:func:`~scarlet2.relative_step`
        """
        self.base = base
        self._base_leaves = jtu.tree_leaves(base)
        self._params = list()
        self._leave_idx = list()

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

    def extract_from(self, root):
        """Extract all parameter arrays from `root`

        Parameters
        ----------
        root: :py:class:`~scarlet2.Module`
            The module to extract parameters from. Can be different from `base`,
            but must have the same Pytree structure

        Returns
        -------
        tuple
            Tuple of parameter arrays in the order listed by this `Parameters` collection.

        """
        # create function that ingests root and returns all nodes
        assert jtu.tree_structure(root) == jtu.tree_structure(self.base)
        root_leaves = jtu.tree_leaves(root)
        return tuple(root_leaves[idx] for idx in self._leave_idx)

    def to_pixels(self, parameter):
        """Convert parameter to pixel coordinates of the model frame

        scarlet2 models are optimized in pixel coordinates (defined by the model
        frame of :py:class:`~scarlet2.Scene`. Therefore parameters (or their priors,
        stepsize, etc) that are defined in :py:mod:`astropy.units` or
        :py:class:`astropy.SkyCoord` need to be transformed to pixel coordinates.

        See details in issue :issue:`51`.

        Parameters
        ----------
        parameter: :py:class:`~scarlet2.Parameter`
            Parameter to transform from sky to pixel coordinates.
        """
        frame = self.base.frame
        used_sky_coords_prior = False

        for fieldname in ["node", "constraint", "prior", "stepsize"]:
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
                setattr(parameter, fieldname, dist.Independent(field, 1))

            used_sky_coords_prior = False

        return parameter


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
        if constraint_is_none or (not constraint_is_none and is_feasible.all()):
            validation_results.append(
                ValidationInfo(
                    f"Parameter {param.name} value is feasible.",
                    check=self.__class__.__name__,
                )
            )
        else:
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
        prior_is_none = param.prior is None
        stepsize_is_none = param.stepsize is None
        if (prior_is_none and not stepsize_is_none) or (not prior_is_none and stepsize_is_none):
            validation_results.append(
                ValidationInfo(
                    f"Parameter {param.name} has prior xor stepsize.",
                    check=self.__class__.__name__,
                )
            )
        if prior_is_none and stepsize_is_none:
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

        return validation_results
