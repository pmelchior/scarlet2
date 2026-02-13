import astropy.units as u
import equinox as eqx
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


def _tree_select(func, pytree, *rest):
    leaves = jtu.tree_leaves(jtu.tree_map(func, pytree, *rest, is_leaf=lambda x: x is None))
    return tuple(_ for _ in leaves if _ is not None)


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

    def get(self, p):
        """Get parameter(s) `p` from this module

        Parameters
        ----------
        p: (:py:class:`Parameters`, :py:class:`Parameter`, str)
            optimization parameters to extract

        Returns
        -------
        tuple
            requested data arrays for `parameters`
        """
        # this method is used during optimization/sampling where the module is a copy without .parameters
        # hence is must be passed as argument
        if isinstance(p, Parameters):
            get_node = lambda node, pnode: node if isinstance(pnode, Parameter) else None
            return _tree_select(get_node, self, p.tree)

        # lookup by name or individual parameter is used for checking/validation
        # as they don't know what module/parameters they belong to, it only works for modules with .parameters
        if not hasattr(self, "parameters"):
            raise AttributeError("Module.get requires parameters attribute to be set for this module.")
        parameters = self.parameters

        if isinstance(p, Parameter):
            get_node = lambda node, pnode: node if pnode is p else None
        elif isinstance(p, str):
            get_node = lambda node, pnode: node if (pnode is not None and pnode.name == p) else None
        plist = _tree_select(get_node, self, parameters.tree)
        if len(plist) == 0:
            raise IndexError(f"Name {p} not found in parameters")
        return plist[0]

    def replace(self, p, v):
        """Replace parameter(s) `p` from this module with `v`

        Parameters
        ----------
        p: (:py:class:`Parameters`, :py:class:`Parameter`, str)
            optimization parameters to replace
        v: (jnp.array, list[jnp.array])
            values to replace parameters with
        """
        # this method is used during optimization/sampling where the module is a copy without .parameters
        # hence is must be passed as argument
        if isinstance(p, Parameters):
            # we con't do jtu.tree_map here because v is a list not a tree
            where = lambda model: model.get(p)  # noqa: E731
            return eqx.tree_at(where, self, replace=v)

        # lookup by name or individual parameter is used for checking/validation
        # as they don't know what module/parameters they belong to, it only works for modules with .parameters
        if not hasattr(self, "parameters"):
            raise AttributeError("Module.get requires parameters attribute to be set for this module.")
        parameters = self.parameters

        if isinstance(p, Parameter):
            replace_node = lambda node, pnode: v if pnode is p else node
        elif isinstance(p, str):
            replace_node = lambda node, pnode: v if (pnode is not None and pnode.name == p) else node
        return jtu.tree_map(replace_node, self, parameters.tree)


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

        # add this source to the active scene
        try:
            base = Parameterization.parameters.base
            parameters = Parameterization.parameters

            # go to pixel frame even if specified in sky coords
            if hasattr(base, "frame"):
                node_ = _to_pixels(base.frame, node)
                # node needs to be updated in base!
                replace_node = lambda n: node_ if n is node else n
                base = jtu.tree_map(replace_node, base)
                base.set_parameters(parameters)

                constraint = _to_pixels(base.frame, constraint) if constraint is not None else None
                # prior = _to_pixels(base.frame, prior) if prior is not None else None
                stepsize = _to_pixels(base.frame, stepsize) if stepsize is not None else None

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
            parameters.__iadd__(node, self)

        except AttributeError as err:
            msg = "A Parameter instance should only be created within the context of Parameters\n"
            msg += "Use 'with Parameters(scene): Parameter(...)'"
            raise RuntimeError(msg) from err

    def __repr__(self):
        # equinox-like formatting
        chunks = []
        for name in ["name", "constraint", "prior", "stepsize"]:
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
        assert isinstance(base, Module)
        self.base = base
        self.tree = jtu.tree_map(lambda node: None, base)  # same treedef as base, but all leaves are None
        # monkey patching parameters onto base
        self.base.set_parameters(self)

    def as_list(self):
        """Return parameter definition as list

        This method uses the same order as :py:func:`~scarlet2.Module.get`.

        Returns
        -------
        tuple
            optimization parameters in order
        """
        get_p = lambda node: node if isinstance(node, Parameter) else None
        return _tree_select(get_p, self.tree)

    def _select(self, fieldname):
        # preserves the order of parameters, in particular the same as Module.get
        return tuple(getattr(p, fieldname) for p in self.as_list())

    @property
    def names(self):
        """Return list of parameter names"""
        return self._select("name")

    @property
    def priors(self):
        """Return list of parameter priors"""
        return self._select("prior")

    @property
    def stepsizes(self):
        """Return list of parameter stepsizes"""
        return self._select("stepsize")

    @property
    def constraints(self):
        """Return list of parameter constraints"""
        return self._select("constraint")

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
        mess += "  list=[\n"
        chunks = []
        for p in self.as_list():
            mess_ = p.__repr__()
            chunk = ""
            for line in mess_.splitlines(keepends=True):
                chunk += "    " + line
            chunks.append(chunk)

        mess += ",\n".join(chunks)
        mess += "\n  ]\n"
        mess += ")"
        return mess

    def __iadd__(self, node, parameter):
        """Add parameter to collection

        Parameters
        ----------
        node: jnp.ndarray
            Parameter array in the base model
        parameter: :py:class:`~scarlet2.Parameter`
            Parameter specification to be added
        """
        select_node = lambda pnode, n: parameter if n is node else pnode
        new_ptree = jtu.tree_map(select_node, self.tree, self.base, is_leaf=lambda x: x is None)
        object.__setattr__(self, "tree", new_ptree)
        return self

    def __isub__(self, name):
        """Remove parameter from collection

        Parameters
        ----------
        name: str
            Name of the parameter to be removed. Silently ignores if named parameter
            is not in the collection.
        """
        select_node = lambda pnode: None if (pnode is not None and pnode.name == name) else pnode
        new_ptree = jtu.tree_map(select_node, self.tree, is_leaf=lambda x: x is None)
        object.__setattr__(self, "tree", new_ptree)
        return self

    def __getitem__(self, i):
        """Access item in collection

        Parameters
        ----------
        i: (int,slice,str)
            Index, slice, or name to access the parameter collection.

        Returns
        -------
        :py:class:`~scarlet2.Parameter`
            If `i` is a slice, returns a subset of the collection.
        """
        if isinstance(i, str):
            select_node = lambda pnode: pnode if (pnode is not None and pnode.name == i) else None
            leaves = jtu.tree_leaves(jtu.tree_map(select_node, self.tree, is_leaf=lambda x: x is None))
            leaves = tuple(_ for _ in leaves if _ is not None)
            if len(leaves) == 0:
                raise IndexError(f"Name {i} not found in parameters")
            return leaves[0]
        return self.as_list()[i]

    def __len__(self):
        """Length of the collection"""
        return len(self.as_list())

    def set_base(self, base):
        """Set base model

        Parameters
        ----------
        base: :py:class:`~scarlet2.Module`
            Base model

        Returns
        -------
        self
        """
        object.__setattr__(self, "base", base)
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

    def __init__(self, parameter: Parameter, parameters: Parameters) -> None:
        """Initialize the SceneValidator.

        Parameters
        ----------
        parameter : Parameter
            The parameter to validate.
        parameters : Parameters
            The parameters collection for `parameter`.

        """
        self.parameter = parameter
        self.parameters = parameters

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
        node = self.parameters.base.get(param)
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
