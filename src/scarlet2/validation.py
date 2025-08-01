from .observation import ObservationValidator
from .scene import FitValidator
from .source import SourceValidator
from .validation_utils import ValidationError


def _check(validation_class, **kwargs) -> list[ValidationError]:
    """Check the object against the validation rules defined in the validation_class.

    Parameters
    ----------
    validation_class : type
        The class containing the validation checks.
    **kwargs : dict
        Keyword arguments to pass to the validation class constructor. These should be
        the inputs required by the validation classes, such as `scene`, `observation`,
        or `source`.

    Returns
    -------
    list[ValidationError]
        A list of validation errors found in the object.
        If no errors are found, the list is empty.
    """
    validator = validation_class(**kwargs)
    validation_errors = []
    for check in validator.validation_checks:
        if error := getattr(validator, check)():
            if isinstance(error, list):
                validation_errors.extend(error)
            else:
                validation_errors.append(error)

    return validation_errors


def check_fit(scene, observation) -> list[ValidationError]:
    """Check the scene after fitting against the various validation rules.

    Parameters
    ----------
    scene : Scene
        The scene object to check.
    observation : Observation
        The observation object to use for checks.

    Returns
    -------
    list[ValidationError]
        A list of validation errors found in the fit of the scene.
        If no errors are found, the list is empty.
    """

    return _check(validation_class=FitValidator, **{"scene": scene, "observation": observation})


def check_observation(observation) -> list[ValidationError]:
    """Check the observation object for consistency

    Parameters
    ----------
    observation: Observation
        The observation object to check.

    Returns
    -------
    list[ValidationError]
        A list of validation errors found in the observation object.
        If no errors are found, the list is empty.
    """

    return _check(validation_class=ObservationValidator, **{"observation": observation})


def check_scene(scene) -> list[ValidationError]:
    """Check the scene against the various validation rules.

    Parameters
    ----------
    scene : Scene
        The scene object to check.

    Returns
    -------
    list[ValidationError]
        A list of validation errors found in the source objects in the scene.
        If no errors are found, the list is empty.
    """

    validation_errors = []
    for source in scene.sources:
        validation_errors.extend(check_source(source))

    return validation_errors


def check_source(source, scene) -> list[ValidationError]:
    """Check the source against the various validation rules.

    Parameters
    ----------
    source : Source
        The source object to check.
    scene : Scene
        The scene that the source is part of.

    Returns
    -------
    list[ValidationError]
        A list of validation errors found in the source object.
        If no errors are found, the list is empty.
    """

    return _check(validation_class=SourceValidator, **{"source": source, "scene": scene})
