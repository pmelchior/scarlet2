from .observation import ObservationValidator
from .scene import FitValidator
from .source import SourceValidator
from .validation_utils import ValidationResult


def _check(validation_class, **kwargs) -> list[ValidationResult]:
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
    list[ValidationResult]
        A list of validation results returned from the validation checks for the
        given object.
    """
    validator = validation_class(**kwargs)
    validation_results = []
    for check in validator.validation_checks:
        if error := getattr(validator, check)():
            if isinstance(error, list):
                validation_results.extend(error)
            else:
                validation_results.append(error)

    return validation_results


def check_fit(scene, observation) -> list[ValidationResult]:
    """Check the scene after fitting against the various validation rules.

    Parameters
    ----------
    scene : Scene
        The scene object to check.
    observation : Observation
        The observation object to use for checks.

    Returns
    -------
    list[ValidationResult]
        A list of validation results returned from the validation checks for the
        scene fit results.
    """

    return _check(validation_class=FitValidator, **{"scene": scene, "observation": observation})


def check_observation(observation) -> list[ValidationResult]:
    """Check the observation object for consistency

    Parameters
    ----------
    observation: Observation
        The observation object to check.

    Returns
    -------
    list[ValidationResult]
        A list of validation results from the validation check of the observation
        object.
    """

    return _check(validation_class=ObservationValidator, **{"observation": observation})


def check_scene(scene) -> list[ValidationResult]:
    """Check the scene against the various validation rules.

    Parameters
    ----------
    scene : Scene
        The scene object to check.
    observation : Observation
        The observation object containing the data to validate against.
    parameters : Parameters
        The parameters of the scene to validate.

    Returns
    -------
    list[ValidationResult]
        A list of validation results from the validation checks of the scene.
    """

    validation_results = []
    for source in scene.sources:
        validation_results.extend(check_source(source))

    return validation_results


def check_source(source) -> list[ValidationResult]:
    """Check the source against the various validation rules.

    Parameters
    ----------
    source : Source
        The source object to check.
    scene : Scene
        The scene that the source is part of.

    Returns
    -------
    list[ValidationResult]
        A list of validation results from the source object checks.
    """

    return _check(validation_class=SourceValidator, **{"source": source})
