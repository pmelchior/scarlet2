from dataclasses import dataclass
from typing import Any, Optional

from .observation import ObservationValidator
from .scene import FitValidator
from .source import SourceValidator


@dataclass
class ValidationError:
    """Represents a validation error. Primarily used to convey the context of the
    error to the user and give an indication of what check failed.
    """

    message: str
    check: str
    context: Optional[Any] = None

    def __str__(self):
        base = f"{self.message} | Check={self.check}"
        if self.context is not None:
            base += f" | Context={self.context})"
        return base


class ValidationMethodCollector(type):
    """Metaclass that collects all validation methods in a class into a single"""

    def __new__(cls, name, bases, namespace):
        """Creates a list of callable methods when a new instances of a class is
        created."""
        cls = super().__new__(cls, name, bases, namespace)
        cls.validation_checks = [
            attr for attr, value in namespace.items() if callable(value) and not attr.startswith("__")
        ]
        return cls


def _check(object, validation_class) -> list[ValidationError]:
    """Check the object against the validation rules defined in the validation_class.

    Parameters
    ----------
    object : Any
        The object to check.
    validation_class : type
        The class containing the validation checks.

    Returns
    -------
    list[ValidationError]
        A list of validation errors found in the object.
        If no errors are found, the list is empty.
    """
    validator = validation_class(object)
    validation_errors = []
    for check in validator.validation_checks:
        if error := getattr(validator, check)():
            validation_errors.append(error)

    return validation_errors


def check_fit(scene) -> list[ValidationError]:
    """Check the scene after fitting against the various validation rules.

    Parameters
    ----------
    scene : Scene
        The scene object to check.

    Returns
    -------
    list[ValidationError]
        A list of validation errors found in the fit of the scene.
        If no errors are found, the list is empty.
    """

    return _check(scene, FitValidator)


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

    return _check(observation, ObservationValidator)


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


def check_source(source) -> list[ValidationError]:
    """Check the source against the various validation rules.

    Parameters
    ----------
    source : Source
        The source object to check.

    Returns
    -------
    list[ValidationError]
        A list of validation errors found in the source object.
        If no errors are found, the list is empty.
    """

    return _check(source, SourceValidator)
