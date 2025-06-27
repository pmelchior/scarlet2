from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp

from scarlet2.observation import Observation
from scarlet2.scene import Scene
from scarlet2.source import Source


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


def check_observation(observation: Observation) -> list[ValidationError]:
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

    validation_errors = []

    observation_checks = ObservationCheck.get_checks()
    for check in observation_checks:
        if error := check(observation).check():
            validation_errors.append(error)

    return validation_errors


def check_scene(scene: Scene) -> list[ValidationError]:
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


def check_source(source: Source) -> list[ValidationError]:
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

    validation_errors = []

    # Perform various checks on the source object
    source_checks = SourceCheck.get_checks()
    for check in source_checks:
        if error := check(source).check():
            validation_errors.append(error)

    return validation_errors


def check_fit(scene: Scene) -> list[ValidationError]:
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

    validation_errors = []
    fit_checks = FitCheck.get_checks()
    for check in fit_checks:
        validation_errors.append(check(scene).check())

    return validation_errors


class BaseCheck:
    """A base class for checks."""

    def __init__(self):
        pass

    @classmethod
    def get_checks(cls):
        """Get the list of all registered checks."""
        if not hasattr(cls, "_checks"):
            cls._checks = []
        return cls._checks

    def check(self) -> Optional[ValidationError]:
        """Run the check on the object."""
        raise NotImplementedError("Subclasses should implement this method.")


class ObservationCheck(BaseCheck):
    """A base class for observation checks."""

    def __init__(self, observation: Observation):
        super().__init__()
        self.observation = observation

    def __init_subclass__(cls):
        """Add all subclasses of ObservationCheck to the list of checks."""
        ObservationCheck.get_checks().append(cls)


class SourceCheck(BaseCheck):
    """A base class for source checks."""

    def __init__(self, source: Source):
        super().__init__()
        self.source = source

    def __init_subclass__(cls):
        """Add all subclasses of SourceCheck to the list of checks."""
        SourceCheck.get_checks().append(cls)


class FitCheck(BaseCheck):
    """A base class for fit checks."""

    def __init__(self, scene: Scene):
        super().__init__()
        self.scene = scene

    def __init_subclass__(cls):
        """Add all subclasses of FitCheck to the list of checks."""
        FitCheck.get_checks().append(cls)


class CheckWeightsNonNegative(ObservationCheck):
    """Check that the weights in the observation are non-negative."""

    def check(self) -> Optional[ValidationError]:
        """Implementation of the check logic.

        Returns
        -------
        ValidationError or None
            Returns a ValidationError if the check fails, otherwise None.
        """
        if (self.observation.weights < 0).any():
            return ValidationError(
                "Weights in the observation must be non-negative.",
                check=self.__class__.__name__,
                #! Placeholder for a meaningful context
                context={"observation.weights": self.observation.weights},
            )
        return None


class CheckWeightIsFinite(ObservationCheck):
    """Check that the weights in the observation are finite."""

    def check(self) -> Optional[ValidationError]:
        """Implementation of the check logic.

        Returns
        -------
        ValidationError or None
            Returns a ValidationError if the check fails, otherwise None.
        """
        if jnp.isinf(self.observation.weights).any():
            return ValidationError(
                "Weights in the observation must be finite.",
                check=self.__class__.__name__,
                #! Placeholder for a meaningful context
                context={"observation.weights": self.observation.weights},
            )
        return None
