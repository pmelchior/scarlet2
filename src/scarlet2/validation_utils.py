import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A global switch that toggles automated validation checks.
VALIDATION_SWITCH = True


def set_validation(state: bool = True):
    """Set the global validation switch.

    Parameters
    ----------
    state : bool, optional
        If True, validation checks will be automatically performed. If False,
        they will be skipped. Defaults to True.
    """

    global VALIDATION_SWITCH
    VALIDATION_SWITCH = state
    logger.info(f"Automated validation checks are now {'enabled' if VALIDATION_SWITCH else 'disabled'}.")


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
    """Metaclass that collects all validation methods in a class into a single list.
    For any class that uses this metaclass, all methods that start with "check_"
    will be automatically collected into a class attribute named `validation_checks`.
    """

    def __new__(cls, name, bases, namespace):
        """Creates a list of callable methods when a new instances of a class is
        created."""
        cls = super().__new__(cls, name, bases, namespace)
        cls.validation_checks = [
            attr for attr, value in namespace.items() if callable(value) and attr.startswith("check_")
        ]
        return cls
