import logging
from dataclasses import dataclass
from typing import Any, Optional

from colorama import Back, Fore, Style

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
class ValidationResult:
    """Represents a validation result. This is the base dataclass that all the
    more specific Validation<Level> dataclasses inherit from. Generally, it should
    not be instantiated directly, but rather through the more specific
    ValidationInfo, ValidationWarning, or ValidationError classes.
    """

    message: str
    check: str
    context: Optional[Any] = None

    def __str__(self):
        base = f"{self.message}"
        if self.context is not None:
            base += f" | Context={self.context})"
        return base


@dataclass
class ValidationInfo(ValidationResult):
    """Represents a validation info message that is informative but not critical."""

    def __str__(self):
        return f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}  INFO   {Style.RESET_ALL} {super().__str__()}"


@dataclass
class ValidationWarning(ValidationResult):
    """Represents a validation warning that is not critical but should be noted."""

    def __str__(self):
        return f"{Style.BRIGHT}{Fore.BLACK}{Back.YELLOW}  WARN   {Style.RESET_ALL} {super().__str__()}"


@dataclass
class ValidationError(ValidationResult):
    """Represents a validation error that is critical and should be addressed."""

    def __str__(self):
        return f"{Style.BRIGHT}{Fore.WHITE}{Back.RED}  ERROR  {Style.RESET_ALL} {super().__str__()}"


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


def print_validation_results(preamble: str, results: list[ValidationResult]):
    """Print the validation results in a formatted manner.

    Parameters
    ----------
    preamble : str
        A string to print before the validation results.
    results : list[_ValidationResult]
        A list of validation results to print.
    """

    print(
        f"{preamble}:\n" + "\n".join(f"[{str(i).zfill(3)}] {str(result)}" for i, result in enumerate(results))
    )
