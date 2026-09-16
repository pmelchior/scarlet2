import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from colorama import Back, Fore, Style

logger = logging.getLogger(__name__)

# The allowed validation modes.
#   "off"     -- validation checks are skipped entirely.
#   "on"      -- validation checks run and only the check result is printed.
#   "verbose" -- validation checks run and both the result and its context are printed.
VALIDATION_MODES = ("off", "on", "verbose")

# A global switch that controls automated validation checks. One of `VALIDATION_MODES`.
VALIDATION_MODE = "on"


def set_validation(state: bool | str = True):
    """Set the global validation mode.

    Parameters
    ----------
    state : bool or str, optional
        Controls automated validation checks. Accepts:

        * ``"off"`` (or ``False``): validation checks are skipped.
        * ``"on"`` (or ``True``): validation checks run and only the check result
          is printed.
        * ``"verbose"``: validation checks run and both the result and its context
          are printed.

        Defaults to ``True`` (equivalent to ``"on"``).
    """

    global VALIDATION_MODE

    mode = ("on" if state else "off") if isinstance(state, bool) else str(state).lower()

    if mode not in VALIDATION_MODES:
        raise ValueError(f"Invalid validation mode {state!r}; expected a bool or one of {VALIDATION_MODES}.")

    VALIDATION_MODE = mode
    logger.info(f"Automated validation checks are now set to '{VALIDATION_MODE}'.")


@dataclass
class ValidationResult:
    """Represents a validation result. This is the base dataclass that all the
    more specific Validation<Level> dataclasses inherit from. Generally, it should
    not be instantiated directly, but rather through the more specific
    ValidationInfo, ValidationWarning, or ValidationError classes.
    """

    message: str
    check: str
    context: Any | None = None

    # Colored label prefix, set by each subclass.
    _label: ClassVar[str] = ""

    def format(self, verbose: bool = True) -> str:
        """Render the result as a string.

        Parameters
        ----------
        verbose : bool, optional
            If True, append the check's context (when present). Defaults to True.
        """

        base = f"{self._label} {self.message}" if self._label else self.message
        if verbose and self.context is not None:
            base += f" | Context={self.context}"
        return base

    def __str__(self):
        return self.format(verbose=True)


@dataclass
class ValidationInfo(ValidationResult):
    """Represents a validation info message that is informative but not critical."""

    _label: ClassVar[str] = f"{Style.BRIGHT}{Fore.BLACK}{Back.GREEN}  INFO   {Style.RESET_ALL}"


@dataclass
class ValidationWarning(ValidationResult):
    """Represents a validation warning that is not critical but should be noted."""

    _label: ClassVar[str] = f"{Style.BRIGHT}{Fore.BLACK}{Back.YELLOW}  WARN   {Style.RESET_ALL}"


@dataclass
class ValidationError(ValidationResult):
    """Represents a validation error that is critical and should be addressed."""

    _label: ClassVar[str] = f"{Style.BRIGHT}{Fore.WHITE}{Back.RED}  ERROR  {Style.RESET_ALL}"


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


def print_validation_results(preamble: str, results: list[ValidationResult], verbose: bool = False):
    """Print the validation results in a formatted manner.

    Parameters
    ----------
    preamble : str
        A string to print before the validation results.
    results : list[ValidationResult]
        A list of validation results to print.
    verbose : bool, optional
        If True, also print the context attached to each result. Defaults to False.
    """
    if len(results) == 0:
        return
    print(
        f"{preamble}:\n"
        + "\n".join(
            f"[{str(i).zfill(3)}] {result.format(verbose=verbose)}" for i, result in enumerate(results)
        )
    )
