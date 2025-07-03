from dataclasses import dataclass
from typing import Any, Optional


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
