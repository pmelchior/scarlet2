import pytest

from scarlet2.validation_utils import (
    ValidationInfo,
    ValidationWarning,
    print_validation_results,
    set_validation,
)


def test_set_validation():
    """Test setting the validation mode. Note that we have to re-import VALIDATION_MODE
    to ensure we are using the current value."""

    set_validation(True)
    from scarlet2.validation_utils import VALIDATION_MODE

    assert VALIDATION_MODE == "on"

    set_validation(False)
    from scarlet2.validation_utils import VALIDATION_MODE

    assert VALIDATION_MODE == "off"

    set_validation("verbose")
    from scarlet2.validation_utils import VALIDATION_MODE

    assert VALIDATION_MODE == "verbose"

    set_validation("on")
    from scarlet2.validation_utils import VALIDATION_MODE

    assert VALIDATION_MODE == "on"


def test_set_validation_invalid():
    """An unknown mode raises a ValueError."""

    with pytest.raises(ValueError):
        set_validation("loud")

    # restore default for other tests
    set_validation(True)


def test_result_format_verbosity():
    """`format` includes the context only when verbose."""

    result = ValidationInfo(message="all good", check="Check", context={"value": 1})

    assert "Context" not in result.format(verbose=False)
    assert "Context={'value': 1}" in result.format(verbose=True)

    # results without a context never print a context section
    no_context = ValidationInfo(message="all good", check="Check")
    assert "Context" not in no_context.format(verbose=True)


def test_print_validation_results_verbose(capsys):
    """`print_validation_results` forwards the verbose flag to each result."""

    results = [ValidationWarning(message="careful", check="Check", context={"value": 2})]

    print_validation_results("Results", results, verbose=False)
    assert "Context" not in capsys.readouterr().out

    print_validation_results("Results", results, verbose=True)
    assert "Context={'value': 2}" in capsys.readouterr().out
