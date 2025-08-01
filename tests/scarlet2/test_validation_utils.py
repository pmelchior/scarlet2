from scarlet2.validation_utils import set_validation


def test_set_validation():
    """Test setting the validation switch. Note that we have to re-import VALIDATION_SWITCH
    to ensure we are using the current value."""

    set_validation(True)
    from scarlet2.validation_utils import VALIDATION_SWITCH

    assert VALIDATION_SWITCH is True

    set_validation(False)
    from scarlet2.validation_utils import VALIDATION_SWITCH

    assert VALIDATION_SWITCH is False
