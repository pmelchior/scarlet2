import os
from pathlib import Path

from ipywidgets import Button
from scarlet2.questionnaire import QuestionnaireWidget, run_questionnaire
from scarlet2.questionnaire.models import Questionnaire
from scarlet2.questionnaire.questionnaire import load_questions


def test_questionnaire_widget_init(example_questionnaire, helpers):
    """Test that the widget initializes correctly with the example questionnaire."""
    widget = QuestionnaireWidget(example_questionnaire)
    assert widget.questions == example_questionnaire.questions
    assert widget.code_output == example_questionnaire.initial_template
    assert widget.commentary == example_questionnaire.initial_commentary

    assert widget.current_question == example_questionnaire.questions[0]
    assert widget.questions_stack == example_questionnaire.questions[1:]
    assert widget.question_answers == []

    helpers.assert_widget_ui_matches_state(widget)


def test_questionnaire_show(example_questionnaire, mocker):
    """Mock the display function to test that show() calls it correctly."""
    mocker.patch("scarlet2.questionnaire.questionnaire.display")

    widget = QuestionnaireWidget(example_questionnaire)
    widget.show()

    from scarlet2.questionnaire.questionnaire import display

    display.assert_called_once_with(widget.ui)


def test_questionnaire_handle_answer_selection(example_questionnaire, helpers):
    """Test that selecting an answer updates the widget correctly."""
    widget = QuestionnaireWidget(example_questionnaire)

    first_question = example_questionnaire.questions[0]
    first_answer = first_question.answers[0]
    first_button = helpers.get_answer_button(widget, 0)  # First answer button

    first_button.click()

    assert widget.code_output == "example_code {{follow}}"
    assert widget.commentary == "This is some commentary."

    assert widget.question_answers == [(first_question, 0)]

    assert widget.current_question == first_answer.followups[0]
    assert widget.questions_stack == first_answer.followups[1:] + example_questionnaire.questions[1:]

    print(widget.question_box.children)

    helpers.assert_widget_ui_matches_state(widget)


def test_questionnaire_complete_all_questions(example_questionnaire, helpers):
    """Test completing the entire questionnaire."""
    widget = QuestionnaireWidget(example_questionnaire)

    answer_inds = [0, 1, 0, 0]

    expected_questions = [
        example_questionnaire.questions[0],
        example_questionnaire.questions[0].answers[0].followups[0],
        example_questionnaire.questions[0].answers[0].followups[1],
        example_questionnaire.questions[1],
    ]

    for i, ans_ind in enumerate(answer_inds):
        current_question = widget.current_question
        assert current_question == expected_questions[i]
        assert widget.question_answers == list(zip(expected_questions[:i], answer_inds[:i], strict=False))
        helpers.assert_widget_ui_matches_state(widget)

        button = helpers.get_answer_button(widget, ans_ind)
        button.click()

    assert widget.current_question is None
    assert widget.question_answers == list(zip(expected_questions, answer_inds, strict=False))
    helpers.assert_widget_ui_matches_state(widget)


def test_questionnaire_switch_variable(example_questionnaire_with_switch, helpers):
    """Test that switch-case logic in the questionnaire works correctly."""

    # Test that the switch-case logic works correctly for the first case.
    widget = QuestionnaireWidget(example_questionnaire_with_switch)

    first_question = example_questionnaire_with_switch.questions[0]
    first_answer = first_question.answers[0]
    first_button = helpers.get_answer_button(widget, 0)  # First answer button

    first_button.click()

    assert widget.code_output == first_answer.templates[0].code
    assert widget.commentary == first_answer.commentary

    assert widget.question_answers == [(first_question, 0)]
    assert widget.variables == {first_question.variable: 0}

    switch = example_questionnaire_with_switch.questions[1]
    case = switch.cases[0]

    assert widget.current_question == case.questions[0]
    assert widget.questions_stack == case.questions[1:] + example_questionnaire_with_switch.questions[2:]

    helpers.assert_widget_ui_matches_state(widget)

    # Now test that the switch-case logic works correctly for the second case.
    widget = QuestionnaireWidget(example_questionnaire_with_switch)
    first_question = example_questionnaire_with_switch.questions[0]
    second_answer = first_question.answers[1]
    second_button = helpers.get_answer_button(widget, 1)  # Second answer button

    second_button.click()

    assert widget.code_output == second_answer.templates[0].code
    assert widget.commentary == second_answer.commentary
    assert widget.question_answers == [(first_question, 1)]
    assert widget.variables == {first_question.variable: 1}

    switch = example_questionnaire_with_switch.questions[1]
    case = switch.cases[1]

    assert widget.current_question == case.questions[0]
    assert widget.questions_stack == case.questions[1:] + example_questionnaire_with_switch.questions[2:]

    helpers.assert_widget_ui_matches_state(widget)

    # Test that the default case is used if no case matches.
    widget = QuestionnaireWidget(example_questionnaire_with_switch)

    first_question = example_questionnaire_with_switch.questions[0]
    third_answer = first_question.answers[2]
    third_button = helpers.get_answer_button(widget, 2)  # Third answer button

    third_button.click()

    assert widget.code_output == third_answer.templates[0].code
    assert widget.commentary == third_answer.commentary
    assert widget.question_answers == [(first_question, 2)]
    assert widget.variables == {first_question.variable: 2}

    switch = example_questionnaire_with_switch.questions[1]
    case = switch.cases[2]  # Default case where value is None

    assert widget.current_question == case.questions[0]
    assert widget.questions_stack == case.questions[1:] + example_questionnaire_with_switch.questions[2:]

    helpers.assert_widget_ui_matches_state(widget)


def test_questionnaire_followup_switch(example_questionnaire_with_followup_switch, helpers):
    """Test that a switch based on a follow-up question works correctly."""

    widget = QuestionnaireWidget(example_questionnaire_with_followup_switch)

    # Select the first answer to the first question (which has a follow-up question)
    first_question = example_questionnaire_with_followup_switch.questions[0]
    first_answer = first_question.answers[0]
    first_button = helpers.get_answer_button(widget, 0)  # First answer button

    first_button.click()

    assert widget.code_output == first_answer.templates[0].code
    assert widget.commentary == first_answer.commentary

    assert widget.question_answers == [(first_question, 0)]

    followup_question = first_answer.followups[0]
    assert widget.current_question == followup_question
    assert widget.questions_stack == example_questionnaire_with_followup_switch.questions[1:]

    helpers.assert_widget_ui_matches_state(widget)

    # Select the first answer to the follow-up question (which sets the switch variable)

    followup_answer = followup_question.answers[0]
    followup_button = helpers.get_answer_button(widget, 0)  # First answer button for the followup question

    followup_button.click()

    assert widget.code_output == "example_code followup_code {{code}}"
    assert widget.commentary == followup_answer.commentary

    assert widget.question_answers == [(first_question, 0), (followup_question, 0)]
    assert widget.variables == {followup_question.variable: 0}

    # Check that the switch question is handled correctly

    switch = example_questionnaire_with_followup_switch.questions[1]
    case = switch.cases[0]

    assert widget.current_question == case.questions[0]
    assert (
        widget.questions_stack
        == case.questions[1:] + example_questionnaire_with_followup_switch.questions[2:]
    )

    helpers.assert_widget_ui_matches_state(widget)

    # Test that the second case of the switch works correctly

    widget = QuestionnaireWidget(example_questionnaire_with_followup_switch)

    # Select the first answer to the first question (which has a follow-up question)
    first_question = example_questionnaire_with_followup_switch.questions[0]
    first_answer = first_question.answers[0]
    first_button = helpers.get_answer_button(widget, 0)  # First answer button

    first_button.click()

    assert widget.code_output == first_answer.templates[0].code
    assert widget.commentary == first_answer.commentary

    assert widget.question_answers == [(first_question, 0)]

    followup_question = first_answer.followups[0]
    assert widget.current_question == followup_question
    assert widget.questions_stack == example_questionnaire_with_followup_switch.questions[1:]

    helpers.assert_widget_ui_matches_state(widget)

    # Select the Second answer to the follow-up question (which sets the switch variable)

    followup_answer = followup_question.answers[1]
    followup_button = helpers.get_answer_button(widget, 1)  # Second answer button for the followup question

    followup_button.click()

    assert widget.code_output == "example_code second_followup_code {{code}}"
    assert widget.commentary == followup_answer.commentary

    assert widget.question_answers == [(first_question, 0), (followup_question, 1)]
    assert widget.variables == {followup_question.variable: 1}

    # Check that the switch question is handled correctly

    switch = example_questionnaire_with_followup_switch.questions[1]
    case = switch.cases[1]

    assert widget.current_question == case.questions[0]
    assert (
        widget.questions_stack
        == case.questions[1:] + example_questionnaire_with_followup_switch.questions[2:]
    )

    helpers.assert_widget_ui_matches_state(widget)

    # Test that the default case of the switch works correctly if the follow-up question is skipped

    widget = QuestionnaireWidget(example_questionnaire_with_followup_switch)

    # Select the second answer to the first question (which does not have a follow-up question)
    first_question = example_questionnaire_with_followup_switch.questions[0]
    second_answer = first_question.answers[1]
    second_button = helpers.get_answer_button(widget, 1)  # Second answer button

    second_button.click()

    assert widget.code_output == second_answer.templates[0].code
    assert widget.commentary == second_answer.commentary
    assert widget.question_answers == [(first_question, 1)]
    assert widget.variables == {}

    switch = example_questionnaire_with_followup_switch.questions[1]
    case = switch.cases[2]  # Default case where value is None

    assert widget.current_question == case.questions[0]
    assert (
        widget.questions_stack
        == case.questions[1:] + example_questionnaire_with_followup_switch.questions[2:]
    )

    helpers.assert_widget_ui_matches_state(widget)


def test_questionnaire_previous_question_navigation(example_questionnaire, helpers):
    """Test that clicking on a previous question button navigates back to that point in the questionnaire."""
    widget = QuestionnaireWidget(example_questionnaire)

    # Complete the first two questions
    answer_inds = [0, 1]
    expected_questions = [
        example_questionnaire.questions[0],
        example_questionnaire.questions[0].answers[0].followups[0],
    ]

    # Answer the first question
    first_button = helpers.get_answer_button(widget, answer_inds[0])
    first_button.click()

    # Answer the second question
    second_button = helpers.get_answer_button(widget, answer_inds[1])
    second_button.click()

    # Verify we're at the expected state after answering two questions
    assert widget.question_answers == list(zip(expected_questions, answer_inds, strict=False))
    assert widget.current_question == example_questionnaire.questions[0].answers[0].followups[1]

    # Now click on the second previous question button to go back to that point
    prev_button = helpers.get_prev_question_button(widget, 1)
    prev_button.click()

    # Verify we're back at the state after answering only the first question
    assert widget.question_answers == [(expected_questions[0], answer_inds[0])]
    assert widget.current_question == expected_questions[1]

    # Answer the second question differently this time
    different_answer_ind = 0  # Different from the original answer_inds[1]
    different_button = helpers.get_answer_button(widget, different_answer_ind)
    different_button.click()

    # Verify the new answer was recorded
    assert widget.question_answers == [(expected_questions[0], answer_inds[0]), 
                                      (expected_questions[1], different_answer_ind)]


def test_read_questions():
    """Test that the questions can be loaded from the packaged YAML file."""
    questions = load_questions()
    assert isinstance(questions, Questionnaire)


def test_questionnaire_feedback_url(example_questionnaire_with_feedback, helpers):
    """Test that the feedback URL is included when present in the questionnaire."""
    widget = QuestionnaireWidget(example_questionnaire_with_feedback)

    # Complete all questions
    _inds = [0, 1, 0, 0]
    for ans_ind in _inds:
        button = helpers.get_answer_button(widget, ans_ind)
        button.click()

    # Check that the questionnaire is completed
    assert widget.current_question is None
    assert widget.feedback_url == example_questionnaire_with_feedback.feedback_url

    # Verify the widget UI matches its state (including feedback URL check)
    helpers.assert_widget_ui_matches_state(widget)


def test_run_questionnaire(example_questionnaire, mocker):
    """Mock the display function to test that run_questionnaire() works correctly."""
    mocker.patch("scarlet2.questionnaire.questionnaire.load_questions")
    mocker.patch("scarlet2.questionnaire.questionnaire.QuestionnaireWidget")

    from scarlet2.questionnaire.questionnaire import QuestionnaireWidget, load_questions

    load_questions.return_value = example_questionnaire
    run_questionnaire()

    load_questions.assert_called_once()
    QuestionnaireWidget.assert_called_once_with(example_questionnaire)
    QuestionnaireWidget.return_value.show.assert_called_once()


def test_save_button_functionality(example_questionnaire, helpers, mocker, tmp_path):
    """Test that the save button functionality works correctly."""
    # Change to the temporary directory for file operations
    original_dir = Path.cwd()
    os.chdir(tmp_path)

    try:
        # Create the widget
        widget = QuestionnaireWidget(example_questionnaire)

        # Mock the open function to test file writing
        mock_open = mocker.patch("builtins.open", mocker.mock_open())

        # Test saving when there are no answers yet
        # Get the save button
        save_button_container = widget.question_box.children[-1]
        save_button = save_button_container.children[-1]
        assert isinstance(save_button, Button)
        assert save_button.description == "Save Answers"

        # Click the save button
        save_button.click()

        # Check that the warning message was set
        assert widget.save_message is None  # Message is reset after rendering

        # Answer a question to have something to save
        first_button = helpers.get_answer_button(widget, 0)
        first_button.click()

        # Click the save button again
        save_button_container = widget.question_box.children[-1]
        save_button = save_button_container.children[-1]
        save_button.click()

        # Check that the file was written with a name matching the expected pattern
        assert mock_open.call_count == 1
        filename = mock_open.call_args[0][0]
        assert filename.startswith("questionnaire_answers_")
        assert filename.endswith(".yaml")

        # Check that the success message was set
        assert widget.save_message is None  # Message is reset after rendering

        # Check that yaml.dump was called with the correct data
        yaml_dump_mock = mocker.patch("yaml.dump")

        # Click the save button again to trigger yaml.dump with our mock
        save_button.click()

        # Check that yaml.dump was called
        yaml_dump_mock.assert_called_once()

        # Check that the first argument to yaml.dump is a dict containing the answers
        dump_args = yaml_dump_mock.call_args[0]
        assert isinstance(dump_args[0], dict)
        assert "answers" in dump_args[0]
        assert len(dump_args[0]["answers"]) == 1
        assert dump_args[0]["answers"][0]["question"] == example_questionnaire.questions[0].question
        assert dump_args[0]["answers"][0]["answer"] == example_questionnaire.questions[0].answers[0].answer
        assert dump_args[0]["answers"][0]["value"] == 0
    finally:
        # Change back to the original directory
        os.chdir(original_dir)
