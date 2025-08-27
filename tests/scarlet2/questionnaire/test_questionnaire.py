from scarlet2.questionnaire import QuestionnaireWidget


def test_questionnaire_widget_init(example_questionnaire, helpers):

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
    widget = QuestionnaireWidget(example_questionnaire)

    first_question = example_questionnaire.questions[0]
    first_answer = first_question.answers[0]
    first_button = widget.question_box.children[1]  # First button after question label

    first_button.click()

    assert widget.code_output == "example_code {{follow}}"
    assert widget.commentary == "This is some commentary."

    assert widget.question_answers == [(first_question, 0)]

    assert widget.current_question == first_answer.followups[0]
    assert widget.questions_stack == first_answer.followups[1:] + example_questionnaire.questions[1:]

    helpers.assert_widget_ui_matches_state(widget)


def test_questionnaire_handle_answer_selection(example_questionnaire, helpers):
    widget = QuestionnaireWidget(example_questionnaire)

    first_question = example_questionnaire.questions[0]
    first_answer = first_question.answers[0]
    first_button = widget.question_box.children[1]  # First button after question label

    first_button.click()

    assert widget.code_output == "example_code {{follow}}"
    assert widget.commentary == "This is some commentary."

    assert widget.question_answers == [(first_question, 0)]

    assert widget.current_question == first_answer.followups[0]
    assert widget.questions_stack == first_answer.followups[1:] + example_questionnaire.questions[1:]

    helpers.assert_widget_ui_matches_state(widget)


def test_questionnaire_complete_all_questions(example_questionnaire, helpers):
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
        assert widget.question_answers == list(zip(expected_questions[:i], answer_inds[:i]))
        helpers.assert_widget_ui_matches_state(widget)

        button = widget.question_box.children[i + 1 + ans_ind]
        button.click()

    assert widget.current_question is None
    assert widget.question_answers == list(zip(expected_questions, answer_inds))
    helpers.assert_widget_ui_matches_state(widget)
