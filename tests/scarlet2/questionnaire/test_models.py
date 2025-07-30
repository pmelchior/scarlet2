import pytest

from scarlet2.questionnaire.models import Questionnaire
from scarlet2.questionnaire.questionnaire import load_questions


def test_validate_model(example_questionnaire_dict):
    questionnaire = Questionnaire.model_validate(example_questionnaire_dict)
    assert questionnaire.initial_template == "{{code}}"
    assert len(questionnaire.questions) == 1
    question = questionnaire.questions[0]
    assert question.question == "Example question?"
    assert len(question.answers) == 2
    answer = question.answers[0]
    assert answer.answer == "Example answer"
    assert len(answer.templates) == 1
    assert answer.templates[0].replacement == "{{code}}"
    assert len(answer.followups) == 1
    followup = answer.followups[0]
    assert followup.question == "Follow-up question?"

def test_model_fails(example_questionnaire_dict):
    invalid_dict = example_questionnaire_dict.copy()
    invalid_dict["initial_template"] = 123

    with pytest.raises(ValueError):
        Questionnaire.model_validate(invalid_dict)

    invalid_dict = example_questionnaire_dict.copy()
    invalid_dict["questions"][0]["question"] = 123

    with pytest.raises(ValueError):
        Questionnaire.model_validate(invalid_dict)

def test_read_questions():
    questions = load_questions()
    assert isinstance(questions, Questionnaire)
