import json
import re
from importlib.resources import files

from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer
from pytest import fixture

from ipywidgets import HTML, HBox, VBox, Button, Label

from scarlet2.questionnaire.models import Questionnaire
from scarlet2.questionnaire.questionnaire import OUTPUT_BOX_LAYOUT, QUESTION_BOX_LAYOUT, VIEWS_PACKAGE_PATH, \
    OUTPUT_BOX_STYLE_FILE


@fixture
def example_questionnaire_dict():
    """An example questionnaire dictionary"""
    return {
        "initial_template": "{{code}}",
        "initial_commentary": "This is an example commentary.",
        "questions": [
            {
                "question": "Example question?",
                "answers": [
                    {
                        "answer": "Example answer",
                        "tooltip": "This is an example tooltip.",
                        "templates": [{"replacement": "code", "code": "example_code {{follow}}"}],
                        "followups": [
                            {
                                "question": "Follow-up question?",
                                "answers": [
                                    {
                                        "answer": "Follow-up answer",
                                        "tooltip": "This is a follow-up tooltip.",
                                        "templates": [{"replacement": "follow", "code": "followup_code\n{{code}}"}],
                                        "followups": [],
                                        "commentary": "",
                                    },
                                    {
                                        "answer": "Second follow-up answer",
                                        "tooltip": "This is a second follow-up tooltip.",
                                        "templates": [{"replacement": "follow", "code": "second_followup_code\n{{code}}"}],
                                    }
                                ],
                            },
                            {
                                "question": "Another follow-up question?",
                                "answers": [
                                    {
                                        "answer": "Another follow-up answer",
                                        "templates": [],
                                    }
                                ],
                            }
                        ],
                        "commentary": "This is some commentary.",
                    },
                    {
                        "answer": "Another answer",
                        "tooltip": "This is another tooltip.",
                        "templates": [{"replacement": "code", "code": "another_code\n{{code}}"}],
                        "followups": [],
                        "commentary": "Some other commentary.",
                    },
                ],
            },
            {
                "question": "Second question?",
                "answers": [
                    {
                        "answer": "Second answer",
                        "tooltip": "This is a second tooltip.",
                        "templates": [{"replacement": "code", "code": "second_code"}],
                        "followups": [],
                        "commentary": "Next commentary.",
                    }
                ],
            }
        ],
    }

@fixture
def example_questionnaire(example_questionnaire_dict):
    return Questionnaire.model_validate(example_questionnaire_dict)

class Helpers:

    @staticmethod
    def assert_widget_ui_matches_state(widget):
        assert isinstance(widget.ui, HBox)
        assert widget.ui.children == (widget.question_box, widget.output_box)

        assert isinstance(widget.output_box, VBox)
        assert widget.output_box.children == (widget.output_container,)
        assert widget.output_box.layout == OUTPUT_BOX_LAYOUT

        assert isinstance(widget.output_container, HTML)

        # check output container contains css from output_box css file
        css_file = files(VIEWS_PACKAGE_PATH).joinpath(OUTPUT_BOX_STYLE_FILE)
        with css_file.open("r") as f:
            css_content = f.read()

        assert css_content in widget.output_container.value

        output_code = re.sub(r"\{\{\s*\w+\s*\}\}", "", widget.code_output)

        html = widget.output_container.value

        # regex to capture the JS argument
        match = re.search(r"navigator\.clipboard\.writeText\((.*?)\)", html)
        assert match, "No copy button found"

        actual_arg = match.group(1)
        expected_arg = json.dumps(output_code)  # exactly how Jinja|tojson would encode it

        assert actual_arg.strip() == expected_arg

        formatter = HtmlFormatter(style="monokai", noclasses=True)
        highlighted_code = highlight(output_code, PythonLexer(), formatter)

        assert highlighted_code in widget.output_container.value

        assert isinstance(widget.question_box, VBox)
        assert widget.question_box.layout == QUESTION_BOX_LAYOUT

        len_cur_answers = len(widget.current_question.answers) if widget.current_question else 0
        expected_children_count = len(widget.question_answers) + 1 + len_cur_answers

        assert len(widget.question_box.children) == expected_children_count
        for i in range(len(widget.question_answers)):
            assert isinstance(widget.question_box.children[i], HTML)
            question = widget.question_answers[i][0]
            assert question.question in widget.question_box.children[i].value
            ans_index = widget.question_answers[i][1]
            assert question.answers[ans_index].answer in widget.question_box.children[i].value

        if widget.current_question is not None:
            qs_ind = len(widget.question_answers)

            assert isinstance(widget.question_box.children[qs_ind], HTML)
            assert widget.current_question.question in widget.question_box.children[qs_ind].value

            for btn, ans in zip(widget.question_box.children[qs_ind + 1:], widget.current_question.answers):
                assert isinstance(btn, Button)
                assert btn.description == ans.answer
                assert btn.tooltip == ans.tooltip

        else:
            assert isinstance(widget.question_box.children[-1], Label)
            assert "You're done" in widget.question_box.children[-1].value


@fixture
def helpers():
    return Helpers()
