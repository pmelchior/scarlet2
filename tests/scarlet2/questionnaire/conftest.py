import json
import re
from importlib.resources import files
from pathlib import Path

import yaml
from ipywidgets import HTML, Button, HBox, Label, VBox
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer
from pytest import fixture
from scarlet2.questionnaire.models import Questionnaire
from scarlet2.questionnaire.questionnaire import (
    OUTPUT_BOX_LAYOUT,
    OUTPUT_BOX_STYLE_FILE,
    QUESTION_BOX_LAYOUT,
    VIEWS_PACKAGE_PATH,
)


@fixture
def data_dir():
    """Path to the data directory containing the example questionnaire YAML file."""
    return Path(__file__).parent / "data"


@fixture
def example_questionnaire_dict(data_dir):
    """An example questionnaire dictionary"""
    yaml_path = data_dir / "example_questionnaire.yaml"
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)


@fixture
def example_questionnaire(example_questionnaire_dict):
    """An example Questionnaire model instance"""
    return Questionnaire.model_validate(example_questionnaire_dict)


@fixture
def example_questionnaire_with_switch_dict(data_dir):
    """An example questionnaire dictionary with a switch question"""
    yaml_path = data_dir / "example_questionnaire_switch.yaml"
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)


@fixture
def example_questionnaire_with_switch(example_questionnaire_with_switch_dict):
    """An example Questionnaire model instance with a switch question"""
    return Questionnaire.model_validate(example_questionnaire_with_switch_dict)


@fixture
def questionnaire_with_followup_switch_example_dict(data_dir):
    """An example questionnaire dictionary with a switch question"""
    yaml_path = data_dir / "example_questionnaire_followup_switch.yaml"
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)


@fixture
def example_questionnaire_with_followup_switch(questionnaire_with_followup_switch_example_dict):
    """An example Questionnaire model instance with a switch question"""
    return Questionnaire.model_validate(questionnaire_with_followup_switch_example_dict)


class Helpers:
    """Helper functions for testing the QuestionnaireWidget"""

    @staticmethod
    def assert_widget_ui_matches_state(widget):
        """Assert that the widget's UI matches its internal state."""
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

            for btn, ans in zip(
                widget.question_box.children[qs_ind + 1 :], widget.current_question.answers, strict=False
            ):
                assert isinstance(btn, Button)
                assert btn.description == ans.answer
                assert btn.tooltip == ans.tooltip

        else:
            assert isinstance(widget.question_box.children[-1], Label)
            assert "You're done" in widget.question_box.children[-1].value


@fixture
def helpers():
    """Provide helper functions for testing."""
    return Helpers()
