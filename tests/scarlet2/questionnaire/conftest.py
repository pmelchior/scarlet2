import json
import re
from importlib.resources import files
from pathlib import Path

import yaml
from ipywidgets import HTML, Button, HBox, VBox
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


@fixture
def example_questionnaire_with_feedback(example_questionnaire):
    """An example Questionnaire model instance with a feedback URL"""
    questionnaire = example_questionnaire.model_copy(deep=True)
    questionnaire.feedback_url = "https://example.com/feedback"
    return questionnaire


class Helpers:
    """Helper functions for testing the QuestionnaireWidget"""

    @staticmethod
    def get_answer_button(widget, answer_index):
        """Get an answer button from the question box children.

        Args:
            widget: The QuestionnaireWidget instance
            answer_index: The index of the answer button to get

        Returns:
            The answer button widget
        """
        # The first element is the CSS snippet
        # Then there are previous question containers
        # Then there's the current question label
        # Then there's the buttons container
        # Finally, there's the save button container
        css_offset = 1
        prev_questions_offset = len(widget.question_answers)
        question_label_offset = 1

        # Get the buttons container which is at index after the question label
        buttons_container_index = css_offset + prev_questions_offset + question_label_offset
        buttons_container = widget.question_box.children[buttons_container_index]

        # Return the specific button from the buttons container
        return buttons_container.children[answer_index]

    @staticmethod
    def get_prev_question_button(widget, question_index):
        """Get a previous question button from the question box children.

        Args:
            widget: The QuestionnaireWidget instance
            question_index: The index of the previous question to get

        Returns:
            The previous question button widget
        """
        # The first element is the CSS snippet
        # Then there are previous question containers
        css_offset = 1

        container = widget.question_box.children[css_offset + question_index]
        # The button is the first child of the container
        return container.children[0]

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
        # Add 1 for the CSS snippet, 1 for the save button container
        css_snippet_count = 1
        save_button_container_count = 1

        # If there's a current question, we have:
        # - CSS snippet
        # - Previous question containers
        # - Current question label
        # - Buttons container
        # - Save button container
        if widget.current_question:
            expected_children_count = css_snippet_count + len(widget.question_answers) + 1 + 1 + save_button_container_count
        # If there's no current question, we have:
        # - CSS snippet
        # - Previous question containers
        # - Final message container
        # - Save button container
        else:
            expected_children_count = css_snippet_count + len(widget.question_answers) + 1 + save_button_container_count

        assert len(widget.question_box.children) == expected_children_count

        # Skip the CSS snippet
        css_offset = 1

        for i in range(len(widget.question_answers)):
            # Add the CSS offset to the index
            child_index = i + css_offset
            assert isinstance(widget.question_box.children[child_index], HBox)
            # The button is the first child of the container
            btn = widget.question_box.children[child_index].children[0]
            question = widget.question_answers[i][0]
            assert question.question in btn.description
            ans_index = widget.question_answers[i][1]
            assert question.answers[ans_index].answer in btn.description

        if widget.current_question is not None:
            # Add the CSS offset to the index
            qs_ind = len(widget.question_answers) + css_offset

            assert isinstance(widget.question_box.children[qs_ind], HTML)
            assert widget.current_question.question in widget.question_box.children[qs_ind].value

            # Get the buttons container which is at index after the question label
            buttons_container_index = qs_ind + 1
            buttons_container = widget.question_box.children[buttons_container_index]
            assert isinstance(buttons_container, VBox)

            # Check each button in the buttons container
            for btn, ans in zip(
                buttons_container.children, widget.current_question.answers, strict=False
            ):
                assert isinstance(btn, Button)
                assert btn.description == ans.answer
                assert btn.tooltip == ans.tooltip

        else:
            # When there's no current question, we have:
            # - CSS snippet
            # - Previous question containers
            # - Final message container
            # - Save button container

            # Get the final message container which is at index before the save button container
            final_message_index = len(widget.question_box.children) - 2
            final_message_container = widget.question_box.children[final_message_index]
            assert isinstance(final_message_container, VBox)

            # The final message is in the HTML child of the container
            final_message_html = final_message_container.children[0]
            assert isinstance(final_message_html, HTML)
            final_message = final_message_html.value
            assert "You're done" in final_message

            # Check for feedback URL if present in the questionnaire
            if widget.feedback_url:
                assert widget.feedback_url in final_message
                assert "feedback form" in final_message

        # Check that the last child is the save button container
        save_button_container = widget.question_box.children[-1]
        assert isinstance(save_button_container, VBox)
        assert len(save_button_container.children) > 0
        save_button = save_button_container.children[-1]
        assert isinstance(save_button, Button)
        assert save_button.description == "Save Answers"


@fixture
def helpers():
    """Provide helper functions for testing."""
    return Helpers()
