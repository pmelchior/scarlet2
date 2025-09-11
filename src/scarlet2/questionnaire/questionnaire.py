import os
import re
from importlib.resources import files

import jinja2
import markdown
import yaml
from IPython.display import display
from ipywidgets import HTML, Button, HBox, Layout, VBox
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from scarlet2.questionnaire.models import (
    Question,
    QuestionAnswer,
    QuestionAnswers,
    Questionnaire,
    Switch,
    Template,
)

PACKAGE_PATH = "scarlet2.questionnaire"
QUESTIONS_FILE_NAME = "questions.yaml"

VIEWS_PACKAGE_PATH = "scarlet2.questionnaire.views"
OUTPUT_BOX_TEMPLATE_FILE = "output_box.html.jinja"
OUTPUT_BOX_STYLE_FILE = "output_box.css"
QUESTION_BOX_STYLE_FILE = "question_box.css"


QUESTION_BOX_LAYOUT = Layout(
    padding="12px",
    backgroundColor="#f9f9f9",
    border="1px solid #ddd",
    borderRadius="10px",
    width="45%",
)

OUTPUT_BOX_LAYOUT = Layout(
    width="50%",
    margin="0 0 0 20px",
)


class QuestionnaireWidget:
    """A widget to run an interactive questionnaire in a Jupyter notebook."""

    def __init__(
        self,
        questionnaire: Questionnaire,
        save_directory: str | None = None,
        initial_answers: QuestionAnswers | None = None,
    ):
        self.questions = questionnaire.questions
        self.initial_template = questionnaire.initial_template
        self.initial_commentary = questionnaire.initial_commentary
        self.feedback_url = questionnaire.feedback_url
        self.save_directory = save_directory

        self._load_resources()

        self._init_state()
        self._init_ui()

        if initial_answers is not None:
            self._start_with_answers(initial_answers)
        else:
            self._render_output_box()
            self._render_next_question()

    def _load_resources(self):
        question_box_css_file = files(VIEWS_PACKAGE_PATH).joinpath(QUESTION_BOX_STYLE_FILE)
        with question_box_css_file.open("r") as f:
            self.question_box_css = f.read()

        output_box_css_file = files(VIEWS_PACKAGE_PATH).joinpath(OUTPUT_BOX_STYLE_FILE)
        with output_box_css_file.open("r") as f:
            self.output_box_css = f.read()

        template_file = files(VIEWS_PACKAGE_PATH).joinpath(OUTPUT_BOX_TEMPLATE_FILE)
        with template_file.open("r") as f:
            template_source = f.read()
        self.output_box_template = jinja2.Template(template_source)

        self.question_box_css_html = HTML(f"<style>{self.question_box_css}</style>")

    def _init_state(self):
        self.current_question = None
        self.questions_stack = []
        self.question_answers = []
        self.variables = {}
        self.code_output = self.initial_template
        self.commentary = self.initial_commentary
        self.save_message = None
        self._add_questions_to_stack(self.questions)

    def _init_ui(self):
        self.output_container = HTML()
        self.question_box = VBox([], layout=QUESTION_BOX_LAYOUT)
        # Add a class to the question box for CSS targeting
        self.question_box.add_class("question-box-container")
        self.output_box = VBox([self.output_container], layout=OUTPUT_BOX_LAYOUT)

        self.ui = HBox([self.question_box, self.output_box])

    def _start_with_answers(self, question_answers: QuestionAnswers):
        # First reset the state
        self._init_state()

        for qa in question_answers.answers:
            # Get the next question
            self.current_question = self._get_next_question()

            # Verify the question matches
            if self.current_question is None or self.current_question.question != qa.question:
                raise ValueError("Provided answers do not match the question flow.")

            if self.current_question.answers[qa.value].answer != qa.answer:
                raise ValueError("Provided answers do not match the question flow.")

            self._handle_answer(qa.value, render=False)

        self._render_output_box()
        self._render_next_question()

    def _add_questions_to_stack(self, questions: list[Question]):
        self.questions_stack = questions + self.questions_stack

    def _get_next_question(self) -> Question | None:
        if len(self.questions_stack) > 0:
            question = self.questions_stack.pop(0)
            if isinstance(question, Question):
                return question
            if isinstance(question, Switch):
                switch = question
                matching_case = None

                # Find the default case (where value is None)
                for case in switch.cases:
                    if case.value is None:
                        matching_case = case
                        break

                variable_value = self.variables.get(switch.switch, None)
                if variable_value is not None:
                    for case in switch.cases:
                        if case.value == variable_value:
                            matching_case = case
                            break
                if matching_case is not None:
                    self._add_questions_to_stack(matching_case.questions)
                return self._get_next_question()
        return None

    def _render_next_question(self):
        self.current_question = self._get_next_question()
        # Reset save message when rendering a new question
        self.save_message = None
        self._render_question_box()

    def _render_question_box(self):
        previous_qs = self._generate_previous_questions()

        # Create save button
        save_button = Button(
            description="Save Answers",
            tooltip="Save all answers to a YAML file",
        )
        save_button.add_class("save-button")
        save_button.on_click(self._save_answers)

        save_components = [save_button]

        if self.save_message:
            message_type = self.save_message["type"]
            message_text = self.save_message["text"]

            # Set color based on message type
            color = "green" if message_type == "success" else "orange"

            # Create message HTML
            message_html = HTML(f'<div class="save-message" style="color: {color}">{message_text}</div>')
            save_components = [message_html, save_button]
            self.save_message = None

        # Create a container for the save button
        save_button_container = VBox(
            save_components,
            layout=Layout(
                width="100%",
                padding="0",
            ),
        )
        save_button_container.add_class("save-button-container")

        if self.current_question is None:
            final_message = "<div>🎉 You're done!</div>"
            if self.feedback_url:
                final_message = (
                    final_message
                    + f"""
                    <div style="font-size: 0.9em;">
                    If you encountered any difficulties or have any suggestions,
                    <a href="{self.feedback_url}" target="_blank"
                    style="text-decoration: underline; color: #0066cc;">
                    please fill out our feedback form here.
                    </a>
                    </div>
                """
                )
            # Wrap the final message in a container
            final_message_container = VBox([HTML(final_message)], layout=Layout(margin="0 0 0 0"))

            self.question_box.children = (
                [self.question_box_css_html] + previous_qs + [final_message_container, save_button_container]
            )
            return

        q_label = HTML(f"<b>{self.current_question.question}</b>")

        buttons = []
        for i, answer in enumerate(self.current_question.answers):
            button = Button(
                description=answer.answer,
                tooltip=answer.tooltip,
                layout=Layout(width="auto", margin="4px 0"),
            )

            def on_click_handler(btn, index=i):
                self._handle_answer(index)

            button.on_click(on_click_handler)

            buttons.append(button)

        # Create a container for the buttons
        buttons_container = VBox(buttons, layout=Layout(margin="0"))

        self.question_box.children = (
            [self.question_box_css_html] + previous_qs + [q_label, buttons_container, save_button_container]
        )

    def _generate_previous_questions(self):
        items = []
        for i, (q, ans_ind) in enumerate(self.question_answers):
            ans = q.answers[ans_ind]

            # The clickable button styled like text
            btn = Button(
                description=f"{q.question} — {ans.answer}",
                tooltip="Click to go back",  # native title tooltip as a fallback
                layout=Layout(width="auto", margin="2px 0"),
                style={"button_color": "transparent", "font_weight": "normal"},
            )
            btn.add_class("prev-btn")

            qas = self._get_question_answers(up_to_index=i)

            def on_click_handler(btn, qas=qas):
                self._start_with_answers(qas)

            btn.on_click(on_click_handler)

            # The custom tooltip node (shown on hover via CSS)
            tip_html = HTML("<div class='tooltip'>Click to go back</div>")

            # Wrap button + tooltip in a positioned container
            container = HBox([btn, tip_html], layout=Layout(position="relative"))
            container.add_class("prev-item")

            items.append(container)

        return items

    def _handle_answer(self, answer_index: int, render: bool = True):
        answer = self.current_question.answers[answer_index]

        self._update_template(answer.templates)
        self.commentary = answer.commentary

        self._add_questions_to_stack(answer.followups)
        self.question_answers.append((self.current_question, answer_index))
        if self.current_question.variable is not None:
            self.variables[self.current_question.variable] = answer_index

        if render:
            self._render_output_box()
            self._render_next_question()

    def _update_template(self, templates: list[Template]):
        for t in templates:
            pattern = r"\{\{\s*" + re.escape(t.replacement) + r"\s*\}\}"
            self.code_output = re.sub(pattern, t.code, self.code_output)

    def _render_output_box(self):
        output_code = re.sub(r"\{\{\s*\w+\s*\}\}", "", self.code_output)
        commentary_html = markdown.markdown(self.commentary, extensions=["extra"])

        formatter = HtmlFormatter(style="monokai", noclasses=True)
        highlighted_code = highlight(output_code, PythonLexer(), formatter)

        # Use pre-loaded template and CSS
        html_content = f"<style>{self.output_box_css}</style>\n" + self.output_box_template.render(
            highlighted_code=highlighted_code,
            raw_code=output_code,
            commentary_html=commentary_html,
        )

        self.output_container.value = html_content

    def _get_question_answers(self, up_to_index=None) -> QuestionAnswers:
        """Get a QuestionAnswers model containing all answers, optionally up to the specified index.

        Args:
            up_to_index (int, optional): The index up to which to include answers.
                If None, includes all answers. Defaults to None.

        Returns:
            QuestionAnswers: The collected question answers.
        """
        question_answers = QuestionAnswers()
        answers_to_process = (
            self.question_answers if up_to_index is None else self.question_answers[:up_to_index]
        )

        for question, answer_index in answers_to_process:
            answer = question.answers[answer_index]
            question_answer = QuestionAnswer(
                question=question.question, answer=answer.answer, value=answer_index
            )
            question_answers.answers.append(question_answer)

        return question_answers

    def _save_answers(self, _):
        """Save all user answers to a YAML file."""
        # Check if there are any answers to save
        if not self.question_answers:
            # Set warning message
            self.save_message = {"type": "warning", "text": "⚠️ No answers to save yet"}
            # Re-render the question box to show the message
            self._render_question_box()
            return None

        # Create a QuestionAnswers model
        question_answers = self._get_question_answers()

        # Create a filename with timestamp to avoid overwriting
        timestamp = question_answers.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"scarlet2_questionnaire_answers_{timestamp}.yaml"

        if self.save_directory is not None:
            filename = os.path.join(self.save_directory, filename)

        # Save to file
        with open(filename, "w") as f:
            yaml.dump(question_answers.model_dump(), f, default_flow_style=False)

        # Set success message
        self.save_message = {"type": "success", "text": f"✅ Answers saved to {filename}"}

        # Re-render the question box to show the message
        self._render_question_box()

        return filename

    def show(self):
        """Display the widget in a Jupyter notebook."""
        display(self.ui)


def load_questions() -> Questionnaire:
    """Load the questionnaire from the packaged YAML file.

    Returns:
        Questionnaire: The loaded questionnaire model.
    """
    questions_path = files(PACKAGE_PATH).joinpath(QUESTIONS_FILE_NAME)
    with questions_path.open("r") as f:
        raw = yaml.safe_load(f)
        return Questionnaire.model_validate(raw)


def run_questionnaire(answer_path=None, *, save_directory=None):
    """Run the Scarlet2 initialization questionnaire in a Jupyter notebook.

    The questionnaire guides the user through a series of questions to set up
    the initialization of a Scarlet2 project that fits their use case.

    The user will be presented with questions and multiple-choice answers, and
    at the end of the questionnaire, a code snippet that can be used as a
    template for initializing Scarlet2 will be generated.
    
    Args:
        answer_path (str, optional): Path to a YAML file with pre-filled answers
            to start the questionnaire from. Defaults to None.
        save_directory (str, optional): Directory where the answers will be saved.
            If None, answers will be saved in the current working directory.
            Defaults to None.
    """
    questions = load_questions()
    if answer_path is not None:
        with open(answer_path, "r") as f:
            raw_answers = yaml.safe_load(f)
            start_answers = QuestionAnswers.model_validate(raw_answers)
            app = QuestionnaireWidget(questions, save_directory=save_directory, initial_answers=start_answers)
    else:
        app = QuestionnaireWidget(questions, save_directory=save_directory)
    app.show()
