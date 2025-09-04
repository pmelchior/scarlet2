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

from scarlet2.questionnaire.models import Question, Questionnaire, Switch, Template

PACKAGE_PATH = "scarlet2.questionnaire"
QUESTIONS_FILE_NAME = "questions.yaml"

VIEWS_PACKAGE_PATH = "scarlet2.questionnaire.views"
OUTPUT_BOX_TEMPLATE_FILE = "output_box.html.jinja"
OUTPUT_BOX_STYLE_FILE = "output_box.css"

_PREV_TOOLTIP_CSS = HTML("""
    <style>
    .prev-item { position: relative; }
    .prev-btn { width: 100%; text-align: left; }

    .prev-item .tooltip {
      position: absolute;
      left: 100%;
      top: 50%;
      transform: translateY(-50%) translateX(8px);
      background: #333;
      color: #fff;
      padding: 6px 8px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.3;
      box-shadow: 0 4px 12px rgba(0,0,0,.2);
      white-space: normal;
      min-width: 160px;
      max-width: 320px;
      opacity: 0;
      visibility: hidden;
      transition: opacity .12s ease, transform .12s ease;
      pointer-events: none;
      z-index: 9999;
    }
    .prev-item:hover .tooltip {
      opacity: 1;
      visibility: visible;
      transform: translateY(-50%) translateX(10px);
    }
    .prev-item .tooltip::before {
      content: "";
      position: absolute;
      left: -6px;
      top: 50%;
      transform: translateY(-50%);
      border-width: 6px;
      border-style: solid;
      border-color: transparent #333 transparent transparent;
    }
    </style>
    """)


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

    def __init__(self, questionnaire: Questionnaire):
        self.questions = questionnaire.questions
        self.code_output = questionnaire.initial_template
        self.commentary = questionnaire.initial_commentary
        self.feedback_url = questionnaire.feedback_url

        self._init_questions()
        self._init_ui()

        self._render_output_box()
        self._render_next_question()

    def _init_questions(self):
        self.current_question = None
        self.questions_stack = []
        self.question_answers = []
        self.variables = {}
        self._add_questions_to_stack(self.questions)

    def _init_ui(self):
        self.output_container = HTML()
        self.question_box = VBox([], layout=QUESTION_BOX_LAYOUT)
        self.output_box = VBox([self.output_container], layout=OUTPUT_BOX_LAYOUT)

        self.ui = HBox([self.question_box, self.output_box])

    def _start_with_answers(self, answers: list[tuple[Question, int]]):
        self._init_questions()
        for question, answer_index in answers:
            current_question = self._get_next_question()
            if current_question != question:
                raise ValueError("Provided answers do not match the question flow.")
            self._handle_answer(answer_index, render=False)
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
        self._render_question_box()

    def _render_question_box(self):
        previous_qs = self._generate_previous_questions()

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
            self.question_box.children = previous_qs + [HTML(final_message)]
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

        self.question_box.children = [_PREV_TOOLTIP_CSS] + previous_qs + [q_label] + buttons

    def _generate_previous_questions(self):
        items = []
        for i, (q, ans_ind) in enumerate(self.question_answers):
            ans = q.answers[ans_ind]

            # The clickable button (Python callback works)
            btn = Button(
                description=f"{q.question}: {ans.answer}",
                tooltip="Click to go back",  # native title tooltip as a fallback
                layout=Layout(width="100%", margin="2px 0"),
            )
            btn.add_class("prev-btn")

            # Python callback
            def _on_click(btn, index=i):
                self._start_with_answers(self.question_answers[:index + 1])

            btn.on_click(_on_click)

            # The custom tooltip node (shown on hover via CSS)
            tip_html = HTML(f"<div class='tooltip'>Click to go back</div>")

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

        # Load Jinja template
        template_file = files(VIEWS_PACKAGE_PATH).joinpath(OUTPUT_BOX_TEMPLATE_FILE)
        with template_file.open("r") as f:
            template_source = f.read()
        template = jinja2.Template(template_source)

        # Load additional CSS for styling
        css_file = files(VIEWS_PACKAGE_PATH).joinpath(OUTPUT_BOX_STYLE_FILE)
        with css_file.open("r") as f:
            css_content = f.read()

        html_content = f"<style>{css_content}</style>\n" + template.render(
            highlighted_code=highlighted_code,
            raw_code=output_code,
            commentary_html=commentary_html,
        )

        self.output_container.value = html_content

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


def run_questionnaire():
    """Run the Scarlet2 initialization questionnaire in a Jupyter notebook.

    The questionnaire guides the user through a series of questions to set up
    the initialization of a Scarlet2 project that fits their use case.

    The user will be presented with questions and multiple-choice answers, and
    at the end of the questionnaire, a code snippet that can be used as a
    template for initializing Scarlet2 will be generated.
    """
    questions = load_questions()
    app = QuestionnaireWidget(questions)
    app.show()
