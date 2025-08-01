from importlib.resources import files
import re

import markdown
import yaml
from ipywidgets import VBox, HBox, Button, Label, HTML, Layout
from IPython.display import display
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

from scarlet2.questionnaire.models import Template, Questionnaire


FILE_PACKAGE_PATH = "scarlet2.questionnaire"
FILE_NAME = "questions.yaml"


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
    def __init__(self, questionnaire):
        self.questions = questionnaire.questions
        self.code_output = questionnaire.initial_template
        self.commentary = ""

        self._init_questions()
        self._init_ui()

        self._render_next_question()

    def _init_questions(self):
        self.current_question = None
        self.questions_stack = []
        self.question_answers = []
        self._add_questions_to_stack(self.questions)

    def _init_ui(self):
        self.output_container = HTML()
        self.question_box = VBox(layout=QUESTION_BOX_LAYOUT)
        self.output_box = VBox([self.output_container], layout=OUTPUT_BOX_LAYOUT)

        self.ui = HBox([self.question_box, self.output_box])

    def _add_questions_to_stack(self, questions):
        self.questions_stack = questions + self.questions_stack

    def _render_next_question(self):
        # Custom styled HTML container for right panel
        self.current_question = self.questions_stack.pop(0) if len(self.questions_stack) > 0 else None
        self._render_question_box()

    def _render_question_box(self):
        previous_qs = [HTML(f"<div style='background_color: #111'><span style='color: #888; padding-right: 10px'>{q.question}</span><span style='color: #555'>{a.answer}</span></div>") for q, a in self.question_answers]

        if self.current_question is None:
            return VBox(previous_qs + [Label("🎉 You're done!")], layout=Layout(
            padding="12px",
            backgroundColor="#f9f9f9",
            border="1px solid #ddd",
            borderRadius="10px",
            width="45%",
        ))

        q_label = HTML(f"<b>{self.current_question.question}</b>")

        buttons = []
        for i, answer in enumerate(self.current_question.answers):
            button = Button(
                description=answer.answer,
                tooltip=answer.tooltip,
                layout=Layout(width="auto", margin="4px 0"),
                button_style="",
            )

            def on_click_handler(btn, index=i):
                self._handle_answer(index)

            button.on_click(on_click_handler)

            buttons.append(button)

        self.question_box.children = previous_qs + [q_label] + buttons

    def _handle_answer(self, answer_index):
        answer = self.current_question.answers[answer_index]

        self._update_template(answer.templates)
        self.commentary = answer.commentary
        self._render_output_box()

        self.questions_stack = answer.followups + self.questions_stack
        self.question_answers.append((self.current_question, answer_index))
        self._render_next_question()

    def _update_template(self, templates: list[Template]):
        for t in templates:
            pattern = r"\{\{\s*" + re.escape(t.replacement) + r"\s*\}\}"
            self.code_output = re.sub(pattern, t.code, self.code_output)

    def _render_output_box(self):
        output_code = re.sub(r"\{\{.*?\}\}", "", self.code_output)
        commentary_text = markdown.markdown(self.commentary, extensions=["extra"])
        commentary_text = re.sub(
            r'<a href="(.*?)">',
            r'<a href="\1" target="_blank" style="color:#0366d6; text-decoration:underline;">',
            commentary_text
        )

        formatter = HtmlFormatter(style="monokai", noclasses=True)
        highlighted_code = highlight(output_code, PythonLexer(), formatter)

        escaped_code = output_code.replace("\\", "\\\\").replace("`", "\\`")

        html_content = f"""
        <div style="
            background-color: #272822;
            color: #f1f1f1;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #444;
            font-family: monospace;
            overflow-x: auto;
        ">
            {highlighted_code}
            <div style="display: flex; justify-content: flex-end; margin-top: 8px;">
                <button onclick="navigator.clipboard.writeText(`{escaped_code}`)"
                    style="
                        font-size: 12px;
                        padding: 4px 8px;
                        background-color: #272822;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    ">
                    📋 Copy
                </button>
            </div>
        </div>
        <div style="
        background-color: #f6f8fa;
        color: #333;
        padding: 8px 12px;
        border-left: 4px solid #0366d6;
        border-radius: 6px;
        font-size: 14px;
        font-family: sans-serif;
    ">
        {commentary_text}
    </div>
        """
        self.output_container.value = html_content

    def show(self):
        display(self.ui)


def load_questions() -> Questionnaire:
    """Load the questionnaire from the packaged YAML file.

    Returns:
        Questionnaire: The loaded questionnaire model.
    """
    questions_path = files(FILE_PACKAGE_PATH).joinpath(FILE_NAME)
    with questions_path.open("r") as f:
        raw = yaml.safe_load(f)
        return Questionnaire.model_validate(raw)


def run_questionnaire():
    questions = load_questions()
    app = QuestionnaireWidget(questions)
    app.show()
