from importlib.resources import files
from functools import partial
import re

import markdown
import yaml
from charset_normalizer.cli import query_yes_no
from ipywidgets import VBox, HBox, Button, Label, HTML, Layout, Output
from IPython.display import display
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

from scarlet2.questionnaire.models import Template, Questionnaire, Switch, Question

FILE_PACKAGE_PATH = "scarlet2.questionnaire"
FILE_NAME = "questions.yaml"


class QuestionnaireWidget:
    def __init__(self, questionnaire):
        self.questions = questionnaire.questions
        self.questions_stack = []
        self.question_answers = []
        self.variables = {}
        self.code_output = questionnaire.initial_template
        self.commentary = ""
        self.output = Output()
        self.output_container = HTML()

        self.ui = VBox(layout=Layout(padding="10px"))
        self._add_questions_to_stack(self.questions)
        self._render_next_question()

    def _add_questions_to_stack(self, questions):
        self.questions_stack = questions + self.questions_stack

    def _get_next_question(self):
        if len(self.questions_stack) > 0:
            question = self.questions_stack.pop(0)
            if isinstance(question, Question):
                return question
            if isinstance(question, Switch):
                variable_value = self.variables.get(question.variable, None)
                if variable_value is not None:
                    for case in question.cases:
                        if case.value == variable_value:
                            self._add_questions_to_stack(case.questions)
                            break
                    else:
                        for case in question.cases:
                            if case.value is None:
                                self._add_questions_to_stack(case.questions)
                                break
                else:
                    for case in question.cases:
                        if case.value is None:
                            self._add_questions_to_stack(case.questions)
                            break
                return self._get_next_question()
        return None

    def _render_next_question(self):
        # Custom styled HTML container for right panel
        question = self._get_next_question()

        left_box = self._render_question_box(question)

        right_box = VBox([self.output_container], layout=Layout(
            width="50%",
            margin="0 0 0 20px",
        ))
        self._show_template()

        self.ui.children = [HBox([left_box, right_box])]

    def _render_question_box(self, question):
        previous_qs = [HTML(f"<div style='background_color: #111'><span style='color: #888; padding-right: 10px'>{q.question}</span><span style='color: #555'>{a.answer}</span></div>") for q, a in self.question_answers]

        if question is None:
            return VBox(previous_qs + [Label("🎉 You're done!")], layout=Layout(
            padding="12px",
            backgroundColor="#f9f9f9",
            border="1px solid #ddd",
            borderRadius="10px",
            width="45%",
        ))

        q_label = HTML(f"<b>{question.question}</b>")

        buttons = []
        for answer in question.answers:
            btn = Button(
                description=answer.answer,
                tooltip=answer.tooltip,
                layout=Layout(width="auto", margin="4px 0"),
                button_style="",
            )

            btn.on_click(partial(self._handle_answer, question, answer))

            buttons.append(btn)

        left_box = VBox(previous_qs + [q_label] + buttons, layout=Layout(
            padding="12px",
            backgroundColor="#f9f9f9",
            border="1px solid #ddd",
            borderRadius="10px",
            width="45%",
        ))

        return left_box

    def _handle_answer(self, question, answer, _=None):
        self._update_template(answer.templates)
        self.commentary = answer.commentary
        self.questions_stack = answer.followups + self.questions_stack
        self.question_answers.append((question, answer))
        if question.variable is not None:
            self.variables[question.variable] = answer.answer
        self._render_next_question()

    def _update_template(self, templates: list[Template]):
        for t in templates:
            pattern = r"\{\{\s*" + re.escape(t.replacement) + r"\s*\}\}"
            self.code_output = re.sub(pattern, t.code, self.code_output)

    def _show_template(self):
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
    questions_path = files(FILE_PACKAGE_PATH).joinpath(FILE_NAME)
    with questions_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        return Questionnaire.model_validate(raw)


def run_questionnaire():
    questions = load_questions()
    app = QuestionnaireWidget(questions)
    app.show()
