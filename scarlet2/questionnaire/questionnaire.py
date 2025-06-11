import importlib
import json
from ipywidgets import VBox, HBox, Button, Output, Label, HTML, Layout, Box
from IPython.display import display

from scarlet2.questionnaire.models import Question

class QuestionnaireWidget:
    def __init__(self, questions):
        self.questions_stack = questions
        self.output = Output()
        self.output_container = HTML()

        self.ui = VBox(layout=Layout(padding="10px"))
        self._render_next_question()

    def _render_next_question(self):
        if not self.questions_stack:
            self.ui.children = [Label("🎉 You're done!")]
            return

        question = self.questions_stack.pop(0)
        q_label = HTML(f"<b>{question.question}</b>")

        buttons = []
        for answer in question.answers:
            btn = Button(
                description=answer.answer,
                tooltip=answer.tooltip,
                layout=Layout(width="auto", margin="4px 0"),
                button_style="",
            )

            def make_on_click(ans):
                def on_click(_):
                    self._show_template(ans)
                    for b in buttons:
                        b.disabled = True
                    self.questions_stack = ans.followups + self.questions_stack
                    self._render_next_question()
                return on_click

            btn.on_click(make_on_click(answer))
            buttons.append(btn)

        left_box = VBox([q_label] + buttons, layout=Layout(
            padding="12px",
            background_color="#f9f9f9",
            border="1px solid #ddd",
            border_radius="10px",
            width="45%",
        ))

        # Custom styled HTML container for right panel
        right_box = VBox([self.output_container], layout=Layout(
            width="50%",
            margin="0 0 0 20px",
        ))

        self.ui.children = [HBox([left_box, right_box])]

    def _show_template(self, answer):
        # Apply inline CSS directly to ensure proper rendering
        html_content = f"""
        <div style="
            background-color: #1e1e1e;
            color: #f1f1f1;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #444;
            font-family: monospace;
            white-space: pre-wrap;
        ">
            <div style="font-weight: bold; margin-bottom: 8px;">Code Template:</div>
            <div>{answer.template.code}</div>
            <div style="font-weight: bold; margin: 16px 0 8px;">Tooltip:</div>
            <div>{answer.template.tooltip}</div>
        </div>
        """
        self.output_container.value = html_content

    def show(self):
        display(self.ui)

def load_questions() -> list[Question]:
    with importlib.resources.files("scarlet2.questionnaire").joinpath("questions.json").open("r") as f:
        raw = json.load(f)
        return [Question.model_validate(q) for q in raw]


def run_questionnaire():
    questions = load_questions()
    app = QuestionnaireWidget(questions)
    app.show()
