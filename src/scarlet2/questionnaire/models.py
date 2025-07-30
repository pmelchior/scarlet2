from pydantic import BaseModel, Field


class Template(BaseModel):
    """Represents a template for code replacement in the questionnaire."""

    replacement: str
    code: str


class Answer(BaseModel):
    """Represents an answer to a question in the questionnaire."""

    answer: str
    tooltip: str = ""
    templates: list[Template]
    followups: list["Question"] = Field(default_factory=list)
    commentary: str = ""


class Question(BaseModel):
    """Represents a question in the questionnaire."""

    question: str
    answers: list[Answer]


Question.model_rebuild()
Answer.model_rebuild()


class Questionnaire(BaseModel):
    """Represents a questionnaire with an initial template and a list of questions."""

    initial_template: str
    questions: list[Question]
