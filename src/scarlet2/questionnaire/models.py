from datetime import datetime
from typing import Union

from pydantic import BaseModel, Field


class Template(BaseModel):
    """Represents a template for code replacement in the questionnaire."""

    replacement: str
    code: str


class Answer(BaseModel):
    """Represents an answer to a question in the questionnaire."""

    answer: str
    tooltip: str = ""
    templates: list[Template] = Field(default_factory=list)
    followups: list[Union["Question", "Switch"]] = Field(default_factory=list)
    commentary: str = ""


class Question(BaseModel):
    """Represents a question in the questionnaire."""

    question: str
    variable: str | None = None
    answers: list[Answer]


class Case(BaseModel):
    """Represents a case in a switch statement within the questionnaire."""

    value: int | None = None
    questions: list[Union[Question, "Switch"]]


class Switch(BaseModel):
    """Represents a switch statement in the questionnaire."""

    switch: str
    cases: list[Case]


# Rebuild models to support self-referencing types and forward references
Question.model_rebuild()
Answer.model_rebuild()
Case.model_rebuild()
Switch.model_rebuild()


class QuestionAnswer(BaseModel):
    """Represents a user's answer to a question."""

    question: str
    answer: str
    value: int


class QuestionAnswers(BaseModel):
    """Represents a collection of user answers to questions."""

    answers: list[QuestionAnswer] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class Questionnaire(BaseModel):
    """Represents a questionnaire with an initial template and a list of questions."""

    initial_template: str
    initial_commentary: str = ""
    feedback_url: str | None = None
    questions: list[Question | Switch]
