from typing import Union

from pydantic import BaseModel, Field


class Template(BaseModel):
    replacement: str
    code: str
    tooltip: str = ""

class Answer(BaseModel):
    answer: str
    tooltip: str = ""
    templates: list[Template] = Field(default_factory=list)
    followups: list[Union["Question", "Switch"]] = Field(default_factory=list)
    commentary: str = ""

class Question(BaseModel):
    question: str
    variable: str = None
    answers: list[Answer]

class Case(BaseModel):
    value: str = None
    questions: list[Union[Question, "Switch"]]

class Switch(BaseModel):
    variable: str
    cases: list[Case]

Question.model_rebuild()
Answer.model_rebuild()
Case.model_rebuild()
Switch.model_rebuild()

class Questionnaire(BaseModel):
    initial_template: str
    questions: list[Question | Switch]
