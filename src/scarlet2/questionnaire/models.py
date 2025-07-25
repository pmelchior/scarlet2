from pydantic import BaseModel, Field


class Template(BaseModel):
    replacement: str
    code: str
    tooltip: str = ""

class Answer(BaseModel):
    answer: str
    tooltip: str = ""
    templates: list[Template]
    followups: list["Question"] = Field(default_factory=list)
    commentary: str = ""

class Question(BaseModel):
    question: str
    answers: list[Answer]

Question.model_rebuild()
Answer.model_rebuild()

class Questionnaire(BaseModel):
    initial_template: str
    questions: list[Question]
