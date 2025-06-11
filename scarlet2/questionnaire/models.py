from pydantic import BaseModel, Field


class Template(BaseModel):
    code: str
    tooltip: str

class Answer(BaseModel):
    answer: str
    tooltip: str
    template: Template
    followups: list["Question"] = Field(default_factory=list)

class Question(BaseModel):
    question: str
    answers: list[Answer]

Question.model_rebuild()
