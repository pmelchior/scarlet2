from importlib.resources import files

import yaml

from scarlet2.questionnaire.models import Questionnaire

FILE_PACKAGE_PATH = "scarlet2.questionnaire"
FILE_NAME = "questions.yaml"


def load_questions() -> Questionnaire:
    """Load the questionnaire from the packaged YAML file.

    Returns:
        Questionnaire: The loaded questionnaire model.
    """
    questions_path = files(FILE_PACKAGE_PATH).joinpath(FILE_NAME)
    with questions_path.open("r") as f:
        raw = yaml.safe_load(f)
        return Questionnaire.model_validate(raw)
