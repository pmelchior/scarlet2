from pytest import fixture

@fixture
def example_questionnaire_dict():
    return {
        "initial_template": "{{code}}",
        "questions": [
            {
                "question": "Example question?",
                "answers": [
                    {
                        "answer": "Example answer",
                        "tooltip": "This is an example tooltip.",
                        "templates": [
                            {"replacement": "{{code}}", "code": "example_code {{follow}}"}
                        ],
                        "followups": [
                            {
                                "question": "Follow-up question?",
                                "answers": [
                                    {
                                        "answer": "Follow-up answer",
                                        "tooltip": "This is a follow-up tooltip.",
                                        "templates": [
                                            {"replacement": "{{follow}}", "code": "followup_code"}
                                        ],
                                        "followups": [],
                                        "commentary": ""
                                    }
                                ]
                            }
                        ],
                        "commentary": ""
                    },
                    {
                        "answer": "Another answer",
                        "tooltip": "This is another tooltip.",
                        "templates": [
                            {"replacement": "{{code}}", "code": "another_code"}
                        ],
                        "followups": [],
                        "commentary": ""
                    }
                ]
            }
        ]
    }
