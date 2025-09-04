# Questionnaire

The Scarlet2 Questionnaire is an interactive tool designed to help you quickly set up a Scarlet2 project tailored to your specific use case. By answering a series of questions about your data and modeling needs, the questionnaire generates a customized code template that you can use as a starting point for your project.

## Running the Questionnaire

To use the questionnaire, simply run the following code in a Jupyter notebook:

```python
from scarlet2.questionnaire import run_questionnaire

run_questionnaire()
```

## How It Works

The questionnaire presents a series of questions with multiple-choice answers. Each question helps determine the appropriate configuration for your Scarlet2 project. The questions adapt based on your previous answers, ensuring that you only see questions relevant to your use case.

As you progress through the questionnaire, you'll see:

1. **Questions**: Each question focuses on a specific aspect of your project setup
2. **Multiple-choice answers**: Select the option that best matches your needs
3. **Tooltips**: Hover over answers for additional information
4. **Commentary**: Explanations about the current step and its implications
5. **Code preview**: A continuously updated code template based on your answers

## Key Features

The questionnaire can help you set up:

- Single or multiple image observations
- Time-domain/transient source modeling
- Different types of sources (point sources, extended sources)
- Appropriate initialization methods
- PSF modeling
- And more...

## Using the Results

When you complete the questionnaire, you'll receive a Python code snippet that:

1. Sets up the appropriate observation objects
2. Creates a modeling frame
3. Initializes sources based on your specifications
4. Provides visualization code

You can copy this code into your own notebook or script and modify it as needed. The generated code serves as a starting point that follows best practices for your specific use case.

## Example Workflow

1. Start a new Jupyter notebook
2. Import and run the questionnaire
3. Answer the questions based on your data and modeling needs
4. Review the generated code template
5. Copy the code to your project
6. Customize the code with your actual data paths and parameters

## Tips for Using the Questionnaire

- Have information about your data ready (number of images, filters, etc.)
- If you have source positions, have them available for initialization
- Consider your modeling goals before starting (e.g., do you need to model transient sources?)
- You can run the questionnaire multiple times to explore different configurations

The questionnaire is designed to be a starting point - you'll likely need to customize the generated code for your specific dataset and research goals.
