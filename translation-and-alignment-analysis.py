import json
from typing import Any
from openai import OpenAI
from dotenv import dotenv_values
import pandas as pd
import os
from enum import Enum
from jsonschema import SchemaError, validate, ValidationError
from dataclasses import asdict, dataclass


class GPTModel(Enum):
    GPT3_5 = "gpt-3.5-turbo"
    GPT4 = "gpt-4-turbo"
    GPT4_o = "gpt-4o"


@dataclass
class JsonSuperClass:

    @property
    def __dict__(self):
        """
        get a python dictionary
        """
        return asdict(self)

    @property
    def json(self):
        """
        get the json formated string
        """
        return json.dumps(self.__dict__)


@dataclass
class OutputData(JsonSuperClass):
    translation: dict[str, Any]
    alignment_results: list[dict[str, Any]]


@dataclass
class RunData(JsonSuperClass):
    input: dict[str, Any]
    output: OutputData
    metadata: dict[str, Any]


INPUT_DIR = 'data/inputs'
OUTPUT_DIR = 'data/outputs'
MODEL = GPTModel.GPT4

CURRENT_DATETIME = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
OUTPUT_RUN_PATH = f'{OUTPUT_DIR}/{CURRENT_DATETIME}'
ERRORS: list[dict[str, Any]] = []

# Load the environment variables from .env file
ENV_VARS = dotenv_values('.env')

# Get the value of the OPENAI_API_KEY key
API_KEY = ENV_VARS.get('OPENAI_API_KEY')
if API_KEY is None:
    raise ValueError('The OPENAI_API_KEY environment variable is not set')


client = OpenAI()


def completion(prompt, model: GPTModel = GPTModel.GPT3_5, output_json=False) -> str:
    completion = client.chat.completions.create(
        model=model.value,
        response_format={"type": "json_object" if output_json else "text"},
        messages=[
            # {"role": "system", "content": "You should only respond with the answer to the prompt. Not with \"Sure, here's ...\""},
            {"role": "system", "content": "Output in JSON format"},
            {"role": "user", "content": prompt}
        ]
    )

    return str(completion.choices[0].message.content)


def prompt_with_variables(prompt: str,
                          variables: dict[str, str],
                          validate_output_with_schema: dict = {},
                          model: GPTModel = GPTModel.GPT3_5) -> dict[str, Any]:
    """
    Replaces the variables, denoted with [[variable]] in the prompt
    with the values in the variables dict.

    The function will try to get a valid output up to 5 times.
    Outputs that do not match the json_schema will be discarded.
    """
    def fill_variables_in_prompt(prompt: str, variables: dict[str, str]) -> str:
        separator = '[[{key}]]'
        for key, value in variables.items():
            # first find if the key is in the prompt
            if separator.format(key=key) not in prompt:
                raise ValueError(f"Variable {key} not found in prompt")
            prompt = prompt.replace(separator.format(key=key), value)
        return prompt

    filled_prompt = fill_variables_in_prompt(prompt, variables)

    MAX_TRIES = 5
    for _ in range(MAX_TRIES):  # if the output is not valid, try again up to 5 times
        completion_json = completion(
            filled_prompt, model=model, output_json=True)
        prompt_output = json.loads(completion_json)
        if not validate_output_with_schema:
            return prompt_output

        try:
            validate(instance=prompt_output,
                     schema=validate_output_with_schema)
            return prompt_output
        except ValidationError as e:
            ERRORS.append({
                "prompt": filled_prompt,
                "output": prompt_output,
                "error": e.message,
            })
        except SchemaError as e:
            raise ValueError(e.message)

    raise ValueError(f"Could not get a valid output in {MAX_TRIES} tries")


def main():

    company_goals_df = pd.read_csv(
        f'{INPUT_DIR}/stated goals.csv', delimiter=';')
    articles_df = pd.read_csv(f'{INPUT_DIR}/articles.csv', delimiter=';')

    # Ensure the directory exists
    os.makedirs(OUTPUT_RUN_PATH, exist_ok=True)

    with open('prompts/translation-into-principles.txt', 'r') as file:
        translation_prompt = file.read()

    # read second prompt file
    with open('prompts/analysis-scoring.txt', 'r') as file:
        analysis_prompt = file.read()

    # read json schema for principles output
    with open('prompts/translation-jsonschema.json', 'r') as file:
        translation_output_schema = json.load(file)

    # read json schema for analysis output
    with open('prompts/analysis-jsonschema.json', 'r') as file:
        analysis_output_schema = json.load(file)

    runs: dict[str, RunData] = dict()

    # Loop over all companies in the stated goals dataframe
    for i, stated_goals in company_goals_df.iterrows():

        company: str = stated_goals['Company']
        print(
            f"Processing company {i} of {len(company_goals_df) - 1}: {company}")
        strategy = stated_goals['Strategy']

        run = RunData(
            input={
                "strategy": strategy
            },
            output=OutputData(
                translation=prompt_with_variables(
                    translation_prompt,
                    variables={
                        'X': strategy
                    },
                    validate_output_with_schema=translation_output_schema,
                    model=MODEL,
                ),
                alignment_results=[]
            ),
            metadata={
                "company": company,
                "goals_source_url": stated_goals['Goals URL'],
                "model": MODEL.value,
                "run_datetime": CURRENT_DATETIME,
            }
        )

        with open(f'{OUTPUT_RUN_PATH}/10-principles-{company}.json', 'w') as file:
            file.write(json.dumps(run.output.translation))

        translation_principles = [principle['Principle']
                                  for principle in run.output.translation['Principles']]

        # find all articles for the company
        company_articles: pd.DataFrame = articles_df[articles_df['Company'] == company]
        results_for_company: list[dict[str, Any]] = []
        for j, article in company_articles.iterrows():
            print(f"Processing article {j} of {len(company_articles) - 1}")
            if 'Article' not in article:
                raise KeyError(
                    "Article column not found in the articles file (this column has the text of the article)")
            else:
                article_text = article['Article']

            analysis = prompt_with_variables(
                analysis_prompt,
                variables={
                    'X': article_text,
                    'principles': json.dumps(run.output.translation)
                },
                validate_output_with_schema=analysis_output_schema,
                model=MODEL,
            )

            # verify that the principles are in the analysis

            analysis_principles = [principle['Principle']
                                   for principle in analysis['Principles']]

            missing_principles = [
                principle for principle in translation_principles if principle not in analysis_principles]

            if missing_principles:
                missing_principles_str = ', '.join(missing_principles)
                ERRORS.append({
                    "prompt": analysis_prompt,
                    "output": analysis,
                    "error": f"Not all principles were found in the analysis: {missing_principles_str}",
                    "error_code": "MISSING_PRINCIPLES"
                })
                print(
                    f"The following principles were not found in the analysis: {missing_principles_str}")

            analysis['URL'] = article['URL']
            results_for_company.append(analysis)

        run.output.alignment_results = results_for_company
        runs[company] = run

        with open(f'{OUTPUT_RUN_PATH}/20-analysis-{company}.json', 'w') as file:
            file.write(json.dumps(
                {"alignment_results": run.output.alignment_results}))

    with open(f'{OUTPUT_RUN_PATH}/30-output.json', 'w') as file:
        file.write(json.dumps(runs, default=lambda x: x.__dict__))


if __name__ == '__main__':
    main()
    if ERRORS:
        with open(f'{OUTPUT_RUN_PATH}/errors.json', 'w') as file:
            file.write(json.dumps({"errors": ERRORS}))
