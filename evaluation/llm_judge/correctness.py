import json
import re
from pathlib import Path

from pydantic import BaseModel, Field
from evaluation.llm_judge.judge import ModelBase
from evaluation.llm_judge.llm_api import LLMApi
import pandas as pd
import click

class LLMOutput(BaseModel):
    correct: bool = Field(..., description="True if the answer is correct, False otherwise.")
    critique: str = Field(..., description="Critiques for the QA pair.")


class LLMCorrectnessEvaluator(ModelBase):
    """Evaluator using an LLM as a judge."""

    def __init__(self, model_name="llama3-8b-8192", temperature=0.1, max_retries=3, prompt_path=None, results_file=None, n_shots=2, backend="openai", num_workers=1):
        super().__init__(results_file)
        if prompt_path is None:
            prompt_path = Path(__file__).parent / 'prompts/grade_qa.yaml'

        self.prompt_path = prompt_path
        self.num_workers = num_workers
        self.generation_key = 'correct'

        self.output_structure = LLMOutput
        self.llm = LLMApi(model_name=model_name, temperature=temperature, parse_response=self.parse_response, process_save_dict=self.process_save_dict, prompt_path=prompt_path, output_structure=self.output_structure, results_file=self.results_file, n_shots=n_shots, generation_key='score', backend=backend, num_workers=num_workers)
        self.results_file = results_file

    @staticmethod
    def parse_response(response, logger):
        """Parse the model response to extract correct boolean and critique."""
        response = response.strip()
        try:
            # First try to parse as valid JSON
            json_match = re.search(r'\{[^{}]*"correct"[^{}]*"critique"[^{}]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    correct = parsed_json.get('correct')
                    critique = parsed_json.get('critique')

                    if correct is not None and critique is not None:
                        return {'correct': correct, 'critique': critique}
                except json.JSONDecodeError:
                    pass

            # Fallback to regex parsing if JSON parsing fails
            correct_match = re.search(r'"correct":\s*(true|false)', response, re.IGNORECASE)
            critique_match = re.search(r'"critique":\s*"([^"]*(?:\\.[^"]*)*)"', response)

            if correct_match and critique_match:
                correct = correct_match.group(1).lower() == 'true'
                critique = critique_match.group(1)
                # Handle escaped characters in critique
                critique = critique.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                return {'correct': correct, 'critique': critique}
            else:
                logger.debug("No valid correct/critique pair found.")
                return None

        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing response: {response}, Error: {e}")
            return None


    def judge(self, sample: dict):
        """Evaluate QA pairs using the LLM."""
        output_dict = self.llm(sample)

        return output_dict[0][self.generation_key]

    def process_dataframe(self, dataset: pd.DataFrame):
        outputs = self.llm.mp_generate(dataset)

        return outputs


    @staticmethod
    def process_save_dict(save_dict, generation_key, prompt_variables)->list[dict]:
        """Process the save_dict and save the results."""
        save_dict = save_dict[0]
        save_dict['critique'] = save_dict[generation_key]['critique']
        save_dict['correct'] = save_dict[generation_key]['correct']
        return [save_dict]




def read_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        data.append(f.readline())
    return data


def get_df(src_file, tgt_file, src_lang, tgt_lang) -> pd.DataFrame:
    src_data = read_file(src_file)
    tgt_data = read_file(tgt_file)
    src_lang_data = [src_lang] * len(src_data)
    tgt_lang_data = [tgt_lang] * len(tgt_data)
    df = pd.DataFrame({'src': src_data, 'tgt': tgt_data, 'src_lang': src_lang_data, 'tgt_lang': tgt_lang_data})
    return df

@click.command()
@click.argument('src_file', type=click.Path(exists=True))
@click.argument('tgt_file', type=click.Path(exists=True))
@click.argument('src_lang', type=str)
@click.argument('tgt_lang', type=str)
@click.argument('out_file', type=str)
def main(src_file, tgt_file, src_lang, tgt_lang, out_file):
    print(out_file)
    df = get_df(src_file, tgt_file, src_lang, tgt_lang)
    evaluator = LLMCorrectnessEvaluator(prompt_path='evaluation/llm_judge/prompts/translation.yaml', results_file=out_file)

    df_eval = evaluator.process_dataframe(df)
    return df_eval



if __name__ == "__main__":
    main()
