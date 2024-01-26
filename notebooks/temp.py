import os
import time
import yaml
import asyncio
from dotenv import load_dotenv
from llm_evals.eval import Eval, Candidate, eval_result_summarizer
from pprint import pprint

load_dotenv()


class AsyncEvaluator:
    def __init__(self, candidate_config_path, eval_config_path):
        with open(candidate_config_path) as f:
            self.cand_config = yaml.safe_load(f)

        with open(eval_config_path) as f:
            self.eval_config = yaml.safe_load(f)

    def fetch_data(self, i, eval_, candidate):
        print(f"{i} Starting data fetch...", flush=True)
        eval_._call_candidate(candidate)
        print(f"{i} Data fetched")
        return {
            'i': i,
            'eval_obj': eval_,
        }

    async def fetch_data_async(self, i, eval_, candidate):
        return self.fetch_data(i, eval_, candidate)

    def run_evaluation(self, async_mode=False):
        loop = asyncio.get_event_loop() if async_mode else None
        tasks = []

        for i in range(10):
            if async_mode:
                task = loop.run_in_executor(
                    None,
                    self.fetch_data,
                    i,
                    Eval(**self.eval_config),
                    Candidate.from_dict(self.cand_config)
                )
            else:
                task = self.fetch_data(
                    i,
                    Eval(**self.eval_config),
                    Candidate.from_dict(self.cand_config),
                )
            tasks.append(task)

        if async_mode:
            results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
        else:
            results = tasks

        for result in results:
            print((round(result['eval_obj'].duration, 1), result['eval_obj'].responses[0][0:50]))

        eval_results = [r['eval_obj']._execute_eval() for r in results]
        for er in eval_results:
            print(eval_result_summarizer(er))


def main():
    evaluator = AsyncEvaluator(
        '../evals/templates/candidate_openai_3.5_1106.yaml',
        '../tests/fake_data/fake_eval_sum_two_numbers.yaml',
    )

    start = time.time()
    evaluator.run_evaluation(async_mode=False)  # Set async_mode to False for synchronous execution
    end = time.time()
    print(f"Total time: {end - start}")


if __name__ == "__main__":
    main()
