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
        eval_._generate_responses(candidate)
        print(f"{i} Data fetched")
        return {
            'i': i,
            'eval_obj': eval_,
        }

    # async def fetch_data_async(self, i, eval_, candidate):
    #     return self.fetch_data(i, eval_, candidate)

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
                    Candidate.from_dict(self.cand_config),
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
            print((round(result['eval_obj']._duration, 1), result['eval_obj']._responses[0][0:50]))

        eval_results = [r['eval_obj']._execute_checks() for r in results]
        for er in eval_results:
            pprint(eval_result_summarizer(er))


# def main():
#     evaluator = AsyncEvaluator(
#         '../evals/templates/candidate_openai_3.5_1106.yaml',
#         '../tests/fake_data/fake_eval_sum_two_numbers.yaml',
#     )

#     start = time.time()
#     evaluator.run_evaluation(async_mode=True)  # Set async_mode to False for synchronous execution
#     end = time.time()
#     print(f"Total time: {end - start}")

def main():

    from llm_evals.candidates import Candidate
    from llm_evals.eval import Eval, EvalHarness, eval_result_summarizer

    eval_harness = EvalHarness(num_cpus=None, run_async=True)
    # eval_harness.add_eval_from_yamls('../evals/evals')
    # eval_harness.add_candidate_from_yamls('../evals/candidates')

    eval_harness.add_eval_from_yaml('../evals/evals/mask_emails.yaml')
    eval_harness.add_eval_from_yaml('../evals/evals/mask_emails.yaml')
    eval_harness.add_candidate_from_yaml('../evals/templates/candidate_openai_3.5_1106.yaml')
    eval_harness.add_candidate_from_yaml('../evals/templates/candidate_openai_3.5_1106.yaml')
    # eval_harness.add_candidate_from_yaml('../evals/templates/candidate_openai_4.0_1106.yaml')


    print('start')
    start = time.time()
    results = eval_harness()
    end = time.time()
    for r in results:
        for a in r:
            print(a)
    print(f"Total time: {end - start}")

    # from llm_evals.checks import CheckType
    # result = results[1][0]
    # code_run_checks = [
    #     r for r in result.all_checks_results
    #     if r.metadata.get('check_type', '') == CheckType.PYTHON_CODE_BLOCKS_RUN.name
    # ]
    # sum(c.metadata['num_code_block_checks'] for c in code_run_checks)
    # sum(c.metadata['num_code_block_checks_successful'] for c in code_run_checks)


if __name__ == "__main__":
    main()
