import os
import time
import yaml
import asyncio
from dotenv import load_dotenv
from llm_evals.eval import Eval, Candidate, eval_result_summarizer
from pprint import pprint

load_dotenv()

def print_result(result):
    print(f"Finished {result.total_time_seconds}", flush=True)

def main():

    from llm_evals.candidates import Candidate
    from llm_evals.eval import Eval, EvalHarness, eval_result_summarizer


    eval_harness = EvalHarness(
        num_cpus=None,
        async_batch_size=50,
        callback=print_result,
    )
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
