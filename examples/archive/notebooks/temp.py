import os
import shutil
import time
from dotenv import load_dotenv
from llm_evals.eval import EvalResult, eval_result_summarizer
from llm_evals.eval import EvalHarness
# import wandb
from pprint import pprint

load_dotenv()

DIR_PATH = "__temp__"

def print_result(result: EvalResult) -> None:
    """Print the result of an evaluation."""
    pprint(eval_result_summarizer(result))
    print(result, flush=True)
    path = f"{DIR_PATH}/{result.candidate_obj.metadata['name']}-{result.eval_obj.metadata['name']}.yaml"  # noqa
    result.to_yaml(path)
    print(f"Finished {result.total_time_seconds}, saved to {path}", flush=True)
    print('-------------------', flush=True)


def main() -> None:
    """Run the main function."""
    # wandb.init(project="test-project")

    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)
    os.makedirs(DIR_PATH)
    assert os.path.exists(DIR_PATH)

    eval_harness = EvalHarness(
        # num_cpus=1,
        # async_batch_size=1,
        callback=print_result,
    )
    # eval_harness.add_eval_from_yamls('../evals/evals')
    # eval_harness.add_candidate_from_yamls('../evals/candidates')

    eval_harness.add_eval_from_yaml('../examples/evals/mask_emails.yaml')
    eval_harness.add_eval_from_yaml('../examples/evals/mask_emails.yaml')
    eval_harness.add_candidate_from_yaml('../examples/candidates/openai_3.5_1106_multiple_parameters.yaml')
    # eval_harness.add_candidate_from_yaml('../examples/candidates/openai_3.5_1106.yaml')
    # eval_harness.add_candidate_from_yaml('../examples/candidates/openai_3.5_1106.yaml')
    # eval_harness.add_candidate_from_yaml('../examples/candidates/openai_4.0_1106.yaml')

    print('start')
    start = time.time()
    results = eval_harness()
    end = time.time()
    # for r in results:
    #     for a in r:
    #         print(a)
    print(f"Total time: {end - start}")


if __name__ == "__main__":
    main()
