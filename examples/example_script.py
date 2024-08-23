"""
Run this script with: python -m examples.example_script.py.

The `OPENAI_API_KEY` environment variable must be set (e.g. in a `.env` file in the root of the
repo).
"""
import os
import shutil
import time
from dotenv import load_dotenv
from llm_eval.eval import EvalResult
from llm_eval.eval import EvalHarness

load_dotenv()
DIR_PATH = "__temp__"


def print_result(result: EvalResult) -> None:
    """Print the result of an evaluation."""
    print(result, flush=True)
    path = f"{DIR_PATH}/{result.candidate_obj.metadata['name']}-{result.eval_obj.metadata['name']}.yaml"  # noqa
    result.to_yaml(path)
    print(f"Finished {result.total_time_seconds}, saved to {path}", flush=True)
    print('-------------------', flush=True)

def main() -> None:
    """Run the main function."""
    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)
    os.makedirs(DIR_PATH)
    assert os.path.exists(DIR_PATH)

    harness = EvalHarness(callback=print_result, num_cpus=None)
    harness.add_evals_from_yamls('examples/evals/*.yaml')
    harness.add_candidate_from_yaml('examples/candidates/openai_4.0.yaml')
    harness.add_candidate_from_yaml('examples/candidates/openai_4o-mini.yaml')

    print("# of Evals: ", len(harness.evals))
    print("# of Candidates: ", len(harness.candidates))

    print("Starting eval_harness")
    start = time.time()
    results = harness()  # run the evals
    end = time.time()
    for r in results:
        for a in r:
            print(a)
    print(f"Total time: {end - start}")


if __name__ == "__main__":
    main()
