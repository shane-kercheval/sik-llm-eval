import os
import shutil
import time
from dotenv import load_dotenv
from llm_evals.eval import EvalResult, eval_result_summarizer
from llm_evals.eval import EvalHarness
# import wandb

load_dotenv()

DIR_PATH = "__temp__"

def print_result(result: EvalResult) -> None:
    """Print the result of an evaluation."""
    print(result)
    path = f"{DIR_PATH}/{result.candidate_obj.uuid}-{result.eval_obj.uuid}.yaml"
    result.to_yaml(path)
    print(f"Finished {result.total_time_seconds}, saved to {path}", flush=True)
    # with open(f"{DIR_PATH}/{result.candidate_obj.uuid}-{result.eval_obj.uuid}.json", "w") as f:
    #     f.write(result.to_json())
    # print(f"Finished {result.total_time_seconds}", flush=True)
    # print(eval_result_summarizer(result))

# def log_weights_and_biases(result: EvalResult) -> None:
#     """Log the result of an evaluation to Weights and Biases."""
#     wandb.log(eval_result_summarizer(result))
#     # wandb.log({"accuracy": 1, "loss": 2})


def main() -> None:
    """Run the main function."""
    # wandb.init(project="test-project")

    if os.path.exists(DIR_PATH):
        shutil.rmtree(DIR_PATH)
    os.makedirs(DIR_PATH)
    assert os.path.exists(DIR_PATH)

    eval_harness = EvalHarness(
        # num_cpus=None,
        # async_batch_size=50,
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
    # for r in results:
    #     for a in r:
    #         print(a)
    print(f"Total time: {end - start}")


if __name__ == "__main__":
    main()
