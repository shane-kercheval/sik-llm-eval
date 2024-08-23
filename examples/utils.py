from llm_eval.eval import EvalResult


def print_result(result: EvalResult) -> None:
    """
    This function is used as a callback and prints the results of each evaluation. It is saved in
    other module so that it can be pickled from the notebook, which is required for
    multiprocessing.

    The callback can also be used, for example, to save the results to a file. If you're
    running a large number of evaluations, you may want to save the results to a file
    periodically in case there are issues/errors before the entire EvalHarness completes.
    """  # noqa: D404
    print(f"Num Checks: {result.num_checks}")
    print(f"Num Successful Checks: {result.num_successful_checks}")
    print('---')
