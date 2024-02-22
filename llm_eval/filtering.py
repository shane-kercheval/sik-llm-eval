"""Classes and functions for filtering evaluation results (EvalResult objects)."""
from llm_eval.checks import PythonCodeBlocksPresent, PythonCodeBlockTests
from llm_eval.eval import Eval, EvalResult


def eval_expects_code_blocks(eval_: Eval) -> bool:
    """
    Return True if the eval object contains a check that tests for the presence of code blocks
    (i.e. Check objects with `PythonCodeBlocksPresent` check type).
    """
    return any(
        isinstance(check, PythonCodeBlocksPresent)
        for test in eval_.test_sequence
        for check in test.checks
    )

def eval_contains_code_block_tests(eval_: Eval) -> bool:
    """
    Return True if the eval object contains a check that tests code blocks (i.e. Check objects with
    `PythonCodeBlockTests` check type).
    """
    return any(
        isinstance(check, PythonCodeBlockTests)
        for test in eval_.test_sequence
        for check in test.checks
    )

def result_expects_code_blocks(result: EvalResult) -> bool:
    """
    Return True if the underlying Eval object in the EvalResult contains a check that tests for the
    presence of code blocks (i.e. Check objects with `PythonCodeBlocksPresent` check type).
    """
    return result.expects_code_blocks

def result_contains_code_block_tests(result: EvalResult) -> bool:
    """
    Return True if the underlying Eval object in the EvalResult contains a check that tests code
    blocks (i.e. Check objects with `PythonCodeBlockTests` check type).
    """
    return result.code_block_tests_result is not None

def filter_expects_code_blocks(results: list[EvalResult]) -> list[EvalResult]:
    """
    Filter results to only include those where the corresponding Eval contains a
    `PythonCodeBlocksPresent` check.
    """
    return [r for r in results if result_expects_code_blocks(r)]

def filter_contains_code_block_tests(results: list[EvalResult]) -> list[EvalResult]:
    """
    Filter results to only include those where the corresponding Eval contains a
    `PythonCodeBlockTests` check.
    """
    return [r for r in results if result_contains_code_block_tests(r)]

def filter_tags(
        results: list[EvalResult],
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None) -> list[EvalResult]:
    """
    Filter results by tags.

    If no `includes` are provided, all EvalResult objects are returned except those with any of the
    `exclude` tags. If `includes` are provided, only EvalResult objects with at least one of the
    `include` tags are returned. If both `include` and `exclude` are provided, the `exclude` tags
    take precedence.

    Args:
        results: List of EvalResult objects.
        include: Tags to include.
        exclude: Tags to exclude.
    """
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]
    include = set(include or [])
    exclude = set(exclude or [])
    if include:
        # include any results that have at least one of the include tags
        filtered = []
        filtered.extend(
            [r for r in results if include & set(r.eval_obj.metadata.get('tags', []))],
        )
    else:
        filtered = results
    if exclude:
        # exclude any results that have at least one of the exclude tags
        # not exclude & set(filtered[0].eval_obj.metadata.get('tags', []))
        filtered = [
            r for r in filtered if not exclude & set(r.eval_obj.metadata.get('tags', []))
        ]
    return filtered
