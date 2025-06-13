"""Classes and functions for filtering evaluation results (EvalResult objects)."""
from sik_llm_eval.checks import PythonCodeBlocksPresent, PythonCodeBlockTests
from sik_llm_eval.eval import Eval, EvalResult


def eval_expects_code_blocks(eval_: Eval) -> bool:
    """
    Return True if the eval object contains a check that tests for the presence of code blocks
    (i.e. Check objects with `PythonCodeBlocksPresent` check type).
    """
    return any(
        isinstance(check, PythonCodeBlocksPresent)
        for check in eval_.checks
    )

def eval_contains_code_block_tests(eval_: Eval) -> bool:
    """
    Return True if the eval object contains a check that tests code blocks (i.e. Check objects with
    `PythonCodeBlockTests` check type).
    """
    return any(
        isinstance(check, PythonCodeBlockTests)
        for check in eval_.checks
    )

def result_expects_code_blocks(result: EvalResult) -> bool:
    """
    Return True if the underlying Eval object in the EvalResult contains a check that tests for the
    presence of code blocks (i.e. Check objects with `PythonCodeBlocksPresent` check type).
    """
    return eval_expects_code_blocks(result.eval)

def result_contains_code_block_tests(result: EvalResult) -> bool:
    """
    Return True if the underlying Eval object in the EvalResult contains a check that tests code
    blocks (i.e. Check objects with `PythonCodeBlockTests` check type).
    """
    return eval_contains_code_block_tests(result.eval)

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

def matches_tags(
        result: EvalResult,
        include: list[str] | str | None = None,
        exclude: list[str] | str | None = None) -> bool:
    """
    Returns boolean if EvalResult matches tags.

    If both `include` and `exclude` are provided, the `exclude` tags take precedence.

    Args:
        result: List of EvalResult objects.
        include: Tags to include.
        exclude: Tags to exclude.
    """
    matches = True
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]
    include = set(include or [])
    exclude = set(exclude or [])
    if include:
        # include any results that have at least one of the include tags
        matches = bool(include & set(result.eval.metadata.get('tags', [])))
    if exclude:
        # exclude any results that have at least one of the exclude tags
        # not exclude & set(filtered[0].eval.metadata.get('tags', []))
        matches = matches and not exclude & set(result.eval.metadata.get('tags', []))
    return matches

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
    return [r for r in results if matches_tags(r, include=include, exclude=exclude)]
