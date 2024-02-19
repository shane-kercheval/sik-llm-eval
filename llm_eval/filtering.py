"""Classes and functions for filtering evaluation results (EvalResult objects)."""
from llm_eval.eval import EvalResult


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
