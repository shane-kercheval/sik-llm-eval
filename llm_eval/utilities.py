"""Utility functions."""

def default_tokenizer(value: str) -> list[str]:
    """
    Tokenize a string by splitting on whitespace.

    Args:
        value: The string to tokenize.

    Returns:
        list[str]: A list of tokens.
    """
    return value.split()


def precision_score(true_pos: int, false_pos: int) -> float:
    """
    "Precision (also called positive predictive value) is the fraction of relevant instances among
    the retrieved instances" (https://en.wikipedia.org/wiki/Precision_and_recall).

    precision = true_pos / pred_pos
    precision = true_pos / (true_pos + false_pos)

    Within the context of classification, precision measures how often the model is correct when
    it predicts a positive outcome.

    Args:
        true_pos: The number of true positive outcomes.
        false_pos: The number of false positive outcomes.
    """
    return precision_score2(true_pos, true_pos + false_pos)


def precision_score2(true_pos: int, pred_pos: int) -> float:
    """
    "Precision (also called positive predictive value) is the fraction of relevant instances among
    the retrieved instances" (https://en.wikipedia.org/wiki/Precision_and_recall).

    precision = true_pos / pred_pos
    precision = true_pos / (true_pos + false_pos)

    Within the context of classification, precision measures how often the model is correct when
    it predicts a positive outcome.

    Args:
        true_pos: The number of true positive outcomes.
        pred_pos: The number of predicted positive outcomes.
    """
    return true_pos / pred_pos if pred_pos > 0 else 0


def recall_score(true_pos: int, false_neg: int) -> float:
    """
    "Recall (also known as sensitivity) is the fraction of relevant instances that were retrieved"
    (https://en.wikipedia.org/wiki/Precision_and_recall).

    Recall is also known as the true positive rate.

    recall = true_pos / actual_pos
    recall = true_pos / (true_pos + false_neg)

    Within the context of classification, recall measures the percent of actual positive outcomes
    (out of all actual positive outcomes) that the model correctly predicted.

    Args:
        true_pos: The number of true positive outcomes.
        false_neg: The number of false negative outcomes.
    """
    return recall_score2(true_pos, true_pos + false_neg)


def recall_score2(true_pos: int, actual_pos: int) -> float:
    """
    "Recall (also known as sensitivity) is the fraction of relevant instances that were retrieved"
    (https://en.wikipedia.org/wiki/Precision_and_recall).

    Recall is also known as the true positive rate.

    Within the context of classification, recall measures the percent of actual positive outcomes
    (out of all actual positive outcomes) that the model correctly predicted.

    Args:
        true_pos: The number of true positive outcomes.
        actual_pos: The number of actual positive outcomes.
    """
    return true_pos / actual_pos if actual_pos > 0 else 0


def f_score(precision: float, recall: float, beta: float) -> float:
    """
    The F-score is a metric that combines precision and recall into a single value. The F-score
    is a generalization of the F1 score and can be adjusted to give more weight to either
    precision or recall.

    The higher the F-score, the better the model is at both precision and recall.

    Two commonly used values for β are 2, which weighs recall higher than precision, and 0.5,
    which weighs recall lower than precision.
    """
    beta_squared = beta ** 2
    denominator = ((beta_squared * precision) + recall)
    if denominator == 0:
        return 0
    return (1 + beta_squared) * (precision * recall) / denominator


def f1_score(precision: float, recall: float) -> float:
    """
    "The F1 score is the harmonic mean of the precision and recall. It thus symmetrically
    represents both precision and recall in one metric. The more generic Fβ score applies
    additional weights, valuing one of precision or recall more than the other.

    The highest possible value of an F-score is 1.0, indicating perfect precision and recall, and
    the lowest possible value is 0, if precision and recall are zero." https://en.wikipedia.org/wiki/F-score

    The beta value for the F1 score is 1, which means that precision and recall are weighted
    equally, the result is the harmonic mean of the two values.

    Args:
        precision: The precision score.
        recall: The recall score.
    """
    return f_score(precision, recall, beta=1)


def precision_score_tokens(expected_tokens: list[str] | set[str], actual_tokens: list[str] | set[str]) -> float:  # noqa
    """
    Calculate the precision score for token comparison.

    Precision measures the accuracy of the generated tokens. It answers the question:
    "Of the tokens we generated, what fraction were actually correct?"

    A high precision score indicates that when the model generates tokens, they are
    often correct, but it doesn't tell us about tokens the model might have missed.

    Args:
    actual_tokens: The actual set of tokens (e.g. the generated tokens from the LLM).
    expected_tokens: The expected set of tokens (e.g. the correct/ideal tokens).
    """
    expected_tokens = set(expected_tokens)
    actual_tokens = set(actual_tokens)
    return precision_score2(
        true_pos=len(actual_tokens.intersection(expected_tokens)),
        pred_pos=len(actual_tokens),
    )


def recall_score_tokens(expected_tokens: list[str] | set[str], actual_tokens: list[str] | set[str]) -> float:  # noqa
    """
    Calculate the recall score for token comparison.

    Recall measures the completeness of the generated tokens. It answers the question:
    "Of the tokens that should have been generated, what fraction did we actually generate?"

    A high recall score indicates that the model is good at finding all the correct tokens,
    but it doesn't tell us if it also included incorrect tokens.

    Args:
    actual_tokens: The list of tokens from the generated text.
    expected_tokens: The list of tokens from the ideal (correct) text.

    Returns:
    float: The recall score, ranging from 0.0 to 1.0.
    """
    expected_tokens = set(expected_tokens)
    actual_tokens = set(actual_tokens)
    return recall_score2(
        true_pos=len(actual_tokens.intersection(expected_tokens)),
        actual_pos=len(expected_tokens),
    )


def f1_score_tokens(expected_tokens: list[str] | set[str], actual_tokens: list[str] | set[str]) -> float:  # noqa
    """
    Calculate the F1 score for token comparison.

    Args:
        actual_tokens: The actual set of tokens (e.g. the generated tokens from the LLM).
        expected_tokens: The expected set of tokens (e.g. the correct/ideal tokens).
    """
    return f1_score(
        precision=precision_score_tokens(actual_tokens, expected_tokens),
        recall=recall_score_tokens(actual_tokens, expected_tokens),
    )
