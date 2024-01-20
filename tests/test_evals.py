"""Tests for the evals module."""
from llm_evals.checks import CheckType, MatchExactCheck
from llm_evals.eval import PromptTest


def test__PromptTest():  # noqa
    test = PromptTest(
        prompt='test',
        ideal_response = None,
        checks = None,
    )
    assert test.prompt == 'test'
    assert test.ideal_response is None
    assert test.checks is None
    assert str(test)
    test_dict = test.to_dict()
    assert test_dict == {'prompt': 'test'}
    assert PromptTest(**test_dict) == test

    test = PromptTest(
        prompt='test1',
        ideal_response='test2',
        checks = [MatchExactCheck(value='test3')],
    )
    assert test.prompt == 'test1'
    assert test.ideal_response == 'test2'
    assert test.checks == [MatchExactCheck(value='test3')]
    assert str(test)
    test_dict = test.to_dict()
    assert test_dict == {
        'prompt': 'test1',
        'ideal_response': 'test2',
        'checks': [{'check_type': CheckType.MATCH_EXACT.name, 'value': 'test3'}],
    }
    assert PromptTest(**test_dict) == test
