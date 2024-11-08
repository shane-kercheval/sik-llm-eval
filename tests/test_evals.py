"""Tests for the evals module."""
import pytest
from copy import deepcopy
import os
from textwrap import dedent
import yaml
from llm_eval.candidates import (
    Candidate,
    CandidateResponse,
)
from llm_eval.checks import (
    Check,
    CheckResult,
    CheckType,
    ContainsCheck,
    LambdaCheck,
    MatchCheck,
    PassFailResult,
    ScoreResult,
    ToolCallsCheck,
)
from llm_eval.eval import (
    CandidateRunResults,
    Eval,
    EvalHarness,
    EvalResult,
    Mode,
)
from llm_eval.internal_utilities import extract_code_blocks
from llm_eval.openai import user_message
from tests.conftest import UnregisteredCandidate, UnregisteredCheck


@pytest.mark.parametrize('input_', [[user_message('test1')], None])
def test__Eval__creation(input_: object):
    eval_obj = Eval(
        input=input_,
        ideal_response='test2',
        checks = [
            MatchCheck(value='test6', metadata={'test': 'test7'}),
            ContainsCheck(value='test8'),
            ContainsCheck(value='test9').to_dict(),
        ],
    )
    assert eval_obj.input == input_
    assert eval_obj.ideal_response == 'test2'
    assert eval_obj.checks == [
        MatchCheck(value='test6', metadata={'test': 'test7'}),
        ContainsCheck(value='test8'),
        ContainsCheck(value='test9'),
    ]
    assert str(eval_obj)

    eval_dict = eval_obj.to_dict()
    expected_dict = {
        'input': input_,
        'ideal_response': 'test2',
        'checks': [
            {'check_type': 'MATCH', 'value': 'test6', 'metadata': {'test': 'test7'}},
            {'check_type': 'CONTAINS', 'value': 'test8'},
            {'check_type': 'CONTAINS', 'value': 'test9'},
        ],
    }
    if not input_:
        del expected_dict['input']
    assert eval_dict == expected_dict
    assert Eval(**eval_dict) == eval_obj

def test__eval__clone(fake_eval_8f9fbf37: dict):
    config = deepcopy(fake_eval_8f9fbf37)
    eval_obj = Eval(**config)
    eval_copy = deepcopy(eval_obj)
    assert eval_obj == eval_copy
    assert eval_obj.to_dict() == eval_copy.to_dict()
    # test-sequence (i.e. PromptTest objects) should be the same prompt tests but different objects
    assert eval_obj.input == eval_copy.input
    assert eval_obj.input is not eval_copy.input
    assert eval_obj.ideal_response == eval_copy.ideal_response
    assert eval_obj.ideal_response is not eval_copy.ideal_response
    assert eval_obj.checks == eval_copy.checks
    assert all(c1 is not c2 for c1, c2 in zip(eval_obj.checks, eval_copy.checks))
    assert eval_obj.metadata == eval_copy.metadata
    assert eval_obj.metadata is not eval_copy.metadata

def test__Eval__from_file():
    """
    For each of our fake evals, test that we can create an Eval object from the yaml file and that
    the object is the same as the one created from the dictionary.
    """
    eval_files = [
        f'tests/fake_data/{f}'
        for f in os.listdir('tests/fake_data')
        if f.startswith('fake_eval') and f.endswith('.yaml')
    ]
    assert len(eval_files) > 1
    for path in eval_files:
        with open(path) as f:
            eval_dict = yaml.safe_load(f)
        eval_obj = Eval.from_file(path)
        assert isinstance(eval_obj, Eval)
        assert eval_obj.to_dict() == eval_dict
        assert Eval(**eval_dict) == eval_obj

@pytest.mark.parametrize(
    ('candidate', 'metadata'),
    [
        (None, None),
        (lambda x: x, {'foo': 'bar'}),
        (UnregisteredCandidate(response="foobar"), None),
    ],
)
def test__Eval__call__result__to_from_dict(candidate, metadata):  # noqa: ANN001
    """Tests the basic case of calling an Eval object and converting it to/from a dict."""
    messages = [user_message('test')]
    eval_obj = Eval(input=messages, checks=[MatchCheck(value='fails')])
    # dict before call should be the same as after call
    assert eval_obj.to_dict() == {'input': messages, 'checks': [{'check_type': 'MATCH', 'value': 'fails'}]}  # noqa: E501
    assert Eval(**eval_obj.to_dict()) == eval_obj
    response = f'response: {eval_obj.input}'
    result = eval_obj(
        response=response,
        metadata=metadata,
        candidate=candidate,
    )
    assert result.eval == eval_obj
    assert result.response == response
    assert result.metadata == metadata
    assert result.check_results[0].value is False
    assert result.check_results[0].success is False
    assert result.check_results[0].metadata['check_type'] == 'MATCH'
    assert result.check_results[0].metadata['check_value'] == 'fails'
    assert result.num_checks == 1
    assert result.timestamp
    if candidate:
        assert result.candidate
    else:
        assert result.candidate is None
    assert Eval(**eval_obj.to_dict()) == eval_obj

    result_dict = result.to_dict()
    assert result_dict['eval'] == eval_obj.to_dict()
    assert result_dict['metadata'] == metadata
    assert result_dict['timestamp']
    assert result_dict['response'] == response
    assert result_dict['check_results'] == [result.check_results[0].to_dict()]
    if candidate:
        if isinstance(candidate, UnregisteredCandidate):
            assert result_dict['candidate']['candidate_type'] == 'UnregisteredCandidate'
        else:
            assert result_dict['candidate']
    else:
        assert result_dict['candidate'] is None
    assert Eval(**result_dict['eval']) == eval_obj
    assert EvalResult(**result_dict) == result
    assert EvalResult(**result_dict).to_dict() == result.to_dict()

def test__Eval__from_objects__minimal():
    eval_obj = Eval(checks=[ContainsCheck(value='a response')])
    response = "This is a response."
    result = eval_obj(response)
    assert result.eval == eval_obj
    assert result.candidate is None
    assert result.response == response
    assert result.num_checks == 1
    assert result.num_successful_checks == 1
    assert result.perc_successful_checks == 1
    assert result.check_results[0].value is True
    assert result.check_results[0].success is True
    assert result.check_results[0].metadata['check_type'] == 'CONTAINS'
    assert result.check_results[0].metadata['check_value'] == 'a response'
    assert result.timestamp

@pytest.mark.parametrize('passes_input', [True, False])
def test__Eval__example_8f9fbf37__callable_candidate(passes_input: bool, fake_eval_8f9fbf37: dict):
    eval_dict = fake_eval_8f9fbf37.copy()
    if not passes_input:
        del eval_dict['input']
    eval_obj = Eval(**eval_dict)
    assert eval_obj.to_dict() == eval_dict
    assert Eval(**eval_dict) == eval_obj

    response = "This is a response with code blocks\n```python\nprint('hello world')\n```"
    eval_result = eval_obj(response)
    assert eval_result.response == response
    assert eval_result.eval.input == eval_obj.input
    assert eval_result.eval.to_dict() == eval_dict
    expected_num_checks = 3
    assert eval_result.num_checks == expected_num_checks
    assert eval_result.num_successful_checks == 2
    assert eval_result.perc_successful_checks == 2 / expected_num_checks
    assert len(eval_result.check_results) == expected_num_checks
    assert eval_result.check_results[-1].metadata['num_code_blocks'] == 1

    eval_result_dict = eval_result.to_dict()
    # we can't check that entire eval_result_dict will recreate the exact eval_result object
    # because the candidate will be slightly different (e.g. if it was a function, it will have
    # been converted to a string; we can't serialize the underlying model/llm)
    assert eval_result_dict['eval'] == eval_dict
    assert Eval(**eval_result_dict['eval']) == eval_obj
    assert eval_result_dict['candidate'] is None
    assert eval_result_dict['response'] == response
    assert eval_result_dict['metadata'] is None
    assert eval_result_dict['timestamp']
    # check that the check result dicts match
    assert eval_result_dict['check_results'] == [r.to_dict() for r in eval_result.check_results]
    # check that the eval_result_dict will recreate the exact eval_result object
    recreated_result = EvalResult(**eval_result_dict)
    assert recreated_result == eval_result
    assert recreated_result.to_dict() == eval_result.to_dict()
    assert recreated_result.eval == eval_result.eval.to_dict()
    assert not recreated_result.candidate
    assert recreated_result.response == eval_result.response
    assert recreated_result.metadata == eval_result.metadata
    assert recreated_result.timestamp == eval_result.timestamp
    assert recreated_result.check_results == eval_result.check_results

def test__Eval__multiple_code_blocks__ensure_code_blocks_run(fake_eval_sum_two_numbers_code_blocks_run: dict):  # noqa: E501
    """
    Use Mock LLM with multiple code blocks (over multiple responses) to ensure code blocks run and
    the check results return the expected values.
    """
    config = fake_eval_sum_two_numbers_code_blocks_run.copy()
    eval_obj = Eval(**config)

    assert eval_obj.checks[-1].code_block_timeout == 5
    assert eval_obj.checks[-1].code_test_timeout == 5

    response = dedent("""
    Certainly! Below is a simple Python function named `sum_two_numbers` that takes two parameters, `a` and `b`, which are intended to be numbers. The function returns the sum of these two numbers.

    ```python
    def sum_two_numbers(a, b):
        return a + b
    ```

    You can use this function by passing two numbers to it, and it will return their sum. For example:

    ```python
    result = sum_two_numbers(100, 5)
    print(result)  # This will print 105
    ```
    """)  # noqa: E501
    expected_code_blocks = extract_code_blocks(response)
    expected_num_code_blocks = len(expected_code_blocks)
    assert expected_num_code_blocks == 2
    expected_successful_code_blocks = len(expected_code_blocks)

    expected_config = deepcopy(config)

    def dummy_candidate(_): return None  # noqa

    eval_result = eval_obj(response, metadata={'foo': 'bar'}, candidate=dummy_candidate)
    assert eval_result.eval == eval_obj
    assert eval_result.candidate == str(dummy_candidate)
    assert eval_result.response == response
    assert eval_result.metadata == {'foo': 'bar'}
    assert eval_result.timestamp

    assert eval_result.eval.to_dict() == expected_config
    assert Eval(**eval_obj.to_dict()) == eval_obj
    assert str(EvalResult(**eval_result.to_dict()).to_dict()) == str(eval_result.to_dict())

    assert eval_result.num_checks == 5
    assert eval_result.num_successful_checks == 2
    assert eval_result.perc_successful_checks == 2 / 5
    assert eval_result.check_results[-1].metadata['num_code_blocks'] == expected_num_code_blocks

    assert len(eval_result.check_results) == 5
    assert isinstance(eval_result.check_results[0], PassFailResult)
    assert isinstance(eval_result.check_results[1], PassFailResult)
    assert isinstance(eval_result.check_results[2], PassFailResult)
    assert isinstance(eval_result.check_results[3], PassFailResult)
    assert isinstance(eval_result.check_results[4], ScoreResult)

    assert eval_result.check_results[0].success
    assert eval_result.check_results[0].metadata['check_type'] == CheckType.CONTAINS.name
    assert not eval_result.check_results[1].success
    assert eval_result.check_results[1].metadata['check_type'] == CheckType.MATCH.name
    assert not eval_result.check_results[2].success
    assert eval_result.check_results[2].metadata['check_type'] == CheckType.CONTAINS.name
    assert eval_result.check_results[3].success
    assert eval_result.check_results[3].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name  # noqa: E501
    assert not eval_result.check_results[4].success
    assert eval_result.check_results[4].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name  # noqa

    # function checks
    expected_code_tests = 5
    expected_successful_code_tests = 3
    expected_total_checks = expected_num_code_blocks + expected_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests

    assert eval_result.check_results[-1].value == expected_successful_checks / expected_total_checks  # noqa: E501
    assert eval_result.check_results[-1].success_threshold == 1
    assert not eval_result.check_results[-1].success
    assert eval_result.check_results[-1].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name  # noqa
    assert eval_result.check_results[-1].metadata['num_code_blocks'] == expected_num_code_blocks
    assert eval_result.check_results[-1].metadata['num_code_blocks_successful'] == expected_successful_code_blocks  # noqa
    assert eval_result.check_results[-1].metadata['code_blocks'] == expected_code_blocks
    assert eval_result.check_results[-1].metadata['code_block_errors'] == [None, None]
    # first function check should have run successfully, but second code block should have failed
    assert eval_result.check_results[-1].metadata['code_test_results'] == [True, True, False, True, False]  # noqa
    assert eval_result.check_results[-1].metadata['num_code_tests'] == expected_code_tests
    assert eval_result.check_results[-1].metadata['num_code_tests_successful'] == expected_successful_code_tests  # noqa
    assert eval_result.check_results[-1].metadata['code_test_errors'][0] is None
    assert eval_result.check_results[-1].metadata['code_test_errors'][1] is None
    assert eval_result.check_results[-1].metadata['code_test_errors'][2] is None
    assert eval_result.check_results[-1].metadata['code_test_errors'][3] is None
    assert eval_result.check_results[-1].metadata['code_test_errors'][4] == {
        'error': 'NameError',
        'message': "name 'variable_does_not_exist' is not defined",
    }

def test__EvalHarness__initializating_with_different_types_gives_same_config(
        fake_eval_subtract_two_numbers: dict,
        fake_eval_sum_two_numbers: dict,
    ):
    subtract_config = fake_eval_subtract_two_numbers.copy()
    sum_config = fake_eval_sum_two_numbers.copy()

    response_subtract = 'This is the response.\n\n```\ndef subtract_two_numbers(a, b):\n    return a - b\n```'  # noqa
    response_sum = 'This is the response.\n\n```\ndef sum_two_numbers(a, b):\n    return a + b\n```'  # noqa
    responses_lookup = {
        fake_eval_subtract_two_numbers['input'][0]['content']: response_subtract,
        fake_eval_sum_two_numbers['input'][0]['content']: response_sum,
    }

    candidate_1_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': 'MockCandidate',
        'responses': responses_lookup,
    }
    candidate_2_dict = deepcopy(candidate_1_dict)
    candidate_2_dict['metadata']['uuid'] = 'candidate_2'

    ####
    # Test that creating the EvalHarness object in different ways still results in the same config
    ####
    eval_harness_via_dicts = EvalHarness(
        evals=[subtract_config, sum_config],
        candidates=[
            candidate_1_dict, candidate_2_dict, MockCandidateCausesError(),
        ],
    )
    eval_harness_via_objects = EvalHarness(
        evals=[Eval(**subtract_config), Eval(**sum_config)],
        candidates=[
            Candidate.from_dict(candidate_1_dict),
            Candidate.from_dict(candidate_2_dict),
            MockCandidateCausesError(),
        ],
    )
    assert eval_harness_via_dicts.evals == eval_harness_via_objects.evals
    assert eval_harness_via_dicts.candidates == eval_harness_via_objects.candidates
    assert all(e1 is not e2 for e1, e2 in zip(eval_harness_via_dicts.evals, eval_harness_via_objects.evals))  # noqa
    assert all(c1 is not c2 for c1, c2 in zip(eval_harness_via_dicts.candidates, eval_harness_via_objects.candidates))  # noqa

    eval_harness_via_dicts_via_add = EvalHarness()
    eval_harness_via_dicts_via_add.add_evals(subtract_config)
    eval_harness_via_dicts_via_add.add_evals(sum_config)
    eval_harness_via_dicts_via_add.add_candidates(candidate_1_dict)
    eval_harness_via_dicts_via_add.add_candidates(candidate_2_dict)
    eval_harness_via_dicts_via_add.add_candidates(MockCandidateCausesError())
    assert eval_harness_via_dicts.evals == eval_harness_via_dicts_via_add.evals
    assert eval_harness_via_dicts.candidates == eval_harness_via_dicts_via_add.candidates
    eval_harness_via_dicts_via_add = EvalHarness()
    eval_harness_via_dicts_via_add.add_evals([subtract_config, sum_config])
    eval_harness_via_dicts_via_add.add_candidates([
        candidate_1_dict, candidate_2_dict, MockCandidateCausesError(),
    ])
    assert eval_harness_via_dicts.evals == eval_harness_via_dicts_via_add.evals

    eval_harness = EvalHarness()
    assert len(eval_harness.evals) == 0
    assert len(eval_harness.candidates) == 0
    eval_harness.add_evals(Eval(**subtract_config))
    eval_harness.add_evals(Eval(**sum_config))
    eval_harness.add_candidates(Candidate.from_dict(candidate_1_dict))
    eval_harness.add_candidates(Candidate.from_dict(candidate_2_dict))
    eval_harness.add_candidates(MockCandidateCausesError())
    assert eval_harness.evals == eval_harness_via_dicts.evals
    assert eval_harness.candidates == eval_harness_via_dicts.candidates

class MockCandidateCausesError(Candidate):  # noqa: D101
    def __call__(self, input: object) -> CandidateResponse:  # noqa
        raise ValueError("This candidate always fails.")

@pytest.mark.parametrize('candidate_type', ['AsyncMockCandidate', 'MockCandidate'])
@pytest.mark.parametrize('num_samples', [1, 5])
@pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
@pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
def test__EvalHarness(
        candidate_type: str,
        num_samples: int,
        response_mode: str,
        eval_mode: str,
        fake_eval_subtract_two_numbers: dict,
        fake_eval_sum_two_numbers: dict,
    ):
    expected_num_eval_results = 2 * num_samples
    subtract_config = fake_eval_subtract_two_numbers.copy()
    sum_config = fake_eval_sum_two_numbers.copy()

    response_subtract = 'This is the response.\n\n```\ndef subtract_two_numbers(a, b):\n    return a - b\n```'  # noqa
    response_sum = 'This is the response.\n\n```\ndef sum_two_numbers(a, b):\n    return a + b\n```'  # noqa
    responses_lookup = {
        fake_eval_subtract_two_numbers['input'][0]['content']: response_subtract,
        fake_eval_sum_two_numbers['input'][0]['content']: response_sum,
    }

    candidate_1_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': candidate_type,
        'responses': responses_lookup,
    }
    candidate_2_dict = deepcopy(candidate_1_dict)
    candidate_2_dict['metadata']['uuid'] = 'candidate_2'

    eval_harness = EvalHarness(
        response_mode=response_mode,
        eval_mode=eval_mode,
        num_samples=num_samples,
    )
    assert len(eval_harness.evals) == 0
    assert len(eval_harness.candidates) == 0
    eval_harness.add_evals(Eval(**subtract_config))
    eval_harness.add_evals(Eval(**sum_config))
    eval_harness.add_candidates(Candidate.from_dict(candidate_1_dict))
    eval_harness.add_candidates(Candidate.from_dict(candidate_2_dict))
    eval_harness.add_candidates(MockCandidateCausesError())
    assert len(eval_harness.evals) == 2
    assert len(eval_harness.candidates) == 3

    ####
    # Now run the eval harness and check the results
    ####
    results = eval_harness()
    assert len(results) == len(eval_harness.candidates)
    assert all(isinstance(x, CandidateRunResults) for x in results)

    # ensure the number of EvalResults is correct for each candidate
    assert all(expected_num_eval_results == len(x.eval_results) for x in results)
    # ensure the results associated with each candidate have both evals
    expected_eval_ids = {
        fake_eval_subtract_two_numbers['metadata']['uuid'],
        fake_eval_sum_two_numbers['metadata']['uuid'],
    }
    assert all(
        expected_eval_ids == {er.eval.metadata['uuid'] for er in r.eval_results}
        for r in results[0:2]  # third candidate causes an error; only check first two
    )

    for eval_result, response_error, eval_error in results[0]:
        assert isinstance(eval_result, EvalResult)
        assert response_error is None
        assert eval_error is None
    assert results[0].candidate == Candidate.from_dict(candidate_1_dict)
    assert results[0].num_errors == 0
    assert results[0].response_errors == [None] * expected_num_eval_results
    assert results[0].eval_errors == [None] * expected_num_eval_results
    # The first list should contain the results for candidate 1 (subtract eval, sum eval)
    assert results[0].eval_results[0].eval == Eval(**subtract_config)
    assert results[0].eval_results[0].candidate == Candidate.from_dict(candidate_1_dict)
    # num_samples indexes into the first occurrence of the second eval
    assert results[0].eval_results[num_samples].eval == Eval(**sum_config)
    assert results[0].eval_results[num_samples].candidate == Candidate.from_dict(candidate_1_dict)

    for eval_result, response_error, eval_error in results[1]:
        assert isinstance(eval_result, EvalResult)
        assert response_error is None
        assert eval_error is None
    assert results[1].candidate == Candidate.from_dict(candidate_2_dict)
    assert results[1].num_errors == 0
    assert results[1].response_errors == [None] * expected_num_eval_results
    assert results[1].eval_errors == [None] * expected_num_eval_results
    # The second list should contain the results for candidate 2 (subtract eval, sum eval)
    assert results[1].candidate == Candidate.from_dict(candidate_2_dict)
    assert results[1].eval_results[0].eval == Eval(**subtract_config)
    assert results[1].eval_results[0].candidate == Candidate.from_dict(candidate_2_dict)
    assert results[1].eval_results[num_samples].eval == Eval(**sum_config)
    assert results[1].eval_results[num_samples].candidate == Candidate.from_dict(candidate_2_dict)

    for eval_result, response_error, eval_error in results[2]:
        assert isinstance(eval_result, EvalResult)
        assert isinstance(response_error, ValueError)
        assert eval_error is None
    assert results[2].candidate == MockCandidateCausesError()
    assert results[2].num_errors == expected_num_eval_results
    assert len(results[2].response_errors) == expected_num_eval_results
    assert all(isinstance(x, ValueError) for x in results[2].response_errors)
    assert results[2].eval_errors == [None] * expected_num_eval_results
    assert results[2].eval_results[0].candidate == MockCandidateCausesError()
    assert results[2].eval_results[0].eval == Eval(**subtract_config)
    assert all(c.success is False for c in results[2].eval_results[0].check_results)
    assert results[2].eval_results[num_samples].candidate == MockCandidateCausesError()
    assert results[2].eval_results[num_samples].eval == Eval(**sum_config)
    assert all(c.success is False for c in results[2].eval_results[num_samples].check_results)

    # eval objects across candidates should have same values (same eval) but different objects
    assert results[0].eval_results[0].eval == results[1].eval_results[0].eval
    assert results[0].eval_results[0].eval is not results[1].eval_results[0].eval
    # when num_samples >1 the first and second indexes will be the same eval but should be
    # different objects
    assert results[0].eval_results[0].eval is not results[1].eval_results[1].eval
    assert results[0].eval_results[num_samples].eval == results[1].eval_results[num_samples].eval
    assert results[0].eval_results[num_samples].eval is not results[1].eval_results[num_samples].eval  # noqa: E501

    # candidate 1 - subtract eval
    cand_1_results_subtract = results[0].eval_results[0]
    assert cand_1_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_1_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 1 - sum eval
    cand_1_results_sum = results[0].eval_results[num_samples]
    assert cand_1_results_sum.response == response_sum
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1
    assert cand_1_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 2 - subtract eval
    cand_2_results_subtract = results[1].eval_results[0]
    assert cand_2_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_2_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 2 - sum eval
    cand_2_results_sum = results[1].eval_results[num_samples]
    assert cand_2_results_sum.response == response_sum
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1
    assert cand_2_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 3 returns an error so there will not be any successful checks
    cand_3_results_subtract = results[2].eval_results[0]
    assert cand_3_results_subtract.response is None
    assert cand_3_results_subtract.num_checks == 3
    assert cand_3_results_subtract.num_successful_checks == 0
    assert cand_3_results_subtract.perc_successful_checks == 0
    assert cand_3_results_subtract.check_results[-1].metadata['num_code_blocks'] == 0
    assert cand_3_results_subtract.check_results[-1].metadata['code_blocks'] == []

    # the eval results of candidate 1 should be the same as the eval results of candidate 2,
    # except the seconds it took to run the evals and the uuid of the candidate
    cand_1_results_subtract_dict = deepcopy(cand_1_results_subtract.to_dict())
    del cand_1_results_subtract_dict['timestamp']
    del cand_1_results_subtract_dict['candidate']['metadata']['uuid']
    del cand_1_results_subtract_dict['metadata']['response_timestamp']
    cand_2_results_subtract_dict = deepcopy(cand_2_results_subtract.to_dict())
    del cand_2_results_subtract_dict['timestamp']
    del cand_2_results_subtract_dict['candidate']['metadata']['uuid']
    del cand_2_results_subtract_dict['metadata']['response_timestamp']
    assert cand_1_results_subtract_dict == cand_2_results_subtract_dict

    cand_1_results_subtract.to_yaml('__temp__.yaml')
    result_from_yaml = cand_1_results_subtract.from_file('__temp__.yaml')
    assert result_from_yaml == cand_1_results_subtract
    assert result_from_yaml.to_dict() == cand_1_results_subtract.to_dict()
    os.remove('__temp__.yaml')

    cand_1_results_subtract.to_json('__temp__.json')
    result_from_json = cand_1_results_subtract.from_file('__temp__.json')
    assert result_from_json == cand_1_results_subtract
    assert result_from_json.to_dict() == cand_1_results_subtract.to_dict()
    os.remove('__temp__.json')

    cand_1_results_sum.to_yaml('__temp__.yaml')
    result_from_yaml = cand_1_results_sum.from_yaml('__temp__.yaml')
    assert result_from_yaml == cand_1_results_sum
    assert result_from_yaml.to_dict() == cand_1_results_sum.to_dict()
    os.remove('__temp__.yaml')

    cand_1_results_sum.to_json('__temp__.json')
    result_from_json = cand_1_results_sum.from_file('__temp__.json')
    assert result_from_json == cand_1_results_sum
    assert result_from_json.to_dict() == cand_1_results_sum.to_dict()
    os.remove('__temp__.json')

    assert subtract_config == fake_eval_subtract_two_numbers  # ensure eval_config wasn't modified
    assert sum_config == fake_eval_sum_two_numbers  # ensure eval_config wasn't modified

class MockCheckCausesError(Check):  # noqa: D101
    def __call__(self, response: object) -> CheckResult:  # noqa
        raise RuntimeError("This check always fails.")

@pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
@pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
def test__EvalHarness__Check_raises_error(
        fake_eval_subtract_two_numbers: dict,
        response_mode: str,
        eval_mode: str,
        ):
    eval_config = fake_eval_subtract_two_numbers.copy()
    check = MockCheckCausesError()
    eval_obj = Eval(**eval_config)
    eval_obj.checks = [check, *eval_obj.checks]
    eval_harness = EvalHarness(
        response_mode=response_mode,
        eval_mode=eval_mode,
        evals=eval_obj,
        candidates=[UnregisteredCandidate(response=eval_obj.input), MockCandidateCausesError()],
    )
    results = eval_harness()
    assert len(results) == 2
    assert all(len(r.eval_results) == 1 for r in results)
    assert all(r.num_errors == 1 for r in results)

    for eval_result, response_error, eval_error in results[0]:
        assert eval_result is None
        assert response_error is None
        assert isinstance(eval_error, RuntimeError)
    assert results[0].candidate == UnregisteredCandidate(response=eval_obj.input)
    assert results[0].num_errors == 1
    assert results[0].response_errors == [None]
    assert len(results[0].eval_errors) == 1
    assert isinstance(results[0].eval_errors[0], RuntimeError)
    assert results[0].eval_results == [None]

    for eval_result, response_error, eval_error in results[1]:
        assert eval_result is None
        assert isinstance(response_error, ValueError)
        assert isinstance(eval_error, RuntimeError)
    assert results[1].candidate == MockCandidateCausesError()
    assert results[1].num_errors == 1
    assert len(results[1].response_errors) == 1
    assert isinstance(results[1].response_errors[0], ValueError)
    assert len(results[1].eval_errors) == 1
    assert isinstance(results[1].eval_errors[0], RuntimeError)

@pytest.mark.parametrize("candidate_type", ["AsyncMockCandidate", "MockCandidate"])
def test__evals__num_samples__greater_than_one__async__via_constructor(
        candidate_type: str,
        fake_eval_subtract_two_numbers: dict,
        fake_eval_sum_two_numbers: dict,
    ):
    """Tests num_samples > 1 for async evals and when we pass num_samples to constructor."""
    subtract_config = fake_eval_subtract_two_numbers.copy()
    sum_config = fake_eval_sum_two_numbers.copy()

    response_subtract = 'This is the response.\n\n```\ndef subtract_two_numbers(a, b):\n    return a - b\n```'  # noqa
    response_sum = 'This is the response.\n\n```\ndef sum_two_numbers(a, b):\n    return a + b\n```'  # noqa
    responses_lookup = {
        fake_eval_subtract_two_numbers['input'][0]['content']: response_subtract,
        fake_eval_sum_two_numbers['input'][0]['content']: response_sum,
    }

    candidate_1_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': candidate_type,
        'responses': responses_lookup,
    }
    candidate_2_dict = deepcopy(candidate_1_dict)
    candidate_2_dict['metadata']['uuid'] = 'candidate_2'

    eval_harness_via_dicts = EvalHarness(
        evals=[subtract_config, sum_config],
        candidates=[candidate_1_dict, candidate_2_dict],
    )
    eval_harness_via_objects = EvalHarness(
        evals=[Eval(**subtract_config), Eval(**sum_config)],
        candidates=[Candidate.from_dict(candidate_1_dict), Candidate.from_dict(candidate_2_dict)],
    )
    assert eval_harness_via_dicts.evals == eval_harness_via_objects.evals
    assert eval_harness_via_dicts.candidates == eval_harness_via_objects.candidates

    num_samples = 3
    eval_harness = EvalHarness(
        num_samples=num_samples,
    )
    assert eval_harness.evals != eval_harness_via_dicts.evals
    assert eval_harness.candidates != eval_harness_via_dicts.candidates
    eval_harness.add_evals(Eval(**subtract_config))
    eval_harness.add_evals(Eval(**sum_config))
    eval_harness.add_candidates(Candidate.from_dict(candidate_1_dict))
    eval_harness.add_candidates(Candidate.from_dict(candidate_2_dict))
    assert eval_harness.evals == eval_harness_via_dicts.evals
    assert eval_harness.candidates == eval_harness_via_dicts.candidates
    num_candidates = len(eval_harness.candidates)
    num_evals = len(eval_harness.evals)

    results = eval_harness()
    assert len(results) == num_candidates
    assert all(len(r.eval_results) == num_evals * num_samples for r in results)
    assert all(r.num_errors == 0 for r in results)
    assert all(r.response_errors == [None] * num_evals * num_samples for r in results)
    assert all(r.eval_errors == [None] * num_evals * num_samples for r in results)
    # The underlying candidate objects should have the same values but should be different objects
    # because each candidate object (against a specific eval) is responsible for storing its own
    # history/conversation and the history should be different for each eval.
    assert results[0].eval_results[0].candidate == results[0].eval_results[1].candidate
    assert results[1].eval_results[0].candidate == results[1].eval_results[1].candidate

    # The first list should contain the results for candidate 1 (subtract eval * 3, sum eval * 3)
    # 3 SAMPLES OF SUBTRACT EVAL
    assert results[0].eval_results[0].eval == Eval(**subtract_config)
    assert results[0].eval_results[0].candidate == Candidate.from_dict(candidate_1_dict)
    assert results[0].eval_results[1].eval == Eval(**subtract_config)
    assert results[0].eval_results[1].candidate == Candidate.from_dict(candidate_1_dict)
    assert results[0].eval_results[2].eval == Eval(**subtract_config)
    assert results[0].eval_results[2].candidate == Candidate.from_dict(candidate_1_dict)
    # 3 SAMPLES OF SUM EVAL
    assert results[0].eval_results[3].eval == Eval(**sum_config)
    assert results[0].eval_results[3].candidate == Candidate.from_dict(candidate_1_dict)
    assert results[0].eval_results[4].eval == Eval(**sum_config)
    assert results[0].eval_results[4].candidate == Candidate.from_dict(candidate_1_dict)
    assert results[0].eval_results[5].eval == Eval(**sum_config)
    assert results[0].eval_results[5].candidate == Candidate.from_dict(candidate_1_dict)

    # The second list should contain the results for candidate 2 (subtract eval, sum eval)
    # 3 SAMPLES OF SUBTRACT EVAL
    assert results[1].eval_results[0].eval == Eval(**subtract_config)
    assert results[1].eval_results[0].candidate == Candidate.from_dict(candidate_2_dict)
    assert results[1].eval_results[1].eval == Eval(**subtract_config)
    assert results[1].eval_results[1].candidate == Candidate.from_dict(candidate_2_dict)
    assert results[1].eval_results[2].eval == Eval(**subtract_config)
    assert results[1].eval_results[2].candidate == Candidate.from_dict(candidate_2_dict)
    # 3 SAMPLES OF SUM EVAL
    assert results[1].eval_results[3].eval == Eval(**sum_config)
    assert results[1].eval_results[3].candidate == Candidate.from_dict(candidate_2_dict)
    assert results[1].eval_results[4].eval == Eval(**sum_config)
    assert results[1].eval_results[4].candidate == Candidate.from_dict(candidate_2_dict)
    assert results[1].eval_results[5].eval == Eval(**sum_config)
    assert results[1].eval_results[5].candidate == Candidate.from_dict(candidate_2_dict)

    # eval objects across candidates should have same values (same eval) but different objects
    assert results[0].eval_results[0].eval == results[1].eval_results[0].eval
    assert results[0].eval_results[0].eval is not results[1].eval_results[0].eval
    assert results[0].eval_results[0].eval == results[0].eval_results[1].eval
    assert results[0].eval_results[0].eval is not results[0].eval_results[1].eval
    assert results[0].eval_results[0].eval == results[0].eval_results[2].eval
    assert results[0].eval_results[0].eval is not results[0].eval_results[2].eval

    assert results[0].eval_results[3].eval == results[1].eval_results[3].eval
    assert results[0].eval_results[3].eval is not results[1].eval_results[3].eval
    assert results[0].eval_results[3].eval == results[0].eval_results[4].eval
    assert results[0].eval_results[3].eval is not results[0].eval_results[4].eval
    assert results[0].eval_results[3].eval == results[0].eval_results[5].eval
    assert results[0].eval_results[3].eval is not results[0].eval_results[5].eval

    # candidate 1 - subtract eval; all 3 should have same results
    cand_1_results_subtract = results[0].eval_results[0]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_1_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    cand_1_results_subtract = results[0].eval_results[1]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_1_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1


    cand_1_results_subtract = results[0].eval_results[2]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_1_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 1 - sum eval; all 3 should have same results
    cand_1_results_sum = results[0].eval_results[3]
    assert cand_1_results_sum.response == response_sum
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1
    assert cand_1_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    cand_1_results_sum = results[0].eval_results[4]
    assert cand_1_results_sum.response == response_sum
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1
    assert cand_1_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    cand_1_results_sum = results[0].eval_results[5]
    assert cand_1_results_sum.response == response_sum
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1
    assert cand_1_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 2 - subtract eval; all 3 should have same results
    cand_2_results_subtract = results[1].eval_results[0]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.response == response_subtract
    assert cand_2_results_subtract.num_checks == 3
    assert cand_2_results_subtract.num_successful_checks == 2
    assert cand_2_results_subtract.perc_successful_checks == 2 / 3
    assert cand_2_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    cand_2_results_subtract = results[1].eval_results[1]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.response == response_subtract
    assert cand_2_results_subtract.num_checks == 3
    assert cand_2_results_subtract.num_successful_checks == 2
    assert cand_2_results_subtract.perc_successful_checks == 2 / 3
    assert cand_2_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    cand_2_results_subtract = results[1].eval_results[2]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.response == response_subtract
    assert cand_2_results_subtract.num_checks == 3
    assert cand_2_results_subtract.num_successful_checks == 2
    assert cand_2_results_subtract.perc_successful_checks == 2 / 3
    assert cand_2_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 2 - sum eval; all 3 should have same results
    cand_2_results_sum = results[1].eval_results[3]
    assert cand_2_results_sum.response == response_sum
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1
    assert cand_2_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    cand_2_results_sum = results[1].eval_results[4]
    assert cand_2_results_sum.response == response_sum
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1
    assert cand_2_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    cand_2_results_sum = results[1].eval_results[5]
    assert cand_2_results_sum.response == response_sum
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1
    assert cand_2_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

def test__Eval_with_numeric_values_loads_correctly(fake_eval_non_string_values: dict):
    """Test that numeric values are converted to strings when loading an Eval object."""
    eval_config = deepcopy(fake_eval_non_string_values)
    eval_ = Eval(**eval_config)
    assert Eval(**eval_.to_dict()) == eval_
    assert eval_.metadata['version'] == 1
    assert eval_.metadata['tags'][0] == 1
    assert isinstance(eval_.input, list)
    assert isinstance(eval_.input[0], dict)
    assert 'role' in eval_.input[0]
    assert 'content' in eval_.input[0]
    assert eval_.input[0]['content'] == 6

@pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
def test__openai_candidates__across_all_modes(
        response_mode: Mode,
        openai_candidate_template: dict,
        fake_eval_sum_two_numbers: dict,
        fake_eval_subtract_two_numbers: dict,
    ):
    """
    The purpose of this test is mainly to ensure we can pickle the openai candidate object when
    using multiprocessing. For example, defining the OpenAI class in the __init__ method of the
    OpenAI Candidate class will cause an error when pickling the object.
    """
    candidate_1 = deepcopy(openai_candidate_template)
    candidate_2 = deepcopy(openai_candidate_template)
    eval_1 = Eval(**fake_eval_sum_two_numbers)
    eval_2 = Eval(**fake_eval_subtract_two_numbers)
    harness = EvalHarness(
        evals=[eval_1, eval_2],
        candidates=[candidate_1, candidate_2],
        response_mode=response_mode,
        eval_mode=Mode.SYNC,
    )
    results = harness()
    assert len(results) == 2
    assert all(len(r.eval_results) == 2 for r in results)
    assert all(r.num_errors == 0 for r in results)
    assert all(r.response_errors == [None, None] for r in results)
    assert all(r.eval_errors == [None, None] for r in results)
    assert results[0].eval_results[0].metadata['response_metadata']['total_tokens'] > 0
    assert results[0].eval_results[1].metadata['response_metadata']['total_tokens'] > 0
    assert results[1].eval_results[0].metadata['response_metadata']['total_tokens'] > 0
    assert results[1].eval_results[1].metadata['response_metadata']['total_tokens'] > 0

def test__Eval__unregistered_check__non_string_prompt_and_response():
    """
    We should be able to use unregistered Check and Candidate classes with non-string prompts and
    responses. These classes won't be able to be saved/loaded from a dictionary, and so we can't
    use them with EvalHarness, but we should be able to use them individually.
    """
    eval_ = Eval(
        input={'prompt': 'Test Prompt'},
        checks=[UnregisteredCheck()],
    )
    assert eval_.to_dict() == {'input': {'prompt': 'Test Prompt'}, 'checks': [{'check_type': 'UnregisteredCheck'}]}  # noqa
    expected_response = {'prompt': {'prompt': 'Test Prompt'}, 'response': 42}
    result = eval_(response=expected_response, candidate=UnregisteredCandidate(42))
    assert result.response == expected_response
    assert len(result.check_results) == 1
    check_result = result.check_results[0]
    assert check_result.value == expected_response
    assert check_result.success is True
    assert check_result.to_dict() == {'value': expected_response, 'success': True, 'result_type': 'UnregisteredCheckResult'}  # noqa
    assert result.num_checks == 1
    assert result.num_successful_checks == 1
    assert result.perc_successful_checks == 1
    # ensure that we can convert the results (which contain unregistered checks/candidates) to
    # a string and dictionary (which call underlying str and to_dict methods on
    # checks/candidates)
    assert result.to_dict()['eval'] == eval_.to_dict()
    assert result.to_dict()['candidate'] == UnregisteredCandidate(42).to_dict()
    assert result.to_dict()['check_results'][0] == check_result.to_dict()

@pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
@pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
def test__EvalHarness__unregistered_check__unregistered_candidate__non_string_prompt_and_response(
        response_mode: Mode,
        eval_mode: Mode,
    ):
        harness = EvalHarness(
            response_mode=response_mode,
            eval_mode=eval_mode,
            evals=[
                Eval(
                    input={'prompt': 'Test Prompt 1'},  # test with dictionary prompt
                    checks=[UnregisteredCheck()],
                ),
                Eval(
                    input={'prompt': 'Test Prompt 2'},  # test with dictionary prompt
                    checks=[UnregisteredCheck()],
                ),
            ],
            candidates = [
                UnregisteredCandidate('Response 1'),
                UnregisteredCandidate('Response 2'),
            ],
        )
        results = harness()
        assert len(results) == 2  # 2 candidates
        assert all(len(r.eval_results) == 2 for r in results)
        assert all(r.num_errors == 0 for r in results)
        assert all(r.response_errors == [None, None] for r in results)
        assert all(r.eval_errors == [None, None] for r in results)

        assert results[0].eval_results[0].response == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 1'}  # noqa
        assert results[0].eval_results[1].response == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 1'}  # noqa
        assert results[1].eval_results[0].response == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 2'}  # noqa
        assert results[1].eval_results[1].response == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 2'}  # noqa
        assert len(results[0].eval_results[0].check_results) == 1
        assert results[0].eval_results[0].perc_successful_checks == 1
        assert results[0].eval_results[0].check_results[0].value == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 1'}  # noqa
        assert results[0].eval_results[0].check_results[0].success is True
        assert len(results[0].eval_results[1].check_results) == 1
        assert results[0].eval_results[1].perc_successful_checks == 1
        assert results[0].eval_results[1].check_results[0].value == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 1'}  # noqa
        assert results[0].eval_results[1].check_results[0].success is True
        assert len(results[1].eval_results[0].check_results) == 1
        assert results[1].eval_results[0].perc_successful_checks == 1
        assert results[1].eval_results[0].check_results[0].value == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 2'}  # noqa
        assert results[1].eval_results[0].check_results[0].success is True
        assert len(results[1].eval_results[1].check_results) == 1
        assert results[1].eval_results[1].perc_successful_checks == 1
        assert results[1].eval_results[1].check_results[0].value == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 2'}  # noqa
        assert results[1].eval_results[1].check_results[0].success is True

        # ensure that we can convert the results (which contain unregistered checks/candidates) to
        # a string and dictionary (which call underlying str and to_dict methods on
        # checks/candidates)
        assert len(str(results[0].eval_results[0])) > 10
        assert results[0].eval_results[0].to_dict()['eval'] == harness.evals[0].to_dict()
        assert results[0].eval_results[0].to_dict()['candidate'] == harness.candidates[0].to_dict()
        assert results[0].eval_results[0].to_dict()['check_results'][0] == results[0].eval_results[0].check_results[0].to_dict()  # noqa
        assert results[0].eval_results[1].to_dict()['eval'] == harness.evals[1].to_dict()
        assert results[0].eval_results[1].to_dict()['candidate'] == harness.candidates[0].to_dict()
        assert results[0].eval_results[1].to_dict()['check_results'][0] == results[0].eval_results[1].check_results[0].to_dict()  # noqa
        assert results[1].eval_results[0].to_dict()['eval'] == harness.evals[0].to_dict()
        assert results[1].eval_results[0].to_dict()['candidate'] == harness.candidates[1].to_dict()
        assert results[1].eval_results[0].to_dict()['check_results'][0] == results[1].eval_results[0].check_results[0].to_dict()  # noqa
        assert results[1].eval_results[1].to_dict()['eval'] == harness.evals[1].to_dict()
        assert results[1].eval_results[1].to_dict()['candidate'] == harness.candidates[1].to_dict()
        assert results[1].eval_results[1].to_dict()['check_results'][0] == results[1].eval_results[1].check_results[0].to_dict()  # noqa

def test__EvalResult__loading_unregistered_candidate_from_dict():
    """
    Test that we can (re)load an EvalResult object that contains an unregistered candidate that
    saved using EvalResult.to_dict(). We should be able to convert the EvalResult object to a
    dictionary and back.
    """
    class MyUnregisteredCandidate(Candidate):

        def __init__(self, custom_state: str) -> None:
            super().__init__()
            self.custom_state = custom_state
            self.metadata = {'some_metadata': 'some_value'}

        def __call__(self, input: str) -> CandidateResponse:  # noqa: A002
            return CandidateResponse(
                response={'input': input, 'response': "This is the response foobar"},
                metadata={'custom_state': self.custom_state},
            )

    eval_ = Eval(
        input="This is the input",
        checks=[LambdaCheck(lambda_str="lambda response: 'foobar' in response['response']")],
        metadata={'foo': 'bar'},
    )
    candidate = MyUnregisteredCandidate('barfoo')
    candidate_response = candidate(eval_.input)
    result = eval_(
        response=candidate_response.response,
        candidate=candidate,
        metadata=candidate_response.metadata,
    )
    result_dict = result.to_dict()
    assert result_dict['eval'] == eval_.to_dict()
    assert result_dict['candidate'] == candidate.to_dict()
    assert result_dict['candidate']['metadata'] == {'some_metadata': 'some_value'}
    assert 'candidate_type' in result_dict['candidate']
    assert result_dict['candidate']['candidate_type'] == 'MyUnregisteredCandidate'
    assert result_dict['response'] == candidate_response.response
    assert result_dict['metadata'] == candidate_response.metadata

    assert len(result_dict['check_results']) == 1
    assert result_dict['check_results'][0]['result_type'] == 'PASS_FAIL'
    assert result_dict['check_results'][0]['value'] is True
    assert result_dict['check_results'][0]['success'] is True

    result_loaded = EvalResult(**result_dict)
    assert result_loaded.to_dict() == result_dict

async def mock_async_candidate_1(prompt):  # noqa
    return CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response1'})

async def mock_async_candidate_2(prompt):  # noqa
    return CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response2'})

def mock_sync_candidate_1(prompt):  # noqa
    return CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response1'})

def mock_sync_candidate_2(prompt):  # noqa
    return CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response2'})

def mock_check_1(data):  # noqa
    return 'Response1' in data.response['response']

def mock_check_2(data):  # noqa
    return 'Response2' in data.response['response']

@pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
@pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
@pytest.mark.parametrize('use_async_candidate', [True, False])
def test__EvalHarness__callable_check__callable_candidate__non_string_prompt_and_response(
        response_mode: Mode,
        eval_mode: Mode,
        use_async_candidate: bool,
    ):
    if use_async_candidate:
        candidates = [mock_async_candidate_1, mock_async_candidate_2]
    else:
        candidates = [mock_sync_candidate_1, mock_sync_candidate_2]

    num_samples = 100
    harness = EvalHarness(
        response_mode=response_mode,
        eval_mode=eval_mode,
        num_samples=num_samples,
        evals=[
            Eval(
                input={'prompt': 'Test Prompt 1'},  # non-string prompt
                checks=[mock_check_1],
            ),
            Eval(
                input={'prompt': 'Test Prompt 2'},  # non-string prompt
                checks=[mock_check_2],
            ),
        ],
        candidates = candidates,
    )
    assert len(harness.evals) == 2
    assert len(harness.candidates) == 2
    results = harness()
    assert len(results) == 2  # 2 candidates
    assert all(len(r.eval_results) == 2 * num_samples for r in results)
    assert all(r.num_errors == 0 for r in results)
    assert all(r.response_errors == [None] * 2 * num_samples for r in results)
    assert all(r.eval_errors == [None] * 2 * num_samples for r in results)

    assert results[0].eval_results[0].response == {'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response1'}  # noqa
    assert results[0].eval_results[num_samples-1].response == {'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response1'}  # noqa
    assert results[0].eval_results[num_samples].response == {'prompt': 'Test Prompt 2', 'response': 'Test Prompt 2 & Response1'}  # noqa
    assert results[1].eval_results[0].response == {'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response2'}  # noqa
    assert results[1].eval_results[num_samples].response == {'prompt': 'Test Prompt 2', 'response': 'Test Prompt 2 & Response2'}  # noqa
    # eval 1 candidate 1
    assert len(results[0].eval_results[0].check_results) == 1
    assert results[0].eval_results[0].num_checks == 1
    assert results[0].eval_results[0].num_successful_checks == 1
    assert results[0].eval_results[0].perc_successful_checks == 1
    assert results[0].eval_results[0].check_results[0].value is True
    assert results[0].eval_results[0].check_results[0].success is True
    # eval 2 candidate 1
    assert len(results[0].eval_results[num_samples].check_results) == 1
    assert results[0].eval_results[num_samples].num_checks == 1
    assert results[0].eval_results[num_samples].num_successful_checks == 0
    assert results[0].eval_results[num_samples].perc_successful_checks == 0
    assert results[0].eval_results[num_samples].check_results[0].value is False
    assert results[0].eval_results[num_samples].check_results[0].success is False
    # eval 1 candidate 2
    assert len(results[1].eval_results[0].check_results) == 1
    assert results[1].eval_results[0].num_checks == 1
    assert results[1].eval_results[0].num_successful_checks == 0
    assert results[1].eval_results[0].perc_successful_checks == 0
    assert results[1].eval_results[0].check_results[0].value is False
    assert results[1].eval_results[0].check_results[0].success is False
    # eval 2 candidate 2
    assert len(results[1].eval_results[1].check_results) == 1
    assert results[1].eval_results[num_samples].num_checks == 1
    assert results[1].eval_results[num_samples].num_successful_checks == 1
    assert results[1].eval_results[num_samples].perc_successful_checks == 1
    assert results[1].eval_results[num_samples].check_results[0].value is True
    assert results[1].eval_results[num_samples].check_results[0].success is True

    # ensure that we can convert the results (which contain unregistered checks/candidates) to
    # a string and dictionary (which call underlying str and to_dict methods on
    # checks/candidates)
    assert len(str(results[0].eval_results[0])) > 10
    assert results[0].eval_results[0].to_dict()['eval'] == harness.evals[0].to_dict()
    assert results[0].eval_results[0].to_dict()['candidate']
    assert results[0].eval_results[0].to_dict()['check_results'][0] == results[0].eval_results[0].check_results[0].to_dict()  # noqa: E501
    assert results[0].eval_results[num_samples].to_dict()['eval'] == harness.evals[1].to_dict()
    assert results[0].eval_results[num_samples].to_dict()['candidate']
    assert results[0].eval_results[num_samples].to_dict()['check_results'][0] == results[0].eval_results[num_samples].check_results[0].to_dict()  # noqa: E501
    assert results[1].eval_results[0].to_dict()['eval'] == harness.evals[0].to_dict()
    assert results[1].eval_results[0].to_dict()['candidate']
    assert results[1].eval_results[0].to_dict()['check_results'][0] == results[1].eval_results[0].check_results[0].to_dict()  # noqa: E501
    assert results[1].eval_results[num_samples].to_dict()['eval'] == harness.evals[1].to_dict()
    assert results[1].eval_results[num_samples].to_dict()['candidate']
    assert results[1].eval_results[num_samples].to_dict()['check_results'][0] == results[1].eval_results[num_samples].check_results[0].to_dict()  # noqa: E501

def test__OpenAIToolsCandidate__ToolsCallCheck(openai_tools_candidate_template: dict):
    """Integration test that tests Evaling a real OpenAITool API call against the ToolsCheck."""
    candidate = Candidate.from_dict(openai_tools_candidate_template)
    eval_ = Eval(
        input=[user_message("What's the weather like in Boston today in degrees F?")],
        checks=[
            ToolCallsCheck(
                function_name='get_current_weather',
                function_arguments={'location': 'Boston, MA', 'unit': 'fahrenheit'},
            ),
        ],
    )
    candidate_response = candidate(eval_.input)
    result = eval_(response=candidate_response.response, candidate=candidate)
    tool_response = result.response[0]
    assert tool_response['name'] == 'get_current_weather'
    assert 'location' in tool_response['arguments']
    assert tool_response['arguments']['location']
    assert isinstance(tool_response['arguments']['location'], str)
    assert 'unit' in tool_response['arguments']
    assert tool_response['arguments']['unit'] in ['celsius', 'fahrenheit']
    # check that it gets at least the function name correctly
    assert result.check_results[0].value >= 0.5
