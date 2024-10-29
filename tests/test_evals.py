"""Tests for the evals module."""
import pytest
from copy import deepcopy
import multiprocessing
import os
from textwrap import dedent
import yaml
from llm_eval.candidates import (
    Candidate,
    CandidateResponse,
    is_async_candidate,
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
    Eval,
    EvalHarness,
    EvalResult,
    ResponseError,
)
from llm_eval.internal_utilities import extract_code_blocks
from llm_eval.openai import user_message
from tests.conftest import MockCandidate


def test__Eval__creation():
    messages = [user_message('test')]
    eval_obj = Eval(input=messages)
    eval_dict = eval_obj.to_dict()
    assert eval_dict == {'input': messages}
    assert Eval(**eval_dict) == eval_obj
    assert str(eval_obj)

    messages = [user_message('test1')]
    eval_obj = Eval(
        input=messages,
        ideal_response='test2',
        checks = [
            MatchCheck(value='test6', metadata={'test': 'test7'}),
            ContainsCheck(value='test8'),
        ],
    )
    assert eval_obj.input == messages
    assert eval_obj.ideal_response == 'test2'
    assert eval_obj.checks == [
        MatchCheck(value='test6', metadata={'test': 'test7'}),
        ContainsCheck(value='test8'),
    ]
    assert str(eval_obj)

    eval_dict = eval_obj.to_dict()
    assert eval_dict == {
        'input': messages,
        'ideal_response': 'test2',
        'checks': [
            {'check_type': 'MATCH', 'value': 'test6', 'metadata': {'test': 'test7'}},
            {'check_type': 'CONTAINS', 'value': 'test8'},
        ],
    }
    assert Eval(**eval_dict) == eval_obj

def test__eval_obj__clone(fake_eval_8f9fbf37: dict):
    config = deepcopy(fake_eval_8f9fbf37)
    eval_obj = Eval(**config)
    eval_cloned = eval_obj.clone()
    assert eval_obj == eval_cloned
    assert eval_obj.to_dict() == eval_cloned.to_dict()
    # test-sequence (i.e. PromptTest objects) should be the same prompt tests but different objects
    assert eval_obj.input == eval_cloned.input
    assert eval_obj.input is not eval_cloned.input
    assert eval_obj.ideal_response == eval_cloned.ideal_response
    assert eval_obj.ideal_response is not eval_cloned.ideal_response
    assert eval_obj.checks == eval_cloned.checks
    assert all(c1 is not c2 for c1, c2 in zip(eval_obj.checks, eval_cloned.checks))
    assert eval_obj.metadata == eval_cloned.metadata
    assert eval_obj.metadata is not eval_cloned.metadata

def test__Eval__call__result__to_from_dict():
    """
    Tests the basic case of calling an Eval object and converting it to/from a dict. No checks are
    passed to the eval.
    """
    messages = [user_message('test')]
    eval_obj = Eval(input=messages)
    # dict before call should be the same as after call
    assert eval_obj.to_dict() == {'input': messages}
    assert Eval(**eval_obj.to_dict()) == eval_obj
    result = eval_obj(lambda x: CandidateResponse(response=f'response: {x}'))
    assert result.eval_obj == eval_obj
    assert result.response == "response: [{'role': 'user', 'content': 'test'}]"
    assert result.response_metadata is None
    assert result.total_time_seconds >= 0
    assert result.check_results == []
    assert Eval(**eval_obj.to_dict()) == eval_obj

    result_dict = result.to_dict()
    assert result_dict['eval_obj'] == eval_obj.to_dict()
    assert result_dict['candidate_obj']
    assert Eval(**result_dict['eval_obj']) == eval_obj
    assert EvalResult(**result_dict) == result
    assert EvalResult(**result_dict).to_dict() == result.to_dict()

def test__Eval__from_objects__minimal():
    def mock_llm(x):  # noqa
        return f'response: {x}'
    prompt = "This is a prompt."
    messages = [user_message(prompt)]
    eval_obj = Eval(input=messages)
    result = eval_obj(lambda x: CandidateResponse(response=mock_llm(x)))
    assert result.eval_obj == eval_obj
    assert result.candidate_obj
    assert result.response == mock_llm(messages)
    assert result.num_checks == 0
    assert result.num_successful_checks == 0
    assert result.perc_successful_checks is None
    assert result.check_results == []
    assert result.timestamp
    assert result.total_time_seconds >= 0

@pytest.mark.parametrize('use_async', [True, False])
def test__Eval__example_8f9fbf37__callable_candidate(use_async: bool, fake_eval_8f9fbf37: dict):
    eval_dict = fake_eval_8f9fbf37.copy()
    eval_obj = Eval(**eval_dict)
    assert eval_obj.to_dict() == eval_dict

    responses = [
        CandidateResponse(response="This is a response with code blocks\n```python\nprint('hello world')\n```"),  # noqa
    ]
    def create_mock_llm(responses, use_async):  # noqa
        if use_async:
            iterator = iter(responses)
            async def mock_llm(_: str):  # noqa
                try:
                    return next(iterator)
                except StopIteration:
                    return None
            return mock_llm
        else:  # noqa: RET505
            iterator = iter(responses)
            def mock_llm(_: str):  # noqa
                try:
                    return next(iterator)
                except StopIteration:
                    return None
        return mock_llm

    mock_llm = create_mock_llm(responses, use_async)
    if use_async:
        assert is_async_candidate(mock_llm)
    else:
        assert not is_async_candidate(mock_llm)
    eval_result = eval_obj(mock_llm)
    assert eval_result.response == responses[0].response
    assert eval_result.eval_obj.input == eval_obj.input
    assert eval_result.eval_obj.to_dict() == eval_dict
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
    assert eval_result_dict['eval_obj'] == eval_dict
    assert Eval(**eval_result_dict['eval_obj']) == eval_obj
    assert eval_result_dict['candidate_obj']
    # check that the check result dicts match
    assert eval_result_dict['check_results'] == [r.to_dict() for r in eval_result.check_results]
    assert eval_result.total_time_seconds > 0
    # check that the eval_result_dict will recreate the exact eval_result object
    recreated_eval = EvalResult(**eval_result_dict)
    assert recreated_eval == eval_result
    assert recreated_eval.to_dict() == eval_result.to_dict()
    assert recreated_eval.eval_obj == eval_result.eval_obj
    assert recreated_eval.candidate_obj
    assert recreated_eval.check_results == eval_result.check_results

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

    def mock_llm(_):  # noqa
        return CandidateResponse(response=response)

    eval_result = eval_obj(mock_llm)
    # we need to strip the code blocks of leading/trailing whitespace to compare them
    expected_config = deepcopy(config)
    expected_config['checks'][-1]['code_tests'] = [
        dedent(x.strip()) for x in
        expected_config['checks'][-1]['code_tests']
    ]
    assert eval_result.eval_obj.to_dict() == expected_config
    assert Eval(**eval_obj.to_dict()) == eval_obj
    # i need to compare strings because underlying error objects (i.e. instances) will not be same
    assert str(EvalResult(**eval_result.to_dict()).to_dict()) == str(eval_result.to_dict())

    assert eval_result.response == response
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

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__Eval__candidate_from_dict(fake_eval_sum_two_numbers: dict, openai_candidate_template: dict):  # noqa: E501
    eval_config = fake_eval_sum_two_numbers.copy()
    eval_obj = Eval(**eval_config)
    result = eval_obj(openai_candidate_template)
    assert result.eval_obj == eval_obj
    assert result.candidate_obj == Candidate.from_dict(openai_candidate_template)
    assert result.candidate_obj.to_dict() == openai_candidate_template
    assert result.response
    assert 'sum_two_numbers' in result.response
    assert len(result.check_results) == 2
    assert result.check_results[0].success
    assert result.check_results[0].metadata['check_type'] == CheckType.CONTAINS.name
    assert result.check_results[1].success
    assert result.check_results[1].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name  # noqa: E501
    assert result.check_results[1].metadata['num_code_blocks'] >= 1
    expected_num_code_blocks = len(result.check_results[1].metadata['code_blocks'])
    assert result.check_results[1].metadata['num_code_blocks'] == expected_num_code_blocks
    assert result.num_checks == 2
    assert result.num_successful_checks == 2
    assert result.perc_successful_checks == 1
    assert EvalResult(**result.to_dict()) == result
    assert EvalResult(**result.to_dict()).to_dict() == result.to_dict()
    assert eval_config == fake_eval_sum_two_numbers

def callback(x: EvalResult) -> None:
    """
    Test the callback function by saving the result to a yaml file in the 'test/temp'
    directory. Assume directory already exists.
    """
    candidate_id = x.candidate_obj.metadata['uuid']
    eval_id = x.eval_obj.metadata['uuid']
    with open(f'tests/__temp__/result-{candidate_id}-{eval_id}.yaml', 'w') as f:
        yaml.dump(x.to_dict(), f, default_flow_style=False, sort_keys=False)

@pytest.mark.parametrize("candidate_type", ["AsyncMockCandidate", "MockCandidate"])
@pytest.mark.parametrize("num_cpus", [-1, 1])
@pytest.mark.parametrize("async_batch_size", [1, 50])
def test__async__EvalHarness__multiple_candidates__multiple_evals(
        candidate_type: str,
        num_cpus: int,
        async_batch_size: int,
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
        'candidate_type': candidate_type,
        'responses': responses_lookup,
    }
    candidate_2_dict = deepcopy(candidate_1_dict)
    candidate_2_dict['metadata']['uuid'] = 'candidate_2'

    eval_harness_via_dicts = EvalHarness(
        evals=[subtract_config, sum_config],
        candidates=[candidate_1_dict, candidate_2_dict],
        num_cpus=num_cpus,
        async_batch_size=async_batch_size,
    )
    eval_harness_via_objects = EvalHarness(
        evals=[Eval(**subtract_config), Eval(**sum_config)],
        candidates=[Candidate.from_dict(candidate_1_dict), Candidate.from_dict(candidate_2_dict)],
        num_cpus=num_cpus,
        async_batch_size=async_batch_size,
    )
    assert eval_harness_via_dicts.evals == eval_harness_via_objects.evals
    assert eval_harness_via_dicts.candidates == eval_harness_via_objects.candidates
    assert all(e1 is not e2 for e1, e2 in zip(eval_harness_via_dicts.evals, eval_harness_via_objects.evals))  # noqa
    assert all(c1 is not c2 for c1, c2 in zip(eval_harness_via_dicts.candidates, eval_harness_via_objects.candidates))  # noqa

    eval_harness_via_dicts_via_add = EvalHarness(
        num_cpus=num_cpus,
        async_batch_size=async_batch_size,
    )
    eval_harness_via_dicts_via_add.add_evals(subtract_config)
    eval_harness_via_dicts_via_add.add_evals(sum_config)
    eval_harness_via_dicts_via_add.add_candidates(candidate_1_dict)
    eval_harness_via_dicts_via_add.add_candidates(candidate_2_dict)
    assert eval_harness_via_dicts.evals == eval_harness_via_dicts_via_add.evals
    assert eval_harness_via_dicts.candidates == eval_harness_via_dicts_via_add.candidates
    eval_harness_via_dicts_via_add = EvalHarness(
        num_cpus=num_cpus,
        async_batch_size=async_batch_size,
    )
    eval_harness_via_dicts_via_add.add_evals([subtract_config, sum_config])
    eval_harness_via_dicts_via_add.add_candidates([candidate_1_dict, candidate_2_dict])
    assert eval_harness_via_dicts.evals == eval_harness_via_dicts_via_add.evals

    eval_harness = EvalHarness(
        num_cpus=num_cpus,
        async_batch_size=async_batch_size,
    )
    assert len(eval_harness.evals) == 0
    assert len(eval_harness.candidates) == 0
    eval_harness.add_evals(Eval(**subtract_config))
    eval_harness.add_evals(Eval(**sum_config))
    eval_harness.add_candidates(Candidate.from_dict(candidate_1_dict))
    eval_harness.add_candidates(Candidate.from_dict(candidate_2_dict))
    assert eval_harness.evals == eval_harness_via_dicts.evals
    assert eval_harness.candidates == eval_harness_via_dicts.candidates

    results = eval_harness()
    assert len(results) == len(eval_harness.candidates)
    assert len(results[0]) == len(eval_harness.evals)
    assert len(results[1]) == len(eval_harness.evals)
    assert results[0][0].candidate_obj == results[0][1].candidate_obj
    assert results[1][0].candidate_obj == results[1][1].candidate_obj
    assert results[0][0].candidate_obj != results[1][0].candidate_obj

    # The first list should contain the results for candidate 1 (subtract eval, sum eval)
    assert results[0][0].eval_obj == Eval(**subtract_config)
    assert results[0][0].candidate_obj == Candidate.from_dict(candidate_1_dict)
    assert results[0][1].eval_obj == Eval(**sum_config)
    assert results[0][1].candidate_obj == Candidate.from_dict(candidate_1_dict)
    # The second list should contain the results for candidate 2 (subtract eval, sum eval)
    assert results[1][0].eval_obj == Eval(**subtract_config)
    assert results[1][0].candidate_obj == Candidate.from_dict(candidate_2_dict)
    assert results[1][1].eval_obj == Eval(**sum_config)
    assert results[1][1].candidate_obj == Candidate.from_dict(candidate_2_dict)

    # eval objects across candidates should have same values (same eval) but different objects
    assert results[0][0].eval_obj == results[1][0].eval_obj
    assert results[0][0].eval_obj is not results[1][0].eval_obj
    assert results[0][1].eval_obj == results[1][1].eval_obj
    assert results[0][1].eval_obj is not results[1][1].eval_obj

    # candidate 1 - subtract eval
    cand_1_results_subtract = results[0][0]
    assert cand_1_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_1_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 1 - sum eval
    cand_1_results_sum = results[0][1]
    assert cand_1_results_sum.response == response_sum
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1
    assert cand_1_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 2 - subtract eval
    cand_2_results_subtract = results[1][0]
    assert cand_2_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_2_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 2 - sum eval
    cand_2_results_sum = results[1][1]
    assert cand_2_results_sum.response == response_sum
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1
    assert cand_2_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    # the eval results of candidate 1 should be the same as the eval results of candidate 2,
    # except the seconds it took to run the evals and the uuid of the candidate
    cand_1_results_subtract_dict = deepcopy(cand_1_results_subtract.to_dict())
    del cand_1_results_subtract_dict['timestamp']
    del cand_1_results_subtract_dict['total_time_seconds']
    del cand_1_results_subtract_dict['candidate_obj']['metadata']['uuid']
    cand_2_results_subtract_dict = deepcopy(cand_2_results_subtract.to_dict())
    del cand_2_results_subtract_dict['timestamp']
    del cand_2_results_subtract_dict['total_time_seconds']
    del cand_2_results_subtract_dict['candidate_obj']['metadata']['uuid']
    assert cand_1_results_subtract_dict == cand_2_results_subtract_dict

    cand_1_results_subtract.to_yaml('__temp__.yaml')
    result_from_yaml = cand_1_results_subtract.from_yaml('__temp__.yaml')
    assert result_from_yaml == cand_1_results_subtract
    assert result_from_yaml.to_dict() == cand_1_results_subtract.to_dict()
    os.remove('__temp__.yaml')

    cand_1_results_subtract.to_json('__temp__.json')
    result_from_json = cand_1_results_subtract.from_json('__temp__.json')
    assert result_from_json == cand_1_results_subtract
    assert result_from_json.to_dict() == cand_1_results_subtract.to_dict()
    os.remove('__temp__.json')

    cand_1_results_sum.to_yaml('__temp__.yaml')
    result_from_yaml = cand_1_results_sum.from_yaml('__temp__.yaml')
    assert result_from_yaml == cand_1_results_sum
    assert result_from_yaml.to_dict() == cand_1_results_sum.to_dict()
    os.remove('__temp__.yaml')

    cand_1_results_sum.to_json('__temp__.json')
    result_from_json = cand_1_results_sum.from_json('__temp__.json')
    assert result_from_json == cand_1_results_sum
    assert result_from_json.to_dict() == cand_1_results_sum.to_dict()
    os.remove('__temp__.json')

    assert subtract_config == fake_eval_subtract_two_numbers  # ensure eval_config wasn't modified
    assert sum_config == fake_eval_sum_two_numbers  # ensure eval_config wasn't modified

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
        num_cpus=1,
        async_batch_size=1,
    )
    eval_harness_via_objects = EvalHarness(
        evals=[Eval(**subtract_config), Eval(**sum_config)],
        candidates=[Candidate.from_dict(candidate_1_dict), Candidate.from_dict(candidate_2_dict)],
        num_cpus=1,
        async_batch_size=1,
    )
    assert eval_harness_via_dicts.evals == eval_harness_via_objects.evals
    assert eval_harness_via_dicts.candidates == eval_harness_via_objects.candidates

    num_samples = 3
    eval_harness = EvalHarness(
        num_cpus=2,
        async_batch_size=2,
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
    assert len(results[0]) == num_evals * num_samples
    assert len(results[1]) == num_evals * num_samples
    # The underlying candidate objects should have the same values but should be different objects
    # because each candidate object (against a specific eval) is responsible for storing its own
    # history/conversation and the history should be different for each eval.
    assert results[0][0].candidate_obj == results[0][1].candidate_obj
    assert results[1][0].candidate_obj == results[1][1].candidate_obj

    # The first list should contain the results for candidate 1 (subtract eval * 3, sum eval * 3)
    # 3 SAMPLES OF SUBTRACT EVAL
    assert results[0][0].eval_obj == Eval(**subtract_config)
    assert results[0][0].candidate_obj == Candidate.from_dict(candidate_1_dict)
    assert results[0][1].eval_obj == Eval(**subtract_config)
    assert results[0][1].candidate_obj == Candidate.from_dict(candidate_1_dict)
    assert results[0][2].eval_obj == Eval(**subtract_config)
    assert results[0][2].candidate_obj == Candidate.from_dict(candidate_1_dict)
    # 3 SAMPLES OF SUM EVAL
    assert results[0][3].eval_obj == Eval(**sum_config)
    assert results[0][3].candidate_obj == Candidate.from_dict(candidate_1_dict)
    assert results[0][4].eval_obj == Eval(**sum_config)
    assert results[0][4].candidate_obj == Candidate.from_dict(candidate_1_dict)
    assert results[0][5].eval_obj == Eval(**sum_config)
    assert results[0][5].candidate_obj == Candidate.from_dict(candidate_1_dict)

    # The second list should contain the results for candidate 2 (subtract eval, sum eval)
    # 3 SAMPLES OF SUBTRACT EVAL
    assert results[1][0].eval_obj == Eval(**subtract_config)
    assert results[1][0].candidate_obj == Candidate.from_dict(candidate_2_dict)
    assert results[1][1].eval_obj == Eval(**subtract_config)
    assert results[1][1].candidate_obj == Candidate.from_dict(candidate_2_dict)
    assert results[1][2].eval_obj == Eval(**subtract_config)
    assert results[1][2].candidate_obj == Candidate.from_dict(candidate_2_dict)
    # 3 SAMPLES OF SUM EVAL
    assert results[1][3].eval_obj == Eval(**sum_config)
    assert results[1][3].candidate_obj == Candidate.from_dict(candidate_2_dict)
    assert results[1][4].eval_obj == Eval(**sum_config)
    assert results[1][4].candidate_obj == Candidate.from_dict(candidate_2_dict)
    assert results[1][5].eval_obj == Eval(**sum_config)
    assert results[1][5].candidate_obj == Candidate.from_dict(candidate_2_dict)

    # eval objects across candidates should have same values (same eval) but different objects
    assert results[0][0].eval_obj == results[1][0].eval_obj
    assert results[0][0].eval_obj is not results[1][0].eval_obj
    assert results[0][0].eval_obj == results[0][1].eval_obj
    assert results[0][0].eval_obj is not results[0][1].eval_obj
    assert results[0][0].eval_obj == results[0][2].eval_obj
    assert results[0][0].eval_obj is not results[0][2].eval_obj

    assert results[0][3].eval_obj == results[1][3].eval_obj
    assert results[0][3].eval_obj is not results[1][3].eval_obj
    assert results[0][3].eval_obj == results[0][4].eval_obj
    assert results[0][3].eval_obj is not results[0][4].eval_obj
    assert results[0][3].eval_obj == results[0][5].eval_obj
    assert results[0][3].eval_obj is not results[0][5].eval_obj

    # candidate 1 - subtract eval; all 3 should have same results
    cand_1_results_subtract = results[0][0]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_1_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    cand_1_results_subtract = results[0][1]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_1_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1


    cand_1_results_subtract = results[0][2]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.response == response_subtract
    assert cand_1_results_subtract.num_checks == 3
    assert cand_1_results_subtract.num_successful_checks == 2
    assert cand_1_results_subtract.perc_successful_checks == 2 / 3
    assert cand_1_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 1 - sum eval; all 3 should have same results
    cand_1_results_sum = results[0][3]
    assert cand_1_results_sum.response == response_sum
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1
    assert cand_1_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    cand_1_results_sum = results[0][4]
    assert cand_1_results_sum.response == response_sum
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1
    assert cand_1_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    cand_1_results_sum = results[0][5]
    assert cand_1_results_sum.response == response_sum
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1
    assert cand_1_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 2 - subtract eval; all 3 should have same results
    cand_2_results_subtract = results[1][0]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.response == response_subtract
    assert cand_2_results_subtract.num_checks == 3
    assert cand_2_results_subtract.num_successful_checks == 2
    assert cand_2_results_subtract.perc_successful_checks == 2 / 3
    assert cand_2_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    cand_2_results_subtract = results[1][1]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.response == response_subtract
    assert cand_2_results_subtract.num_checks == 3
    assert cand_2_results_subtract.num_successful_checks == 2
    assert cand_2_results_subtract.perc_successful_checks == 2 / 3
    assert cand_2_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    cand_2_results_subtract = results[1][2]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.response == response_subtract
    assert cand_2_results_subtract.num_checks == 3
    assert cand_2_results_subtract.num_successful_checks == 2
    assert cand_2_results_subtract.perc_successful_checks == 2 / 3
    assert cand_2_results_subtract.check_results[-1].metadata['num_code_blocks'] == 1

    # candidate 2 - sum eval; all 3 should have same results
    cand_2_results_sum = results[1][3]
    assert cand_2_results_sum.response == response_sum
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1
    assert cand_2_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    cand_2_results_sum = results[1][4]
    assert cand_2_results_sum.response == response_sum
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1
    assert cand_2_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

    cand_2_results_sum = results[1][5]
    assert cand_2_results_sum.response == response_sum
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1
    assert cand_2_results_sum.check_results[-1].metadata['num_code_blocks'] == 1

def test__Eval_with_numeric_values_loads_correctly(fake_eval_non_string_values: dict):
    """Test that numeric values are converted to strings when loading an Eval object."""
    eval_config = deepcopy(fake_eval_non_string_values)
    eval_obj = Eval(**eval_config)
    assert Eval(**eval_obj.to_dict()) == eval_obj
    assert eval_obj.metadata['version'] == 1
    assert eval_obj.metadata['tags'][0] == 1
    assert isinstance(eval_obj.input, list)
    assert isinstance(eval_obj.input[0], dict)
    assert 'role' in eval_obj.input[0]
    assert 'content' in eval_obj.input[0]
    assert eval_obj.input[0]['content'] == 6

class ErrorCallbackHandler:
    """
    ErrorCallbackHandler is responsible for managing a shared list of errors (or other data)
    across multiple processes. It provides a callback function that can be passed to different
    processes and safely appends items to the shared list.

    This class is designed to work in a multiprocessing environment where the callback function
    needs to be pickled and passed to different processes. By encapsulating the shared list
    and the callback function in a class, we ensure that the callback function can be pickled
    without issues.
    """

    def __init__(self, shared_list):  # noqa
        """
        Initializes the ErrorCallbackHandler with a shared list.

        Args:
            shared_list (multiprocessing.Manager().list): A list managed by multiprocessing.Manager
            that can be shared across multiple processes.
        """
        self.shared_list = shared_list

    def callback(self, exception: Exception, eval_obj: Eval, candidate_obj: Candidate):  # noqa
        self.shared_list.append((exception, eval_obj, candidate_obj))

error_callback_global_list = multiprocessing.Manager().list()
multi_processing_error_handler = ErrorCallbackHandler(error_callback_global_list)
multi_processing_error_callback = multi_processing_error_handler.callback

@pytest.mark.parametrize('num_cpus', [1, None])
def test__EvalHarness__candidate_has_error_generating_response_multi_processing(
        num_cpus: int | None,
        fake_eval_sum_two_numbers_code_blocks_run: dict,
    ):
    """
    Tests that the EvalHarness captures errors generated by the candidate. If no error_callback
    is set, the harness should raise the error and stop processing the evals. If an error_callback
    is set, the harness should call the error_callback and the candidate object and continue
    processing the remaining evals.
    """
    eval_config = deepcopy(fake_eval_sum_two_numbers_code_blocks_run)
    prompt = eval_config['input'][0]['content']
    response = '```\ndef sum_two_numbers(a, b): return a+b\n```'

    # Create mock candidates 1 and 2; candidate 1 will raise an error when generating a response
    # candidate 2 will generate the correct response
    # Test that both mock candidates give the expected results before running the harness
    candidate_1 = MockCandidate(
        metadata={'name': 'candidate_1'},
        responses={
            prompt: ValueError('Fake Rate Limit Error for prompt_1'),
        },
    )
    with pytest.raises(ValueError, match='Fake Rate Limit Error for prompt_1'):
        candidate_1(eval_config['input'])

    candidate_2 = MockCandidate(
        metadata={'name': 'candidate_2'},
        responses={
            prompt: response,
        },
    )
    assert candidate_2(eval_config['input']).response == response

    eval_1 = Eval(**eval_config)
    eval_2 = Eval(**eval_config)
    harness = EvalHarness(
        evals=[eval_1, eval_2],
        candidates=[candidate_1, candidate_2],
        num_cpus=num_cpus,
    )
    # this should raise an error because the candidate has an error generating a response
    # and we have not set the error_callaback to capture/ignore errors
    with pytest.raises(ResponseError) as error:  # noqa: PT012
        _ = harness()
        error = error.value  # noqa
        assert isinstance(error.exception, ValueError)
        assert error.exception.args[0] == 'Fake Rate Limit Error'
        assert error.eval_obj == eval_1
        assert error.candidate_obj == candidate_1

    if num_cpus == 1:
        # NOTE: not happy with this solution but it works for now
        test_harness_callback_errors = []
        def local_error_callback(exception: Exception, eval_obj: Eval, candidate_obj: Candidate) -> None:  # noqa: E501
            test_harness_callback_errors.append((exception, eval_obj, candidate_obj))
        harness.error_callback = local_error_callback
    else:
        harness.error_callback = multi_processing_error_callback

    results = harness()
    assert len(results) == 2
    if num_cpus == 1:
        errors = test_harness_callback_errors
    else:
        errors = list(multi_processing_error_handler.shared_list)
    # both evals for candidate 1 should have an error
    assert len(errors) == 2
    # candidate 1 will raise an error for both evals
    assert errors[0][0].args[0] == results[0][0].response_metadata['harness_exception'].args[0]
    assert errors[1][0].args[0] == results[0][1].response_metadata['harness_exception'].args[0]

    if num_cpus != 1:
        # if num_cpus is not 1, the order of the errors is not guaranteed
        # sort errors by candidate name so we can compare them
        errors = sorted(errors, key=lambda x: x[2].metadata['name'])
    # first item in tuple is the exception, second is the eval, third is the candidate
    assert errors[0][0].args[0] == 'Fake Rate Limit Error for prompt_1'
    assert errors[0][0].args[0] == results[0][0].response_metadata['harness_exception'].args[0]
    assert errors[0][1] == eval_1
    assert errors[0][2].metadata['name'] == candidate_1.metadata['name']
    assert errors[1][0].args[0] == 'Fake Rate Limit Error for prompt_1'
    assert errors[1][0].args[0] == results[0][1].response_metadata['harness_exception'].args[0]
    assert errors[1][1] == eval_2
    assert errors[1][2].metadata['name'] == candidate_1.metadata['name']

    # test that the CheckResult objects have the correct values (should be failing)
    # in the first two evals, the prompt should fail
    expected_num_checks = len(eval_config['checks'])
    expected_num_code_tests = len(eval_config['checks'][-1]['code_tests'])
    for i in range(2):
        assert not any(x.success for x in results[0][i].check_results)
        assert results[0][i].num_checks == expected_num_checks
        assert results[0][i].num_successful_checks == 0
        # no code block were generated because the first prompt failed
        assert results[0][i].check_results[-1].metadata['num_code_blocks'] == 0
        assert results[0][i].check_results[-1].metadata['num_code_blocks_successful'] == 0
        assert results[0][i].check_results[-1].metadata['num_code_tests'] == expected_num_code_tests  # noqa
        assert results[0][i].check_results[-1].metadata['num_code_tests_successful'] == 0

    # in the second two evals, the prompt should pass
    for i in range(2):
        assert results[1][i].check_results[0].success  # checks for sum_two_numbers
        assert not results[1][i].check_results[1].success
        assert not results[1][i].check_results[2].success
        assert results[1][i].check_results[3].success
        assert not results[1][i].check_results[4].success

        assert results[1][i].num_checks == expected_num_checks
        assert results[1][i].num_successful_checks == 2
        # code block were generated
        assert results[1][i].check_results[-1].metadata['num_code_blocks'] == 1
        assert results[1][i].check_results[-1].metadata['num_code_blocks_successful'] == 1
        assert results[1][i].check_results[-1].metadata['num_code_tests'] == expected_num_code_tests  # noqa: E501
        assert results[1][i].check_results[-1].metadata['num_code_tests_successful'] > 0

def test__MultiProcessing_openai_candidates(
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
        num_cpus=2,
    )
    results = harness()
    assert len(results) == 2
    assert len(results[0]) == 2
    assert len(results[1]) == 2
    assert results[0][0].response_metadata['total_tokens'] > 0
    assert results[0][1].response_metadata['total_tokens'] > 0
    assert results[1][0].response_metadata['total_tokens'] > 0
    assert results[1][1].response_metadata['total_tokens'] > 0

class UnregisteredCheckResult(CheckResult):  # noqa
    pass

class UnregisteredCheck(Check):  # noqa
    def _call(self, value: str) -> UnregisteredCheckResult:
        return UnregisteredCheckResult(
            success=value is not None,
            value=value,
            metadata={},
        )

    def clone(self) -> Check:  # noqa
        return UnregisteredCheck()

class UnregisteredCandidate(Candidate):  # noqa
    def __init__(self, response: object) -> None:
        super().__init__()
        self.response = response

    def __call__(self, prompt: dict) -> dict:  # noqa
        # returns dictionary instead of string
        return CandidateResponse(response={'prompt': prompt, 'response': self.response})

def test__Eval__unregistered_check__unregistered_candidate__non_string_prompt_and_response():
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
    assert UnregisteredCandidate(42).to_dict() == {'candidate_type': 'UnregisteredCandidate'}
    result = eval_(UnregisteredCandidate(42))
    assert result.response == {'prompt': {'prompt': 'Test Prompt'}, 'response': 42}
    assert len(result.check_results) == 1
    check_result = result.check_results[0]
    assert check_result.value == {'prompt': {'prompt': 'Test Prompt'}, 'response': 42}
    assert check_result.success is True
    assert check_result.to_dict() == {'value': {'prompt': {'prompt': 'Test Prompt'}, 'response': 42}, 'success': True, 'result_type': 'UnregisteredCheckResult'}  # noqa
    assert result.num_checks == 1
    assert result.num_successful_checks == 1
    assert result.perc_successful_checks == 1
    # ensure that we can convert the results (which contain unregistered checks/candidates) to
    # a string and dictionary (which call underlying str and to_dict methods on
    # checks/candidates)
    assert result.to_dict()['eval_obj'] == eval_.to_dict()
    assert result.to_dict()['candidate_obj'] == UnregisteredCandidate(42).to_dict()
    assert result.to_dict()['check_results'][0] == check_result.to_dict()

def test__EvalHarness__unregistered_check__unregistered_candidate__non_string_prompt_and_response():  # noqa: E501
        harness = EvalHarness(
            # num_cpus=1, async_batch_size=1,
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
        assert len(results[0]) == 2  # 2 evals
        assert len(results[1]) == 2  # same 2 evals
        assert results[0][0].response == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 1'}  # noqa
        assert results[0][1].response == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 1'}  # noqa
        assert results[1][0].response == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 2'}  # noqa
        assert results[1][1].response == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 2'}  # noqa
        assert len(results[0][0].check_results) == 1
        assert results[0][0].perc_successful_checks == 1
        assert results[0][0].check_results[0].value == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 1'}  # noqa
        assert results[0][0].check_results[0].success is True
        assert len(results[0][1].check_results) == 1
        assert results[0][1].perc_successful_checks == 1
        assert results[0][1].check_results[0].value == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 1'}  # noqa
        assert results[0][1].check_results[0].success is True
        assert len(results[1][0].check_results) == 1
        assert results[1][0].perc_successful_checks == 1
        assert results[1][0].check_results[0].value == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 2'}  # noqa
        assert results[1][0].check_results[0].success is True
        assert len(results[1][1].check_results) == 1
        assert results[1][1].perc_successful_checks == 1
        assert results[1][1].check_results[0].value == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 2'}  # noqa
        assert results[1][1].check_results[0].success is True

        # ensure that we can convert the results (which contain unregistered checks/candidates) to
        # a string and dictionary (which call underlying str and to_dict methods on
        # checks/candidates)
        assert len(str(results[0][0])) > 10
        assert results[0][0].to_dict()['eval_obj'] == harness.evals[0].to_dict()
        assert results[0][0].to_dict()['candidate_obj'] == harness.candidates[0].to_dict()
        assert results[0][0].to_dict()['check_results'][0] == results[0][0].check_results[0].to_dict()  # noqa
        assert results[0][1].to_dict()['eval_obj'] == harness.evals[1].to_dict()
        assert results[0][1].to_dict()['candidate_obj'] == harness.candidates[0].to_dict()
        assert results[0][1].to_dict()['check_results'][0] == results[0][1].check_results[0].to_dict()  # noqa
        assert results[1][0].to_dict()['eval_obj'] == harness.evals[0].to_dict()
        assert results[1][0].to_dict()['candidate_obj'] == harness.candidates[1].to_dict()
        assert results[1][0].to_dict()['check_results'][0] == results[1][0].check_results[0].to_dict()  # noqa
        assert results[1][1].to_dict()['eval_obj'] == harness.evals[1].to_dict()
        assert results[1][1].to_dict()['candidate_obj'] == harness.candidates[1].to_dict()
        assert results[1][1].to_dict()['check_results'][0] == results[1][1].check_results[0].to_dict()  # noqa

def test__Eval__callable_check__callable_candidate__non_string_prompt_and_response():
    """
    We should be able to use callable Checks and Candidates (e.g. functions) with non-string
    prompts and responses. Lambdas can't be pickled, so we can't use them with EvalHarness (with
    multi-processing), but we can use them with Evals individually.
    """
    eval_ = Eval(
        input={'prompt': 'Test Prompt'},  # non-string prompt
        checks=[
            lambda data: 'Response' in data.response['response'],  # should pass
            lambda data: 'does not exist' in data.response['response'],  # should fail
        ],
    )
    assert 'input' in eval_.to_dict()
    assert eval_.to_dict()['input'] == {'prompt': 'Test Prompt'}
    assert len(eval_.to_dict()['checks']) == 2

    # return dictionary instead of string
    result = eval_(lambda prompt: CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response'}))  # noqa
    assert result.response == {'prompt': 'Test Prompt', 'response': 'Test Prompt & Response'}
    assert len(result.check_results) == 2
    check_result_1 = result.check_results[0]
    assert check_result_1.value is True
    assert check_result_1.success is True
    assert check_result_1.to_dict() == {'value': True, 'success': True, 'result_type': 'PASS_FAIL'}
    check_result_2 = result.check_results[1]
    assert check_result_2.value is False
    assert check_result_2.success is False
    assert check_result_2.to_dict() == {'value': False, 'success': False, 'result_type': 'PASS_FAIL'}  # noqa

    assert len(result.check_results) == 2
    assert result.perc_successful_checks == 0.5
    assert result.check_results[0].value == check_result_1.value
    assert result.check_results[0].success == check_result_1.success
    assert result.check_results[1].value == check_result_2.value
    assert result.check_results[1].success == check_result_2.success

    assert result.num_checks == 2
    assert result.num_successful_checks == 1
    assert result.perc_successful_checks == 0.5

    assert result.to_dict()['eval_obj'] == eval_.to_dict()
    assert result.to_dict()['candidate_obj']
    assert result.to_dict()['check_results'][0] == check_result_1.to_dict()
    assert result.to_dict()['check_results'][1] == check_result_2.to_dict()

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
    result = eval_(candidate)
    result_dict = result.to_dict()
    assert result_dict['eval_obj'] == eval_.to_dict()
    assert result_dict['candidate_obj'] == candidate.to_dict()
    assert result_dict['candidate_obj']['metadata'] == {'some_metadata': 'some_value'}
    assert 'candidate_type' in result_dict['candidate_obj']
    assert result_dict['candidate_obj']['candidate_type'] == 'MyUnregisteredCandidate'
    assert result_dict['response'] == {'input': 'This is the input', 'response': 'This is the response foobar'}  # noqa: E501
    assert result_dict['response_metadata'] == {'custom_state': 'barfoo'}

    assert len(result_dict['check_results']) == 1
    assert result_dict['check_results'][0]['result_type'] == 'PASS_FAIL'
    assert result_dict['check_results'][0]['value'] is True
    assert result_dict['check_results'][0]['success'] is True

    result_loaded = EvalResult(**result_dict)
    assert result_loaded.to_dict() == result_dict

@pytest.mark.parametrize('use_async', [True, False])
def test__EvalHarness__callable_check__callable_candidate__non_string_prompt_and_response(use_async: bool):  # noqa: E501
    if use_async:
        async def async_candidate_1(prompt):  # noqa
            return CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response1'})  # noqa

        async def async_candidate_2(prompt):  # noqa
            return CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response2'})  # noqa

        candidates = [async_candidate_1, async_candidate_2]
    else:
        candidates = [
            lambda prompt: CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response1'}),  # noqa
            lambda prompt: CandidateResponse(response=prompt | {'response': prompt['prompt'] + ' & Response2'}),  # noqa
        ]

    harness = EvalHarness(
        num_cpus=1, async_batch_size=1,
        evals=[
            Eval(
                input={'prompt': 'Test Prompt 1'},  # non-string prompt
                checks=[lambda data: 'Response1' in data.response['response']],
            ),
            Eval(
                input={'prompt': 'Test Prompt 2'},  # non-string prompt
                checks=[lambda data: 'Response2' in data.response['response']],
            ),
        ],
        candidates = candidates,
    )
    num_samples = 100
    assert len(harness.evals) == 2
    assert len(harness.candidates) == 2
    results = harness(num_samples=num_samples)
    assert len(results) == 2  # 2 candidates
    assert len(results[0]) == 2 * num_samples  # 2 evals
    assert len(results[1]) == 2 * num_samples # same 2 evals
    assert results[0][0].response == {'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response1'}  # noqa
    assert results[0][num_samples-1].response == {'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response1'}  # noqa
    assert results[0][num_samples].response == {'prompt': 'Test Prompt 2', 'response': 'Test Prompt 2 & Response1'}  # noqa
    assert results[1][0].response == {'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response2'}  # noqa
    assert results[1][num_samples].response == {'prompt': 'Test Prompt 2', 'response': 'Test Prompt 2 & Response2'}  # noqa
    # eval 1 candidate 1
    assert len(results[0][0].check_results) == 1
    assert results[0][0].num_checks == 1
    assert results[0][0].num_successful_checks == 1
    assert results[0][0].perc_successful_checks == 1
    assert results[0][0].check_results[0].value is True
    assert results[0][0].check_results[0].success is True
    # eval 2 candidate 1
    assert len(results[0][num_samples].check_results) == 1
    assert results[0][num_samples].num_checks == 1
    assert results[0][num_samples].num_successful_checks == 0
    assert results[0][num_samples].perc_successful_checks == 0
    assert results[0][num_samples].check_results[0].value is False
    assert results[0][num_samples].check_results[0].success is False
    # eval 1 candidate 2
    assert len(results[1][0].check_results) == 1
    assert results[1][0].num_checks == 1
    assert results[1][0].num_successful_checks == 0
    assert results[1][0].perc_successful_checks == 0
    assert results[1][0].check_results[0].value is False
    assert results[1][0].check_results[0].success is False
    # eval 2 candidate 2
    assert len(results[1][1].check_results) == 1
    assert results[1][num_samples].num_checks == 1
    assert results[1][num_samples].num_successful_checks == 1
    assert results[1][num_samples].perc_successful_checks == 1
    assert results[1][num_samples].check_results[0].value is True
    assert results[1][num_samples].check_results[0].success is True

    # ensure that we can convert the results (which contain unregistered checks/candidates) to
    # a string and dictionary (which call underlying str and to_dict methods on
    # checks/candidates)
    assert len(str(results[0][0])) > 10
    assert results[0][0].to_dict()['eval_obj'] == harness.evals[0].to_dict()
    assert results[0][0].to_dict()['candidate_obj']
    assert results[0][0].to_dict()['check_results'][0] == results[0][0].check_results[0].to_dict()
    assert results[0][num_samples].to_dict()['eval_obj'] == harness.evals[1].to_dict()
    assert results[0][num_samples].to_dict()['candidate_obj']
    assert results[0][num_samples].to_dict()['check_results'][0] == results[0][num_samples].check_results[0].to_dict()  # noqa: E501
    assert results[1][0].to_dict()['eval_obj'] == harness.evals[0].to_dict()
    assert results[1][0].to_dict()['candidate_obj']
    assert results[1][0].to_dict()['check_results'][0] == results[1][0].check_results[0].to_dict()
    assert results[1][num_samples].to_dict()['eval_obj'] == harness.evals[1].to_dict()
    assert results[1][num_samples].to_dict()['candidate_obj']
    assert results[1][num_samples].to_dict()['check_results'][0] == results[1][num_samples].check_results[0].to_dict()  # noqa: E501

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
    result = eval_(candidate)
    tool_response = result.response[0]
    assert tool_response['name'] == 'get_current_weather'
    assert 'location' in tool_response['arguments']
    assert tool_response['arguments']['location']
    assert isinstance(tool_response['arguments']['location'], str)
    assert 'unit' in tool_response['arguments']
    assert tool_response['arguments']['unit'] in ['celsius', 'fahrenheit']
    # check that it gets at least the function name correctly
    assert result.check_results[0].value >= 0.5
