"""Tests for the evals module."""
import glob
import json
import tempfile
from time import perf_counter
import time
from pydantic import BaseModel
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
    LLMCheck,
    LambdaCheck,
    MatchCheck,
    PassFailResult,
    ScoreResult,
    ToolCallsCheck,
)
from llm_eval.delayed_semaphore import DelayedSemaphore
from llm_eval.eval import (
    CandidateRunResults,
    Eval,
    EvalHarness,
    EvalResult,
    Mode,
)
from llm_eval.internal_utilities import extract_code_blocks
from llm_eval.openai import user_message
from tests.conftest import (
    OPENAI_DEFAULT_MODEL,
    AsyncMockCandidateCausesError,
    MockCandidateCausesError,
    MockCheckCausesError,
    MockRetryTestCandidate,
    MockRetryTestCheck,
    UnregisteredCandidate,
    UnregisteredCheck,
)

class TestEval:
    """Test Eval object."""

    @pytest.mark.parametrize('input_', [[user_message('test1')], None])
    def test__Eval__creation(self, input_: object):
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

    def test__eval__clone(self, fake_eval_8f9fbf37: dict):
        config = deepcopy(fake_eval_8f9fbf37)
        eval_obj = Eval(**config)
        eval_copy = deepcopy(eval_obj)
        assert eval_obj == eval_copy
        assert eval_obj.to_dict() == eval_copy.to_dict()
        # test-sequence (i.e. PromptTest objects) should be the same prompt tests but different objects  # noqa: E501
        assert eval_obj.input == eval_copy.input
        assert eval_obj.input is not eval_copy.input
        assert eval_obj.ideal_response == eval_copy.ideal_response
        assert eval_obj.ideal_response is not eval_copy.ideal_response
        assert eval_obj.checks == eval_copy.checks
        assert all(c1 is not c2 for c1, c2 in zip(eval_obj.checks, eval_copy.checks))
        assert eval_obj.metadata == eval_copy.metadata
        assert eval_obj.metadata is not eval_copy.metadata

    def test__Eval__from_file(self):
        """
        For each of our fake evals, test that we can create an Eval object from the yaml file and
        that the object is the same as the one created from the dictionary.
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
    def test__Eval__call__result__to_from_dict(self, candidate, metadata):  # noqa: ANN001
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

    def test__Eval__from_objects__minimal(self):
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
    def test__Eval__example_8f9fbf37__callable_candidate(self, passes_input: bool, fake_eval_8f9fbf37: dict):  # noqa: E501
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
        assert eval_result_dict['check_results'] == [r.to_dict() for r in eval_result.check_results]  # noqa: E501
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

    def test__Eval__multiple_code_blocks__ensure_code_blocks_run(self, fake_eval_sum_two_numbers_code_blocks_run: dict):  # noqa: E501
        """
        Use Mock LLM with multiple code blocks (over multiple responses) to ensure code blocks run
        and the check results return the expected values.
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
        assert eval_result.check_results[-1].metadata['num_code_blocks'] == expected_num_code_blocks  # noqa: E501

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
        assert eval_result.check_results[-1].metadata['num_code_blocks'] == expected_num_code_blocks  # noqa: E501
        assert eval_result.check_results[-1].metadata['num_code_blocks_successful'] == expected_successful_code_blocks  # noqa
        assert eval_result.check_results[-1].metadata['code_blocks'] == expected_code_blocks
        assert eval_result.check_results[-1].metadata['code_block_errors'] == [None, None]
        # first function check should have run successfully, but second code block should have failed  # noqa: E501
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


class TestEvalHarness:
    """Test EvalHarness object."""

    def test__EvalHarness__initializating_with_different_types_gives_same_config(
            self,
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
        # Test that creating the EvalHarness object in different ways still results in the same config  # noqa: E501
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


    @pytest.mark.parametrize('candidate_type', ['AsyncMockCandidate', 'MockCandidate'])
    @pytest.mark.parametrize('num_samples', [1, 5])
    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test__EvalHarness(
            self,
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
        assert results[0].eval_results[num_samples].candidate == Candidate.from_dict(candidate_1_dict)  # noqa: E501

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
        assert results[1].eval_results[num_samples].candidate == Candidate.from_dict(candidate_2_dict)  # noqa: E501

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
        assert results[0].eval_results[num_samples].eval == results[1].eval_results[num_samples].eval  # noqa: E501
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

        assert subtract_config == fake_eval_subtract_two_numbers  # ensure eval_config wasn't modified  # noqa: E501
        assert sum_config == fake_eval_sum_two_numbers  # ensure eval_config wasn't modified

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test__EvalHarness__Check_raises_error(
            self,
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
            candidates=[UnregisteredCandidate(response=eval_obj.input), MockCandidateCausesError()],  # noqa: E501
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
            self,
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
            candidates=[Candidate.from_dict(candidate_1_dict), Candidate.from_dict(candidate_2_dict)],  # noqa: E501
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
        # The underlying candidate objects should have the same values but should be different objects  # noqa: E501
        # because each candidate object (against a specific eval) is responsible for storing its own  # noqa: E501
        # history/conversation and the history should be different for each eval.
        assert results[0].eval_results[0].candidate == results[0].eval_results[1].candidate
        assert results[1].eval_results[0].candidate == results[1].eval_results[1].candidate

        # The first list should contain the results for candidate 1 (subtract eval * 3, sum eval * 3)  # noqa: E501
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

    def test__Eval_with_numeric_values_loads_correctly(self, fake_eval_non_string_values: dict):
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
            self,
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

    def test__Eval__unregistered_check__non_string_prompt_and_response(self):
        """
        We should be able to use unregistered Check and Candidate classes with non-string prompts and
        responses. These classes won't be able to be saved/loaded from a dictionary, and so we can't
        use them with EvalHarness, but we should be able to use them individually.
        """  # noqa: E501
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
    def test__EvalHarness__unregistered_check__unregistered_candidate__non_string_prompt_and_response(  # noqa: E501
            self,
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

            # ensure that we can convert the results (which contain unregistered checks/candidates) to  # noqa: E501
            # a string and dictionary (which call underlying str and to_dict methods on
            # checks/candidates)
            assert len(str(results[0].eval_results[0])) > 10
            assert results[0].eval_results[0].to_dict()['eval'] == harness.evals[0].to_dict()
            assert results[0].eval_results[0].to_dict()['candidate'] == harness.candidates[0].to_dict()  # noqa: E501
            assert results[0].eval_results[0].to_dict()['check_results'][0] == results[0].eval_results[0].check_results[0].to_dict()  # noqa
            assert results[0].eval_results[1].to_dict()['eval'] == harness.evals[1].to_dict()
            assert results[0].eval_results[1].to_dict()['candidate'] == harness.candidates[0].to_dict()  # noqa: E501
            assert results[0].eval_results[1].to_dict()['check_results'][0] == results[0].eval_results[1].check_results[0].to_dict()  # noqa
            assert results[1].eval_results[0].to_dict()['eval'] == harness.evals[0].to_dict()
            assert results[1].eval_results[0].to_dict()['candidate'] == harness.candidates[1].to_dict()  # noqa: E501
            assert results[1].eval_results[0].to_dict()['check_results'][0] == results[1].eval_results[0].check_results[0].to_dict()  # noqa
            assert results[1].eval_results[1].to_dict()['eval'] == harness.evals[1].to_dict()
            assert results[1].eval_results[1].to_dict()['candidate'] == harness.candidates[1].to_dict()  # noqa: E501
            assert results[1].eval_results[1].to_dict()['check_results'][0] == results[1].eval_results[1].check_results[0].to_dict()  # noqa

    def test__EvalResult__loading_unregistered_candidate_from_dict(self):
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

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('use_async_candidate', [True, False])
    def test__EvalHarness__callable_check__callable_candidate__non_string_prompt_and_response(
            self,
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

    def test__OpenAIToolsCandidate__ToolsCallCheck(self, openai_tools_candidate_template: dict):
        """
        Integration test that tests Evaling a real OpenAITool API call against the
        ToolsCheck.
        """
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


class TestLogging:
    """Test logging (i.e. saving eval) logic for EvalHarness."""

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test__EvalHarness__log_directory(self, response_mode: Mode, eval_mode: Mode):
        """Test that EvalHarness correctly saves eval results to the specified log directory."""
        # Create a temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple eval with basic checks
            eval_1 = Eval(
                input="Test input 1",
                checks=[ContainsCheck(value="Test response 1 for candidate 1")],
            )
            eval_2 = Eval(
                input="Test input 2",
                checks=[ContainsCheck(value="Test response 2 for candidate 2")],
            )
            # Create simple candidates with unique identifiable responses
            candidate_1 = UnregisteredCandidate(response="Test response 1 for candidate 1", metadata={'uuid': 'candidate_1'})  # noqa: E501
            candidate_2 = UnregisteredCandidate(response="Test response 2 for candidate 2", metadata={'uuid': 'candidate_2'})  # noqa: E501
            # Create the harness with the temp directory as log_directory
            harness = EvalHarness(
                evals=[eval_1, eval_2],
                candidates=[candidate_1, candidate_2],
                log_directory=temp_dir,
                response_mode=response_mode,
                eval_mode=eval_mode,
            )
            # Run the harness
            results = harness()
            # Verify that results were generated
            assert len(results) == 2
            assert len(results[0].eval_results) == 2
            assert results[0].num_errors == 0
            # Check that files were created in the log directory
            log_files = sorted(glob.glob(os.path.join(temp_dir, "eval_result_*.json")))
            assert len(log_files) == 4  # 2 candidates * 2 evals

            # Sort results by candidate UUID to ensure consistent ordering
            sorted_results = sorted(results, key=lambda r: r.candidate.metadata['uuid'])

            # First read all log files into a list
            logged_results = []
            for log_file in log_files:
                with open(log_file) as f:
                    logged_results.append(json.load(f))

            # For each result (candidate), verify its log files
            for result in sorted_results:
                # Find log files for this candidate's results
                candidate_uuid = result.candidate.metadata['uuid']
                candidate_logs = [
                    log for log in logged_results
                    if log['candidate']['metadata']['uuid'] == candidate_uuid
                ]
                assert len(candidate_logs) == 2  # Each candidate should have 2 eval results

                # For each eval result for this candidate
                for eval_result in result.eval_results:
                    # Find the matching logged result for this eval
                    matching_log = next(
                        log for log in candidate_logs
                        if log['eval']['input'] == eval_result.eval.input
                    )
                    # Verify the logged result matches the actual result
                    actual_result = eval_result.to_dict()
                    assert matching_log['eval'] == actual_result['eval']
                    assert matching_log['response'] == actual_result['response']
                    assert matching_log['metadata'] == actual_result['metadata']
                    assert matching_log['check_results'] == actual_result['check_results']
                    assert matching_log['timestamp'] == actual_result['timestamp']
                    assert matching_log['candidate'] == actual_result['candidate']

            # Verify the log directory is cleaned up when the context manager exits
            assert os.path.exists(temp_dir)
        assert not os.path.exists(temp_dir)  # Directory should be cleaned up after manager exits

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test__EvalHarness__log_directory__with_errors(self, response_mode: Mode, eval_mode: Mode):
        """Test that EvalHarness correctly saves eval results to the specified log directory even when errors occur."""  # noqa: E501
        # Create a temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple eval with basic checks
            eval_1 = Eval(
                input="Test input 1",
                checks=[ContainsCheck(value="Test response 1 for candidate 1")],
            )
            eval_2 = Eval(
                input="Test input 2",
                checks=[ContainsCheck(value="Test response 2 for candidate 2")],
            )
            # Create simple candidates - one that works and one that fails
            candidate_1 = UnregisteredCandidate(response="Test response 1 for candidate 1", metadata={'uuid': 'candidate_1'})  # noqa: E501
            candidate_2 = AsyncMockCandidateCausesError()  # This will raise an error

            # Create the harness with the temp directory as log_directory
            harness = EvalHarness(
                evals=[eval_1, eval_2],
                candidates=[candidate_1, candidate_2],
                log_directory=temp_dir,
                response_mode=response_mode,
                eval_mode=eval_mode,
            )
            # Run the harness
            results = harness()

            # Verify that results were generated
            assert len(results) == 2
            assert len(results[0].eval_results) == 2  # First candidate has 2 successful results
            assert len(results[1].eval_results) == 2  # Second candidate has 2 results with errors
            assert results[0].num_errors == 0  # First candidate has no errors
            assert results[1].num_errors == 2  # Second candidate has errors for both evals

            # Check that files were created in the log directory
            log_files = sorted(glob.glob(os.path.join(temp_dir, "eval_result_*.json")))
            assert len(log_files) == 4  # 2 candidates * 2 evals

            # Read all log files first
            logged_results = []
            for log_file in log_files:
                with open(log_file) as f:
                    logged_results.append(json.load(f))

            # For each result (candidate), verify its log files
            for result in results:
                # Find log files for this result's candidate
                candidate_logs = []
                for logged_result in logged_results:
                    # For successful candidate
                    if isinstance(result.candidate, UnregisteredCandidate):
                        # Match by candidate type and metadata
                        if (logged_result['candidate'].get('candidate_type') == 'UnregisteredCandidate' and  # noqa: E501
                            logged_result['candidate'].get('metadata', {}).get('uuid') == 'candidate_1'):  # noqa: E501
                            candidate_logs.append(logged_result)
                            # Verify no error information in metadata for successful candidate
                            assert 'error' not in logged_result['metadata']
                            assert 'error_type' not in logged_result['metadata']
                    # For error candidate
                    elif isinstance(result.candidate, AsyncMockCandidateCausesError):  # noqa: SIM102
                        # Match by candidate type
                        if logged_result['candidate'].get('candidate_type') == 'AsyncMockCandidateCausesError':  # noqa: E501
                            candidate_logs.append(logged_result)
                            # Verify error information in metadata for failed candidate
                            assert 'error' in logged_result['metadata']
                            assert 'error_type' in logged_result['metadata']
                            assert logged_result['metadata']['error'] == "This candidate always fails."  # noqa: E501
                            assert logged_result['metadata']['error_type'] == "ValueError"

                # Verify we found the expected number of log files for this result
                assert len(candidate_logs) == 2  # Each candidate should have 2 eval results

                # Verify the contents of each log file
                for logged_result in candidate_logs:
                    # Find the matching eval result
                    matching_result = next(
                        er for er in result.eval_results
                        if er.eval.input == logged_result['eval']['input']
                    )
                    # Verify the logged result matches the actual result
                    actual_result = matching_result.to_dict()
                    assert logged_result['eval'] == actual_result['eval']
                    assert logged_result['response'] == actual_result['response']
                    assert logged_result['metadata'] == actual_result['metadata']
                    assert logged_result['check_results'] == actual_result['check_results']
                    assert logged_result['timestamp'] == actual_result['timestamp']
                    assert logged_result['candidate'] == actual_result['candidate']

            # Verify the log directory is cleaned up when the context manager exits
            assert os.path.exists(temp_dir)
        assert not os.path.exists(temp_dir)  # Directory should be cleaned up after manager exits

    # @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    # @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test__EvalHarness__log_directory__with_LLMCheck(self):
        """
        Test that EvalHarness correctly saves eval results when using LLMCheck, which is causing
        `TypeError: Object of type ModelMetaclass is not JSON serializable` error because
        the serialization process tries to convert this class to JSON, it encounters the
        ModelMetaclass which is the metaclass used by Pydantic's BaseModel - and this metaclass
        can't be serialized to JSON.
        """
        # Create a temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            class ContainsToxicity(BaseModel):
                contains_toxicity: bool
                toxicity_phrase: str | None = None

            ####
            # Check Toxicity
            ####
            check = LLMCheck(
                eval_prompt="Check if the response contains toxicity.",
                response_format=ContainsToxicity,
                openai_model_name=OPENAI_DEFAULT_MODEL,
            )
            eval_1 = Eval(
                input="Test input 1",
                checks=[check],
            )
            # Create simple candidates with unique identifiable responses
            candidate_1 = UnregisteredCandidate(
                response="Test response 1 for candidate 1",
                metadata={'uuid': 'candidate_1'},
            )
            # Create the harness with the temp directory as log_directory
            harness = EvalHarness(
                evals=[eval_1],
                candidates=[candidate_1],
                log_directory=temp_dir,
            )
            results = harness()
            assert len(results) == 1
            log_files = sorted(glob.glob(os.path.join(temp_dir, "eval_result_*.json")))
            assert len(log_files) == 1  # 1 candidate * 1 eval
            # Read the log file
            with open(log_files[0]) as f:
                logged_result = json.load(f)
            # Verify the logged result matches the actual result
            actual_result = results[0].eval_results[0].to_dict()
            # the actual response_format in the non-logged result should be the class itself;
            # however, python typtes aren't json serializable so we have to save the type as a
            # string when logging to json
            actual_response_format = actual_result['eval']['checks'][0].pop('response_format')
            assert actual_response_format == ContainsToxicity
            logged_response_format = logged_result['eval']['checks'][0].pop('response_format')
            assert 'ContainsToxicity' in logged_response_format
            # without response_format, the rest of the check should be the same; therefore the
            # eval should be the same; and actuall the entire logged result should be the same
            # as the actual result aside from the response_format
            assert logged_result == actual_result


@pytest.mark.asyncio
class TestRetryAsync:
    """Test retry functionality in EvalHarness."""

    async def test_generate_single_response_async_retry_basic(self):
        """Test basic retry functionality with fixed delay."""
        max_retries = 3
        fail_until = 2
        attempts = 0

        class RetryTestCandidate(Candidate):
            async def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                attempts += 1
                if attempts <= fail_until:  # Fail first two attempts
                    raise ValueError("Simulated failure")
                return CandidateResponse(response="success")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()
        semaphore = DelayedSemaphore(1)
        response, error = await EvalHarness._generate_single_response_async(
            semaphore=semaphore,
            candidate=candidate,
            eval=eval_obj,
            max_retries=max_retries,
            retry_delay=0.1,
            retry_backoff=1.0,  # No backoff
            max_retry_delay=1.0,
        )
        assert response.response == "success"
        assert error is None
        assert attempts == max_retries  # Should succeed on third attempt

    async def test_generate_single_response_async_retry_backoff(self):
        """Test retry functionality with exponential backoff."""
        attempts = 0
        start_times = []

        class RetryTestCandidate(Candidate):
            async def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                start_times.append(perf_counter())
                attempts += 1
                if attempts < 4:  # Fail first three attempts
                    raise ValueError("Simulated failure")
                return CandidateResponse(response="success")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()
        semaphore = DelayedSemaphore(1)

        response, error = await EvalHarness._generate_single_response_async(
            semaphore=semaphore,
            candidate=candidate,
            eval=eval_obj,
            max_retries=3,
            retry_delay=0.1,
            retry_backoff=2.0,  # Double delay each time
            max_retry_delay=1.0,
        )

        assert response.response == "success"
        assert error is None
        assert attempts == 4  # Should succeed on fourth attempt

        # Calculate actual delays between attempts
        delays = [start_times[i] - start_times[i-1] for i in range(1, len(start_times))]

        # Verify delays increase (with some tolerance for timing variations)
        assert delays[0] >= 0.1  # First retry after 0.1s
        assert delays[1] >= 0.2  # Second retry after 0.2s
        assert delays[2] >= 0.4  # Third retry after 0.4s

    async def test_generate_single_response_async_retry_max_delay(self):
        """Test retry functionality respects maximum delay."""
        attempts = 0
        start_times = []

        class RetryTestCandidate(Candidate):
            async def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                start_times.append(perf_counter())
                attempts += 1
                if attempts < 4:
                    raise ValueError("Simulated failure")
                return CandidateResponse(response="success")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()
        semaphore = DelayedSemaphore(1)

        response, error = await EvalHarness._generate_single_response_async(
            semaphore=semaphore,
            candidate=candidate,
            eval=eval_obj,
            max_retries=3,
            retry_delay=0.1,
            retry_backoff=10.0,  # Would grow very large without max_retry_delay
            max_retry_delay=0.3,  # Cap at 0.3s
        )

        assert response.response == "success"
        assert error is None
        assert attempts == 4

        # Calculate actual delays between attempts
        delays = [start_times[i] - start_times[i-1] for i in range(1, len(start_times))]

        # All delays should be <= max_retry_delay (with small tolerance)
        assert all(delay <= 0.35 for delay in delays)  # Using 0.35 to allow for small timing variations  # noqa: E501

    async def test_generate_single_response_async_retry_exhaustion(self):
        """Test behavior when retries are exhausted."""
        attempts = 0
        max_retries = 2

        class RetryTestCandidate(Candidate):
            async def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                attempts += 1
                raise ValueError("Simulated failure")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()
        semaphore = DelayedSemaphore(1)

        response, error = await EvalHarness._generate_single_response_async(
            semaphore=semaphore,
            candidate=candidate,
            eval=eval_obj,
            max_retries=max_retries,
            retry_delay=0.1,
            retry_backoff=1.0,
            max_retry_delay=1.0,
        )

        assert response is None
        assert isinstance(error, ValueError)
        assert str(error) == "Simulated failure"
        assert attempts == max_retries + 1  # Initial attempt + 2 retries

    async def test_generate_single_response_async_no_retries(self):
        """Test behavior when max_retries is 0."""
        attempts = 0

        class RetryTestCandidate(Candidate):
            async def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                attempts += 1
                raise ValueError("Simulated failure")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()
        semaphore = DelayedSemaphore(1)

        response, error = await EvalHarness._generate_single_response_async(
            semaphore=semaphore,
            candidate=candidate,
            eval=eval_obj,
            max_retries=0,  # No retries
            retry_delay=0.1,
            retry_backoff=1.0,
            max_retry_delay=1.0,
        )

        assert response is None
        assert isinstance(error, ValueError)
        assert str(error) == "Simulated failure"
        assert attempts == 1  # Only initial attempt, no retries

    async def test_run_single_eval_async_retry_basic(self):
        """Test async retry functionality when evaluating results."""
        max_retries = 2  # Will retry twice for total of three attempts
        fail_until_attempt = 2  # will succeed on third (last) attempt
        attempts = 0

        class RetryTestCheck(Check):
            def __call__(self, response: object) -> CheckResult:  # noqa: ARG002
                nonlocal attempts
                attempts += 1
                if attempts <= fail_until_attempt:
                    raise ValueError("Simulated check failure")
                return PassFailResult(value=True)

        eval_obj = Eval(
            input="test",
            checks=[RetryTestCheck()],
        )
        candidate = UnregisteredCandidate(response="test response")
        candidate_response = CandidateResponse(response="test response")

        eval_result, error = await EvalHarness._run_single_eval_async(
            semaphore=DelayedSemaphore(value=1, batch_delay=0),
            eval=eval_obj,
            candidate=candidate,
            candidate_response=candidate_response,
            candidate_error=None,
            max_retries=max_retries,
            retry_delay=0.01,  # Small delay for faster tests
            retry_backoff=1.0,
            max_retry_delay=0.01,
            log_directory=None,
        )

        assert error is None
        assert eval_result is not None
        assert attempts == fail_until_attempt + 1  # Failed twice, succeeded on third try

    async def test_run_single_eval_async_retry_exhaustion(self):
        """Test that async retry mechanism gives up after max_retries."""
        max_retries = 2
        attempts = 0

        class RetryTestCheck(Check):
            def __call__(self, response: object) -> CheckResult:  # noqa: ARG002
                nonlocal attempts
                attempts += 1
                raise ValueError("Simulated persistent check failure")

        eval_obj = Eval(
            input="test",
            checks=[RetryTestCheck()],
        )
        candidate = UnregisteredCandidate(response="test response")
        candidate_response = CandidateResponse(response="test response")

        eval_result, error = await EvalHarness._run_single_eval_async(
            semaphore=DelayedSemaphore(value=1, batch_delay=0),
            eval=eval_obj,
            candidate=candidate,
            candidate_response=candidate_response,
            candidate_error=None,
            max_retries=max_retries,
            retry_delay=0.01,  # Small delay for faster tests
            retry_backoff=1.0,
            max_retry_delay=0.01,
            log_directory=None,
        )

        assert error is not None
        assert isinstance(error, ValueError)
        assert eval_result is None
        assert attempts == max_retries + 1  # Initial attempt + retries


class TestRetrySync:
    """Test retry functionality with synchronous methods in EvalHarness."""

    def test_generate_single_response_sync_retry_basic(self):
        """Test basic retry functionality with fixed delay."""
        max_retries = 3
        fail_until = 2
        attempts = 0

        class RetryTestCandidate(Candidate):
            def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                attempts += 1
                if attempts <= fail_until:  # Fail first two attempts
                    raise ValueError("Simulated failure")
                return CandidateResponse(response="success")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()

        response, error = EvalHarness._generate_single_response(
            candidate=candidate,
            eval=eval_obj,
            max_retries=max_retries,
            retry_delay=0.1,
            retry_backoff=1.0,  # No backoff
            max_retry_delay=1.0,
        )

        assert response.response == "success"
        assert error is None
        assert attempts == max_retries  # Should succeed on third attempt

    def test_generate_single_response_sync_retry_backoff(self):
        """Test retry functionality with exponential backoff."""
        attempts = 0
        start_times = []

        class RetryTestCandidate(Candidate):
            def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                start_times.append(perf_counter())
                attempts += 1
                if attempts < 4:  # Fail first three attempts
                    raise ValueError("Simulated failure")
                return CandidateResponse(response="success")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()

        response, error = EvalHarness._generate_single_response(
            candidate=candidate,
            eval=eval_obj,
            max_retries=3,
            retry_delay=0.1,
            retry_backoff=2.0,  # Double delay each time
            max_retry_delay=1.0,
        )

        assert response.response == "success"
        assert error is None
        assert attempts == 4  # Should succeed on fourth attempt

        # Calculate actual delays between attempts
        delays = [start_times[i] - start_times[i-1] for i in range(1, len(start_times))]

        # Verify delays increase (with some tolerance for timing variations)
        assert delays[0] >= 0.1  # First retry after 0.1s
        assert delays[1] >= 0.2  # Second retry after 0.2s
        assert delays[2] >= 0.4  # Third retry after 0.4s

    def test_generate_single_response_sync_retry_max_delay(self):
        """Test maximum delay cap for synchronous response generation."""
        attempts = 0
        delays = []

        # Replace time.sleep with a function that records the delay
        original_sleep = time.sleep

        def mock_sleep(seconds) -> None:  # noqa: ANN001
            nonlocal delays
            delays.append(seconds)
            # Don't actually sleep in tests

        # Patch time.sleep temporarily
        time.sleep = mock_sleep

        try:
            class RetryTestCandidate(Candidate):
                def __call__(self, input_):  # noqa: ANN001, ARG002
                    nonlocal attempts
                    attempts += 1
                    if attempts <= 3:  # Fail first 3 attempts
                        raise ValueError("Simulated response failure")
                    return CandidateResponse(response="success")

            eval_obj = Eval(input="test", checks=[])
            candidate = RetryTestCandidate()

            response, error = EvalHarness._generate_single_response(
                candidate=candidate,
                eval=eval_obj,
                max_retries=3,
                retry_delay=0.2,
                retry_backoff=3.0,  # Triple the delay each time: 0.2, 0.6, 1.8
                max_retry_delay=0.5,  # Cap at 0.5
            )

            assert response is not None
            assert error is None
            assert attempts == 4  # Initial + 3 retries
            assert len(delays) == 3  # Should have 3 delays recorded

            # Verify delays with max cap (0.2, 0.5, 0.5)
            assert delays[0] == 0.2
            assert delays[1] == 0.5  # Should be capped at 0.5 (would be 0.6)
            assert delays[2] == 0.5  # Should be capped at 0.5 (would be 1.8)
        finally:
            # Restore original sleep function
            time.sleep = original_sleep

    def test_generate_single_response_sync_retry_exhaustion(self):
        """Test behavior when retries are exhausted."""
        attempts = 0
        max_retries = 2

        class RetryTestCandidate(Candidate):
            def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                attempts += 1
                raise ValueError("Simulated failure")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()

        response, error = EvalHarness._generate_single_response(
            candidate=candidate,
            eval=eval_obj,
            max_retries=max_retries,
            retry_delay=0.1,
            retry_backoff=1.0,
            max_retry_delay=1.0,
        )

        assert response is None
        assert isinstance(error, ValueError)
        assert str(error) == "Simulated failure"
        assert attempts == max_retries + 1  # Initial attempt + 2 retries

    def test_generate_single_response_sync_no_retries(self):
        """Test behavior when max_retries is 0."""
        attempts = 0

        class RetryTestCandidate(Candidate):
            def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal attempts
                attempts += 1
                raise ValueError("Simulated failure")

        eval_obj = Eval(input="test", checks=[])
        candidate = RetryTestCandidate()

        response, error = EvalHarness._generate_single_response(
            candidate=candidate,
            eval=eval_obj,
            max_retries=0,  # No retries
            retry_delay=0.1,
            retry_backoff=1.0,
            max_retry_delay=1.0,
        )

        assert response is None
        assert isinstance(error, ValueError)
        assert str(error) == "Simulated failure"
        assert attempts == 1  # Only initial attempt, no retries

    def test_run_single_eval_retry_basic(self):
        """Test retry functionality when evaluating results."""
        max_retries = 3
        fail_until = 2
        attempts = 0

        class RetryTestCheck(Check):
            def __call__(self, response: object) -> CheckResult:  # noqa: ARG002
                nonlocal attempts
                attempts += 1
                if attempts <= fail_until:
                    raise ValueError("Simulated check failure")
                return PassFailResult(value=True)

        eval_obj = Eval(
            input="test",
            checks=[RetryTestCheck()],
        )
        candidate = UnregisteredCandidate(response="test response")
        candidate_response = CandidateResponse(response="test response")

        eval_result, error = EvalHarness._run_single_eval(
            eval=eval_obj,
            candidate=candidate,
            candidate_response=candidate_response,
            candidate_error=None,
            max_retries=max_retries,
            retry_delay=0.01,  # Small delay for faster tests
            retry_backoff=1.0,
            max_retry_delay=0.01,
            log_directory=None,
        )

        assert error is None
        assert eval_result is not None
        assert attempts == fail_until + 1  # Failed twice, succeeded on third try

    def test_run_single_eval_retry_exhaustion(self):
        """Test that retry mechanism gives up after max_retries."""
        max_retries = 2
        attempts = 0

        class RetryTestCheck(Check):
            def __call__(self, response: object) -> CheckResult:  # noqa: ARG002
                nonlocal attempts
                attempts += 1
                raise ValueError("Simulated persistent check failure")

        eval_obj = Eval(
            input="test",
            checks=[RetryTestCheck()],
        )
        candidate = UnregisteredCandidate(response="test response")
        candidate_response = CandidateResponse(response="test response")

        eval_result, error = EvalHarness._run_single_eval(
            eval=eval_obj,
            candidate=candidate,
            candidate_response=candidate_response,
            candidate_error=None,
            max_retries=max_retries,
            retry_delay=0.01,  # Small delay for faster tests
            retry_backoff=1.0,
            max_retry_delay=0.01,
            log_directory=None,
        )

        assert error is not None
        assert isinstance(error, ValueError)
        assert eval_result is None
        assert attempts == max_retries + 1  # Initial attempt + retries


class TestRetryEvalHarness:
    """Test retry functionality in EvalHarness directly."""

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC])
    def test_end_to_end_retry_with_harness(self, response_mode: Mode, eval_mode: Mode):
        """
        Test end-to-end retry functionality through the EvalHarness. We are not testing parallel
        here because of nonlocal variables used in the test.
        """
        # Track attempts for both response generation and evaluation
        response_attempts = 0
        eval_attempts = 0

        class RetryTestCandidate(Candidate):
            def __call__(self, input_):  # noqa: ANN001, ARG002
                nonlocal response_attempts
                response_attempts += 1
                # Fail first attempt, succeed on second
                if response_attempts == 1:
                    raise ValueError("Simulated response failure")
                return CandidateResponse(response="success")

        class RetryTestCheck(Check):
            def __call__(self, response: object) -> CheckResult:  # noqa: ARG002
                nonlocal eval_attempts
                eval_attempts += 1
                # Fail first attempt, succeed on second
                if eval_attempts == 1:
                    raise ValueError("Simulated check failure")
                return PassFailResult(value=True)

        # Create harness with retry settings
        harness = EvalHarness(
            evals=[Eval(input="test", checks=[RetryTestCheck()])],
            candidates=[RetryTestCandidate()],
            max_retries=2,  # Allow up to 2 retries (3 attempts total)
            retry_delay=0.1,
            retry_backoff=1.0,
            response_mode=response_mode,
            eval_mode=eval_mode,
        )

        # Run the harness and check results
        results = harness()

        # Verify both retries happened
        assert response_attempts == 2  # Initial failure + successful retry
        assert eval_attempts == 2      # Initial failure + successful retry

        # Verify results
        assert len(results) == 1
        assert results[0].num_errors == 0
        assert len(results[0].eval_results) == 1
        assert results[0].eval_results[0].check_results[0].value is True
        assert results[0].eval_results[0].check_results[0].success is True

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test_EvalHarness_async__generation_error__recovers(self, response_mode: Mode, eval_mode: Mode):  # noqa
        """Test async retry functionality when evaluating results."""
        max_retries = 2
        total_attempts = max_retries + 1

        eval_obj = Eval(input="test", checks=[])
        # fail the first two attempts and succeed on the third (i.e. last retry)
        candidate = MockRetryTestCandidate(fail_until_attempt=max_retries)
        harness = EvalHarness(
            evals=[eval_obj],
            candidates=[candidate],
            response_mode=response_mode,
            eval_mode=eval_mode,
            max_retries=max_retries,
            retry_delay=0.01,
            retry_backoff=1.0,  # No backoff
            max_retry_delay=1.0,
        )
        results = harness()
        assert len(results) == 1
        candidate_result = results[0]
        assert candidate_result.eval_results[0].metadata['response_metadata']['attempts'] == total_attempts  # noqa: E501
        assert len(candidate_result.eval_results) == 1
        assert isinstance(candidate_result.eval_results[0], EvalResult)
        assert candidate_result.num_errors == 0
        assert candidate_result.response_errors == [None]
        assert candidate_result.eval_errors == [None]
        assert candidate_result.eval_results[0].response == "success"

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test_EvalHarness_async__generation_error__exhaustion(self, response_mode: Mode, eval_mode: Mode):  # noqa
        """Test behavior when retries are exhausted."""
        max_retries = 2
        total_attempts = max_retries + 1

        eval_obj = Eval(input="test", checks=[])
        # fail all attempts
        candidate = MockRetryTestCandidate(fail_until_attempt=total_attempts)

        harness = EvalHarness(
            evals=[eval_obj],
            candidates=[candidate],
            response_mode=response_mode,
            eval_mode=eval_mode,
            max_retries=max_retries,
            retry_delay=0.01,  # Small delay for faster tests
            retry_backoff=1.0,  # No backoff
            max_retry_delay=0.01,
        )
        results = harness()
        assert len(results) == 1
        candidate_result = results[0]
        assert len(candidate_result.eval_results) == 1
        assert isinstance(candidate_result.eval_results[0], EvalResult)
        assert candidate_result.num_errors == 1
        assert isinstance(candidate_result.response_errors[0], ValueError)
        assert candidate_result.eval_errors == [None]
        assert candidate_result.eval_results[0].response is None
        assert candidate_result.eval_results[0].metadata['error'] == "Simulated failure"
        assert candidate_result.eval_results[0].metadata['error_type'] == "ValueError"

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test_EvalHarness_async__eval_error__recovers(self, response_mode: Mode, eval_mode: Mode):
        """Test async retry functionality when evaluating results."""
        max_retries = 2
        total_attempts = max_retries + 1

        eval_obj = Eval(input="test", checks=[MockRetryTestCheck(fail_until_attempt=max_retries)])
        candidate = UnregisteredCandidate(response="test response")

        harness = EvalHarness(
            evals=[eval_obj],
            candidates=[candidate],
            response_mode=response_mode,
            eval_mode=eval_mode,
            max_retries=max_retries,
            retry_delay=0.01,
            retry_backoff=1.0,  # No backoff
            max_retry_delay=1.0,
        )
        results = harness()
        assert len(results) == 1
        candidate_result = results[0]
        assert candidate_result.num_errors == 0
        assert len(candidate_result.eval_results) == 1
        assert isinstance(candidate_result.eval_results[0], EvalResult)
        assert candidate_result.response_errors == [None]
        assert candidate_result.eval_errors == [None]
        assert candidate_result.eval_results[0].response == {'prompt': 'test', 'response': 'test response'}  # noqa: E501
        assert candidate_result.eval_results[0].check_results[0].value is True
        assert candidate_result.eval_results[0].check_results[0].metadata['attempts'] == total_attempts  # noqa: E501

    @pytest.mark.parametrize('response_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    @pytest.mark.parametrize('eval_mode', [Mode.SYNC, Mode.ASYNC, Mode.PARALLEL])
    def test_EvalHarness_async__eval_error__exhaustion(self, response_mode: Mode, eval_mode: Mode):
        """Test async retry functionality when evaluating results."""
        max_retries = 2
        total_attempts = max_retries + 1

        eval_obj = Eval(input="test", checks=[MockRetryTestCheck(fail_until_attempt=total_attempts)])  # noqa: E501
        candidate = UnregisteredCandidate(response="test response")

        harness = EvalHarness(
            evals=[eval_obj],
            candidates=[candidate],
            response_mode=response_mode,
            eval_mode=eval_mode,
            max_retries=max_retries,
            retry_delay=0.01,
            retry_backoff=1.0,  # No backoff
            max_retry_delay=1.0,
        )
        results = harness()
        assert len(results) == 1
        candidate_result = results[0]
        assert candidate_result.num_errors == 1
        assert len(candidate_result.eval_results) == 1
        assert candidate_result.eval_results[0]is None
        assert candidate_result.response_errors == [None]
        assert isinstance(candidate_result.eval_errors[0], ValueError)
        assert str(candidate_result.eval_errors[0]) == 'Simulated failure'
