"""Tests for the evals module."""
from copy import deepcopy
import os
import shutil
from textwrap import dedent
import pytest
import yaml
from llm_eval.candidates import CallableCandidate, Candidate, CandidateType
from llm_eval.checks import CheckType, ContainsCheck, MatchCheck, PassFailResult, ScoreResult
from llm_eval.eval import Eval, EvalHarness, EvalResult, PromptTest
from llm_eval.utilities.internal_utilities import extract_code_blocks


def test__PromptTest():  # noqa
    test = PromptTest(
        prompt='test',
        ideal_response = None,
        checks = None,
    )
    assert test.prompt == 'test'
    assert test.ideal_response is None
    assert test.checks == []
    assert str(test)
    test_dict = test.to_dict()
    assert PromptTest(**test_dict) == test
    assert test_dict == {'prompt': 'test'}

    test = PromptTest(
        prompt='test',
        ideal_response = None,
        checks = [],
    )
    assert test.prompt == 'test'
    assert test.ideal_response is None
    assert test.checks == []
    assert str(test)
    test_dict = test.to_dict()
    assert PromptTest(**test_dict) == test
    assert test_dict == {'prompt': 'test'}

    test = PromptTest(
        prompt='test1',
        ideal_response='test2',
        checks = [MatchCheck(value='test3')],
    )
    assert test.prompt == 'test1'
    assert test.ideal_response == 'test2'
    assert test.checks == [MatchCheck(value='test3')]
    assert str(test)
    test_dict = test.to_dict()
    assert PromptTest(**test_dict) == test
    assert test_dict == {
        'prompt': 'test1',
        'ideal_response': 'test2',
        'checks': [{'check_type': CheckType.MATCH.name, 'value': 'test3'}],
    }

    test = PromptTest(
        prompt='test1',
        ideal_response='test2',
        checks = [MatchCheck(value='test3'), MatchCheck(value='test4')],
    )
    assert test.prompt == 'test1'
    assert test.ideal_response == 'test2'
    assert test.checks == [MatchCheck(value='test3'), MatchCheck(value='test4')]
    assert str(test)
    test_dict = test.to_dict()
    assert PromptTest(**test_dict) == test
    assert test_dict == {
        'prompt': 'test1',
        'ideal_response': 'test2',
        'checks': [
            {'check_type': CheckType.MATCH.name, 'value': 'test3'},
            {'check_type': CheckType.MATCH.name, 'value': 'test4'},
        ],
    }

def test__PromptTest__none_list_check():  # noqa
    test = PromptTest(
        prompt='test',
        ideal_response = None,
        checks = None,
    )
    assert test.prompt == 'test'
    assert test.ideal_response is None
    assert test.checks == []
    test = PromptTest(
        prompt='test',
        ideal_response = None,
        checks = {'value': 'a', 'check_type': 'MATCH'},
    )
    assert test.prompt == 'test'
    assert test.ideal_response is None
    assert test.checks[0].value == 'a'
    test = PromptTest(
        prompt='test',
        ideal_response = None,
        checks = MatchCheck(value='a'),
    )
    assert test.prompt == 'test'
    assert test.ideal_response is None
    assert test.checks[0].value == 'a'

def test__Eval__creation():  # noqa
    eval_obj = Eval(test_sequence=PromptTest(prompt='test'))
    eval_dict = eval_obj.to_dict()
    assert eval_dict == {'test_sequence': [{'prompt': 'test'}]}
    assert Eval(**eval_dict) == eval_obj
    assert str(eval_obj)

    eval_obj = Eval(
        test_sequence=[
            PromptTest(
                prompt='test1',
                ideal_response='test2',
                checks = [MatchCheck(value='test3')],
            ),
            PromptTest(
                prompt='test4',
                ideal_response='test5',
                checks = [
                    MatchCheck(value='test6', metadata={'test': 'test7'}),
                    ContainsCheck(value='test8'),
                ],
            ),
        ],
    )
    assert eval_obj.test_sequence[0].prompt == 'test1'
    assert eval_obj.test_sequence[0].ideal_response == 'test2'
    assert eval_obj.test_sequence[0].checks == [MatchCheck(value='test3')]
    assert eval_obj.test_sequence[1].prompt == 'test4'
    assert eval_obj.test_sequence[1].ideal_response == 'test5'
    assert eval_obj.test_sequence[1].checks == [
        MatchCheck(value='test6', metadata={'test': 'test7'}),
        ContainsCheck(value='test8'),
    ]
    assert str(eval_obj)

    eval_dict = eval_obj.to_dict()
    assert eval_dict == {
        'test_sequence': [
            {
                'prompt': 'test1',
                'ideal_response': 'test2',
                'checks': [
                    {
                        'check_type': CheckType.MATCH.name,
                        'value': 'test3',
                    },
                ],
            },
            {
                'prompt': 'test4',
                'ideal_response': 'test5',
                'checks': [
                    {
                        'metadata': {'test': 'test7'},
                        'check_type': CheckType.MATCH.name,
                        'value': 'test6',
                    },
                    {
                        'check_type': CheckType.CONTAINS.name,
                        'value': 'test8',
                    },
                ],
            },
        ],
    }
    assert Eval(**eval_dict) == eval_obj

def test__eval_obj__clone(fake_eval_8f9fbf37):  #noqa
    config = deepcopy(fake_eval_8f9fbf37)
    eval_obj = Eval(**config)
    eval_cloned = eval_obj.clone()
    assert eval_obj == eval_cloned
    assert eval_obj.to_dict() == eval_cloned.to_dict()
    # test-sequence (i.e. PromptTest objects) should be the same prompt tests but different objects
    assert eval_obj.test_sequence == eval_cloned.test_sequence
    assert eval_obj.test_sequence[0] == eval_cloned.test_sequence[0]
    assert eval_obj.test_sequence[0] is not eval_cloned.test_sequence[0]
    assert eval_obj.test_sequence[1] == eval_cloned.test_sequence[1]
    assert eval_obj.test_sequence[1] is not eval_cloned.test_sequence[1]

def test__Eval__call__result__to_from_dict():  # noqa
    """
    Tests the basic case of calling an Eval object and converting it to/from a dict. No checks are
    passed to the eval.
    """
    eval_obj = Eval(test_sequence=PromptTest(prompt='test'))
    # dict before call should be the same as after call
    assert eval_obj.to_dict() == {'test_sequence': [{'prompt': 'test'}]}
    assert Eval(**eval_obj.to_dict()) == eval_obj
    result = eval_obj(lambda x: f'response: {x}')
    assert result.response_characters == len('response: test')
    assert eval_obj.to_dict() == {'test_sequence': [{'prompt': 'test'}]}
    assert Eval(**eval_obj.to_dict()) == eval_obj

    result_dict = result.to_dict()
    assert result_dict['eval_obj'] == eval_obj.to_dict()
    assert result_dict['candidate_obj'] == {
        'metadata': {'function': 'def <lambda>(x)'},
        'candidate_type': CandidateType.CALLABLE_NO_SERIALIZE.name,
        }
    assert Eval(**result_dict['eval_obj']) == eval_obj
    assert Candidate.from_dict(result_dict['candidate_obj']) == result.candidate_obj
    assert EvalResult(**result_dict) == result
    assert EvalResult(**result_dict).to_dict() == result.to_dict()

def test__Eval__from_objects__minimal():  # noqa
    candidate = CallableCandidate(model=lambda x: x)
    prompt = "This is a prompt."
    eval_obj = Eval(test_sequence={'prompt': prompt})
    result = eval_obj(candidate)
    assert str(result)  # make sure __str__ works
    assert result.eval_obj == eval_obj
    assert result.candidate_obj == candidate
    assert result.responses == [prompt]
    assert result.prompts == [prompt]
    assert result.ideal_responses == [None]
    assert result.response_characters == len(prompt)
    assert result.num_checks == 0
    assert result.num_successful_checks == 0
    assert result.perc_successful_checks is None
    assert result.results == [[]]
    assert result.cost is None
    assert result.timestamp
    assert result.num_code_blocks == 0
    assert result.all_check_results == []
    assert not result.expects_code_blocks
    assert result.code_block_tests_result is None
    assert result.total_time_seconds > 0

def test__Eval__example_8f9fbf37__callable_candidate(fake_eval_8f9fbf37: dict):  # noqa
    eval_dict = fake_eval_8f9fbf37.copy()
    eval_obj = Eval(**eval_dict)
    assert eval_obj.to_dict() == eval_dict

    responses = [
        "This is the first response",
        "This is a response with code blocks\n```python\nprint('hello world')\n```",
    ]
    def mock_llm():  # noqa
        yield from responses
    mock_llm_instance = mock_llm()
    eval_result = eval_obj(lambda _: next(mock_llm_instance))
    assert eval_result.responses == responses
    assert eval_result.prompts == [test.prompt for test in eval_obj.test_sequence]
    assert eval_result.ideal_responses == [test.ideal_response for test in eval_obj.test_sequence]
    assert eval_result.eval_obj.to_dict() == eval_dict
    assert eval_result.cost is None
    assert eval_result.num_checks == 4
    assert eval_result.num_successful_checks == 2
    assert eval_result.perc_successful_checks == 2 / 4
    assert len(eval_result.results) == 2
    assert len(eval_result.results[0]) == 3
    assert len(eval_result.results[1]) == 1
    assert eval_result.response_characters == sum(len(r) for r in responses)
    assert eval_result.num_code_blocks == 1

    eval_result_dict = eval_result.to_dict()
    # we can't check that entire eval_result_dict will recreate the exact eval_result object
    # because the candidate will be slightly different (e.g. if it was a function, it will have
    # been converted to a string; we can't serialize the underlying model/llm)
    assert eval_result_dict['eval_obj'] == eval_dict
    assert Eval(**eval_result_dict['eval_obj']) == eval_obj
    assert eval_result_dict['candidate_obj'] == {
        'metadata': {'function': 'def <lambda>(_)'},
        'candidate_type': CandidateType.CALLABLE_NO_SERIALIZE.name,
    }
    # check that the check result dicts match
    flatted_check_results = [r for tests in eval_result_dict['results'] for r in tests]
    assert flatted_check_results == [r.to_dict() for r in eval_result.all_check_results]
    assert eval_result.expects_code_blocks
    assert eval_result.code_block_tests_result is None
    assert eval_result.total_time_seconds > 0
    # check that the eval_result_dict will recreate the exact eval_result object
    recreated_eval = EvalResult(**eval_result_dict)
    assert recreated_eval == eval_result
    assert recreated_eval.to_dict() == eval_result.to_dict()
    assert recreated_eval.eval_obj == eval_result.eval_obj
    assert recreated_eval.candidate_obj == eval_result.candidate_obj
    assert recreated_eval.results == eval_result.results
    flatted_checks = [r for test in eval_obj.test_sequence for r in test.checks]
    for c, r in zip(flatted_checks, eval_result.all_check_results, strict=True):
        assert c.check_type == r.metadata['check_type']
    assert eval_result.expects_code_blocks
    assert eval_result.code_block_tests_result is None

def test__Eval__multiple_code_blocks__ensure_code_blocks_run(fake_eval_sum_two_numbers_code_blocks_run):  # noqa
    """
    Use Mock LLM with multiple code blocks (over multiple responses) to ensure code blocks run and
    the check results return the expected values.
    """
    config = fake_eval_sum_two_numbers_code_blocks_run.copy()
    eval_obj = Eval(**config)

    assert eval_obj.test_sequence[1].checks[-1].code_block_timeout == 5
    assert eval_obj.test_sequence[1].checks[-1].code_test_timeout == 5

    response_1 = dedent("""
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
    response_2 = dedent("""
    To test the `sum_two_numbers` function, you can use assertion statements in Python. Assertions are a great way to ensure that your function behaves as expected. Here are some examples:

    ```python
    assert sum_two_numbers(2, 3) == 5, "Test failed for inputs 2 and 3"
    assert sum_two_numbers(-1, 1) == 0, "Test failed for inputs -1 and 1"
    assert sum_two_numbers(0, 0) == 0, "Test failed for inputs 0 and 0"
    assert sum_two_numbers(-2, -3) == -5, "Test failed for inputs -2 and -3"
    assert sum_two_numbers(1.5, 2.5) == 4.0, "Test failed for inputs 1.5 and 2.5"
    my_value = 1  # ensure code block runs
    ```

    Each assertion checks a specific case:

    1) Adding two positive integers.
    2) Adding a negative integer and a positive integer.
    3) Adding two zeros.
    4) Adding two negative integers.
    5) Adding two floating-point numbers.

    If the function returns the expected value, the program will continue without any interruption. If any assertion fails, Python raises an `AssertionError` with the specified message, indicating that the test case failed.
    """)  # noqa: E501
    responses = [response_1, response_2]

    expected_code_blocks = extract_code_blocks(response_1)
    expected_code_blocks.extend(extract_code_blocks(response_2))

    expected_num_code_blocks = len(expected_code_blocks)
    expected_successful_code_blocks = len(expected_code_blocks)

    def mock_llm():  # noqa
        yield from responses
    mock_llm_instance = mock_llm()
    eval_result = eval_obj(lambda _: next(mock_llm_instance))

    # we need to strip the code blocks of leading/trailing whitespace to compare them
    expected_config = deepcopy(config)
    expected_config['test_sequence'][1]['checks'][-1]['code_tests'] = [
        dedent(x.strip()) for x in
        expected_config['test_sequence'][1]['checks'][-1]['code_tests']
    ]
    assert eval_result.eval_obj.to_dict() == expected_config
    assert Eval(**eval_obj.to_dict()) == eval_obj
    # i need to compare strings because underlying error objects (i.e. instances) will not be same
    assert str(EvalResult(**eval_result.to_dict()).to_dict()) == str(eval_result.to_dict())

    assert eval_result.responses == responses
    assert eval_result.prompts == [test.prompt for test in eval_obj.test_sequence]
    assert eval_result.ideal_responses == [test.ideal_response for test in eval_obj.test_sequence]
    assert eval_result.response_characters == sum(len(r) for r in responses)
    assert eval_result.num_checks == 7
    assert eval_result.num_successful_checks == 4
    assert eval_result.perc_successful_checks == 4 / 7
    assert eval_result.num_code_blocks == expected_num_code_blocks

    assert len(eval_result.results) == 2
    assert len(eval_result.results[0]) == 4
    assert isinstance(eval_result.results[0][0], PassFailResult)
    assert isinstance(eval_result.results[0][0], PassFailResult)
    assert isinstance(eval_result.results[0][1], PassFailResult)
    assert isinstance(eval_result.results[0][1], PassFailResult)
    assert len(eval_result.results[1]) == 3
    assert isinstance(eval_result.results[1][0], PassFailResult)
    assert isinstance(eval_result.results[1][1], PassFailResult)
    assert isinstance(eval_result.results[1][2], ScoreResult)
    assert len(eval_result.all_check_results) == 7
    assert eval_result.expects_code_blocks
    assert eval_result.code_block_tests_result is not None

    # Check 0.0
    assert eval_result.results[0][0].success
    assert eval_result.results[0][0].metadata['check_type'] == CheckType.CONTAINS.name
    # Check 0.1
    assert not eval_result.results[0][1].success
    assert eval_result.results[0][1].metadata['check_type'] == CheckType.MATCH.name
    # Check 0.2
    assert not eval_result.results[0][2].success
    assert eval_result.results[0][2].metadata['check_type'] == CheckType.CONTAINS.name
    # Check 0.3
    assert eval_result.results[0][3].success
    assert eval_result.results[0][3].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name  # noqa: E501
    # Check 1.0
    assert eval_result.results[1][0].success
    assert eval_result.results[1][0].metadata['check_type'] == CheckType.CONTAINS.name
    # Check 1.1
    assert eval_result.results[1][1].success
    assert eval_result.results[1][1].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name  # noqa
    # Check 1.2
    assert not eval_result.results[1][2].success
    assert eval_result.results[1][2].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name  # noqa

    # function checks
    expected_code_tests = 6
    expected_successful_code_tests = 4
    expected_total_checks = expected_num_code_blocks + expected_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests

    assert eval_result.results[1][2].value == expected_successful_checks / expected_total_checks
    assert eval_result.results[1][2].success_threshold == 1
    assert not eval_result.results[1][2].success
    assert eval_result.results[1][2].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name  # noqa
    assert eval_result.results[1][2].metadata['num_code_blocks'] == expected_num_code_blocks
    assert eval_result.results[1][2].metadata['num_code_blocks_successful'] == expected_successful_code_blocks  # noqa
    assert eval_result.results[1][2].metadata['code_blocks'] == expected_code_blocks
    assert eval_result.results[1][2].metadata['code_block_errors'] == [None, None, None]
    # first function check should have run successfully, but second code block should have failed
    assert eval_result.results[1][2].metadata['code_test_results'] == [True, True, False, True, True, False]  # noqa
    assert eval_result.results[1][2].metadata['num_code_tests'] == expected_code_tests
    assert eval_result.results[1][2].metadata['num_code_tests_successful'] == expected_successful_code_tests  # noqa
    assert eval_result.results[1][2].metadata['code_test_errors'][0] is None
    assert eval_result.results[1][2].metadata['code_test_errors'][1] is None
    assert eval_result.results[1][2].metadata['code_test_errors'][2] is None
    assert eval_result.results[1][2].metadata['code_test_errors'][3] is None
    assert eval_result.results[1][2].metadata['code_test_errors'][4] is None
    assert eval_result.results[1][2].metadata['code_test_errors'][5] == {'error': 'NameError', 'message': "name 'variable_does_not_exist' is not defined"}  # noqa

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__Eval__candidate_from_dict(fake_eval_sum_two_numbers, openai_candidate_template):  # noqa
    eval_config = fake_eval_sum_two_numbers.copy()
    eval_obj = Eval(**eval_config)
    result = eval_obj(openai_candidate_template)
    assert result.eval_obj == eval_obj
    assert result.candidate_obj == Candidate.from_dict(openai_candidate_template)
    assert result.candidate_obj.to_dict() == openai_candidate_template
    assert len(result.responses) == 1
    assert 'sum_two_numbers' in result.responses[0]
    assert len(result.prompts) == 1
    assert result.prompts[0] == eval_config['test_sequence'][0]['prompt']
    assert len(result.results) == 1
    assert result.cost == result.candidate_obj.cost
    assert 'cost' in result.to_dict()
    assert result.cost == result.to_dict()['cost']
    assert len(result.all_check_results) == 2
    assert result.expects_code_blocks
    assert result.code_block_tests_result is None
    assert result.all_check_results[0].success
    assert result.all_check_results[0].metadata['check_type'] == CheckType.CONTAINS.name
    assert result.all_check_results[1].success
    assert result.all_check_results[1].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name  # noqa: E501
    assert result.all_check_results[1].metadata['num_code_blocks'] >= 1
    expected_num_code_blocks = len(result.all_check_results[1].metadata['code_blocks'])
    assert result.all_check_results[1].metadata['num_code_blocks'] == expected_num_code_blocks
    assert result.num_checks == 2
    assert result.num_successful_checks == 2
    assert result.perc_successful_checks == 1
    assert str(result)
    assert EvalResult(**result.to_dict()) == result
    assert EvalResult(**result.to_dict()).to_dict() == result.to_dict()
    assert eval_config == fake_eval_sum_two_numbers  # make sure eval_config wasn't modified

@Candidate.register('MockCandidate')
class MockCandidate(Candidate):
    """
    This class needs to be outside of the test function so that we can test multi-processing, which
    requires that the class be picklable, which requires that it be defined at the top level of the
    module.

    This candidate takes a dictionary of prompts (keys) and responses (values) and returns the
    response for the given prompt.
    """  # noqa: D404

    def __init__(
            self,
            responses: dict,
            metadata: dict | None = None,
            parameters: dict | None = None):
        super().__init__(metadata=metadata, parameters=parameters)
        self.responses = responses.copy()

    def __call__(self, prompt: str) -> str:
        """Returns the response for the given prompt."""
        return self.responses[prompt]

    def to_dict(self) -> dict:
        """Need to add `responses` to enable proper to_dict values."""
        value = super().to_dict()
        value['responses'] = self.responses
        return value

def test__EvalHarness__multiple_candidates__multiple_evals(fake_eval_subtract_two_numbers, fake_eval_sum_two_numbers):  # noqa
    subtract_config = fake_eval_subtract_two_numbers.copy()
    sum_config = fake_eval_sum_two_numbers.copy()

    response_subtract_0 = 'This is the response.\n\n```\ndef subtract_two_numbers(a, b):\n    return a - b\n```'  # noqa
    response_subtract_1 = 'This is the assertion statement.\n\n```\nassert subtract_two_numbers(2, 3) == -1\n```'  # noqa
    response_sum_0 = 'This is the response.\n\n```\ndef sum_two_numbers(a, b):\n    return a + b\n```'  # noqa
    responses_lookup = {
        fake_eval_subtract_two_numbers['test_sequence'][0]['prompt']: response_subtract_0,
        fake_eval_subtract_two_numbers['test_sequence'][1]['prompt']: response_subtract_1,
        fake_eval_sum_two_numbers['test_sequence'][0]['prompt']: response_sum_0,
    }

    candidate_1_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': 'MockCandidate',
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

    eval_harness = EvalHarness(
        num_cpus=1,
        async_batch_size=1,
    )
    assert eval_harness.evals != eval_harness_via_dicts.evals
    assert eval_harness.candidates != eval_harness_via_dicts.candidates
    eval_harness.add_eval(Eval(**subtract_config))
    eval_harness.add_eval(Eval(**sum_config))
    eval_harness.add_candidate(Candidate.from_dict(candidate_1_dict))
    eval_harness.add_candidate(Candidate.from_dict(candidate_2_dict))
    assert eval_harness.evals == eval_harness_via_dicts.evals
    assert eval_harness.candidates == eval_harness_via_dicts.candidates

    results = eval_harness()
    assert len(results) == 2
    assert len(results[0]) == 2
    assert len(results[1]) == 2
    # The underlying candidate objects should have the same values but should be different objects
    # because each candidate object (against a specific eval) is responsible for storing its own
    # history/conversation and the history should be different for each eval.
    assert results[0][0].candidate_obj == results[0][1].candidate_obj
    assert results[0][0].candidate_obj is not results[0][1].candidate_obj
    assert results[1][0].candidate_obj == results[1][1].candidate_obj
    assert results[1][0].candidate_obj is not results[1][1].candidate_obj

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
    assert cand_1_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_1_results_subtract.num_code_blocks == 2
    assert cand_1_results_subtract.num_checks == 5
    assert cand_1_results_subtract.num_successful_checks == 4
    assert cand_1_results_subtract.perc_successful_checks == 4 / 5

    # candidate 1 - sum eval
    cand_1_results_sum = results[0][1]
    assert cand_1_results_sum.responses == [response_sum_0]
    assert cand_1_results_sum.num_code_blocks == 1
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1

    # candidate 2 - subtract eval
    cand_2_results_subtract = results[1][0]
    assert cand_2_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_2_results_subtract.num_code_blocks == 2
    assert cand_2_results_subtract.num_checks == 5
    assert cand_2_results_subtract.num_successful_checks == 4
    assert cand_2_results_subtract.perc_successful_checks == 4 / 5

    # candidate 2 - sum eval
    cand_2_results_sum = results[1][1]
    assert cand_2_results_sum.responses == [response_sum_0]
    assert cand_2_results_sum.num_code_blocks == 1
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1

    # the eval results of candidate 1 should be the same as the eval results of candidate 2,
    # except the seconds it took to run the evals and the uuid of the candidate
    cand_1_results_subtract_dict = deepcopy(cand_1_results_subtract.to_dict())
    del cand_1_results_subtract_dict['total_time_seconds']
    del cand_1_results_subtract_dict['candidate_obj']['metadata']['uuid']
    cand_2_results_subtract_dict = deepcopy(cand_2_results_subtract.to_dict())
    del cand_2_results_subtract_dict['total_time_seconds']
    del cand_2_results_subtract_dict['candidate_obj']['metadata']['uuid']
    assert cand_1_results_subtract_dict == cand_2_results_subtract_dict

    cand_1_results_subtract.to_yaml('__temp__.yaml')
    result_from_yaml = cand_1_results_subtract.from_yaml('__temp__.yaml')
    assert result_from_yaml == cand_1_results_subtract
    assert result_from_yaml.to_dict() == cand_1_results_subtract.to_dict()
    os.remove('__temp__.yaml')

    cand_1_results_sum.to_yaml('__temp__.yaml')
    result_from_yaml = cand_1_results_sum.from_yaml('__temp__.yaml')
    assert result_from_yaml == cand_1_results_sum
    assert result_from_yaml.to_dict() == cand_1_results_sum.to_dict()
    os.remove('__temp__.yaml')

    assert subtract_config == fake_eval_subtract_two_numbers  # ensure eval_config wasn't modified
    assert sum_config == fake_eval_sum_two_numbers  # ensure eval_config wasn't modified

def test__multiline_eval__dedents_prompt():  # noqa
    prompt = """
        - This is a multiline prompt.
            - It needs to be dedented.
    """
    eval_obj = Eval(test_sequence=[PromptTest(prompt=prompt)])
    result = eval_obj(lambda x: x)
    assert result.responses[0] == dedent(prompt)

def callback(x: EvalResult) -> None:
    """
    Test the callback function by saving the result to a yaml file in the 'test/temp'
    directory. Assume directory already exists.
    """
    candidate_id = x.candidate_obj.metadata['uuid']
    eval_id = x.eval_obj.metadata['uuid']
    with open(f'tests/__temp__/result-{candidate_id}-{eval_id}.yaml', 'w') as f:
        yaml.dump(x.to_dict(), f, default_flow_style=False, sort_keys=False)

def test__EvalHarness__multi_prossing_async__vs__not(fake_eval_subtract_two_numbers, fake_eval_sum_two_numbers_code_blocks_run):  # noqa
    subtract_config = fake_eval_subtract_two_numbers.copy()
    sum_config = fake_eval_sum_two_numbers_code_blocks_run.copy()

    dir_path = "tests/__temp__"
    def recreate_temp_dir() -> None:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        assert os.path.exists(dir_path)
    recreate_temp_dir()

    response_subtract_0 = 'This is the response.\n\n```\ndef subtract_two_numbers(a, b):\n    return a - b\n```'  # noqa
    response_subtract_1 = 'This is the assertion statement.\n\n```\nassert subtract_two_numbers(2, 3) == -1\n```'  # noqa
    response_sum_0 = 'This is the response.\n\n```\ndef sum_two_numbers(a, b):\n    return a + b\n```'  # noqa
    responses_lookup = {
        fake_eval_subtract_two_numbers['test_sequence'][0]['prompt']: response_subtract_0,
        fake_eval_subtract_two_numbers['test_sequence'][1]['prompt']: response_subtract_1,
        fake_eval_sum_two_numbers_code_blocks_run['test_sequence'][0]['prompt']: response_sum_0,
    }

    candidate_1_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': 'MockCandidate',
        'responses': responses_lookup,
    }
    candidate_2_dict = deepcopy(candidate_1_dict)
    candidate_2_dict['metadata']['uuid'] = 'candidate_2'

    eval_harness_sequential = EvalHarness(
        evals=[subtract_config, sum_config],
        candidates=[candidate_1_dict, candidate_2_dict],
        num_cpus=1,
        async_batch_size=1,
    )
    eval_harness_async_multiprocessing = EvalHarness(
        evals=[subtract_config, sum_config],
        candidates=[candidate_1_dict, candidate_2_dict],
        num_cpus=200,
        async_batch_size=200,
    )
    eval_harness_multi_with_callback = EvalHarness(
        evals=[subtract_config, sum_config],
        candidates=[candidate_1_dict, candidate_2_dict],
        num_cpus=200,
        async_batch_size=200,
        callback=callback,
    )

    results_sequential = eval_harness_sequential()
    results_async_multiprocessing = eval_harness_async_multiprocessing()
    results_multi_with_callback = eval_harness_multi_with_callback()

    assert len(results_sequential) == 2
    assert len(results_sequential[0]) == 2
    assert len(results_sequential[1]) == 2

    assert len(results_async_multiprocessing) == 2
    assert len(results_async_multiprocessing[0]) == 2
    assert len(results_async_multiprocessing[1]) == 2

    assert len(results_multi_with_callback) == 2
    assert len(results_multi_with_callback[0]) == 2
    assert len(results_multi_with_callback[1]) == 2

    # check that the results have been saved and are the same as the results from the sequential
    eval_ids = [
        fake_eval_subtract_two_numbers['metadata']['uuid'],
        fake_eval_sum_two_numbers_code_blocks_run['metadata']['uuid'],
    ]
    candidate_ids = ['candidate_1', 'candidate_2']
    for eval_index, eval_id in enumerate(eval_ids):
        for candidate_index, candidate_id in enumerate(candidate_ids):
            path = f'{dir_path}/result-{candidate_id}-{eval_id}.yaml'
            assert os.path.exists(path)
            with open(path) as f:
                result = yaml.safe_load(f)
            expected_dict = deepcopy(results_sequential[candidate_index][eval_index].to_dict())
            del expected_dict['total_time_seconds']
            del result['total_time_seconds']
            del expected_dict['timestamp']
            del result['timestamp']
            assert result == expected_dict

    # for each result, the dictionary (which contains eval, candidate, and results/checks) should
    # be the same (except for the total_time_seconds)
    s_dict = deepcopy(results_sequential[0][0].to_dict())
    del s_dict['total_time_seconds']
    del s_dict['timestamp']
    am_dict = deepcopy(results_async_multiprocessing[0][0].to_dict())
    del am_dict['total_time_seconds']
    del am_dict['timestamp']
    c_dict = deepcopy(results_multi_with_callback[0][0].to_dict())
    del c_dict['total_time_seconds']
    del c_dict['timestamp']
    assert s_dict == am_dict
    assert s_dict == c_dict

    s_dict = deepcopy(results_sequential[0][1].to_dict())
    del s_dict['total_time_seconds']
    del s_dict['timestamp']
    am_dict = deepcopy(results_async_multiprocessing[0][1].to_dict())
    del am_dict['total_time_seconds']
    del am_dict['timestamp']
    c_dict = deepcopy(results_multi_with_callback[0][1].to_dict())
    del c_dict['total_time_seconds']
    del c_dict['timestamp']
    assert s_dict == am_dict
    assert s_dict == c_dict

    s_dict = deepcopy(results_sequential[1][0].to_dict())
    del s_dict['total_time_seconds']
    del s_dict['timestamp']
    am_dict = deepcopy(results_async_multiprocessing[1][0].to_dict())
    del am_dict['total_time_seconds']
    del am_dict['timestamp']
    c_dict = deepcopy(results_multi_with_callback[1][0].to_dict())
    del c_dict['total_time_seconds']
    del c_dict['timestamp']
    assert s_dict == am_dict
    assert s_dict == c_dict

    s_dict = deepcopy(results_sequential[1][1].to_dict())
    del s_dict['total_time_seconds']
    del s_dict['timestamp']
    am_dict = deepcopy(results_async_multiprocessing[1][1].to_dict())
    del am_dict['total_time_seconds']
    del am_dict['timestamp']
    c_dict = deepcopy(results_multi_with_callback[1][1].to_dict())
    del c_dict['total_time_seconds']
    del c_dict['timestamp']
    assert s_dict == am_dict
    assert s_dict == c_dict

    # The underlying candidate objects should have the same values but should be different objects
    # because each candidate object (against a specific eval) is responsible for storing its own
    # history/conversation and the history should be different for each eval.
    assert results_sequential[0][0].candidate_obj == results_async_multiprocessing[0][0].candidate_obj  # noqa
    assert results_sequential[0][0].candidate_obj is not results_async_multiprocessing[0][0].candidate_obj  # noqa
    assert results_sequential[0][1].candidate_obj == results_async_multiprocessing[0][1].candidate_obj  # noqa
    assert results_sequential[0][1].candidate_obj is not results_async_multiprocessing[0][1].candidate_obj  # noqa
    assert results_sequential[1][0].candidate_obj == results_async_multiprocessing[1][0].candidate_obj  # noqa
    assert results_sequential[1][0].candidate_obj is not results_async_multiprocessing[1][0].candidate_obj  # noqa
    assert results_sequential[1][1].candidate_obj == results_async_multiprocessing[1][1].candidate_obj  # noqa
    assert results_sequential[1][1].candidate_obj is not results_async_multiprocessing[1][1].candidate_obj  # noqa

    # eval objects across candidates should have same values (same eval) but different objects
    assert results_sequential[0][0].eval_obj == results_async_multiprocessing[0][0].eval_obj
    assert results_sequential[0][0].eval_obj is not results_async_multiprocessing[0][0].eval_obj
    assert results_sequential[0][1].eval_obj == results_async_multiprocessing[0][1].eval_obj
    assert results_sequential[0][1].eval_obj is not results_async_multiprocessing[0][1].eval_obj
    assert results_sequential[1][0].eval_obj == results_async_multiprocessing[1][0].eval_obj
    assert results_sequential[1][0].eval_obj is not results_async_multiprocessing[1][0].eval_obj
    assert results_sequential[1][1].eval_obj == results_async_multiprocessing[1][1].eval_obj
    assert results_sequential[1][1].eval_obj is not results_async_multiprocessing[1][1].eval_obj


    # test the callback when running sequentially since it's a different execution path
    # there was actually a bug where we weren't passing in the callback
    eval_harness_sequential_with_callback = EvalHarness(
        evals=[subtract_config, sum_config],
        candidates=[candidate_1_dict, candidate_2_dict],
        num_cpus=1,
        async_batch_size=1,
        callback=callback,
    )
    recreate_temp_dir()
    eval_harness_sequential_with_callback()
    # check that the results have been saved and are the same as the results from the sequential
    eval_ids = [
        fake_eval_subtract_two_numbers['metadata']['uuid'],
        fake_eval_sum_two_numbers_code_blocks_run['metadata']['uuid'],
    ]
    candidate_ids = ['candidate_1', 'candidate_2']
    for eval_index, eval_id in enumerate(eval_ids):
        for candidate_index, candidate_id in enumerate(candidate_ids):
            path = f'{dir_path}/result-{candidate_id}-{eval_id}.yaml'
            assert os.path.exists(path)
            with open(path) as f:
                result = yaml.safe_load(f)
            expected_dict = deepcopy(results_sequential[candidate_index][eval_index].to_dict())
            del expected_dict['total_time_seconds']
            del result['total_time_seconds']
            del expected_dict['timestamp']
            del result['timestamp']
            assert result == expected_dict

    shutil.rmtree(dir_path)

def test__EvalHarness__adding_candidates_with_multi_value_parameters_should_create_multiple_candidates():  # noqa
    test_params = {
        'param_1': 'param_a',
        'param_2': 'param_b',
        'param_3': ['param_c', 'param_d', 'param_e'],
    }
    expected_params = [
        {'param_1': 'param_a', 'param_2': 'param_b', 'param_3': 'param_c'},
        {'param_1': 'param_a', 'param_2': 'param_b', 'param_3': 'param_d'},
        {'param_1': 'param_a', 'param_2': 'param_b', 'param_3': 'param_e'},
    ]
    candidate_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': 'MockCandidate',
        'responses': {'prompt_a': 'response1', 'prompt_b': 'response2'},
        'parameters': test_params,
    }
    eval_harness = EvalHarness()
    assert eval_harness.candidates == []
    eval_harness.add_candidate(candidate_dict)
    assert len(eval_harness.candidates) == 3
    for c, e in zip(eval_harness.candidates, expected_params):
        assert c.parameters == e
        assert c.responses == {'prompt_a': 'response1', 'prompt_b': 'response2'}
        assert c.metadata == {'uuid': 'candidate_1'}
    assert eval_harness.candidates[0].metadata is not eval_harness.candidates[1].metadata
    assert eval_harness.candidates[0].metadata is not eval_harness.candidates[2].metadata
    assert eval_harness.candidates[1].metadata is not eval_harness.candidates[2].metadata

    # ensure that the candidates are the same when added from yaml
    # save to yaml and load from yaml
    with open('__temp__.yaml', 'w') as f:
        yaml.dump(candidate_dict, f, default_flow_style=False, sort_keys=False)
    eval_harness_from_yaml = EvalHarness()
    try:
        eval_harness_from_yaml.add_candidate_from_yaml('__temp__.yaml')
    finally:
        os.remove('__temp__.yaml')
    assert eval_harness_from_yaml.candidates == eval_harness.candidates

def test__cannot_add_more_than_one_code_blocks_run_check():  # noqa
    eval_config = {
        'metadata': {'uuid': 'eval_1'},
        'test_sequence': [
            {
                'prompt': 'This is a prompt',
                'ideal_response': 'This is the ideal response',
                'checks': [
                    {
                        'check_type': CheckType.PYTHON_CODE_BLOCKS_PRESENT.name,
                    },
                ],
            },
            {
                'prompt': 'This is a prompt',
                'ideal_response': 'This is the ideal response',
                'checks': [
                    {
                        'check_type': CheckType.PYTHON_CODE_BLOCK_TESTS.name,
                        'code_tests': ['print("hello world")'],
                    },
                ],
            },
        ],
    }
    # this should work
    _ = Eval(**eval_config)
    # this should raise a ValueError because we are adding more than one code_blocks_run check
    # to a single test
    new_config = deepcopy(eval_config)
    new_config['test_sequence'][1]['checks'].append(
        {
            'check_type': CheckType.PYTHON_CODE_BLOCK_TESTS.name,
            'code_tests': ['print("hello world")'],
        },
    )
    with pytest.raises(ValueError):  # noqa: PT011
        Eval(**new_config)

    # this should raise a ValueError because we are adding more than one code_blocks_run check
    # across multiple tests
    new_config = deepcopy(eval_config)
    new_config['test_sequence'][0]['checks'].append(
        {
            'check_type': CheckType.PYTHON_CODE_BLOCK_TESTS.name,
            'code_tests': ['print("hello world")'],
        },
    )
    with pytest.raises(ValueError):  # noqa: PT011
        Eval(**new_config)
