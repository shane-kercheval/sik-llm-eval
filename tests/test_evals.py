"""Tests for the evals module."""
import pytest
from copy import deepcopy
import multiprocessing
import os
from textwrap import dedent
import yaml
from llm_eval.candidates import (
    CallableCandidate,
    Candidate,
    CandidateType,
    ChatModelCandidate,
    is_async_candidate,
)
from llm_eval.checks import (
    Check,
    CheckResult,
    CheckType,
    ContainsCheck,
    MatchCheck,
    PassFailResult,
    ScoreResult,
    ToolCallsCheck,
)
from llm_eval.eval import (
    Eval,
    EvalHarness,
    EvalResult,
    MultiEval,
    PromptComparison,
    PromptTest,
    ResponseError,
)
from llm_eval.llms.base import ChatModel
from llm_eval.llms.message_formatters import LlamaMessageFormatter, openai_message_formatter
from llm_eval.utilities.internal_utilities import extract_code_blocks
from tests.conftest import MockCandidate


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
    eval_obj = Eval(prompt_sequence=PromptTest(prompt='test'))
    eval_dict = eval_obj.to_dict()
    assert eval_dict == {'prompt_sequence': [{'prompt': 'test'}]}
    assert Eval(**eval_dict) == eval_obj
    assert str(eval_obj)

    eval_obj = Eval(
        prompt_sequence=[
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
    assert eval_obj.prompt_sequence[0].prompt == 'test1'
    assert eval_obj.prompt_sequence[0].ideal_response == 'test2'
    assert eval_obj.prompt_sequence[0].checks == [MatchCheck(value='test3')]
    assert eval_obj.prompt_sequence[1].prompt == 'test4'
    assert eval_obj.prompt_sequence[1].ideal_response == 'test5'
    assert eval_obj.prompt_sequence[1].checks == [
        MatchCheck(value='test6', metadata={'test': 'test7'}),
        ContainsCheck(value='test8'),
    ]
    assert str(eval_obj)

    eval_dict = eval_obj.to_dict()
    assert eval_dict == {
        'prompt_sequence': [
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
    assert eval_obj.prompt_sequence == eval_cloned.prompt_sequence
    assert eval_obj.prompt_sequence[0] == eval_cloned.prompt_sequence[0]
    assert eval_obj.prompt_sequence[0] is not eval_cloned.prompt_sequence[0]
    assert eval_obj.prompt_sequence[1] == eval_cloned.prompt_sequence[1]
    assert eval_obj.prompt_sequence[1] is not eval_cloned.prompt_sequence[1]

def test__Eval__call__result__to_from_dict():  # noqa
    """
    Tests the basic case of calling an Eval object and converting it to/from a dict. No checks are
    passed to the eval.
    """
    eval_obj = Eval(prompt_sequence=PromptTest(prompt='test'))
    # dict before call should be the same as after call
    assert eval_obj.to_dict() == {'prompt_sequence': [{'prompt': 'test'}]}
    assert Eval(**eval_obj.to_dict()) == eval_obj
    result = eval_obj(lambda x: f'response: {x}')
    assert result.response_characters == len('response: test')
    assert eval_obj.to_dict() == {'prompt_sequence': [{'prompt': 'test'}]}
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
    eval_obj = Eval(prompt_sequence={'prompt': prompt})
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
    assert result.get_code_block_tests_result() is None
    assert result.total_time_seconds >= 0

@pytest.mark.parametrize('use_async', [True, False])
def test__Eval__example_8f9fbf37__callable_candidate(use_async: bool, fake_eval_8f9fbf37: dict):  # noqa
    eval_dict = fake_eval_8f9fbf37.copy()
    eval_obj = Eval(**eval_dict)
    assert eval_obj.to_dict() == eval_dict

    responses = [
        "This is the first response",
        "This is a response with code blocks\n```python\nprint('hello world')\n```",
    ]
    def create_mock_llm(responses, use_async):  # noqa
        if use_async:
            response_iter = iter(responses)
            async def mock_llm(_: str):  # noqa
                try:
                    return next(response_iter)
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
    assert eval_result.responses == responses
    assert eval_result.prompts == [test.prompt for test in eval_obj.prompt_sequence]
    assert eval_result.ideal_responses == [
        test.ideal_response for test in eval_obj.prompt_sequence
    ]
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
        'metadata': {'function': 'def mock_llm(_: str)'},
        'candidate_type': CandidateType.CALLABLE_NO_SERIALIZE.name,
    }
    # check that the check result dicts match
    flatted_check_results = [r for tests in eval_result_dict['results'] for r in tests]
    assert flatted_check_results == [r.to_dict() for r in eval_result.all_check_results]
    assert eval_result.expects_code_blocks
    assert eval_result.get_code_block_tests_result() is None
    assert eval_result.total_time_seconds > 0
    # check that the eval_result_dict will recreate the exact eval_result object
    recreated_eval = EvalResult(**eval_result_dict)
    assert recreated_eval == eval_result
    assert recreated_eval.to_dict() == eval_result.to_dict()
    assert recreated_eval.eval_obj == eval_result.eval_obj
    assert recreated_eval.candidate_obj == eval_result.candidate_obj
    assert recreated_eval.results == eval_result.results
    flatted_checks = [r for test in eval_obj.prompt_sequence for r in test.checks]
    for c, r in zip(flatted_checks, eval_result.all_check_results, strict=True):
        assert c.check_type == r.metadata['check_type']
    assert eval_result.expects_code_blocks
    assert eval_result.get_code_block_tests_result() is None

def test__Eval__multiple_code_blocks__ensure_code_blocks_run(fake_eval_sum_two_numbers_code_blocks_run):  # noqa
    """
    Use Mock LLM with multiple code blocks (over multiple responses) to ensure code blocks run and
    the check results return the expected values.
    """
    config = fake_eval_sum_two_numbers_code_blocks_run.copy()
    eval_obj = Eval(**config)

    assert eval_obj.prompt_sequence[1].checks[-1].code_block_timeout == 5
    assert eval_obj.prompt_sequence[1].checks[-1].code_test_timeout == 5

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
    # save the eval_result to a file as string str(eval_result) to check formatting
    print(eval_result)
    with open('tests/eval/eval_result.txt', 'w') as f:
        f.write(str(eval_result))

    # we need to strip the code blocks of leading/trailing whitespace to compare them
    expected_config = deepcopy(config)
    expected_config['prompt_sequence'][1]['checks'][-1]['code_tests'] = [
        dedent(x.strip()) for x in
        expected_config['prompt_sequence'][1]['checks'][-1]['code_tests']
    ]
    assert eval_result.eval_obj.to_dict() == expected_config
    assert Eval(**eval_obj.to_dict()) == eval_obj
    # i need to compare strings because underlying error objects (i.e. instances) will not be same
    assert str(EvalResult(**eval_result.to_dict()).to_dict()) == str(eval_result.to_dict())

    assert eval_result.responses == responses
    assert eval_result.prompts == [test.prompt for test in eval_obj.prompt_sequence]
    assert eval_result.ideal_responses == [test.ideal_response for test in eval_obj.prompt_sequence]  # noqa
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
    assert eval_result.get_code_block_tests_result() is not None

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
    assert result.prompts[0] == eval_config['prompt_sequence'][0]['prompt']
    assert len(result.results) == 1
    assert result.cost == result.candidate_obj.cost
    assert 'cost' in result.to_dict()
    assert result.cost == result.to_dict()['cost']
    assert len(result.all_check_results) == 2
    assert result.expects_code_blocks
    assert result.get_code_block_tests_result() is None
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

@pytest.mark.skip("We removed the functionality where multiple model parameters (as a list) are supported. We should add this functionality back in with a new format/syntax. Let's keep this test for now. We can remove in the future if we don't think we will add the functionality back in.")  # noqa
def test__EvalHarness__add_multiple_candidates_from_single_dict(fake_eval_subtract_two_numbers, fake_eval_sum_two_numbers):  # noqa
    candidate_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': 'MockCandidateCannedResponse',
        'parameters': {
            'temperature': [0.0, 0.5, 1.0],
        },
    }
    eval_harness = EvalHarness(
        evals=[fake_eval_subtract_two_numbers, fake_eval_sum_two_numbers],
        candidates=candidate_dict,
        num_cpus=1, async_batch_size=1,
    )
    # should produce 3 candidates with same 2 evals (3 outer lists each having a list of 2 evals)
    results = eval_harness()
    assert len(results) == 3
    assert len(results[0]) == 2
    assert len(results[1]) == 2
    assert len(results[2]) == 2
    assert results[0][0].eval_obj == Eval(**fake_eval_subtract_two_numbers)
    assert results[0][0].candidate_obj.metadata == candidate_dict['metadata']
    assert results[0][0].candidate_obj.parameters == {'temperature': 0.0}
    assert results[0][1].eval_obj == Eval(**fake_eval_sum_two_numbers)
    assert results[0][1].candidate_obj.metadata == candidate_dict['metadata']
    assert results[0][1].candidate_obj.parameters == {'temperature': 0.0}

    assert results[1][0].eval_obj == Eval(**fake_eval_subtract_two_numbers)
    assert results[1][0].candidate_obj.metadata == candidate_dict['metadata']
    assert results[1][0].candidate_obj.parameters == {'temperature': 0.5}
    assert results[1][1].eval_obj == Eval(**fake_eval_sum_two_numbers)
    assert results[1][1].candidate_obj.metadata == candidate_dict['metadata']
    assert results[1][1].candidate_obj.parameters == {'temperature': 0.5}

    assert results[2][0].eval_obj == Eval(**fake_eval_subtract_two_numbers)
    assert results[2][0].candidate_obj.metadata == candidate_dict['metadata']
    assert results[2][0].candidate_obj.parameters == {'temperature': 1.0}
    assert results[2][1].eval_obj == Eval(**fake_eval_sum_two_numbers)
    assert results[2][1].candidate_obj.metadata == candidate_dict['metadata']
    assert results[2][1].candidate_obj.parameters == {'temperature': 1.0}

def test__multiline_eval__dedents_prompt():  # noqa
    prompt = """
        - This is a multiline prompt.
            - It needs to be dedented.
    """
    eval_obj = Eval(prompt_sequence=[PromptTest(prompt=prompt)])
    result = eval_obj(lambda x: x)
    assert result.responses[0] == dedent(prompt).lstrip()

def callback(x: EvalResult) -> None:
    """
    Test the callback function by saving the result to a yaml file in the 'test/temp'
    directory. Assume directory already exists.
    """
    candidate_id = x.candidate_obj.metadata['uuid']
    eval_id = x.eval_obj.metadata['uuid']
    with open(f'tests/__temp__/result-{candidate_id}-{eval_id}.yaml', 'w') as f:
        yaml.dump(x.to_dict(), f, default_flow_style=False, sort_keys=False)

@pytest.mark.skip("We removed the functionality where multiple model parameters (as a list) are supported. We should add this functionality back in with a new format/syntax. Let's keep this test for now. We can remove in the future if we don't think we will add the functionality back in.")  # noqa
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
    eval_harness.add_candidates(candidate_dict)
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

@pytest.mark.parametrize("candidate_type", ["AsyncMockCandidate", "MockCandidate"])
@pytest.mark.parametrize("num_cpus", [-1, 1])
@pytest.mark.parametrize("async_batch_size", [1, 50])
def test__async__EvalHarness__multiple_candidates__multiple_evals(candidate_type, num_cpus, async_batch_size, fake_eval_subtract_two_numbers, fake_eval_sum_two_numbers):  # noqa
    subtract_config = fake_eval_subtract_two_numbers.copy()
    sum_config = fake_eval_sum_two_numbers.copy()

    response_subtract_0 = 'This is the response.\n\n```\ndef subtract_two_numbers(a, b):\n    return a - b\n```'  # noqa
    response_subtract_1 = 'This is the assertion statement.\n\n```\nassert subtract_two_numbers(2, 3) == -1\n```'  # noqa
    response_sum_0 = 'This is the response.\n\n```\ndef sum_two_numbers(a, b):\n    return a + b\n```'  # noqa
    responses_lookup = {
        fake_eval_subtract_two_numbers['prompt_sequence'][0]['prompt']: response_subtract_0,
        fake_eval_subtract_two_numbers['prompt_sequence'][1]['prompt']: response_subtract_1,
        fake_eval_sum_two_numbers['prompt_sequence'][0]['prompt']: response_sum_0,
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
    assert eval_harness.evals != eval_harness_via_dicts.evals
    assert eval_harness.candidates != eval_harness_via_dicts.candidates
    eval_harness.add_evals(Eval(**subtract_config))
    eval_harness.add_evals(Eval(**sum_config))
    eval_harness.add_candidates(Candidate.from_dict(candidate_1_dict))
    eval_harness.add_candidates(Candidate.from_dict(candidate_2_dict))
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

def test__cannot_add_more_than_one_code_blocks_run_check():  # noqa
    eval_config = {
        'metadata': {'uuid': 'eval_1'},
        'prompt_sequence': [
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
    new_config['prompt_sequence'][1]['checks'].append(
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
    new_config['prompt_sequence'][0]['checks'].append(
        {
            'check_type': CheckType.PYTHON_CODE_BLOCK_TESTS.name,
            'code_tests': ['print("hello world")'],
        },
    )
    with pytest.raises(ValueError):  # noqa: PT011
        Eval(**new_config)

def test__evals__num_samples__greater_than_one__async__via_constructor(fake_eval_subtract_two_numbers, fake_eval_sum_two_numbers):  # noqa
    """Tests num_samples > 1 for async evals and when we pass num_samples to constructor."""
    subtract_config = fake_eval_subtract_two_numbers.copy()
    sum_config = fake_eval_sum_two_numbers.copy()

    response_subtract_0 = 'This is the response.\n\n```\ndef subtract_two_numbers(a, b):\n    return a - b\n```'  # noqa
    response_subtract_1 = 'This is the assertion statement.\n\n```\nassert subtract_two_numbers(2, 3) == -1\n```'  # noqa
    response_sum_0 = 'This is the response.\n\n```\ndef sum_two_numbers(a, b):\n    return a + b\n```'  # noqa
    responses_lookup = {
        fake_eval_subtract_two_numbers['prompt_sequence'][0]['prompt']: response_subtract_0,
        fake_eval_subtract_two_numbers['prompt_sequence'][1]['prompt']: response_subtract_1,
        fake_eval_sum_two_numbers['prompt_sequence'][0]['prompt']: response_sum_0,
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
    assert results[0][0].candidate_obj is not results[0][1].candidate_obj
    assert results[1][0].candidate_obj == results[1][1].candidate_obj
    assert results[1][0].candidate_obj is not results[1][1].candidate_obj

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
    assert cand_1_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_1_results_subtract.num_code_blocks == 2
    assert cand_1_results_subtract.num_checks == 5
    assert cand_1_results_subtract.num_successful_checks == 4
    assert cand_1_results_subtract.perc_successful_checks == 4 / 5

    cand_1_results_subtract = results[0][1]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_1_results_subtract.num_code_blocks == 2
    assert cand_1_results_subtract.num_checks == 5
    assert cand_1_results_subtract.num_successful_checks == 4
    assert cand_1_results_subtract.perc_successful_checks == 4 / 5

    cand_1_results_subtract = results[0][2]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_1_results_subtract.num_code_blocks == 2
    assert cand_1_results_subtract.num_checks == 5
    assert cand_1_results_subtract.num_successful_checks == 4
    assert cand_1_results_subtract.perc_successful_checks == 4 / 5

    # candidate 1 - sum eval; all 3 should have same results
    cand_1_results_sum = results[0][3]
    assert cand_1_results_sum.responses == [response_sum_0]
    assert cand_1_results_sum.num_code_blocks == 1
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1

    cand_1_results_sum = results[0][4]
    assert cand_1_results_sum.responses == [response_sum_0]
    assert cand_1_results_sum.num_code_blocks == 1
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1

    cand_1_results_sum = results[0][5]
    assert cand_1_results_sum.responses == [response_sum_0]
    assert cand_1_results_sum.num_code_blocks == 1
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1

    # candidate 2 - subtract eval; all 3 should have same results
    cand_2_results_subtract = results[1][0]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_2_results_subtract.num_code_blocks == 2
    assert cand_2_results_subtract.num_checks == 5
    assert cand_2_results_subtract.num_successful_checks == 4
    assert cand_2_results_subtract.perc_successful_checks == 4 / 5

    cand_2_results_subtract = results[1][1]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_2_results_subtract.num_code_blocks == 2
    assert cand_2_results_subtract.num_checks == 5
    assert cand_2_results_subtract.num_successful_checks == 4
    assert cand_2_results_subtract.perc_successful_checks == 4 / 5

    cand_2_results_subtract = results[1][2]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_2_results_subtract.num_code_blocks == 2
    assert cand_2_results_subtract.num_checks == 5
    assert cand_2_results_subtract.num_successful_checks == 4
    assert cand_2_results_subtract.perc_successful_checks == 4 / 5

    # candidate 2 - sum eval; all 3 should have same results
    cand_2_results_sum = results[1][3]
    assert cand_2_results_sum.responses == [response_sum_0]
    assert cand_2_results_sum.num_code_blocks == 1
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1

    cand_2_results_sum = results[1][4]
    assert cand_2_results_sum.responses == [response_sum_0]
    assert cand_2_results_sum.num_code_blocks == 1
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1

    cand_2_results_sum = results[1][5]
    assert cand_2_results_sum.responses == [response_sum_0]
    assert cand_2_results_sum.num_code_blocks == 1
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1

def test__evals__num_samples__greater_than_one__non_async__via_call(fake_eval_subtract_two_numbers, fake_eval_sum_two_numbers):  # noqa
    """Tests num_samples > 1 for non-async evals and when we pass num_samples to __call__()."""
    subtract_config = fake_eval_subtract_two_numbers.copy()
    sum_config = fake_eval_sum_two_numbers.copy()

    response_subtract_0 = 'This is the response.\n\n```\ndef subtract_two_numbers(a, b):\n    return a - b\n```'  # noqa
    response_subtract_1 = 'This is the assertion statement.\n\n```\nassert subtract_two_numbers(2, 3) == -1\n```'  # noqa
    response_sum_0 = 'This is the response.\n\n```\ndef sum_two_numbers(a, b):\n    return a + b\n```'  # noqa
    responses_lookup = {
        fake_eval_subtract_two_numbers['prompt_sequence'][0]['prompt']: response_subtract_0,
        fake_eval_subtract_two_numbers['prompt_sequence'][1]['prompt']: response_subtract_1,
        fake_eval_sum_two_numbers['prompt_sequence'][0]['prompt']: response_sum_0,
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

    num_samples = 3
    eval_harness = EvalHarness(
        num_cpus=1,
        async_batch_size=1,
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

    results = eval_harness(num_samples=num_samples)
    assert len(results) == num_candidates
    assert len(results[0]) == num_evals * num_samples
    assert len(results[1]) == num_evals * num_samples
    # The underlying candidate objects should have the same values but should be different objects
    # because each candidate object (against a specific eval) is responsible for storing its own
    # history/conversation and the history should be different for each eval.
    assert results[0][0].candidate_obj == results[0][1].candidate_obj
    assert results[0][0].candidate_obj is not results[0][1].candidate_obj
    assert results[1][0].candidate_obj == results[1][1].candidate_obj
    assert results[1][0].candidate_obj is not results[1][1].candidate_obj

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
    assert cand_1_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_1_results_subtract.num_code_blocks == 2
    assert cand_1_results_subtract.num_checks == 5
    assert cand_1_results_subtract.num_successful_checks == 4
    assert cand_1_results_subtract.perc_successful_checks == 4 / 5

    cand_1_results_subtract = results[0][1]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_1_results_subtract.num_code_blocks == 2
    assert cand_1_results_subtract.num_checks == 5
    assert cand_1_results_subtract.num_successful_checks == 4
    assert cand_1_results_subtract.perc_successful_checks == 4 / 5

    cand_1_results_subtract = results[0][2]
    cand_1_results_subtract.to_dict()
    assert cand_1_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_1_results_subtract.num_code_blocks == 2
    assert cand_1_results_subtract.num_checks == 5
    assert cand_1_results_subtract.num_successful_checks == 4
    assert cand_1_results_subtract.perc_successful_checks == 4 / 5

    # candidate 1 - sum eval; all 3 should have same results
    cand_1_results_sum = results[0][3]
    assert cand_1_results_sum.responses == [response_sum_0]
    assert cand_1_results_sum.num_code_blocks == 1
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1

    cand_1_results_sum = results[0][4]
    assert cand_1_results_sum.responses == [response_sum_0]
    assert cand_1_results_sum.num_code_blocks == 1
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1

    cand_1_results_sum = results[0][5]
    assert cand_1_results_sum.responses == [response_sum_0]
    assert cand_1_results_sum.num_code_blocks == 1
    assert cand_1_results_sum.num_checks == 2
    assert cand_1_results_sum.num_successful_checks == 2
    assert cand_1_results_sum.perc_successful_checks == 1

    # candidate 2 - subtract eval; all 3 should have same results
    cand_2_results_subtract = results[1][0]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_2_results_subtract.num_code_blocks == 2
    assert cand_2_results_subtract.num_checks == 5
    assert cand_2_results_subtract.num_successful_checks == 4
    assert cand_2_results_subtract.perc_successful_checks == 4 / 5

    cand_2_results_subtract = results[1][1]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_2_results_subtract.num_code_blocks == 2
    assert cand_2_results_subtract.num_checks == 5
    assert cand_2_results_subtract.num_successful_checks == 4
    assert cand_2_results_subtract.perc_successful_checks == 4 / 5

    cand_2_results_subtract = results[1][2]
    cand_2_results_subtract.to_dict()
    assert cand_2_results_subtract.responses == [response_subtract_0, response_subtract_1]
    assert cand_2_results_subtract.num_code_blocks == 2
    assert cand_2_results_subtract.num_checks == 5
    assert cand_2_results_subtract.num_successful_checks == 4
    assert cand_2_results_subtract.perc_successful_checks == 4 / 5

    # candidate 2 - sum eval; all 3 should have same results
    cand_2_results_sum = results[1][3]
    assert cand_2_results_sum.responses == [response_sum_0]
    assert cand_2_results_sum.num_code_blocks == 1
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1

    cand_2_results_sum = results[1][4]
    assert cand_2_results_sum.responses == [response_sum_0]
    assert cand_2_results_sum.num_code_blocks == 1
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1

    cand_2_results_sum = results[1][5]
    assert cand_2_results_sum.responses == [response_sum_0]
    assert cand_2_results_sum.num_code_blocks == 1
    assert cand_2_results_sum.num_checks == 2
    assert cand_2_results_sum.num_successful_checks == 2
    assert cand_2_results_sum.perc_successful_checks == 1

def test__Evals__sysem_message_previous_messages__load(fake_eval_with_previous_messages):  # noqa
    config = deepcopy(fake_eval_with_previous_messages)
    system_message = 'Custom System Message'
    assert config['system_message'] == system_message
    user_message_1 = 'User Message 1'
    assert config['previous_messages'][0]['user'] == user_message_1
    assistant_response_1 = 'Assistant Response 1'
    assert config['previous_messages'][0]['assistant'] == assistant_response_1
    user_message_2 = 'User Message 2'
    assert config['previous_messages'][1]['user'] == user_message_2
    assistant_response_2 = 'Assistant Response 2'
    assert config['previous_messages'][1]['assistant'] == assistant_response_2

    eval_obj = Eval(**config)
    assert eval_obj.system_message == system_message
    assert eval_obj.previous_messages[0]['user'] == user_message_1
    assert eval_obj.previous_messages[0]['assistant'] == assistant_response_1
    assert eval_obj.previous_messages[1]['user'] == user_message_2
    assert eval_obj.previous_messages[1]['assistant'] == assistant_response_2
    assert eval_obj.to_dict() == config
    assert Eval(**eval_obj.to_dict()) == eval_obj

def test__Evals__sysem_message_previous_messages__run_base_candidate(fake_eval_with_previous_messages):  # noqa
    """
    Tests the `system_message` and `previous_messages` options in the Eval object.

    These options should not effect other Evals (i.e. if the candidate is reused by other Evals).
    """
    config = deepcopy(fake_eval_with_previous_messages)
    initial_system_message = "Initial System Message"
    system_message = config['system_message']
    formatter = LlamaMessageFormatter()

    eval_response_1 = 'Eval Response 1'
    eval_response_2 = 'Eval Response 2'

    # actual formatted prompts send to the model
    actual_prompts = []
    class MockChatModel(ChatModel):
        def __init__(self):
            self.responses = [eval_response_1, eval_response_2]
            super().__init__(
                system_message=initial_system_message,
                message_formatter=formatter,
                token_calculator=len,
            )

        def _run(self, prompt: str) -> tuple[str | list[str] | object, dict]:
            actual_prompts.append(prompt)
            return self.responses.pop(0), {}

    @Candidate.register('MOCK_CHAT_CANDIDATE')
    class MockChatCandidate(ChatModelCandidate):
        def __init__(self):
            super().__init__(model=MockChatModel())

    ####
    # test that the system message and previous messages are set correctly for the candidate object
    ####
    try:
        candidate = MockChatCandidate()
        assert candidate.model.system_message == initial_system_message
        assert not candidate.model.chat_history
        assert not candidate.model._previous_messages

        eval_obj = Eval(**config)
        result = eval_obj(candidate)
        assert result.responses == [eval_response_1, eval_response_2]
        # system message and chat_history should not have changed; Candidate should be cloned and
        # not used directly so it can be reused for other evals without side effects from evals
        assert candidate.model.system_message == initial_system_message
        assert not candidate.model.chat_history
        assert not candidate.model._previous_messages

        # the cloned candidate that was used should have the system message and the expected chat
        # history
        assert result.candidate_obj.model.system_message == config['system_message']
        assert result.candidate_obj.model.chat_history[0].prompt == config['previous_messages'][0]['user']  # noqa
        assert result.candidate_obj.model.chat_history[0].response == config['previous_messages'][0]['assistant']  # noqa
        assert result.candidate_obj.model.chat_history[1].prompt == config['previous_messages'][1]['user']  # noqa
        assert result.candidate_obj.model.chat_history[1].response == config['previous_messages'][1]['assistant']  # noqa
        assert result.candidate_obj.model.chat_history[2].prompt == eval_obj.prompt_sequence[0].prompt  # noqa
        assert result.candidate_obj.model.chat_history[2].response == eval_response_1
        assert result.candidate_obj.model.chat_history[3].prompt == eval_obj.prompt_sequence[1].prompt  # noqa
        assert result.candidate_obj.model.chat_history[3].response == eval_response_2

        expected_prompt_1 = formatter(
            system_message=system_message,
            messages=config['previous_messages'],
            prompt=eval_obj.prompt_sequence[0].prompt,
        )
        assert actual_prompts[0] == expected_prompt_1
        # we now expected the "previous_messages" from the eval object plus the prompt on the eval
        # and the new response from the assistant
        expected_messages = config['previous_messages'] \
            + [{'user': config['prompt_sequence'][0]['prompt'], 'assistant': eval_response_1}]
        expected_prompt_2 = formatter(
            system_message=system_message,
            messages=expected_messages,
            prompt=eval_obj.prompt_sequence[1].prompt,
        )
        assert actual_prompts[1] == expected_prompt_2
        assert result.candidate_obj.model._previous_messages == expected_prompt_2
        # remove prompts from actual_prompts so we can test the next eval
        actual_prompts.clear()

        ####
        # Test that an eval without a system message or previous messages does not change (e.g. set
        # to null) the candidate's system message or previous messages; and test that the candidate
        # is cloned and unaffected by the eval and can be reused for other evals without side
        # effects
        ####
        # do not create a new candidate object; test that the candidate object is unaffected
        # candidate = MockChatCandidate()
        assert candidate.model.system_message == initial_system_message
        assert candidate.model._previous_messages is None
        assert not candidate.model.chat_history

        no_messages_config = deepcopy(config)
        del no_messages_config['system_message']
        del no_messages_config['previous_messages']
        eval_obj = Eval(**no_messages_config)
        result = eval_obj(candidate)
        assert result.responses == [eval_response_1, eval_response_2]
        # system message and chat_history should not have changed; Candidate should be cloned and
        # not used directly so it can be reused for other evals without side effects from evals
        assert candidate.model.system_message == initial_system_message
        assert not candidate.model.chat_history
        assert not candidate.model._previous_messages
        # the cloned candidate that was used should have the system message and the expected chat
        # history
        assert result.candidate_obj.model.system_message == initial_system_message
        assert result.candidate_obj.model.chat_history[0].prompt == eval_obj.prompt_sequence[0].prompt  # noqa
        assert result.candidate_obj.model.chat_history[0].response == eval_response_1
        assert result.candidate_obj.model.chat_history[1].prompt == eval_obj.prompt_sequence[1].prompt  # noqa
        assert result.candidate_obj.model.chat_history[1].response == eval_response_2
        expected_message = formatter(
            system_message=initial_system_message,
            messages=[],
            prompt=eval_obj.prompt_sequence[0].prompt,
        )
        assert actual_prompts[0] == expected_message
        expected_message = formatter(
            system_message=initial_system_message,
            messages=[(eval_obj.prompt_sequence[0].prompt, eval_response_1)],
            prompt=eval_obj.prompt_sequence[1].prompt,
        )
        assert result.candidate_obj.model._previous_messages == expected_message
        assert actual_prompts[1] == expected_message
    finally:
        # unregister the candidate class so it doesn't interfere with other tests
        Candidate.registry._registry.pop('MOCK_CHAT_CANDIDATE')

@pytest.fixture()
def eval_fixture(request):  # noqa
    return request.getfixturevalue(request.param)

@pytest.mark.parametrize(
        'eval_fixture',
        ['fake_eval_with_previous_messages', 'fake_eval_non_string_values'],
        indirect=True,
)
def test__Evals__sysem_message_previous_messages__run_base_candidate_with_EvalHarness(eval_fixture):  # noqa
    """
    Tests the `system_message` and `previous_messages` options in the Eval object but uses the
    EvalHarness to run the evals rather than calling the eval object directly.

    These options should not effect other Evals (i.e. if the candidate is reused by other Evals).
    """
    eval_config = deepcopy(eval_fixture)
    initial_system_message = "Initial System Message"
    system_message = eval_config['system_message']
    formatter = LlamaMessageFormatter()

    eval_response_1 = 'Eval Response 1'
    eval_response_2 = 'Eval Response 2'

    # actual formatted prompts send to the model
    actual_prompts = []
    class MockChatModel(ChatModel):
        def __init__(self):
            self.responses = [eval_response_1, eval_response_2]
            super().__init__(
                system_message=initial_system_message,
                message_formatter=formatter,
                token_calculator=len,
            )

        def _run(self, prompt: str) -> tuple[str | list[str] | object, dict]:
            actual_prompts.append(prompt)
            return self.responses.pop(0), {}

    @Candidate.register('MOCK_CHAT_CANDIDATE')
    class MockChatCandidate(ChatModelCandidate):
        def __init__(self):
            super().__init__(model=MockChatModel())

    try:
        ####
        # test that the system message and previous messages are set correctly for the candidate
        ####
        candidate = MockChatCandidate()
        assert candidate.model.system_message == initial_system_message
        assert not candidate.model.chat_history
        assert not candidate.model._previous_messages

        no_messages_config = deepcopy(eval_config)
        del no_messages_config['system_message']
        del no_messages_config['previous_messages']
        harness = EvalHarness(
            evals=[eval_config, no_messages_config],
            candidates=[candidate],
            num_cpus=1,
        )
        results = harness()
        result_with_messages = results[0][0]
        result_without_messages = results[0][1]
        assert result_with_messages.responses == [eval_response_1, eval_response_2]
        assert result_without_messages.responses == [eval_response_1, eval_response_2]

        # system message and chat_history should not have changed; Candidate should be cloned and
        # not used directly so it can be reused for other evals without side effects from evals
        assert candidate.model.system_message == initial_system_message
        assert not candidate.model.chat_history
        assert not candidate.model._previous_messages

        # the cloned candidate that was used should have the system message and the expected chat
        # history
        assert result_with_messages.candidate_obj.model.system_message == str(eval_config['system_message'])  # noqa
        assert result_with_messages.candidate_obj.model.chat_history[0].prompt == str(eval_config['previous_messages'][0]['user'])  # noqa
        assert result_with_messages.candidate_obj.model.chat_history[0].response == str(eval_config['previous_messages'][0]['assistant'])  # noqa
        assert result_with_messages.candidate_obj.model.chat_history[1].prompt == str(eval_config['previous_messages'][1]['user'])  # noqa
        assert result_with_messages.candidate_obj.model.chat_history[1].response == str(eval_config['previous_messages'][1]['assistant'])  # noqa
        assert result_with_messages.candidate_obj.model.chat_history[2].prompt == eval_config['prompt_sequence'][0]['prompt']  # noqa
        assert result_with_messages.candidate_obj.model.chat_history[2].response == eval_response_1
        assert result_with_messages.candidate_obj.model.chat_history[3].prompt == eval_config['prompt_sequence'][1]['prompt']  # noqa
        assert result_with_messages.candidate_obj.model.chat_history[3].response == eval_response_2

        formatted_messages = [
            {k:str(v) for k, v in x.items()}
            for x in eval_config['previous_messages']
        ]
        expected_prompt_1 = formatter(
            system_message=str(system_message),
            messages=formatted_messages,
            prompt=eval_config['prompt_sequence'][0]['prompt'],
        )
        assert actual_prompts[0] == expected_prompt_1
        # we now expected the "previous_messages" from the eval object plus the prompt on the eval
        # and the new response from the assistant
        expected_messages = [
            *formatted_messages,
            {'user': eval_config['prompt_sequence'][0]['prompt'], 'assistant': eval_response_1},
        ]
        expected_prompt_2 = formatter(
            system_message=system_message,
            messages=expected_messages,
            prompt=eval_config['prompt_sequence'][1]['prompt'],
        )
        assert actual_prompts[1] == expected_prompt_2
        assert result_with_messages.candidate_obj.model._previous_messages == expected_prompt_2

        ####
        # Test that an eval without a system message or previous messages does not change (e.g. set
        # to null) the candidate's system message or previous messages; and test that the candidate
        # is cloned and unaffected by the eval and can be reused for other evals without side
        # effects
        ####
        assert result_without_messages.candidate_obj.model.system_message == initial_system_message
        assert result_without_messages.candidate_obj.model.chat_history[0].prompt == eval_config['prompt_sequence'][0]['prompt']  # noqa
        assert result_without_messages.candidate_obj.model.chat_history[0].response == eval_response_1  # noqa
        assert result_without_messages.candidate_obj.model.chat_history[1].prompt == eval_config['prompt_sequence'][1]['prompt']  # noqa
        assert result_without_messages.candidate_obj.model.chat_history[1].response == eval_response_2  # noqa
        expected_message = formatter(
            system_message=initial_system_message,
            messages=[],
            prompt=eval_config['prompt_sequence'][0]['prompt'],
        )
        assert actual_prompts[2] == expected_message
        expected_message = formatter(
            system_message=initial_system_message,
            messages=[(eval_config['prompt_sequence'][0]['prompt'], eval_response_1)],
            prompt=eval_config['prompt_sequence'][1]['prompt'],
        )
        assert result_without_messages.candidate_obj.model._previous_messages == expected_message
        assert actual_prompts[3] == expected_message
    finally:
        # unregister the candidate class so it doesn't interfere with other tests
        Candidate.registry._registry.pop('MOCK_CHAT_CANDIDATE')

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__Evals__sysem_message_previous_messages__run_with_OpenAI(fake_eval_with_previous_messages, openai_candidate_template):  # noqa
    """
    Tests the `system_message` and `previous_messages` options in the Eval object.

    These options should not effect other Evals (i.e. if the candidate is reused by other Evals).
    """
    eval_config = fake_eval_with_previous_messages.copy()
    candidate = Candidate.from_dict(openai_candidate_template)
    initial_system_message = candidate.model.system_message
    assert not candidate.model.chat_history
    assert not candidate.model._previous_messages

    formatter = openai_message_formatter

    ####
    # test that the system message and previous messages are set correctly for the candidate object
    ####
    eval_obj = Eval(**eval_config)
    result = eval_obj(candidate)
    assert len(result.responses) == 2
    # system message and chat_history should not have changed; Candidate should be cloned and
    # not used directly so it can be reused for other evals without side effects from evals
    assert candidate.model.system_message == initial_system_message
    assert not candidate.model.chat_history
    assert not candidate.model._previous_messages

    # the cloned candidate that was used should have the system message and the expected chat
    # history
    assert result.candidate_obj.model.system_message == eval_config['system_message']
    assert result.candidate_obj.model.chat_history[0].prompt == eval_config['previous_messages'][0]['user']  # noqa
    assert result.candidate_obj.model.chat_history[0].response == eval_config['previous_messages'][0]['assistant']  # noqa
    assert result.candidate_obj.model.chat_history[1].prompt == eval_config['previous_messages'][1]['user']  # noqa
    assert result.candidate_obj.model.chat_history[1].response == eval_config['previous_messages'][1]['assistant']  # noqa
    assert result.candidate_obj.model.chat_history[2].prompt == eval_obj.prompt_sequence[0].prompt
    assert result.candidate_obj.model.chat_history[2].response == result.responses[0]
    assert result.candidate_obj.model.chat_history[3].prompt == eval_obj.prompt_sequence[1].prompt
    assert result.candidate_obj.model.chat_history[3].response == result.responses[1]

    expected_prompt_1 = formatter(
        system_message=eval_config['system_message'],
        messages=eval_config['previous_messages'],
        prompt=eval_obj.prompt_sequence[0].prompt,
    )
    # the left side of the comparison is the actual prompt sent to the model
    assert result.candidate_obj.model.chat_history[2].metadata['messages'] == expected_prompt_1

    # we now expected the "previous_messages" from the eval object plus the prompt on the eval
    # and the new response from the assistant
    expected_messages = eval_config['previous_messages'] \
        + [{'user': eval_config['prompt_sequence'][0]['prompt'], 'assistant': result.responses[0]}]
    expected_prompt_2 = formatter(
        system_message=eval_config['system_message'],
        messages=expected_messages,
        prompt=eval_obj.prompt_sequence[1].prompt,
    )
    # the left side of the comparison is the actual prompt sent to the model
    assert result.candidate_obj.model.chat_history[3].metadata['messages'] == expected_prompt_2
    assert result.candidate_obj.model._previous_messages == expected_prompt_2

    ####
    # Test that an eval without a system message or previous messages does not change (e.g. set to
    # null) the candidate's system message or previous messages; and test that the candidate is
    # cloned and unaffected by the eval and can be reused for other evals without side effects
    ####
    # do not create a new candidate object; we want to test that the candidate object is unaffected
    # candidate = MockChatCandidate()
    assert candidate.model.system_message == initial_system_message
    assert candidate.model._previous_messages is None
    assert not candidate.model.chat_history

    no_messages_config = deepcopy(eval_config)
    del no_messages_config['system_message']
    del no_messages_config['previous_messages']
    eval_obj = Eval(**no_messages_config)
    result = eval_obj(candidate)
    assert len(result.responses) == 2
    # system message and chat_history should not have changed; Candidate should be cloned and
    # not used directly so it can be reused for other evals without side effects from evals
    assert candidate.model.system_message == initial_system_message
    assert not candidate.model.chat_history
    assert not candidate.model._previous_messages
    # the cloned candidate that was used should have the system message and the expected chat
    # history
    assert result.candidate_obj.model.system_message == initial_system_message
    assert result.candidate_obj.model.chat_history[0].prompt == eval_obj.prompt_sequence[0].prompt
    assert result.candidate_obj.model.chat_history[0].response == result.responses[0]
    assert result.candidate_obj.model.chat_history[1].prompt == eval_obj.prompt_sequence[1].prompt
    assert result.candidate_obj.model.chat_history[1].response == result.responses[1]
    expected_message = formatter(
        system_message=initial_system_message,
        messages=[],
        prompt=eval_obj.prompt_sequence[0].prompt,
    )
    # the left side of the comparison is the actual prompt sent to the model
    assert result.candidate_obj.model.chat_history[0].metadata['messages'] == expected_message

    expected_message = formatter(
        system_message=initial_system_message,
        messages=[(eval_obj.prompt_sequence[0].prompt, result.responses[0])],
        prompt=eval_obj.prompt_sequence[1].prompt,
    )
    # the left side of the comparison is the actual prompt sent to the model
    assert result.candidate_obj.model.chat_history[1].metadata['messages'] == expected_message
    assert result.candidate_obj.model._previous_messages == expected_message

def test__Eval_with_previous_messages_not_in_correct_format_raise_exception():  # noqa
    # test that we can create a basic eval object (so we can test the exception)
    eval_obj = Eval(prompt_sequence=[{'prompt': 'Prompt 1'}])
    assert eval_obj
    eval_obj = Eval(
        prompt_sequence=[{'prompt': 'Prompt 1'}],
        previous_messages=[{'user': 'message', 'assistant': 'message'}],
    )
    assert eval_obj
    eval_obj = Eval(
        prompt_sequence=[{'prompt': 'Prompt 1'}],
        previous_messages=[('User 1', 'Assistant 1')],
    )
    assert eval_obj
    with pytest.raises(AssertionError):
        Eval(
            prompt_sequence=[{'prompt': 'Prompt 1'}],
            # missing user key
            previous_messages=[{'assistant': 'message'}],
        )
    with pytest.raises(AssertionError):
        Eval(
            prompt_sequence=[{'prompt': 'Prompt 1'}],
            # missing assistant key
            previous_messages=[{'user': 'message'}],
        )
    with pytest.raises(AssertionError):
        Eval(
            prompt_sequence=[{'prompt': 'Prompt 1'}],
            # invalid user key
            previous_messages=[{'user 1': 'User 1', 'assistant': 'Assistant 1'}],
        )
    with pytest.raises(AssertionError):
        Eval(
            prompt_sequence=[{'prompt': 'Prompt 1'}],
            # invalid assistant key
            previous_messages=[{'user': 'User 1', 'assistant 1': 'Assistant 1'}],
        )
    # test tuples
    with pytest.raises(AssertionError):
        Eval(
            prompt_sequence=[{'prompt': 'Prompt 1'}],
            # only 1 item in tuple
            previous_messages=[('User 1')],
        )
    # test tuples
    with pytest.raises(AssertionError):
        Eval(
            prompt_sequence=[{'prompt': 'Prompt 1'}],
            # 3 items in tuple
            previous_messages=[('User 1', 'Message 1', 'Extra')],
        )

def test__Eval_with_numeric_values_loads_correctly(fake_eval_non_string_values):  # noqa
    """Test that numeric values are converted to strings when loading an Eval object."""
    eval_config = deepcopy(fake_eval_non_string_values)
    eval_obj = Eval(**eval_config)
    assert Eval(**eval_obj.to_dict()) == eval_obj
    assert isinstance(eval_obj.system_message, str)
    assert eval_obj.system_message == str(eval_config['system_message'])
    assert isinstance(eval_obj.previous_messages, list)
    assert isinstance(eval_obj.previous_messages[0], dict)
    assert isinstance(eval_obj.previous_messages[0]['user'], str)
    assert eval_obj.previous_messages[0]['user'] == str(eval_config['previous_messages'][0]['user'])  # noqa
    assert isinstance(eval_obj.previous_messages[0]['assistant'], str)
    assert eval_obj.previous_messages[0]['assistant'] == str(eval_config['previous_messages'][0]['assistant'])  # noqa
    assert isinstance(eval_obj.previous_messages[1], dict)
    assert isinstance(eval_obj.previous_messages[1]['user'], str)
    assert eval_obj.previous_messages[1]['user'] == str(eval_config['previous_messages'][1]['user'])  # noqa
    assert isinstance(eval_obj.previous_messages[1]['assistant'], str)
    assert eval_obj.previous_messages[1]['assistant'] == str(eval_config['previous_messages'][1]['assistant'])  # noqa
    assert isinstance(eval_obj.prompt_sequence[0].prompt, int)
    assert eval_obj.prompt_sequence[0].prompt == eval_config['prompt_sequence'][0]['prompt']
    assert isinstance(eval_obj.prompt_sequence[1].prompt, int)
    assert eval_obj.prompt_sequence[1].prompt == eval_config['prompt_sequence'][1]['prompt']

error_list_manager = multiprocessing.Manager()
test_harness_callback_errors = error_list_manager.list()
def error_callback(exception: Exception, eval_obj: Eval, candidate_obj: Candidate) -> None:
    """
    This is a callback function that will be called when an error occurs in the EvalHarness. It
    needs to be defined outside of the test so that it can be pickled and used in multiprocessing
    when testing num_cpus != 1. `test_harness_callback_errors` is a global variable that is defined
    in the test and will be used to store the errors that occur in the callback.
    """  # noqa
    test_harness_callback_errors.append((exception, eval_obj, candidate_obj))

@pytest.mark.parametrize('num_cpus', [1, None])
def test__EvalHarness__candidate_has_error_generating_response_multi_processing(num_cpus, fake_eval_sum_two_numbers_code_blocks_run):  # noqa
    """
    Tests that the EvalHarness captures errors generated by the candidate. If no error_callback
    is set, the harness should raise the error and stop processing the evals. If an error_callback
    is set, the harness should call the error_callback and the candidate object and continue
    processing the remaining evals.
    """
    eval_config = deepcopy(fake_eval_sum_two_numbers_code_blocks_run)
    prompt_1 = eval_config['prompt_sequence'][0]['prompt']
    prompt_2 = eval_config['prompt_sequence'][1]['prompt']
    response_1 = '```\ndef sum_two_numbers(a, b): return a+b\n```'
    response_2 = '```\nCode Block 2\n```'
    candidate_1 = MockCandidate(
        metadata={'name': 'candidate_1'},
        responses={
            prompt_1: ValueError('Fake Rate Limit Error for prompt_1'),
            prompt_2: response_2,
        },
    )
    with pytest.raises(ValueError, match='Fake Rate Limit Error for prompt_1'):
        candidate_1(prompt_1)
    assert candidate_1(prompt_2) == response_2
    candidate_2 = MockCandidate(
        metadata={'name': 'candidate_2'},
        responses={
            prompt_1: response_1,
            prompt_2: ValueError('Fake Rate Limit Error for prompt_2'),
        },
    )
    assert candidate_2(prompt_1) == response_1
    with pytest.raises(ValueError, match='Fake Rate Limit Error for prompt_2'):
        candidate_2(prompt_2)

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

    global test_harness_callback_errors  # noqa
    if num_cpus == 1:
        # NOTE: not happy with this solution but it works for now
        test_harness_callback_errors = []
        def local_error_callback(exception: Exception, eval_obj: Eval, candidate_obj: Candidate) -> None:  # noqa: E501
            test_harness_callback_errors.append((exception, eval_obj, candidate_obj))
        harness.error_callback = local_error_callback
    else:
        # test_harness_callback_errors = error_list_manager.list()
        harness.error_callback = error_callback
    results = harness()
    assert len(results) == 2
    errors = list(test_harness_callback_errors)
    assert len(errors) == len(harness.evals) * len(harness.candidates)
    if num_cpus != 1:
        # if num_cpus is not 1, the order of the errors is not guaranteed
        # sort errors by candidate name so we can compare them
        errors = sorted(errors, key=lambda x: x[2].metadata['name'])
    assert errors[0][0].args[0] == 'Fake Rate Limit Error for prompt_1'
    assert errors[0][0].args[0] == results[0][0].harness_exception.args[0]
    assert errors[0][1] == eval_1
    assert errors[0][2].metadata['name'] == candidate_1.metadata['name']
    assert errors[1][0].args[0] == 'Fake Rate Limit Error for prompt_1'
    assert errors[1][0].args[0] == results[0][1].harness_exception.args[0]
    assert errors[1][1] == eval_2
    assert errors[1][2].metadata['name'] == candidate_1.metadata['name']
    assert errors[2][0].args[0] == 'Fake Rate Limit Error for prompt_2'
    assert errors[2][0].args[0] == results[1][0].harness_exception.args[0]
    assert errors[2][1] == eval_1
    assert errors[2][2].metadata['name'] == candidate_2.metadata['name']
    assert errors[3][0].args[0] == 'Fake Rate Limit Error for prompt_2'
    assert errors[3][0].args[0] == results[1][1].harness_exception.args[0]
    assert errors[3][1] == eval_2
    assert errors[3][2].metadata['name'] == candidate_2.metadata['name']

    # test that the CheckResult objects have the correct values (should be failing)
    # in the first two evals, the first prompt should fail and the second prompt should pass
    # so no code was generated
    expected_num_checks = len(eval_config['prompt_sequence'][0]['checks']) \
        + len(eval_config['prompt_sequence'][1]['checks'])
    expected_num_code_tests = len(eval_config['prompt_sequence'][-1]['checks'][-1]['code_tests'])
    for i in range(2):
        assert not any(x.success for x in results[0][i].all_check_results)
        assert results[0][i].num_checks == expected_num_checks
        assert results[0][i].num_successful_checks == 0
        # no code block were generated because the first prompt failed
        assert results[0][i].num_code_blocks == 0
        assert results[0][i].get_num_code_blocks_successful() == 0
        assert expected_num_code_tests > 1
        assert results[0][i].get_num_code_tests_defined() == expected_num_code_tests
        assert results[0][i].get_num_code_tests_successful() == 0

    # in the second two evals, the first prompt should pass and the second prompt should fail
    # so code was generated for the first prompt
    for i in range(2):
        assert results[1][i].all_check_results[0].success  # checks for sum_two_numbers
        assert not results[1][i].all_check_results[1].success
        assert not results[1][i].all_check_results[2].success
        assert results[1][i].all_check_results[3].success  # checks for code blocks
        assert not results[1][i].all_check_results[4].success  # checks that response contains sum_two_numbers but no response was returned  # noqa
        assert not results[1][i].all_check_results[5].success
        assert not results[1][i].all_check_results[6].success

        assert results[1][i].num_checks == expected_num_checks
        assert results[1][i].num_successful_checks == 2
        # code block were generated because the first prompt passed
        assert results[1][i].num_code_blocks == 1
        assert results[1][i].get_num_code_blocks_successful() == 1
        assert expected_num_code_tests > 1
        assert results[1][i].get_num_code_tests_defined() == expected_num_code_tests
        # There was actually code generated on the first prompt and for example,
        # the first test is sum_two_numbers(2, 3) == 5
        assert results[1][i].get_num_code_tests_successful() > 0

@pytest.mark.parametrize('eval_fixture', [
    'fake_multi_eval',
    'fake_multi_eval_non_string_values',
], indirect=True)
def test__PromptComparison(eval_fixture):  # noqa
    expected_prompt_parameters = eval_fixture['prompt_comparison']['prompt_parameters']
    expected_prompts = eval_fixture['prompt_comparison']['prompts']
    expected_checks = eval_fixture['prompt_comparison']['checks']
    expected_ideal_response = eval_fixture['prompt_comparison']['ideal_response']
    config = deepcopy(eval_fixture)

    prompt_parameters = config['prompt_comparison']['prompt_parameters']
    prompts = config['prompt_comparison']['prompts']
    checks = config['prompt_comparison']['checks']
    ideal_response = config['prompt_comparison']['ideal_response']

    # test with prompt as list of dicts and prompt_parameters is None
    prompt_comparison = PromptComparison(
        prompts=prompts,
        prompt_parameters=None,
        checks=checks,
        ideal_response=ideal_response,
    )
    prompt_tests = prompt_comparison()
    assert len(prompt_tests) == len(expected_prompts)
    for i, prompt_test in enumerate(prompt_tests):
        assert prompt_test.prompt == dedent(str(expected_prompts[i]['prompt']))
        # all PromptTests should share the same checks and ideal_response
        assert isinstance(prompt_test.checks, list)
        assert len(prompt_test.checks) == len(expected_checks)
        assert isinstance(prompt_test.checks[0], Check)
        assert prompt_test.to_dict()['checks'] == expected_checks
        assert prompt_test.ideal_response == str(expected_ideal_response)

    # test with prompt as list of strings and prompt_parameters is None
    prompt_comparison = PromptComparison(
        prompts=[x['prompt'] for x in prompts],
        prompt_parameters=None,
        checks=checks,
        ideal_response=ideal_response,
    )
    prompt_tests = prompt_comparison()
    assert len(prompt_tests) == len(expected_prompts)
    for i, prompt_test in enumerate(prompt_tests):
        assert prompt_test.prompt == dedent(str(expected_prompts[i]['prompt']))
        # all PromptTests should share the same checks and ideal_response
        assert isinstance(prompt_test.checks, list)
        assert len(prompt_test.checks) == len(expected_checks)
        assert isinstance(prompt_test.checks[0], Check)
        assert prompt_test.to_dict()['checks'] == expected_checks
        assert prompt_test.ideal_response == str(expected_ideal_response)

    # test with prompt as list of dicts and prompt_parameters is not None
    prompt_comparison = PromptComparison(
        prompts=prompts,
        prompt_parameters=prompt_parameters,
        checks=checks,
        ideal_response=ideal_response,
    )
    prompt_tests = prompt_comparison()
    assert len(prompt_tests) == len(expected_prompts)
    for i, prompt_test in enumerate(prompt_tests):
        assert prompt_test.prompt == dedent(str(expected_prompts[i]['prompt'])).\
            format(**expected_prompt_parameters)
        # all PromptTests should share the same checks and ideal_response
        assert isinstance(prompt_test.checks, list)
        assert len(prompt_test.checks) == len(expected_checks)
        assert isinstance(prompt_test.checks[0], Check)
        assert prompt_test.to_dict()['checks'] == expected_checks
        assert prompt_test.ideal_response == str(expected_ideal_response)

    # test with prompt as list of strings and prompt_parameters is not None
    prompt_comparison = PromptComparison(
        prompts=[x['prompt'] for x in prompts],
        prompt_parameters=prompt_parameters,
        checks=checks,
        ideal_response=ideal_response,
    )
    prompt_tests = prompt_comparison()
    assert len(prompt_tests) == len(expected_prompts)
    for i, prompt_test in enumerate(prompt_tests):
        assert prompt_test.prompt == dedent(str(expected_prompts[i]['prompt'])).\
            format(**expected_prompt_parameters)
        # all PromptTests should share the same checks and ideal_response
        assert isinstance(prompt_test.checks, list)
        assert len(prompt_test.checks) == len(expected_checks)
        assert isinstance(prompt_test.checks[0], Check)
        assert prompt_test.to_dict()['checks'] == expected_checks
        assert prompt_test.ideal_response == str(expected_ideal_response)

    # test with prompt as list of dicts and prompt_parameters is not None
    prompt_comparison = PromptComparison(
        prompts=prompts,
        prompt_parameters=prompt_parameters,
        checks=None,
        ideal_response=None,
    )
    prompt_tests = prompt_comparison()
    assert len(prompt_tests) == len(expected_prompts)
    for i, prompt_test in enumerate(prompt_tests):
        assert prompt_test.prompt == dedent(str(expected_prompts[i]['prompt'])).\
            format(**expected_prompt_parameters)
        # all PromptTests should share the same checks and ideal_response
        assert isinstance(prompt_test.checks, list)
        assert len(prompt_test.checks) == 0
        assert prompt_test.ideal_response is None
    # ensure we didn't change config
    assert config == eval_fixture

def test__PromptComparison__prompt_parameters():  # noqa
    comparison = PromptComparison(
        prompts=[
            'Prompt 1: {param1}',
            'Prompt 2: {param2} | {param1}',
            'Prompt 2: {param2}',
            'Prompt 3: No pameters',
            '{param3}',
        ],
        prompt_parameters={
            'param1': 'Param 1',
            'param2': 2,
            'param3': False,
            'param4': 'not used',
        },
    )
    prompt_tests = comparison()
    assert len(prompt_tests) == 5
    assert prompt_tests[0].prompt == 'Prompt 1: Param 1'
    assert prompt_tests[1].prompt == 'Prompt 2: 2 | Param 1'
    assert prompt_tests[2].prompt == 'Prompt 2: 2'
    assert prompt_tests[3].prompt == 'Prompt 3: No pameters'
    assert prompt_tests[4].prompt == 'False'

@pytest.mark.parametrize('eval_fixture', [
    'fake_eval_8f9fbf37',
    'fake_eval_subtract_two_numbers',
    'fake_eval_sum_two_numbers',
    'fake_eval_sum_two_numbers_code_blocks_run',
    'fake_eval_no_code_blocks',
    'fake_eval_with_previous_messages',
    'fake_eval_non_string_values',
], indirect=True)
def test__MultiEval__with_regular_eval(eval_fixture):  # noqa
    """
    Check that an equivalent Eval object is created when using MultiEval with a single Eval
    congig/yaml.
    """
    config = deepcopy(eval_fixture)
    evals = MultiEval.from_dict(config)()
    assert len(evals) == 1
    assert evals[0] == Eval(**config)
    # evals[0].to_dict() == Eval(**config).to_dict()
    # ensure we didn't change config
    assert config == eval_fixture

@pytest.mark.parametrize('eval_fixture', [
    'fake_multi_eval',
    'fake_multi_eval_non_string_values',
], indirect=True)
def test__MultiEval(eval_fixture):  # noqa
    expected_metadata = eval_fixture['metadata']
    expected_system_messages = [str(x) for x in eval_fixture['system_message']]
    expected_previous_messages = eval_fixture['previous_messages']
    expected_previous_messages = [
        [{k:str(v) for k, v in x.items()} for x in prev]
        for prev in expected_previous_messages
    ]
    expected_prompt_parameters = eval_fixture['prompt_comparison']['prompt_parameters']
    expected_prompts = eval_fixture['prompt_comparison']['prompts']
    expected_checks = eval_fixture['prompt_comparison']['checks']
    expected_ideal_response = eval_fixture['prompt_comparison']['ideal_response']

    config = deepcopy(eval_fixture)
    multi_eval = MultiEval.from_dict(config)
    evals = multi_eval()

    # all combinations
    assert len(evals) == len(expected_system_messages) * len(expected_prompts) * len(expected_previous_messages)  # noqa
    # all evals should have the same metadata, checks, and ideal_response
    for eval_ in evals:
        assert eval_.metadata == expected_metadata
        assert eval_.to_dict()['prompt_sequence'][0]['checks'] == expected_checks
        assert eval_.prompt_sequence[0].ideal_response == str(expected_ideal_response)
    # NOTE: this logic depends on the order that the combinations are created which is currently
    # system_message, then previous_messages, then prompt
    assert evals[0].system_message == expected_system_messages[0]
    assert evals[0].previous_messages == expected_previous_messages[0]
    assert evals[0].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[1].system_message == expected_system_messages[0]
    assert evals[1].previous_messages == expected_previous_messages[0]
    assert evals[1].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[2].system_message == expected_system_messages[0]
    assert evals[2].previous_messages == expected_previous_messages[0]
    assert evals[2].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[3].system_message == expected_system_messages[0]
    assert evals[3].previous_messages == expected_previous_messages[1]
    assert evals[3].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[4].system_message == expected_system_messages[0]
    assert evals[4].previous_messages == expected_previous_messages[1]
    assert evals[4].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[5].system_message == expected_system_messages[0]
    assert evals[5].previous_messages == expected_previous_messages[1]
    assert evals[5].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[6].system_message == expected_system_messages[1]
    assert evals[6].previous_messages == expected_previous_messages[0]
    assert evals[6].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[7].system_message == expected_system_messages[1]
    assert evals[7].previous_messages == expected_previous_messages[0]
    assert evals[7].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[8].system_message == expected_system_messages[1]
    assert evals[8].previous_messages == expected_previous_messages[0]
    assert evals[8].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[9].system_message == expected_system_messages[1]
    assert evals[9].previous_messages == expected_previous_messages[1]
    assert evals[9].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[10].system_message == expected_system_messages[1]
    assert evals[10].previous_messages == expected_previous_messages[1]
    assert evals[10].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[11].system_message == expected_system_messages[1]
    assert evals[11].previous_messages == expected_previous_messages[1]
    assert evals[11].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
        format(**expected_prompt_parameters)

    # ensure we didn't change config
    assert config == eval_fixture

def test_MultiEval_with_prompt_sequence(fake_multi_eval_with_prompt_sequence):  # noqa
    """
    Rather than a prompt_comparison key, the MultiEval object has a prompt_sequence key, which
    indicates we are testing a list of PromptTests rather than comparing prompts.
    """
    expected_metadata = fake_multi_eval_with_prompt_sequence['metadata']
    expected_system_messages = [str(x) for x in fake_multi_eval_with_prompt_sequence['system_message']]  # noqa
    expected_previous_messages = fake_multi_eval_with_prompt_sequence['previous_messages']
    expected_previous_messages = [
        [{k:str(v) for k, v in x.items()} for x in prev]
        for prev in expected_previous_messages
    ]
    expected_prompt_sequence = fake_multi_eval_with_prompt_sequence['prompt_sequence']

    config = deepcopy(fake_multi_eval_with_prompt_sequence)
    multi_eval = MultiEval.from_dict(config)
    evals = multi_eval()

    # all combinations
    assert len(evals) == len(expected_system_messages) * len(expected_previous_messages)
    # all evals should have the same metadata
    for eval_ in evals:
        eval_.metadata == expected_metadata

    # NOTE: this logic depends on the order that the combinations are created which is currently
    # system_message, then previous_messages, then prompt
    assert evals[0].system_message == expected_system_messages[0]
    assert evals[0].previous_messages == expected_previous_messages[0]
    assert evals[0].prompt_sequence[0] == PromptTest(**expected_prompt_sequence[0])
    assert evals[0].prompt_sequence[1] == PromptTest(**expected_prompt_sequence[1])

    assert evals[1].system_message == expected_system_messages[0]
    assert evals[1].previous_messages == expected_previous_messages[1]
    assert evals[1].prompt_sequence[0] == PromptTest(**expected_prompt_sequence[0])
    assert evals[1].prompt_sequence[1] == PromptTest(**expected_prompt_sequence[1])

    assert evals[2].system_message == expected_system_messages[1]
    assert evals[2].previous_messages == expected_previous_messages[0]
    assert evals[2].prompt_sequence[0] == PromptTest(**expected_prompt_sequence[0])
    assert evals[2].prompt_sequence[1] == PromptTest(**expected_prompt_sequence[1])

    assert evals[3].system_message == expected_system_messages[1]
    assert evals[3].previous_messages == expected_previous_messages[1]
    assert evals[3].prompt_sequence[0] == PromptTest(**expected_prompt_sequence[0])
    assert evals[3].prompt_sequence[1] == PromptTest(**expected_prompt_sequence[1])
    # ensure we didn't change config
    assert config == fake_multi_eval_with_prompt_sequence

def test__MultiEval__system_message_edge_cases__prompts__list_PromptTest(fake_multi_eval):  # noqa
    """A list of PromptTest/dict objects assumes a single Eval with multiple sequential prompts."""
    prompts = [
        PromptTest(
            prompt='Prompt 1',
            checks = [ContainsCheck(value='_')],
        ),
        PromptTest(prompt='Prompt 2'),
    ]
    expected_metadata = fake_multi_eval['metadata']
    expected_system_messages = fake_multi_eval['system_message'][0]
    expected_previous_messages = fake_multi_eval['previous_messages']
    expected_previous_messages = [
        [{k:str(v) for k, v in x.items()} for x in prev]
        for prev in expected_previous_messages
    ]
    config = deepcopy(fake_multi_eval)
    tests = [
        prompts,
        [p.to_dict() for p in prompts],
    ]
    for test in tests:
        multi_eval = MultiEval(
            prompts=test,
            system_message=config['system_message'][0],
            previous_messages=config['previous_messages'][0],
            metadata=config['metadata'],
        )
        evals = multi_eval()
        # all combinations; only 1 Eval object
        assert len(evals) == 1
        assert evals[0].metadata == expected_metadata
        assert evals[0].system_message == expected_system_messages
        assert evals[0].previous_messages == expected_previous_messages[0]
        assert evals[0].prompt_sequence[0] == prompts[0]
        assert evals[0].prompt_sequence[1] == prompts[1]
        # ensure we didn't change config
        assert config == fake_multi_eval

def test__MultiEval__system_message_edge_cases__prompts__PromptComparison(fake_multi_eval):  # noqa
    """
    A single PromptComparison object assumes multiple Eval objects with the same system_message,
    etc. but different prompts. Also testing a single dict object representing a PromptComparison.
    """
    prompt_comparison_dict = fake_multi_eval['prompt_comparison']
    prompt_comparison_obj = PromptComparison(**prompt_comparison_dict)

    expected_metadata = fake_multi_eval['metadata']
    expected_system_messages = [str(x) for x in fake_multi_eval['system_message']]
    expected_previous_messages = fake_multi_eval['previous_messages']
    expected_previous_messages = [
        [{k:str(v) for k, v in x.items()} for x in prev]
        for prev in expected_previous_messages
    ]
    config = deepcopy(fake_multi_eval)

    expected_prompt_1 = prompt_comparison_dict['prompts'][0]['prompt'].\
            format(**prompt_comparison_dict['prompt_parameters'])
    expected_prompt_2 = prompt_comparison_dict['prompts'][1]['prompt'].\
            format(**prompt_comparison_dict['prompt_parameters'])
    expected_prompt_3 = prompt_comparison_dict['prompts'][2]['prompt'].\
            format(**prompt_comparison_dict['prompt_parameters'])

    for test in [prompt_comparison_dict, prompt_comparison_obj]:
        multi_eval = MultiEval(
            prompts=test,
            system_message=config['system_message'],
            previous_messages=config['previous_messages'][0],
            metadata=config['metadata'],
        )
        evals = multi_eval()
        # all combinations; only 1 Eval object
        assert len(evals) == len(expected_system_messages) * len(prompt_comparison_dict['prompts'])
        # all evals should have the same metadata, previous_messages (only 1 set),
        # ideal_response, and checks
        for eval_ in evals:
            assert eval_.metadata == expected_metadata
            assert eval_.previous_messages == expected_previous_messages[0]
            assert eval_.to_dict()['prompt_sequence'][0]['checks'] == prompt_comparison_dict['checks']  # noqa
            assert eval_.prompt_sequence[0].ideal_response == str(prompt_comparison_dict['ideal_response'])  # noqa

        assert evals[0].system_message == expected_system_messages[0]
        assert evals[0].prompt_sequence[0].prompt == expected_prompt_1
        assert evals[1].system_message == expected_system_messages[0]
        assert evals[1].prompt_sequence[0].prompt == expected_prompt_2
        assert evals[2].system_message == expected_system_messages[0]
        assert evals[2].prompt_sequence[0].prompt == expected_prompt_3
        assert evals[3].system_message == expected_system_messages[1]
        assert evals[3].prompt_sequence[0].prompt == expected_prompt_1
        assert evals[4].system_message == expected_system_messages[1]
        assert evals[4].prompt_sequence[0].prompt == expected_prompt_2
        assert evals[5].system_message == expected_system_messages[1]
        assert evals[5].prompt_sequence[0].prompt == expected_prompt_3
        # ensure we didn't change config
        assert config == fake_multi_eval

def test__MultiEval__system_message_edge_cases__prompts__PromptTest(fake_multi_eval_with_prompt_sequence):  # noqa
    """
    A single PromptTest object (or dict representing PromptTest) assumes a single Eval object.
    Multiple previous_messages should still generate multiple Evals.
    """
    prompt_test_dict = fake_multi_eval_with_prompt_sequence['prompt_sequence'][0]
    prompt_test_obj = PromptTest(**prompt_test_dict)

    expected_metadata = fake_multi_eval_with_prompt_sequence['metadata']
    expected_system_messages = [str(x) for x in fake_multi_eval_with_prompt_sequence['system_message']]  # noqa
    expected_previous_messages = fake_multi_eval_with_prompt_sequence['previous_messages']
    expected_previous_messages = [
        [{k:str(v) for k, v in x.items()} for x in prev]
        for prev in expected_previous_messages
    ]
    config = deepcopy(fake_multi_eval_with_prompt_sequence)


    for test in [prompt_test_dict, prompt_test_obj]:
        multi_eval = MultiEval(
            prompts=test,
            system_message=config['system_message'],
            previous_messages=config['previous_messages'],
            metadata=config['metadata'],
        )
        evals = multi_eval()
        # all combinations; only 1 Eval object
        assert len(evals) == len(expected_system_messages) * len(expected_previous_messages)
        # all evals should have the same metadata, and PromptTest
        for eval_ in evals:
            assert eval_.metadata == expected_metadata
            assert eval_.prompt_sequence[0].prompt == prompt_test_dict['prompt']
            assert eval_.to_dict()['prompt_sequence'][0]['checks'] == prompt_test_dict['checks']

        assert evals[0].system_message == expected_system_messages[0]
        assert evals[0].previous_messages == expected_previous_messages[0]
        assert evals[1].system_message == expected_system_messages[0]
        assert evals[1].previous_messages == expected_previous_messages[1]
        assert evals[2].system_message == expected_system_messages[1]
        assert evals[2].previous_messages == expected_previous_messages[0]
        assert evals[3].system_message == expected_system_messages[1]
        assert evals[3].previous_messages == expected_previous_messages[1]
        # ensure we didn't change config
        assert config == fake_multi_eval_with_prompt_sequence

@pytest.mark.parametrize('eval_fixture', [
    'fake_multi_eval',
    'fake_multi_eval_non_string_values',
], indirect=True)
def test__MultiEval__only_prompt_comparison(eval_fixture):  # noqa
    expected_prompt_parameters = eval_fixture['prompt_comparison']['prompt_parameters']
    expected_prompts = eval_fixture['prompt_comparison']['prompts']
    expected_checks = eval_fixture['prompt_comparison']['checks']
    expected_ideal_response = eval_fixture['prompt_comparison']['ideal_response']
    config = deepcopy(eval_fixture)

    multi_eval = MultiEval(
        prompts=config['prompt_comparison'],
        system_message=None,
        previous_messages=None,
        metadata=None,
    )
    evals = multi_eval()
    # all combinations
    assert len(evals) == len(expected_prompts)
    for eval_ in evals:
        assert not eval_.metadata
        assert not eval_.system_message
        assert not eval_.previous_messages
        assert eval_.to_dict()['prompt_sequence'][0]['checks'] == expected_checks
        assert eval_.prompt_sequence[0].ideal_response == str(expected_ideal_response)

    expected_prompt_1 = str(expected_prompts[0]['prompt']).format(**expected_prompt_parameters)
    expected_prompt_2 = str(expected_prompts[1]['prompt']).format(**expected_prompt_parameters)
    expected_prompt_3 = str(expected_prompts[2]['prompt']).format(**expected_prompt_parameters)
    assert evals[0].prompt_sequence[0].prompt == expected_prompt_1
    assert evals[1].prompt_sequence[0].prompt == expected_prompt_2
    assert evals[2].prompt_sequence[0].prompt == expected_prompt_3
    # ensure we didn't change config
    assert config == eval_fixture

@pytest.mark.parametrize('previous_messages', [
    [
        {'user': 'Question 1', 'assistant': 'Response 1'},
        {'user': 'Question 2', 'assistant': 'Response 2'},
    ],
    [
        ('Question 1', 'Response 1'),
        ('Question 2', 'Response 2'),
    ],
])
def test__MultiEval__previous_messages__list_of_dict_tuple(previous_messages, fake_multi_eval_with_prompt_sequence):  # noqa
    """Test that previous_messages can be a list of dictionaries or tuples."""
    prompt_test_dict = fake_multi_eval_with_prompt_sequence['prompt_sequence']
    expected_metadata = fake_multi_eval_with_prompt_sequence['metadata']
    expected_system_messages = [str(x) for x in fake_multi_eval_with_prompt_sequence['system_message']]  # noqa
    config = deepcopy(fake_multi_eval_with_prompt_sequence)

    multi_eval = MultiEval(
        prompts=prompt_test_dict,
        system_message=config['system_message'],
        previous_messages=previous_messages,
        metadata=config['metadata'],
    )
    evals = multi_eval()
    # all combinations; 2 system messages but only 1 *set* of PromptTests (dict) and
    # only 1 *set* of previous messages
    assert len(evals) == len(expected_system_messages)
    # all evals should have the same metadata, and PromptTest
    for eval_ in evals:
        assert eval_.metadata == expected_metadata
        assert eval_.prompt_sequence[0].prompt == prompt_test_dict[0]['prompt']
        assert eval_.to_dict()['prompt_sequence'][0]['checks'] == prompt_test_dict[0]['checks']
        assert eval_.prompt_sequence[1].prompt == prompt_test_dict[1]['prompt']
        # last check has code_tests and whitespace is stripped so can't directly compare and this
        # functionality is tested elsewhere so just check the length
        assert len(eval_.to_dict()['prompt_sequence'][1]['checks']) == len(prompt_test_dict[1]['checks'])  # noqa

    expected_previous_messages = previous_messages
    if isinstance(previous_messages[0], tuple):
        expected_previous_messages = [
            {'user': x[0], 'assistant': x[1]} for x in previous_messages
        ]
    assert evals[0].system_message == expected_system_messages[0]
    assert evals[0].previous_messages == expected_previous_messages
    assert evals[1].system_message == expected_system_messages[1]
    assert evals[1].previous_messages == expected_previous_messages
    # ensure we didn't change config
    assert config == fake_multi_eval_with_prompt_sequence

@pytest.mark.parametrize('previous_messages', [
    [
        [
            {'user': 'Question 1', 'assistant': 'Response 1'},
            {'user': 'Question 2', 'assistant': 'Response 2'},
        ],
        [
            {'user': 'Question 3', 'assistant': 'Response 3'},
            {'user': 'Question 4', 'assistant': 'Response 4'},
        ],
    ],
    [
        [
            ('Question 1', 'Response 1'),
            ('Question 2', 'Response 2'),
        ],
        [
            ('Question 3', 'Response 3'),
            ('Question 4', 'Response 4'),
        ],
    ],
])
def test__MultiEval__previous_messages__list_of_list_of_dict_tuple(previous_messages, fake_multi_eval_with_prompt_sequence):  # noqa
    """
    Test that previous_messages can be a list of list of dictionaries or tuples, which will create
    multiple Evals.
    """
    prompt_test_dict = fake_multi_eval_with_prompt_sequence['prompt_sequence']
    expected_metadata = fake_multi_eval_with_prompt_sequence['metadata']
    expected_system_messages = [str(x) for x in fake_multi_eval_with_prompt_sequence['system_message']]  # noqa
    config = deepcopy(fake_multi_eval_with_prompt_sequence)

    multi_eval = MultiEval(
        prompts=prompt_test_dict,
        system_message=config['system_message'],
        previous_messages=previous_messages,
        metadata=config['metadata'],
    )
    evals = multi_eval()
    # all combinations; 2 system messages but only 1 *set* of PromptTests (dict) and
    # only 1 *set* of previous messages
    assert len(evals) == len(expected_system_messages) * len(previous_messages)
    # all evals should have the same metadata, and PromptTest
    for eval_ in evals:
        assert eval_.metadata == expected_metadata
        assert eval_.prompt_sequence[0].prompt == prompt_test_dict[0]['prompt']
        assert eval_.to_dict()['prompt_sequence'][0]['checks'] == prompt_test_dict[0]['checks']
        assert eval_.prompt_sequence[1].prompt == prompt_test_dict[1]['prompt']
        # last check has code_tests and whitespace is stripped so can't directly compare and this
        # functionality is tested elsewhere so just check the length
        assert len(eval_.to_dict()['prompt_sequence'][1]['checks']) == len(prompt_test_dict[1]['checks'])  # noqa

    expected_previous_messages = previous_messages
    if isinstance(previous_messages[0][0], tuple):
        expected_previous_messages = [
        [{'user': x[0], 'assistant': x[1]} for x in message_set]
        for message_set in previous_messages
        ]
    assert evals[0].system_message == expected_system_messages[0]
    assert evals[0].previous_messages == expected_previous_messages[0]
    assert evals[1].system_message == expected_system_messages[0]
    assert evals[1].previous_messages == expected_previous_messages[1]
    assert evals[2].system_message == expected_system_messages[1]
    assert evals[2].previous_messages == expected_previous_messages[0]
    assert evals[3].system_message == expected_system_messages[1]
    assert evals[3].previous_messages == expected_previous_messages[1]
    # ensure we didn't change config
    assert config == fake_multi_eval_with_prompt_sequence

@pytest.mark.parametrize('eval_fixture', [
    'fake_multi_eval',
    'fake_multi_eval_non_string_values',
], indirect=True)
def test__EvalHarness__MultEval_object_and_dict(eval_fixture, fake_eval_with_previous_messages):  # noqa
    expected_metadata = eval_fixture['metadata']
    expected_system_messages = [str(x) for x in eval_fixture['system_message']]
    expected_prompt_parameters = eval_fixture['prompt_comparison']['prompt_parameters']
    expected_previous_messages = eval_fixture['previous_messages']
    expected_previous_messages = [
        [{k:str(v) for k, v in x.items()} for x in prev]
        for prev in expected_previous_messages
    ]
    prompt_comparison = eval_fixture['prompt_comparison']
    expected_prompts = prompt_comparison['prompts']
    expected_num_evals_from_prompt_comparison = len(expected_system_messages) \
        * len(expected_previous_messages) \
        * len(prompt_comparison['prompts'])
    config = deepcopy(eval_fixture)

    harness = EvalHarness(num_cpus=1, async_batch_size=1)
    harness.add_evals(MultiEval.from_dict(config))
    assert len(harness.evals) == expected_num_evals_from_prompt_comparison

    evals = harness.evals
    # all evals should have the same metadata, checks, and ideal_response
    for eval_ in evals:
        assert eval_.metadata == expected_metadata
        assert eval_.to_dict()['prompt_sequence'][0]['checks'] == prompt_comparison['checks']
        assert eval_.prompt_sequence[0].ideal_response == str(prompt_comparison['ideal_response'])
    # NOTE: this logic depends on the order that the combinations are created which is currently
    # system_message, then previous_messages, then prompt
    assert evals[0].system_message == expected_system_messages[0]
    assert evals[0].previous_messages == expected_previous_messages[0]
    assert evals[0].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[1].system_message == expected_system_messages[0]
    assert evals[1].previous_messages == expected_previous_messages[0]
    assert evals[1].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[2].system_message == expected_system_messages[0]
    assert evals[2].previous_messages == expected_previous_messages[0]
    assert evals[2].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[3].system_message == expected_system_messages[0]
    assert evals[3].previous_messages == expected_previous_messages[1]
    assert evals[3].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[4].system_message == expected_system_messages[0]
    assert evals[4].previous_messages == expected_previous_messages[1]
    assert evals[4].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[5].system_message == expected_system_messages[0]
    assert evals[5].previous_messages == expected_previous_messages[1]
    assert evals[5].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[6].system_message == expected_system_messages[1]
    assert evals[6].previous_messages == expected_previous_messages[0]
    assert evals[6].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[7].system_message == expected_system_messages[1]
    assert evals[7].previous_messages == expected_previous_messages[0]
    assert evals[7].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[8].system_message == expected_system_messages[1]
    assert evals[8].previous_messages == expected_previous_messages[0]
    assert evals[8].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[9].system_message == expected_system_messages[1]
    assert evals[9].previous_messages == expected_previous_messages[1]
    assert evals[9].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[10].system_message == expected_system_messages[1]
    assert evals[10].previous_messages == expected_previous_messages[1]
    assert evals[10].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
        format(**expected_prompt_parameters)

    assert evals[11].system_message == expected_system_messages[1]
    assert evals[11].previous_messages == expected_previous_messages[1]
    assert evals[11].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
        format(**expected_prompt_parameters)
    # ensure we didn't change config
    assert config == eval_fixture

    harness.add_evals(fake_eval_with_previous_messages)
    assert len(harness.evals) == expected_num_evals_from_prompt_comparison + 1

    harness.add_evals(config)
    assert len(harness.evals) == (expected_num_evals_from_prompt_comparison * 2) + 1

    candidate_1_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': 'MockCandidateCannedResponse',
    }
    candidate_2_dict = deepcopy(candidate_1_dict)
    candidate_2_dict['metadata']['uuid'] = 'candidate_2'
    harness.add_candidates([candidate_1_dict, candidate_2_dict])
    assert len(harness.evals) == (expected_num_evals_from_prompt_comparison * 2) + 1
    results = harness()
    assert len(results) == 2
    assert len(results[0]) == (expected_num_evals_from_prompt_comparison * 2) + 1
    assert len(results[1]) == (expected_num_evals_from_prompt_comparison * 2) + 1

@pytest.mark.parametrize('eval_fixture', [
    'fake_multi_eval',
    'fake_multi_eval_non_string_values',
], indirect=True)
def test__EvalHarness__list_MultEval_object_and_dict__from_constructor(eval_fixture, fake_eval_with_previous_messages):  # noqa
    expected_metadata = eval_fixture['metadata']
    expected_system_messages = [str(x) for x in eval_fixture['system_message']]
    expected_prompt_parameters = eval_fixture['prompt_comparison']['prompt_parameters']
    expected_previous_messages = eval_fixture['previous_messages']
    expected_previous_messages = [
        [{k:str(v) for k, v in x.items()} for x in prev]
        for prev in expected_previous_messages
    ]
    prompt_comparison = eval_fixture['prompt_comparison']
    expected_prompts = prompt_comparison['prompts']
    expected_num_evals_from_prompt_comparison = len(expected_system_messages) \
        * len(expected_previous_messages) \
        * len(prompt_comparison['prompts'])
    config = deepcopy(eval_fixture)

    harness = EvalHarness(
        evals=[config, MultiEval.from_dict(config)],
        num_cpus=1, async_batch_size=1,
    )

    assert len(harness.evals) == expected_num_evals_from_prompt_comparison * 2
    evals = harness.evals
    # all evals should have the same metadata, checks, and ideal_response
    for eval_ in evals:
        assert eval_.metadata == expected_metadata
        assert eval_.to_dict()['prompt_sequence'][0]['checks'] == prompt_comparison['checks']
        assert eval_.prompt_sequence[0].ideal_response == str(prompt_comparison['ideal_response'])
    # NOTE: this logic depends on the order that the combinations are created which is currently
    # system_message, then previous_messages, then prompt
    for i in range(2):
        i = i * 12  # add either 0 or 12 to the index  # noqa
        assert evals[0+i].system_message == expected_system_messages[0]
        assert evals[0+i].previous_messages == expected_previous_messages[0]
        assert evals[0+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[1+i].system_message == expected_system_messages[0]
        assert evals[1+i].previous_messages == expected_previous_messages[0]
        assert evals[1+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[2+i].system_message == expected_system_messages[0]
        assert evals[2+i].previous_messages == expected_previous_messages[0]
        assert evals[2+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[3+i].system_message == expected_system_messages[0]
        assert evals[3+i].previous_messages == expected_previous_messages[1]
        assert evals[3+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[4+i].system_message == expected_system_messages[0]
        assert evals[4+i].previous_messages == expected_previous_messages[1]
        assert evals[4+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[5+i].system_message == expected_system_messages[0]
        assert evals[5+i].previous_messages == expected_previous_messages[1]
        assert evals[5+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[6+i].system_message == expected_system_messages[1]
        assert evals[6+i].previous_messages == expected_previous_messages[0]
        assert evals[6+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[7+i].system_message == expected_system_messages[1]
        assert evals[7+i].previous_messages == expected_previous_messages[0]
        assert evals[7+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[1]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[8+i].system_message == expected_system_messages[1]
        assert evals[8+i].previous_messages == expected_previous_messages[0]
        assert evals[8+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[2]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[9+i].system_message == expected_system_messages[1]
        assert evals[9+i].previous_messages == expected_previous_messages[1]
        assert evals[9+i].prompt_sequence[0].prompt == dedent(str(expected_prompts[0]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[10+i].system_message == expected_system_messages[1]
        assert evals[10+i].previous_messages == expected_previous_messages[1]
        assert evals[10+i].prompt_sequence[0].prompt == \
            dedent(str(expected_prompts[1]['prompt'])).\
            format(**expected_prompt_parameters)

        assert evals[11+i].system_message == expected_system_messages[1]
        assert evals[11+i].previous_messages == expected_previous_messages[1]
        assert evals[11+i].prompt_sequence[0].prompt == \
            dedent(str(expected_prompts[2]['prompt'])).\
            format(**expected_prompt_parameters)
    # ensure we didn't change config
    assert config == eval_fixture

    harness.add_evals(fake_eval_with_previous_messages)
    assert len(harness.evals) == (expected_num_evals_from_prompt_comparison * 2) + 1

    candidate_1_dict = {
        'metadata': {'uuid': 'candidate_1'},
        'candidate_type': 'MockCandidateCannedResponse',
    }
    candidate_2_dict = deepcopy(candidate_1_dict)
    candidate_2_dict['metadata']['uuid'] = 'candidate_2'
    harness.add_candidates([candidate_1_dict, candidate_2_dict])
    assert len(harness.evals) == (expected_num_evals_from_prompt_comparison * 2) + 1
    results = harness()
    assert len(results) == 2
    assert len(results[0]) == (expected_num_evals_from_prompt_comparison * 2) + 1
    assert len(results[1]) == (expected_num_evals_from_prompt_comparison * 2) + 1

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
        self.has_executed = False
        self.response = response

    def __call__(self, prompt: dict) -> dict:  # noqa
        # Candidates should not be reused in the same Eval
        if self.has_executed:
            raise Exception('Candidate should not be called more than once')
        self.has_executed = True
        # returns dictionary instead of string
        return {'prompt': prompt, 'response': self.response}

    def set_system_message(self, system_message: str) -> None:  # noqa
        pass

    def set_message_history(self, messages: list[dict] | list[tuple]) -> None:  # noqa
        pass

    def clone(self) -> 'UnregisteredCandidate':  # noqa
        return UnregisteredCandidate(response=self.response)

def test__Eval__unregistered_check__unregistered_candidate__non_string_prompt_and_response():  # noqa
    """
    We should be able to use unregistered Check and Candidate classes with non-string prompts and
    responses. These classes won't be able to be saved/loaded from a dictionary, and so we can't
    use them with EvalHarness, but we should be able to use them individually.
    """
    eval_ = Eval(
        prompt_sequence=PromptTest(
            prompt={'prompt': 'Test Prompt'},
            checks=[UnregisteredCheck()],
        ),
    )
    assert eval_.to_dict() == {'prompt_sequence': [{'prompt': {'prompt': 'Test Prompt'}, 'checks': [{'check_type': 'UnregisteredCheck'}]}]}  # noqa
    assert UnregisteredCandidate(42).to_dict() == {'candidate_type': 'UnregisteredCandidate'}
    result = eval_(UnregisteredCandidate(42))
    assert len(result.responses) == 1
    assert result.responses[0] == {'prompt': {'prompt': 'Test Prompt'}, 'response': 42}
    assert len(result.results) == 1
    assert len(result.results[0]) == 1
    check_result = result.results[0][0]
    assert check_result.value == {'prompt': {'prompt': 'Test Prompt'}, 'response': 42}
    assert check_result.success is True
    assert check_result.to_dict() == {'value': {'prompt': {'prompt': 'Test Prompt'}, 'response': 42}, 'success': True, 'result_type': 'UnregisteredCheckResult'}  # noqa
    assert len(result.all_check_results) == 1
    assert result.perc_successful_checks == 1
    assert result.all_check_results[0].value == {'prompt': {'prompt': 'Test Prompt'}, 'response': 42}  # noqa
    assert result.all_check_results[0].success is True
    assert result.prompts == [{'prompt': 'Test Prompt'}]
    assert result.response_characters is None  # only applicable for string responses
    assert result.characters_per_second is None  # only applicable for string responses
    assert result.expects_code_blocks is False
    assert result.get_code_block_tests_result() is None
    assert result.get_num_code_blocks_successful() is None
    assert result.get_num_code_tests_defined() is None
    assert result.get_num_code_tests_successful() is None
    # ensure that we can convert the results (which contain unregistered checks/candidates) to
    # a string and dictionary (which call underlying str and to_dict methods on
    # checks/candidates)
    assert len(str(result)) > 10
    assert result.to_dict()['eval_obj'] == eval_.to_dict()
    assert result.to_dict()['candidate_obj'] == UnregisteredCandidate(42).to_dict()
    assert result.to_dict()['results'][0][0] == check_result.to_dict()

def test__EvalHarness__unregistered_check__unregistered_candidate__non_string_prompt_and_response():  # noqa
        harness = EvalHarness(
            # num_cpus=1, async_batch_size=1,
            evals=[
                Eval(
                    prompt_sequence=PromptTest(
                        prompt={'prompt': 'Test Prompt 1'},  # test with dictionary prompt
                        checks=[UnregisteredCheck()],
                    ),
                ),
                Eval(
                    prompt_sequence=PromptTest(
                        prompt={'prompt': 'Test Prompt 2'},  # test with dictionary prompt
                        checks=[UnregisteredCheck()],
                    ),
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
        assert results[0][0].responses == [{'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 1'}]  # noqa
        assert results[0][1].responses == [{'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 1'}]  # noqa
        assert results[1][0].responses == [{'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 2'}]  # noqa
        assert results[1][1].responses == [{'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 2'}]  # noqa
        assert len(results[0][0].all_check_results) == 1
        assert results[0][0].perc_successful_checks == 1
        assert results[0][0].all_check_results[0].value == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 1'}  # noqa
        assert results[0][0].all_check_results[0].success is True
        assert len(results[0][1].all_check_results) == 1
        assert results[0][1].perc_successful_checks == 1
        assert results[0][1].all_check_results[0].value == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 1'}  # noqa
        assert results[0][1].all_check_results[0].success is True
        assert len(results[1][0].all_check_results) == 1
        assert results[1][0].perc_successful_checks == 1
        assert results[1][0].all_check_results[0].value == {'prompt': {'prompt': 'Test Prompt 1'}, 'response': 'Response 2'}  # noqa
        assert results[1][0].all_check_results[0].success is True
        assert len(results[1][1].all_check_results) == 1
        assert results[1][1].perc_successful_checks == 1
        assert results[1][1].all_check_results[0].value == {'prompt': {'prompt': 'Test Prompt 2'}, 'response': 'Response 2'}  # noqa
        assert results[1][1].all_check_results[0].success is True

        assert results[0][0].prompts == [{'prompt': 'Test Prompt 1'}]
        assert results[0][1].prompts == [{'prompt': 'Test Prompt 2'}]
        assert results[1][0].prompts == [{'prompt': 'Test Prompt 1'}]
        assert results[1][1].prompts == [{'prompt': 'Test Prompt 2'}]

        # if these work on the first result, they should work on the rest
        assert results[0][0].response_characters is None  # only applicable for string responses
        assert results[0][0].characters_per_second is None  # only applicable for string responses
        assert results[0][0].expects_code_blocks is False
        assert results[0][0].get_code_block_tests_result() is None
        assert results[0][0].get_num_code_blocks_successful() is None
        assert results[0][0].get_num_code_tests_defined() is None
        assert results[0][0].get_num_code_tests_successful() is None
        # ensure that we can convert the results (which contain unregistered checks/candidates) to
        # a string and dictionary (which call underlying str and to_dict methods on
        # checks/candidates)
        assert len(str(results[0][0])) > 10
        assert results[0][0].to_dict()['eval_obj'] == harness.evals[0].to_dict()
        assert results[0][0].to_dict()['candidate_obj'] == harness.candidates[0].to_dict()
        assert results[0][0].to_dict()['results'][0][0] == results[0][0].results[0][0].to_dict()
        assert results[0][1].to_dict()['eval_obj'] == harness.evals[1].to_dict()
        assert results[0][1].to_dict()['candidate_obj'] == harness.candidates[0].to_dict()
        assert results[0][1].to_dict()['results'][0][0] == results[0][1].results[0][0].to_dict()
        assert results[1][0].to_dict()['eval_obj'] == harness.evals[0].to_dict()
        assert results[1][0].to_dict()['candidate_obj'] == harness.candidates[1].to_dict()
        assert results[1][0].to_dict()['results'][0][0] == results[1][0].results[0][0].to_dict()
        assert results[1][1].to_dict()['eval_obj'] == harness.evals[1].to_dict()
        assert results[1][1].to_dict()['candidate_obj'] == harness.candidates[1].to_dict()
        assert results[1][1].to_dict()['results'][0][0] == results[1][1].results[0][0].to_dict()

def test__Eval__callable_check__callable_candidate__non_string_prompt_and_response():  # noqa
    """
    We should be able to use callable Checks and Candidates (e.g. functions) with non-string
    prompts and responses. Lambdas can't be pickled, so we can't use them with EvalHarness (with
    multi-processing), but we can use them with Evals individually.
    """
    eval_ = Eval(
        prompt_sequence=PromptTest(
            prompt={'prompt': 'Test Prompt'},  # non-string prompt
            checks=[
                lambda data: 'Response' in data.response['response'],  # should pass
                lambda data: 'does not exist' in data.response['response'],  # should fail
            ],
        ),
    )
    assert 'prompt_sequence' in eval_.to_dict()
    assert eval_.to_dict()['prompt_sequence'][0]['prompt'] == {'prompt': 'Test Prompt'}
    assert len(eval_.to_dict()['prompt_sequence'][0]['checks']) == 2

    # return dictionary instead of string
    result = eval_(lambda prompt: prompt | {'response': prompt['prompt'] + ' & Response'})
    assert len(result.responses) == 1
    assert result.responses[0] == {'prompt': 'Test Prompt', 'response': 'Test Prompt & Response'}
    assert len(result.results) == 1
    assert len(result.results[0]) == 2
    check_result_1 = result.results[0][0]
    assert check_result_1.value is True
    assert check_result_1.success is True
    assert check_result_1.to_dict() == {'value': True, 'success': True, 'result_type': 'PASS_FAIL'}
    check_result_2 = result.results[0][1]
    assert check_result_2.value is False
    assert check_result_2.success is False
    assert check_result_2.to_dict() == {'value': False, 'success': False, 'result_type': 'PASS_FAIL'}  # noqa

    assert len(result.all_check_results) == 2
    assert result.perc_successful_checks == 0.5
    assert result.all_check_results[0].value == check_result_1.value
    assert result.all_check_results[0].success == check_result_1.success
    assert result.all_check_results[1].value == check_result_2.value
    assert result.all_check_results[1].success == check_result_2.success

    assert result.prompts == [{'prompt': 'Test Prompt'}]
    assert result.response_characters is None  # only applicable for string responses
    assert result.characters_per_second is None  # only applicable for string responses
    assert result.expects_code_blocks is False
    assert result.get_code_block_tests_result() is None
    assert result.get_num_code_blocks_successful() is None
    assert result.get_num_code_tests_defined() is None
    assert result.get_num_code_tests_successful() is None
    # ensure that we can convert the results (which contain unregistered checks/candidates) to
    # a string and dictionary (which call underlying str and to_dict methods on
    # checks/candidates)
    assert len(str(result)) > 10
    assert result.to_dict()['eval_obj'] == eval_.to_dict()
    assert result.to_dict()['candidate_obj']['candidate_type'] == 'CALLABLE_NO_SERIALIZE'
    assert result.to_dict()['results'][0][0] == check_result_1.to_dict()
    assert result.to_dict()['results'][0][1] == check_result_2.to_dict()

@pytest.mark.parametrize('use_async', [True, False])
def test__EvalHarness__callable_check__callable_candidate__non_string_prompt_and_response(use_async):  # noqa
    if use_async:
        async def async_candidate_1(prompt):  # noqa
            return prompt | {'response': prompt['prompt'] + ' & Response1'}

        async def async_candidate_2(prompt):  # noqa
            return prompt | {'response': prompt['prompt'] + ' & Response2'}

        candidates = [async_candidate_1, async_candidate_2]
    else:
        candidates = [
            lambda prompt: prompt | {'response': prompt['prompt'] + ' & Response1'},
            lambda prompt: prompt | {'response': prompt['prompt'] + ' & Response2'},
        ]

    harness = EvalHarness(
        num_cpus=1, async_batch_size=1,
        evals=[
            Eval(
                prompt_sequence=PromptTest(
                    prompt={'prompt': 'Test Prompt 1'},  # test with dictionary prompt
                    checks=[lambda data: 'Response1' in data.response['response']],
                ),
            ),
            Eval(
                prompt_sequence=PromptTest(
                    prompt={'prompt': 'Test Prompt 2'},  # test with dictionary prompt
                    checks=[lambda data: 'Response2' in data.response['response']],
                ),
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
    assert results[0][0].responses == [{'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response1'}]  # noqa
    assert results[0][num_samples-1].responses == [{'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response1'}]  # noqa
    assert results[0][num_samples].responses == [{'prompt': 'Test Prompt 2', 'response': 'Test Prompt 2 & Response1'}]  # noqa
    assert results[1][0].responses == [{'prompt': 'Test Prompt 1', 'response': 'Test Prompt 1 & Response2'}]  # noqa
    assert results[1][num_samples].responses == [{'prompt': 'Test Prompt 2', 'response': 'Test Prompt 2 & Response2'}]  # noqa
    # eval 1 candidate 1
    assert len(results[0][0].all_check_results) == 1
    assert results[0][0].perc_successful_checks == 1
    assert results[0][0].all_check_results[0].value is True
    assert results[0][0].all_check_results[0].success is True
    # eval 2 candidate 1
    assert len(results[0][num_samples].all_check_results) == 1
    assert results[0][num_samples].perc_successful_checks == 0
    assert results[0][num_samples].all_check_results[0].value is False
    assert results[0][num_samples].all_check_results[0].success is False
    # eval 1 candidate 2
    assert len(results[1][0].all_check_results) == 1
    assert results[1][0].perc_successful_checks == 0
    assert results[1][0].all_check_results[0].value is False
    assert results[1][0].all_check_results[0].success is False
    # eval 2 candidate 2
    assert len(results[1][1].all_check_results) == 1
    assert results[1][num_samples].perc_successful_checks == 1
    assert results[1][num_samples].all_check_results[0].value is True
    assert results[1][num_samples].all_check_results[0].success is True

    assert results[0][0].prompts == [{'prompt': 'Test Prompt 1'}]
    assert results[0][num_samples].prompts == [{'prompt': 'Test Prompt 2'}]
    assert results[1][0].prompts == [{'prompt': 'Test Prompt 1'}]
    assert results[1][num_samples].prompts == [{'prompt': 'Test Prompt 2'}]

    # if these work on the first result, they should work on the rest
    assert results[0][0].response_characters is None  # only applicable for string responses
    assert results[0][0].characters_per_second is None  # only applicable for string responses
    assert results[0][0].expects_code_blocks is False
    assert results[0][0].get_code_block_tests_result() is None
    assert results[0][0].get_num_code_blocks_successful() is None
    assert results[0][0].get_num_code_tests_defined() is None
    assert results[0][0].get_num_code_tests_successful() is None
    # ensure that we can convert the results (which contain unregistered checks/candidates) to
    # a string and dictionary (which call underlying str and to_dict methods on
    # checks/candidates)
    assert len(str(results[0][0])) > 10
    assert results[0][0].to_dict()['eval_obj'] == harness.evals[0].to_dict()
    assert results[0][0].to_dict()['candidate_obj'] == harness.candidates[0].to_dict()
    assert results[0][0].to_dict()['results'][0][0] == results[0][0].results[0][0].to_dict()
    assert results[0][num_samples].to_dict()['eval_obj'] == harness.evals[1].to_dict()
    assert results[0][num_samples].to_dict()['candidate_obj'] == harness.candidates[0].to_dict()
    assert results[0][num_samples].to_dict()['results'][0][0] == results[0][num_samples].results[0][0].to_dict()  # noqa: E501
    assert results[1][0].to_dict()['eval_obj'] == harness.evals[0].to_dict()
    assert results[1][0].to_dict()['candidate_obj'] == harness.candidates[1].to_dict()
    assert results[1][0].to_dict()['results'][0][0] == results[1][0].results[0][0].to_dict()
    assert results[1][num_samples].to_dict()['eval_obj'] == harness.evals[1].to_dict()
    assert results[1][num_samples].to_dict()['candidate_obj'] == harness.candidates[1].to_dict()
    assert results[1][num_samples].to_dict()['results'][0][0] == results[1][num_samples].results[0][0].to_dict()  # noqa: E501

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAIToolsCandidate__ToolsCallCheck(openai_tools_candidate_template):  # noqa
    """Integration test that tests Evaling a real OpenAITool API call against the ToolsCheck."""
    candidate = Candidate.from_dict(openai_tools_candidate_template)
    candidate
    eval_ = Eval(
        prompt_sequence=PromptTest(
            prompt="What's the weather like in Boston today in degrees F?",
            checks=[
                ToolCallsCheck(
                    function_name='get_current_weather',
                    function_arguments={'location': 'Boston, MA', 'unit': 'fahrenheit'},
                ),
            ],
        ),
    )
    result = eval_(candidate)
    tool_response = result.responses[0][0]
    assert tool_response['name'] == 'get_current_weather'
    assert 'location' in tool_response['arguments']
    assert tool_response['arguments']['location']
    assert isinstance(tool_response['arguments']['location'], str)
    assert 'unit' in tool_response['arguments']
    assert tool_response['arguments']['unit'] in ['celsius', 'fahrenheit']
    # check that it gets at least the function name correctly
    assert result.all_check_results[0].value >= 0.5
