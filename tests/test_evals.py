"""Tests for the evals module."""
import os
from textwrap import dedent
import pytest
from llm_evals.candidates import Candidate, CandidateType
from llm_evals.checks import CheckType, ContainsCheck, MatchCheck, PassFailResult, ScoreResult
from llm_evals.eval import Eval, EvalResult, PromptTest
from llm_evals.utilities.internal_utilities import extract_code_blocks


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
    assert flatted_check_results == [r.to_dict() for r in eval_result.all_checks_results]
    assert eval_result.total_time_seconds > 0
    # check that the eval_result_dict will recreate the exact eval_result object
    recreated_eval = EvalResult(**eval_result_dict)
    assert recreated_eval == eval_result
    assert recreated_eval.to_dict() == eval_result.to_dict()
    assert recreated_eval.eval_obj == eval_result.eval_obj
    assert recreated_eval.candidate_obj == eval_result.candidate_obj
    assert recreated_eval.results == eval_result.results
    flatted_checks = [r for test in eval_obj.test_sequence for r in test.checks]
    for c, r in zip(flatted_checks, eval_result.all_checks_results, strict=True):
        assert c.check_type == r.metadata['check_type']

def test__Eval__multiple_code_blocks__ensure_code_blocks_run(fake_eval_sum_two_numbers_code_blocks_run):  # noqa
    """
    Use Mock LLM with multiple code blocks (over multiple responses) to ensure code blocks run and
    the check results return the expected values.
    """
    config = fake_eval_sum_two_numbers_code_blocks_run.copy()
    eval_obj = Eval(**config)

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

    assert eval_result.eval_obj.to_dict() == config
    assert Eval(**eval_obj.to_dict()) == eval_obj
    assert EvalResult(**eval_result.to_dict()) == eval_result

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
    assert len(eval_result.all_checks_results) == 7

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
    assert eval_result.results[1][2].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_RUN.name  # noqa

    # function checks
    expected_function_checks = 6
    expected_successful_function_checks = 4
    expected_total_checks = expected_num_code_blocks + expected_function_checks
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_function_checks

    assert eval_result.results[1][2].value == expected_successful_checks / expected_total_checks
    assert eval_result.results[1][2].success_threshold == 1
    assert not eval_result.results[1][2].success
    assert eval_result.results[1][2].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_RUN.name  # noqa
    assert eval_result.results[1][2].metadata['num_code_blocks'] == expected_num_code_blocks
    assert eval_result.results[1][2].metadata['num_code_blocks_successful'] == expected_successful_code_blocks  # noqa
    assert eval_result.results[1][2].metadata['code_blocks'] == expected_code_blocks
    assert eval_result.results[1][2].metadata['code_block_errors'] == [None, None, None]
    # first function check should have run successfully, but second code block should have failed
    assert eval_result.results[1][2].metadata['function_check_results'] == [True, True, False, True, True, False]  # noqa
    assert eval_result.results[1][2].metadata['num_function_checks'] == expected_function_checks
    assert eval_result.results[1][2].metadata['num_function_checks_successful'] == expected_successful_function_checks  # noqa
    assert eval_result.results[1][2].metadata['function_check_errors'][0] is None
    assert eval_result.results[1][2].metadata['function_check_errors'][1] is None
    assert eval_result.results[1][2].metadata['function_check_errors'][2] is None
    assert eval_result.results[1][2].metadata['function_check_errors'][3] is None
    assert eval_result.results[1][2].metadata['function_check_errors'][4] is None
    assert isinstance(eval_result.results[1][2].metadata['function_check_errors'][5], NameError)

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
    assert len(result.all_checks_results) == 2
    assert result.all_checks_results[0].success
    assert result.all_checks_results[0].metadata['check_type'] == CheckType.CONTAINS.name
    assert result.all_checks_results[1].success
    assert result.all_checks_results[1].metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name  # noqa: E501
    assert result.all_checks_results[1].metadata['num_code_blocks'] >= 1
    expected_num_code_blocks = len(result.all_checks_results[1].metadata['code_blocks'])
    assert result.all_checks_results[1].metadata['num_code_blocks'] == expected_num_code_blocks
    assert result.num_checks == 2
    assert result.num_successful_checks == 2
    assert result.perc_successful_checks == 1
    assert str(result)
    assert EvalResult(**result.to_dict()) == result
    assert EvalResult(**result.to_dict()).to_dict() == result.to_dict()
    assert eval_config == fake_eval_sum_two_numbers  # make sure eval_config wasn't modified
