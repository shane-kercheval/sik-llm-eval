"""Tests for the evals module."""
import os
import pytest
from llm_evals.candidates import Candidate, CandidateType
from llm_evals.checks import CheckType, ContainsCheck, MatchCheck
from llm_evals.eval import Eval, EvalResult, PromptTest, eval_result_summarizer


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

def test_Eval__example_8f9fbf37__callable_candidate(fake_eval_8f9fbf37: dict):  # noqa
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

    assert eval_result_summarizer(eval_result)

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test_Eval__candidate_from_dict(fake_eval_sum_two_numbers, openai_candidate_template):  # noqa
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
