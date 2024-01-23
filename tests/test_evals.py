"""Tests for the evals module."""
from llm_evals.checks import CheckType, ContainsCheck, MatchCheck
from llm_evals.eval import Candidate, Eval, EvalResult, PromptTest


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

def test__candidate__creation():  # noqa
    assert Candidate().to_dict() == {}
    assert Candidate(**Candidate().to_dict()) == Candidate()

    candidate = Candidate(model=lambda x: x)
    candidate_dict = candidate.to_dict()
    assert candidate_dict == {}
    assert Candidate(**candidate_dict) == candidate
    assert candidate('test') == 'test'

    candidate = Candidate(model=lambda x: x, metadata={'test': 'test'})
    candidate_dict = candidate.to_dict()
    assert candidate_dict == {'metadata': {'test': 'test'}}
    assert Candidate(**candidate_dict) == candidate

    candidate = Candidate(
        uuid='test_uuid',
        model=lambda x: x,
        metadata={'test': 'test'},
        model_type='TEST_MODEL_TYPE',
        parameters={'param_1': 'param_a'},
        system_info={'system_1': 'system_a'},
    )
    candidate_dict = candidate.to_dict()
    assert candidate_dict == {
        'uuid': 'test_uuid',
        'metadata': {'test': 'test'},
        'model_type': 'TEST_MODEL_TYPE',
        'parameters': {'param_1': 'param_a'},
        'system_info': {'system_1': 'system_a'},
    }
    assert Candidate(**candidate_dict) == candidate

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
    assert result_dict['candidate_obj'] == {'metadata': {'function': 'def <lambda>(x)'}}
    assert Eval(**result_dict['eval_obj']) == eval_obj
    assert EvalResult(**result_dict) == result
    assert EvalResult(**result_dict).to_dict() == result.to_dict()

def test_EVAL_(fake_eval_8f9fbf37: dict):  # noqa
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
    result = eval_obj(lambda _: next(mock_llm_instance))
    assert result.responses == responses
    assert result.prompts == [test.prompt for test in eval_obj.test_sequence]
    assert result.ideal_responses == [test.ideal_response for test in eval_obj.test_sequence]
    assert result.eval_obj.to_dict() == eval_dict

    result_dict = result.to_dict()
    assert result_dict['eval_obj'] == eval_dict
    assert Eval(**result_dict['eval_obj']) == eval_obj
    assert result_dict['candidate_obj'] == {'metadata': {'function': 'def <lambda>(_)'}}
    assert result.total_time_seconds > 0
    assert EvalResult(**result_dict).to_dict() == result.to_dict()
    recreated_eval = EvalResult(**result_dict)
    assert recreated_eval.eval_obj == result.eval_obj
    assert recreated_eval.candidate_obj == result.candidate_obj

    assert recreated_eval.results == result.results

    flatted_checks = [r for test in eval_obj.test_sequence for r in test.checks]
    for c, r in zip(flatted_checks, result.all_checks_results(), strict=True):
        assert c.check_type == r.metadata['check_type']


    result.save_yaml('test.yaml')

    # save results to yaml
    

