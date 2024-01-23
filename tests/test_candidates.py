"""Tests for the candidates module."""
import os
import pytest
from llm_evals.candidates import CallableCandidate, Candidate, CandidateType, OpenAICandidate


class MockLMM:
    """Mock class representing an LLM."""

    def __init__(self, **kwargs: dict):
        self.llm_parameters = kwargs
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        """Caches prompts for unit tests."""
        self.prompts.append(prompt)
        return prompt


@Candidate.register('MOCK_MODEL')
class MockCandidate(Candidate):
    """Mock class representing a Candidate."""

    def __init__(self, **kwargs: dict) -> None:
        """Initialize a MockCandidate object."""
        uuid = kwargs.pop('uuid', None)
        metadata = kwargs.pop('metadata', None)
        parameters = kwargs.pop('parameters', None)
        system_info = kwargs.pop('system_info', None)
        super().__init__(parameters=parameters, metadata=metadata, uuid=uuid, system_info=system_info)  # noqa
        self.model = None
        if parameters is not None:
            self.model = MockLMM(**parameters)

    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""
        return self.model(prompt)


def test__CallableCandidate():  # noqa
    candidate = CallableCandidate(model=lambda x: x)
    assert candidate('test') == 'test'
    assert candidate.to_dict() == {'candidate_type': CandidateType.CALLABLE_NO_SERIALIZE.name}

def test__candidate__registration():  # noqa
    assert 'MOCK_MODEL' in Candidate.registry
    assert 'mock_model' in Candidate.registry

    assert MockCandidate().model is None  # only create model when called with parameters
    assert MockCandidate() == Candidate.from_dict({'candidate_type': 'MOCK_MODEL'})
    assert MockCandidate().to_dict() == {'candidate_type': 'MOCK_MODEL'}

    candidate = MockCandidate(parameters={'param_1': 'param_a'}, metadata={'test': 'test'})
    candidate.parameters == {'param_1': 'param_a'}
    candidate.metadata == {'test': 'test'}
    assert isinstance(candidate.model, MockLMM)
    assert candidate.model.llm_parameters == {'param_1': 'param_a'}

    # test underlying call mechanism so that we can cache prompts and ensure a new model is created
    # for each new candidate instance
    response = candidate('test_1')
    assert response == 'test_1'
    response = candidate('test_2')
    assert response == 'test_2'
    assert candidate.model.prompts == ['test_1', 'test_2']

    candidate_2 = MockCandidate(parameters={'param_1': 'param_b'})
    assert candidate_2.model.llm_parameters == {'param_1': 'param_b'}
    assert candidate.model.llm_parameters == {'param_1': 'param_a'}
    assert candidate.model.prompts == ['test_1', 'test_2']

    response = candidate_2('test_3')
    assert response == 'test_3'
    assert candidate.model.prompts == ['test_1', 'test_2']

def test__candidate__to_from_dict():  # noqa
    candidate_dict = {
        'candidate_type': 'MOCK_MODEL',
        'uuid': 'test_uuid',
        'metadata': {'name': 'test name'},
        'parameters': {'param_1': 'param_a', 'param_2': 'param_b'},
        'system_info': {'system_1': 'system_a', 'system_2': 'system_b'},
    }
    candidate_dict_no_type = candidate_dict.copy()
    candidate_dict_no_type.pop('candidate_type')

    candidate = MockCandidate(**candidate_dict_no_type)
    assert candidate.to_dict() == candidate_dict
    assert candidate == Candidate.from_dict(candidate_dict)
    assert candidate == Candidate.from_dict(candidate.to_dict())
    with pytest.raises(ValueError):  # noqa: PT011
        Candidate.from_dict(candidate_dict_no_type)

    assert isinstance(candidate.model, MockLMM)
    assert candidate.model.llm_parameters == {'param_1': 'param_a', 'param_2': 'param_b'}
    response = candidate('test')
    assert response == 'test'
    assert candidate.model.prompts == ['test']
    # make sure there is no shared state between candidates
    another_candidate = Candidate.from_dict(candidate_dict)
    assert another_candidate == candidate
    assert another_candidate.model.llm_parameters == candidate.model.llm_parameters
    assert another_candidate.model.prompts != candidate.model.prompts
    response = another_candidate('test_another')
    assert response == 'test_another'
    assert another_candidate.model.prompts == ['test_another']
    assert candidate.model.prompts == ['test']

def test__candidate__clone():  #noqa
    candidate_dict = {
        'candidate_type': 'MOCK_MODEL',
        'uuid': 'test_uuid',
        'metadata': {'name': 'test name'},
        'parameters': {'param_1': 'param_a', 'param_2': 'param_b'},
        'system_info': {'system_1': 'system_a', 'system_2': 'system_b'},
    }
    candidate = Candidate.from_dict(candidate_dict)
    response = candidate('test')
    assert response == 'test'
    assert candidate.model.prompts == ['test']

    clone = candidate.clone()
    assert candidate.clone() == candidate
    assert candidate.to_dict() == clone.to_dict()
    # the "objects" i.e. dictionaries should match but the model objects should not
    assert candidate.model.prompts == ['test']
    assert clone.model.prompts == []
    # ensure that changing values on the clone doesn't affect the original
    clone.uuid = 'test_uuid_2'
    clone.metadata['name'] = 'test name 2'
    clone.parameters['param_1'] = 'param_a_2'
    clone.system_info['system_1'] = 'system_a_2'
    assert clone.to_dict() != candidate.to_dict()
    assert candidate.to_dict() == candidate_dict
    # ensure that using the clone doesn't affect the original
    response = clone('test another')
    assert response == 'test another'
    assert clone.model.prompts == ['test another']
    assert candidate.model.prompts == ['test']

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI():  # noqa
    candidate = OpenAICandidate()
    response = candidate("What is the capital of France?")
    assert 'Paris' in response

    assert candidate.to_dict() == {'candidate_type': CandidateType.OPENAI.name}
    recreated_candidate = Candidate.from_dict(candidate.to_dict())
    assert candidate == recreated_candidate
    # ensure that the recreated candidate doesn't share history with the original
    assert len(candidate.model.history()) == 1
    assert len(recreated_candidate.model.history()) == 0
    response = recreated_candidate("What is the capital of Spain?")
    assert 'Madrid' in response
    assert len(candidate.model.history()) == 1
    assert len(recreated_candidate.model.history()) == 1
    # ensure that the cloned candidate doesn't share history with the original
    cloned_candidate = recreated_candidate.clone()
    assert cloned_candidate == recreated_candidate
    assert len(cloned_candidate.model.history()) == 0
    response = cloned_candidate("What is the capital of Germany?")
    assert 'Berlin' in response
    assert len(candidate.model.history()) == 1
    assert len(recreated_candidate.model.history()) == 1
    assert len(cloned_candidate.model.history()) == 1
