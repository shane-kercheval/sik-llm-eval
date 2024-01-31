"""Tests for the candidates module."""
import os
from copy import deepcopy
from openai import BadRequestError
import pytest
from llm_evals.candidates import (
    CallableCandidate,
    Candidate,
    CandidateType,
    OpenAICandidate,
)
from llm_evals.llms.hugging_face import HuggingFaceRequestError


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
        metadata = kwargs.pop('metadata', None)
        model_parameters = kwargs.pop('model_parameters', None)
        super().__init__(model_parameters=model_parameters, metadata=metadata)
        self.model = None
        if model_parameters is not None:
            self.model = MockLMM(**model_parameters)

    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""
        return self.model(prompt)


def test__Candidate__from_yaml(openai_candidate_template: dict):  # noqa
    candidate = Candidate.from_yaml('examples/candidates/openai_3.5_1106.yaml')
    assert candidate.candidate_type == CandidateType.OPENAI.name
    assert candidate.to_dict() == openai_candidate_template

def test__candidate__registration():  # noqa
    assert 'MOCK_MODEL' in Candidate.registry
    assert 'mock_model' in Candidate.registry

    assert MockCandidate().model is None  # only create model when called with parameters
    assert MockCandidate() == Candidate.from_dict({'candidate_type': 'MOCK_MODEL'})
    assert MockCandidate().to_dict() == {'candidate_type': 'MOCK_MODEL'}

    candidate = MockCandidate(model_parameters={'param_1': 'param_a'}, metadata={'test': 'test'})
    candidate.model_parameters == {'param_1': 'param_a'}
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

    candidate_2 = MockCandidate(model_parameters={'param_1': 'param_b'})
    assert candidate_2.model.llm_parameters == {'param_1': 'param_b'}
    assert candidate.model.llm_parameters == {'param_1': 'param_a'}
    assert candidate.model.prompts == ['test_1', 'test_2']

    response = candidate_2('test_3')
    assert response == 'test_3'
    assert candidate.model.prompts == ['test_1', 'test_2']

def test__candidate__to_from_dict():  # noqa
    candidate_dict = {
        'candidate_type': 'MOCK_MODEL',
        'metadata': {'name': 'test name'},
        'model_parameters': {'param_1': 'param_a', 'param_2': 'param_b'},
    }
    candidate_dict_no_type = deepcopy(candidate_dict)
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
        'metadata': {'name': 'test name'},
        'model_parameters': {'param_1': 'param_a', 'param_2': 'param_b'},
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
    clone.metadata['name'] = 'test name 2'
    clone.model_parameters['param_1'] = 'param_a_2'
    assert clone.to_dict() != candidate.to_dict()
    assert candidate.to_dict() == candidate_dict
    # ensure that using the clone doesn't affect the original
    response = clone('test another')
    assert response == 'test another'
    assert clone.model.prompts == ['test another']
    assert candidate.model.prompts == ['test']

def test__CallableCandidate():  # noqa
    candidate = CallableCandidate(model=lambda x: x)
    assert candidate('test') == 'test'
    assert candidate.to_dict() == {'candidate_type': CandidateType.CALLABLE_NO_SERIALIZE.name}

def test__candidate__multiple_model_params_returns_multiple_candidates():  # noqa
    pass

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__default__no_model_parameters():  # noqa
    candidate = OpenAICandidate()
    candidate.model.model_parameters == {}
    response = candidate("What is the capital of France?")
    assert 'Paris' in response
    assert candidate.total_tokens > 0
    assert candidate.total_tokens == candidate.model.total_tokens
    assert candidate.response_tokens > 0
    assert candidate.response_tokens == candidate.model.response_tokens
    assert candidate.input_tokens > 0
    assert candidate.input_tokens == candidate.model.input_tokens
    assert candidate.cost > 0
    assert candidate.cost == candidate.model.cost
    assert candidate.to_dict() == {'candidate_type': CandidateType.OPENAI.name}
    # test that the model generated from the dict is the same as the original
    # but that they don't share history (i.e. there is a new underlying object for the model)
    recreated_candidate = Candidate.from_dict(candidate.to_dict())
    recreated_candidate.model.model_parameters == {}
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
    cloned_candidate.model.model_parameters == {}
    assert cloned_candidate == recreated_candidate
    assert len(cloned_candidate.model.history()) == 0
    response = cloned_candidate("What is the capital of Germany?")
    assert 'Berlin' in response
    assert len(candidate.model.history()) == 1
    assert len(recreated_candidate.model.history()) == 1
    assert len(cloned_candidate.model.history()) == 1

def test__OpenAI__config():  # noqa
    """Test that the various config options for an OpenAI candidate work."""
    config = {
        'metadata': {'name': 'Test Name'},
        'candidate_type': CandidateType.OPENAI.name,
        'model_parameters': {
            'model_name': 'test model name',
            'system_message': 'test system message',
            'temperature': -1,
            'max_tokens': -2,
            'seed': -3,
        },
    }
    candidate = Candidate.from_dict(config)
    assert candidate.metadata == config['metadata']
    assert candidate.candidate_type == CandidateType.OPENAI.name
    assert candidate.model_parameters == config['model_parameters']
    # test that the underlying model parameters that are sent to OpenAI are correct
    expected_model_param_names = ['temperature', 'max_tokens']
    expected_model_parameters = {
        k:v for k, v in config['model_parameters'].items()
        if k in expected_model_param_names
    }
    assert candidate.model.model_parameters == expected_model_parameters

    assert candidate.to_dict() == config
    assert candidate.from_dict(candidate.to_dict()) == candidate

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__template__model_parameters(openai_candidate_template):  # noqa
    """Test that the template for an OpenAI candidate works."""
    template = deepcopy(openai_candidate_template)
    expected_model_param_names = ['temperature', 'max_tokens']
    expected_model_parameters = {
        k:v for k, v in template['model_parameters'].items()
        if k in expected_model_param_names
    }
    candidate = Candidate.from_dict(template)
    assert candidate.model.model_parameters == expected_model_parameters
    assert candidate.model.model_name == template['model_parameters']['model_name']
    assert candidate.model.system_message == template['model_parameters']['system_message']
    assert candidate.to_dict() == template

    response = candidate("What is the capital of France?")
    assert 'Paris' in response
    assert candidate.model.model_name == template['model_parameters']['model_name']
    assert candidate.model.history()[-1].metadata['model_name'] == template['model_parameters']['model_name']  # noqa: E501
    assert candidate.model.history()[-1].metadata['model_parameters'] == expected_model_parameters
    # after all tests, the dict_copy shoudl be the same as the original i.e. no side effects from
    # other functions we are passing dict4 to
    assert template == openai_candidate_template

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__invalid_model_parameters(openai_candidate_template):  # noqa
    """Test invalid parameters so that we know we're actually sending them."""
    template = deepcopy(openai_candidate_template)
    template['model_parameters']['temperature'] = -10  # invalid value
    candidate = Candidate.from_dict(template)
    with pytest.raises(BadRequestError):
        _ = candidate("What is the capital of France?")

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test__HuggingFaceEndpoint__template(hugging_face_candidate_template):  # noqa
    """Test that the various config options for a Hugging Face Endpoint candidate work."""
    template = deepcopy(hugging_face_candidate_template)
    expected_model_param_names = ['temperature', 'max_tokens', 'seed']
    expected_model_parameters = {
        k:v for k, v in template['model_parameters'].items()
        if k in expected_model_param_names
    }
    candidate = Candidate.from_dict(template)
    assert candidate.to_dict() == template
    assert Candidate.from_dict(candidate.to_dict()) == candidate

    # check .model_parameters on candidate
    expected_candidate_parameters = deepcopy(template['model_parameters'])
    expected_candidate_parameters.pop('system_format')
    expected_candidate_parameters.pop('prompt_format')
    expected_candidate_parameters.pop('response_format')
    assert candidate.model_parameters == expected_candidate_parameters

    # check .model_parameters on model
    assert candidate.model.model_parameters == expected_model_parameters

    # ensure message is created correctly
    expected_message = template['model_parameters']['system_format'].format(system_message='a') \
        + template['model_parameters']['prompt_format'].format(prompt='b') \
        + template['model_parameters']['response_format'].format(response='c') \
        + template['model_parameters']['prompt_format'].format(prompt='d')
    assert candidate.model._message_formatter('a', [('b', 'c')], 'd') == expected_message

    # test that the dictionary hasn't changed after passing the dict to various functions
    # i.e. test no side effects against dict
    assert candidate.to_dict() == template
    assert template == hugging_face_candidate_template
    assert Candidate.from_dict(candidate.to_dict()) == candidate

    # test response
    response = candidate("What is the capital of France?")
    assert 'Paris' in response
    assert candidate.total_tokens > 0
    assert candidate.total_tokens == candidate.model.total_tokens
    assert candidate.response_tokens > 0
    assert candidate.response_tokens == candidate.model.response_tokens
    assert candidate.input_tokens > 0
    assert candidate.input_tokens == candidate.model.input_tokens
    # test that the model generated from the dict is the same as the original
    # but that they don't share history (i.e. there is a new underlying object for the model)
    assert candidate.to_dict() == template
    recreated_candidate = Candidate.from_dict(candidate.to_dict())
    assert candidate == recreated_candidate
    # ensure that the recreated candidate doesn't share history with the original
    assert len(candidate.model.history()) == 1
    assert len(recreated_candidate.model.history()) == 0
    response = recreated_candidate("What is the capital of Spain?")
    assert 'Madrid' in response
    assert len(candidate.model.history()) == 1
    assert len(recreated_candidate.model.history()) == 1
    assert recreated_candidate.to_dict() == template
    # ensure that the cloned candidate doesn't share history with the original
    cloned_candidate = recreated_candidate.clone()
    assert cloned_candidate == candidate
    assert len(cloned_candidate.model.history()) == 0
    response = cloned_candidate("What is the capital of Germany?")
    assert 'Berlin' in response
    assert cloned_candidate.to_dict() == template
    assert len(candidate.model.history()) == 1
    assert len(recreated_candidate.model.history()) == 1
    assert len(cloned_candidate.model.history()) == 1

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test__HuggingFaceEndpointCandidate__invalid_model_parameters(hugging_face_candidate_template):  # noqa
    """Test invalid parameters so that we know we're actually sending them."""
    template = deepcopy(hugging_face_candidate_template)
    template['model_parameters']['temperature'] = -10  # invalid value
    candidate = Candidate.from_dict(template)
    with pytest.raises(HuggingFaceRequestError) as exception:
        _ = candidate("What is the capital of France?")
    exception = exception.value
    assert exception.error_type.lower() == 'validation'
    assert 'temperature' in exception.error_message
