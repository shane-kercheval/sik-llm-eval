"""Tests for the candidates module."""
import os
from copy import deepcopy
from openai import BadRequestError
import pytest
from sik_llms import Tool, ToolPrediction, user_message
from sik_llm_eval.candidates import (
    Candidate,
    CandidateResponse,
    CandidateType,
    OpenAICandidate,
    is_async_candidate,
)

class MockLMM:
    """Mock class representing an LLM."""

    def __init__(self, **kwargs: dict):
        self.llm_parameters = kwargs
        self.prompts = []

    def __call__(self, input: str) -> str:  # noqa: A002
        """Caches prompts for unit tests."""
        self.prompts.append(input)
        return input


@Candidate.register('MOCK_MODEL')
class MockCandidate(Candidate):
    """Mock class representing a Candidate."""

    def __init__(self, **kwargs: dict) -> None:
        """Initialize a MockCandidate object."""
        metadata = kwargs.pop('metadata', None)
        parameters = kwargs.pop('parameters', None)
        super().__init__(parameters=parameters, metadata=metadata)
        self.model = None
        if parameters is not None:
            self.model = MockLMM(**parameters)
        else:
            self.model = MockLMM()

    def __call__(self, input: str) -> str:  # noqa: A002
        """Invokes the underlying model with the input and returns the response."""
        return self.model(input)


def test__is_async_candidate():
    async def async_function():  # noqa
        pass
    def sync_function():  # noqa
        pass
    class AsyncCallable:
        async def __call__(self):
            pass
    class SyncCallable:
        def __call__(self):
            pass

    assert is_async_candidate(async_function)
    assert not is_async_candidate(sync_function)
    assert is_async_candidate(AsyncCallable())
    assert not is_async_candidate(SyncCallable())

def test__Candidate__from_yaml(openai_candidate_template: dict):
    candidate = Candidate.from_yaml('examples/candidates/openai_4o-mini.yaml')
    assert candidate.candidate_type == CandidateType.OPENAI.name
    assert candidate.to_dict() == openai_candidate_template

def test__candidate__registration():
    assert not Candidate.is_registered('NotRegistered')

    assert 'MOCK_MODEL' in Candidate.registry
    assert Candidate.is_registered('MOCK_MODEL')
    assert 'mock_model' in Candidate.registry
    assert Candidate.is_registered('mock_model')

    assert MockCandidate().model.llm_parameters == {}
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

def test__candidate__to_from_dict():
    candidate_dict = {
        'candidate_type': 'MOCK_MODEL',
        'metadata': {'name': 'test name'},
        'parameters': {'param_1': 'param_a', 'param_2': 'param_b'},
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

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__default__no_parameters(openai_model: str):
    candidate = OpenAICandidate(model_name=openai_model)
    assert candidate.to_dict() == {
        'candidate_type': CandidateType.OPENAI.name,
        'model_name': openai_model,
    }
    messages = [user_message("What is the capital of France?")]
    response = candidate(messages)
    assert 'Paris' in response.response
    assert response.metadata['input_tokens'] > 0
    assert response.metadata['output_tokens'] > 0
    assert response.metadata['total_tokens'] > 0
    assert response.metadata['input_cost'] > 0
    assert response.metadata['output_cost'] > 0
    assert response.metadata['total_cost'] > 0
    assert response.metadata['output_characters'] > 0
    # test that the model generated from the dict is the same as the original
    # but that they don't share history (i.e. there is a new underlying object for the model)
    recreated_candidate = Candidate.from_dict(candidate.to_dict())
    assert candidate == recreated_candidate
    assert recreated_candidate.to_dict() == {
        'candidate_type': CandidateType.OPENAI.name,
        'model_name': openai_model,
    }
    messages = [user_message("What is the capital of Germany?")]
    response = recreated_candidate(messages)
    assert 'Berlin' in response.response

def test__OpenAI__config():
    """Test that the various config options for an OpenAI candidate work."""
    config = {
        'metadata': {'name': 'Test Name'},
        'candidate_type': CandidateType.OPENAI.name,
        'model_name': 'test model name',
        'parameters': {
            'temperature': -1,
            'max_tokens': -2,
            'seed': -3,
        },
    }
    candidate = Candidate.from_dict(config)
    assert candidate.metadata == config['metadata']
    assert candidate.candidate_type == CandidateType.OPENAI.name
    assert candidate.parameters == config['parameters']
    assert candidate.to_dict() == config
    assert candidate.from_dict(candidate.to_dict()) == candidate

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__template__parameters(openai_candidate_template: dict):
    """Test that the template for an OpenAI candidate works."""
    template = deepcopy(openai_candidate_template)
    expected_model_param_names = ['temperature', 'max_tokens']
    expected_parameters = {
        k:v for k, v in template['parameters'].items()
        if k in expected_model_param_names
    }
    candidate = Candidate.from_dict(template)
    candidate.parameters == expected_parameters
    assert candidate.to_dict() == template

    messages = [user_message("What is the capital of France?")]
    response = candidate(messages)
    assert 'Paris' in response.response

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__invalid_parameters(openai_candidate_template: dict):
    """Test invalid parameters so that we know we're actually sending them."""
    template = deepcopy(openai_candidate_template)
    template['parameters']['temperature'] = -10  # invalid value
    candidate = Candidate.from_dict(template)
    messages = [user_message("What is the capital of France?")]
    with pytest.raises(BadRequestError):
        _ = candidate(messages)

def test__OpenAIToolsCandidate__from_yaml(
            openai_tools_candidate_template: dict,
            weather_tool: Tool, stocks_tool: Tool,
        ):
    candidate = Candidate.from_yaml('examples/candidates/openai_tools_4o-mini.yaml')
    assert candidate.candidate_type == CandidateType.OPENAI_TOOLS.name
    assert candidate.tools[0].model_dump() == weather_tool.model_dump()
    assert candidate.tools[1].model_dump() == stocks_tool.model_dump()
    assert candidate.to_dict() == openai_tools_candidate_template

    response = candidate([user_message("What's the weather like in Boston today in degrees F?")])
    assert isinstance(response, CandidateResponse)
    assert isinstance(response.response, ToolPrediction)
    tool_prediction = response.response
    assert tool_prediction.name == 'get_current_weather'
    arguments = tool_prediction.arguments
    assert 'location' in arguments
    assert arguments['location']
    assert isinstance(arguments['location'], str)
    assert 'unit' in arguments
    assert arguments['unit'] in ['celsius', 'fahrenheit']

    assert response.metadata['input_tokens'] > 0
    assert response.metadata['output_tokens'] > 0
    assert response.metadata['total_tokens'] > 0
    assert response.metadata['input_cost'] > 0
    assert response.metadata['output_cost'] > 0
    assert response.metadata['total_cost'] > 0
