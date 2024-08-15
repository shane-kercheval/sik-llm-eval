"""Tests for the candidates module."""
import os
from copy import deepcopy
from openai import BadRequestError
import pytest
from llm_eval.candidates import (
    Candidate,
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


def test__is_async_candidate():  # noqa
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

def test__Candidate__from_yaml(openai_candidate_template: dict):  # noqa
    candidate = Candidate.from_yaml('examples/candidates/openai_3.5.yaml')
    assert candidate.candidate_type == CandidateType.OPENAI.name
    assert candidate.to_dict() == openai_candidate_template

def test__candidate__registration():  # noqa
    assert 'MOCK_MODEL' in Candidate.registry
    assert 'mock_model' in Candidate.registry

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

def test__candidate__to_from_dict():  # noqa
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


from pydantic import BaseModel
class CandidateResponse(BaseModel):
    content: str
    metadata: dict | None = None
 # e.g. openai candidate can store input/output characters, cost, etc. in metadata


@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__default__no_parameters(openai_model_name):  # noqa
    candidate = OpenAICandidate(model_name=openai_model_name)
    assert candidate.client.model_parameters == {}




    response = candidate([{'role': 'user', 'content': "What is the capital of France?"}])
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
    recreated_candidate.model.parameters == {}
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
    cloned_candidate.model.parameters == {}
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
        'parameters': {
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
    assert candidate.parameters == config['parameters']
    # test that the underlying model parameters that are sent to OpenAI are correct
    expected_model_param_names = ['temperature', 'max_tokens']
    expected_parameters = {
        k:v for k, v in config['parameters'].items()
        if k in expected_model_param_names
    }
    assert candidate.model.parameters == expected_parameters

    assert candidate.to_dict() == config
    assert candidate.from_dict(candidate.to_dict()) == candidate

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__template__parameters(openai_candidate_template):  # noqa
    """Test that the template for an OpenAI candidate works."""
    template = deepcopy(openai_candidate_template)
    expected_model_param_names = ['temperature', 'max_tokens']
    expected_parameters = {
        k:v for k, v in template['parameters'].items()
        if k in expected_model_param_names
    }
    candidate = Candidate.from_dict(template)
    assert candidate.model.parameters == expected_parameters
    assert candidate.model.model_name == template['parameters']['model_name']
    assert candidate.model.system_message == template['parameters']['system_message']
    assert candidate.to_dict() == template

    response = candidate("What is the capital of France?")
    assert 'Paris' in response
    assert candidate.model.model_name == template['parameters']['model_name']
    assert candidate.model.history()[-1].metadata['model_name'] == template['parameters']['model_name']  # noqa: E501
    assert candidate.model.history()[-1].metadata['parameters'] == expected_parameters
    # after all tests, the dict_copy shoudl be the same as the original i.e. no side effects from
    # other functions we are passing dict4 to
    assert template == openai_candidate_template

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__OpenAI__invalid_parameters(openai_candidate_template):  # noqa
    """Test invalid parameters so that we know we're actually sending them."""
    template = deepcopy(openai_candidate_template)
    template['parameters']['temperature'] = -10  # invalid value
    candidate = Candidate.from_dict(template)
    with pytest.raises(BadRequestError):
        _ = candidate("What is the capital of France?")

@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_API_KEY'), reason="HUGGING_FACE_API_KEY is not set")  # noqa
@pytest.mark.skipif(not os.environ.get('HUGGING_FACE_ENDPOINT_UNIT_TESTS'), reason="HUGGING_FACE_ENDPOINT_UNIT_TESTS is not set")  # noqa
def test__HuggingFaceEndpoint__template(hugging_face_candidate_template):  # noqa
    """Test that the various config options for a Hugging Face Endpoint candidate work."""
    template = deepcopy(hugging_face_candidate_template)
    expected_model_param_names = ['temperature', 'max_tokens', 'seed']
    expected_parameters = {
        k:v for k, v in template['parameters'].items()
        if k in expected_model_param_names
    }
    candidate = Candidate.from_dict(template)
    assert candidate.to_dict() == template
    assert Candidate.from_dict(candidate.to_dict()) == candidate

    # check .parameters on candidate
    expected_candidate_parameters = deepcopy(template['parameters'])
    expected_candidate_parameters.pop('system_format')
    expected_candidate_parameters.pop('prompt_format')
    expected_candidate_parameters.pop('response_prefix')
    assert candidate.parameters == expected_candidate_parameters

    # check .parameters on model
    model_parameters = candidate.model.parameters.copy()
    del model_parameters['return_full_text']
    assert model_parameters == expected_parameters

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
def test__HuggingFaceEndpointCandidate__invalid_parameters(hugging_face_candidate_template):  # noqa
    """Test invalid parameters so that we know we're actually sending them."""
    template = deepcopy(hugging_face_candidate_template)
    template['parameters']['temperature'] = -10  # invalid value
    candidate = Candidate.from_dict(template)
    with pytest.raises(HuggingFaceRequestError) as exception:
        _ = candidate("What is the capital of France?")
    exception = exception.value
    assert exception.error_type.lower() == 'validation'
    assert 'temperature' in exception.error_message

def test__OpenAICandidate__from_yaml(openai_tools_candidate_template: dict, tool_weather, tool_stocks):  # noqa
    candidate = Candidate.from_yaml('examples/candidates/openai_tools_3.5.yaml')
    assert candidate.candidate_type == CandidateType.OPENAI_TOOLS.name
    assert candidate.to_dict() == openai_tools_candidate_template
    assert candidate.model.model_name == candidate.to_dict()['parameters']['model_name']
    assert isinstance(candidate.model.tools, list)
    assert len(candidate.model.tools) == 2
    assert isinstance(candidate.model.tools[0], dict)
    assert candidate.model.tools[0]['function'] == tool_weather
    assert isinstance(candidate.model.tools[1], dict)
    assert candidate.model.tools[1]['function'] == tool_stocks

    response = candidate("What's the weather like in Boston today in degrees F?")
    assert isinstance(response, list)
    assert len(response) == 1
    # ensure the response is from the weather tool and contains the correct parameters
    assert response[0]['name'] == 'get_current_weather'
    assert 'location' in response[0]['arguments']
    assert response[0]['arguments']['location']
    assert isinstance(response[0]['arguments']['location'], str)
    assert 'unit' in response[0]['arguments']
    assert response[0]['arguments']['unit'] in ['celsius', 'fahrenheit']
