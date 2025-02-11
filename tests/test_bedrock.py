"""Tests for the Bedrock API."""
import os
import pytest
from openai import AsyncOpenAI, OpenAI
from llm_eval.bedrock import AsyncBedrockCompletion, BedrockCompletion

BEDROCK_API_KEY = os.getenv('BEDROCK_API_KEY')
BEDROCK_URL = os.getenv('BEDROCK_API_URL')
DEFAULT_MODEL = 'anthropic.claude-3-haiku-20240307-v1:0'


@pytest.mark.skipif('BEDROCK_API_KEY' not in os.environ, reason="BEDROCK_API_KEY not set in environment variables")  # noqa
@pytest.mark.skipif('BEDROCK_API_URL' not in os.environ, reason="BEDROCK_API_URL not set in environment variables")  # noqa
def test__bedrock_claude():
    client = OpenAI(base_url=BEDROCK_URL, api_key=BEDROCK_API_KEY)
    assert client.api_key is not None  # via load_dotenv in conftest.py
    model = BedrockCompletion(
        client=client,
        model=DEFAULT_MODEL,
    )
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat exactly: '{expected_response}'"}]
    response = model(messages=messages)
    assert expected_response in response.content
    assert response.model == DEFAULT_MODEL
    assert response.created
    assert response.duration_seconds
    assert response.usage
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0


@pytest.mark.skipif('BEDROCK_API_KEY' not in os.environ, reason="BEDROCK_API_KEY not set in environment variables")  # noqa
@pytest.mark.skipif('BEDROCK_API_URL' not in os.environ, reason="BEDROCK_API_URL not set in environment variables")  # noqa
@pytest.mark.asyncio
async def test__bedrock_claude__async():
    client = AsyncOpenAI(base_url=BEDROCK_URL, api_key=BEDROCK_API_KEY)
    assert client.api_key is not None  # via load_dotenv in conftest.py
    model = AsyncBedrockCompletion(
        client=client,
        model=DEFAULT_MODEL,
    )
    expected_response = "testing testing"
    messages = [{"role": "user", "content": f"Repeat exactly: '{expected_response}'"}]
    response = await model(messages=messages)
    assert expected_response in response.content
    assert response.model == DEFAULT_MODEL
    assert response.created
    assert response.duration_seconds
    assert response.usage
    assert response.usage['prompt_tokens'] > 0
    assert response.usage['completion_tokens'] > 0
    assert response.usage['total_tokens'] > 0
