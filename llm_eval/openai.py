"""Contains helper functions for interacting with OpenAI models."""
from abc import ABC, abstractmethod
import time
from typing import Callable
from functools import cache
from openai import OpenAI
from pydantic import BaseModel, SerializeAsAny
import tiktoken
from tiktoken import Encoding


CHAT_MODEL_COST_PER_TOKEN = {
    # LATEST MODELS
    'gpt-4o-2024-05-13': {'input': 5.00 / 1_000_000, 'output': 15.00 / 1_000_000},
    'gpt-4o-mini-2024-07-18':  {'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000},
    # GPT-4-Turbo 128K
    'gpt-4-turbo-2024-04-09': {'input': 10.00 / 1_000_000, 'output': 30.00 / 1_000_000},
    'gpt-4-0125-preview': {'input': 0.01 / 1_000, 'output': 0.03 / 1_000},
    # GPT-3.5 Turbo 16K
    'gpt-3.5-turbo-0125': {'input': 0.50 / 1_000_000, 'output': 1.50 / 1_000_000},

    # LEGACY MODELS
    # GPT-4-Turbo 128K
    # 'gpt-4-1106-preview': {'input': 0.01 / 1_000, 'output': 0.03 / 1_000},
    # GPT-3.5-Turbo 16K
    # 'gpt-3.5-turbo-1106': {'input': 0.001 / 1_000, 'output': 0.002 / 1_000},
    # GPT-4
    'gpt-4-0613': {'input': 0.03 / 1_000, 'output': 0.06 / 1_000},
    # GPT-4- 32K
    # 'gpt-4-32k-0613': {'input': 0.06 / 1_000, 'output': 0.12 / 1_000},
    # GPT-3.5-Turbo 4K
    # 'gpt-3.5-turbo-0613': {'input': 0.0015 / 1_000, 'output': 0.002 / 1_000},
    # GPT-3.5-Turbo 16K
    # 'gpt-3.5-turbo-16k-0613': {'input': 0.003 / 1_000, 'output': 0.004 / 1_000},
}

EMBEDDING_MODEL_COST_PER_TOKEN = {
    # "Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens
    # is about 750 words. This paragraph is 35 tokens."
    # https://openai.com/pricing
    # https://platform.openai.com/docs/models
    ####
    # Embedding models
    ####
    # LATEST MODELS
    # https://openai.com/blog/new-embedding-models-and-api-updates
    'text-embedding-3-small': 0.02 / 1_000_000,
    'text-embedding-3-large': 0.13 / 1_000_000,
    # LEGACY MODELS
    'text-embedding-ada-002': 0.1 / 1_000_000,
}

MODEL_COST_PER_TOKEN = CHAT_MODEL_COST_PER_TOKEN | EMBEDDING_MODEL_COST_PER_TOKEN


@cache
def _get_encoding_for_model(model_name: str) -> Encoding:
    """Gets the encoding for a given model so that we can calculate the number of tokens."""
    return tiktoken.encoding_for_model(model_name)


def num_tokens(model_name: str, value: str) -> int:
    """For a given model, returns the number of tokens based on the str `value`."""
    return len(_get_encoding_for_model(model_name=model_name).encode(value))


def num_tokens_from_messages(model_name: str, messages: list[dict]) -> int:
    """
    Copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    Returns the number of tokens used by a list of messages.
    """
    if model_name in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        # todo: verify once .ipynb is updated
        "gpt-4-1106-preview",
        "gpt-3.5-turbo-1106",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model_name == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model_name:
        # Warning: gpt-3.5-turbo may update over time.
        # Returning num tokens assuming gpt-3.5-turbo-0613
        return num_tokens_from_messages(model_name="gpt-3.5-turbo-1106", messages=messages)
    elif "gpt-4" in model_name:
        # Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.
        return num_tokens_from_messages(model_name="gpt-4-1106-preview", messages=messages)
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model_name}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")  # noqa
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(_get_encoding_for_model(model_name=model_name).encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens



class OpenAIResponse(BaseModel):
    """Base class for OpenAI responses."""

    object_name: str
    model: str
    created: int
    metadata: dict | SerializeAsAny[BaseModel] | None = None
    # https://platform.openai.com/docs/guides/chat-completions/response-format
    # stop: API returned complete message, or a message terminated by one of the stop sequences
    # provided via the stop parameter
    # length: Incomplete model output due to max_tokens parameter or token limit
    # function_call: The model decided to call a function
    # content_filter: Omitted content due to a flag from our content filters
    # null: API response still in progress or incomplete
    finish_reason: str | None = None


class OpenAIChunkResponse(OpenAIResponse):
    """Stores the result of an OpenAI chat completion chunk."""

    content: str


class OpenAIChatResponse(OpenAIChunkResponse):
    """Stores the result of an OpenAI chat completion."""

    content: str
    role: str | None = None
    usage: dict | None = None
    duration_seconds: float | None = None
    logprobs_tokens: list[str] | None = None
    logprobs: list[float] | None = None


class OpenAICompletionWrapperBase(ABC):
    """
    Wrapper for OpenAI API.

    The user can specify the model name, timeout, stream, and other parameters for the API call
    either in the constructor or when calling the object. If the latter, the parameters specified
    when calling the object will override the parameters specified in the constructor.
    """

    def __init__(
            self,
            client: OpenAI,
            model: str,
            stream_callback: Callable | None = None,
            **model_kwargs: dict,
            ) -> None:
        self.client = client
        self.model = model
        self.stream_callback = stream_callback
        self.model_parameters = model_kwargs or {}

    @staticmethod
    def _parse_response(response) -> OpenAIChatResponse | OpenAIChunkResponse:  # noqa: ANN001
        # chat.completion is the latest response type
        # 'chat.completion.chunk' indicates streaming
        if len(response.choices) != 1:
            raise ValueError(f"Currently only handling one choice, received {len(response.choices)}")  # noqa: E501

        is_streaming = response.object == 'chat.completion.chunk'
        is_legacy_streaming = (response.object == 'text_completion') \
            and hasattr(response.choices[0], 'delta') \
            and response.choices[0].delta
        is_non_streaming = response.object == 'chat.completion'
        is_legacy_non_streaming = (response.object == 'text_completion') \
            and hasattr(response.choices[0], 'message') \
            and response.choices[0].message

        assert is_streaming or is_legacy_streaming or is_non_streaming or is_legacy_non_streaming, f"Unexpected response object: {response.object}"  # noqa: E501

        if is_streaming or is_legacy_streaming:
            return OpenAIChunkResponse(
                object_name=response.object,
                model=response.model,
                created=response.created,
                content=response.choices[0].delta.content,
                finish_reason=response.choices[0].finish_reason,
            )
        if is_non_streaming or is_legacy_non_streaming:
            if response.choices[0].logprobs is not None:
                logprobs_tokens = [x.token for x in response.choices[0].logprobs.content]
                logprobs = [x.logprob for x in response.choices[0].logprobs.content]
            else:
                logprobs_tokens = None
                logprobs = None
            return OpenAIChatResponse(
                object_name=response.object,
                model=response.model,
                created=response.created,
                content=response.choices[0].message.content,
                finish_reason=response.choices[0].finish_reason,
                role=response.choices[0].message.role,
                usage=dict(response.usage) if response.usage else None,
                logprobs_tokens=logprobs_tokens,
                logprobs=logprobs,
            )
        raise ValueError(f"Unexpected response object: {response.object}")

    @abstractmethod
    def __call__(
            self,
            messages: list[str],
            model: str | None = None,
            stream_callback: Callable | None = None,
            **model_kwargs: dict,
            ) -> OpenAIChatResponse | OpenAIChunkResponse:
        """
        Calls the client's chat.completions.create method. Returns the parsed response. If any
        of the parameters are specified when calling the object, they will override the parameters
        specified in the constructor.
        """


class OpenAICompletionWrapper(OpenAICompletionWrapperBase):
    """Non-Async wrapper for OpenAI API."""

    def __call__(
            self,
            messages: list[str],
            model: str | None = None,
            stream_callback: Callable | None = None,
            **model_kwargs: dict,
        ) -> OpenAIChatResponse | OpenAIChunkResponse:
        """Non-Async __call__."""
        model = model or self.model
        stream_callback = stream_callback or self.stream_callback
        model_parameters = model_kwargs or self.model_parameters
        if stream_callback:
            chunks = []
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **model_parameters,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    chunk_parsed = OpenAICompletionWrapper._parse_response(chunk)
                    stream_callback(chunk_parsed)
                    chunks.append(chunk_parsed)
            # send a final chunk with no content to indicate the end of the stream
            stream_callback(OpenAIChunkResponse(
                object_name=chunk.object,
                model=chunk.model,
                created=chunk.created,
                content="",
                finish_reason=chunk.choices[0].finish_reason,  # last finish reason
            ))
            end_time = time.time()
            return OpenAIChatResponse(
                object_name='chat.completion',
                model=chunks[0].model,
                created=chunks[-1].created,
                role='assistant',
                duration_seconds=end_time - start_time,
                content="".join([chunk.content for chunk in chunks]),
                finish_reason=chunk.choices[0].finish_reason,
            )
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **model_parameters,
        )
        end_time = time.time()
        response = OpenAICompletionWrapper._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response


class AsyncOpenAICompletionWrapper(OpenAICompletionWrapperBase):
    """Async wrapper for OpenAI API."""

    async def __call__(
            self,
            messages: list[str],
            model: str | None = None,
            stream_callback: Callable | None = None,
            **model_kwargs: dict,
            ) -> OpenAIChatResponse | OpenAIChunkResponse:
        """Async __call__."""
        model = model or self.model
        stream_callback = stream_callback or self.stream_callback
        model_parameters = model_kwargs or self.model_parameters
        if stream_callback:
            chunks = []
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **model_parameters,
            )
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    chunk_parsed = OpenAICompletionWrapper._parse_response(chunk)
                    await stream_callback(chunk_parsed)
                    chunks.append(chunk_parsed)
            # send a final chunk with no content to indicate the end of the stream
            await stream_callback(OpenAIChunkResponse(
                object_name=chunk.object,
                model=chunk.model,
                created=chunk.created,
                content="",
                finish_reason=chunk.choices[0].finish_reason,  # last finish reason
            ))
            end_time = time.time()
            return OpenAIChatResponse(
                object_name='chat.completion',
                model=chunks[0].model,
                created=chunks[-1].created,
                role='assistant',
                duration_seconds=end_time - start_time,
                content="".join([chunk.content for chunk in chunks]),
                finish_reason=chunk.choices[0].finish_reason,
            )
        start_time = time.time()
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            **model_parameters,
        )
        end_time = time.time()
        response = OpenAICompletionWrapper._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response
