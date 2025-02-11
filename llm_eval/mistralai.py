"""Contains helper functions for interacting with MistralAI models."""

from abc import ABC
import time
from typing import Callable, Literal
import json

from mistralai import Mistral
from mistralai.models.chatcompletionresponse import ChatCompletionResponse
from pydantic import BaseModel, SerializeAsAny

# https://mistral.ai/technology/#pricing
CHAT_MODEL_COST_PER_TOKEN = {
    # large
    'mistral-large-latest': {'input': 2.00 / 1_000_000, 'output': 6.00 / 1_000_000},
    'mistral-large-2407': {'input': 2.00 / 1_000_000, 'output': 6.00 / 1_000_000},
    # small
    'mistral-small-latest': {'input': 0.20 / 1_000_000, 'output': 0.60 / 1_000_000},
    'mistral-small-2409': {'input': 0.20 / 1_000_000, 'output': 0.60 / 1_000_000},
    # codestral
    'codestral-latest': {'input': 0.20 / 1_000_000, 'output': 0.60 / 1_000_000},
    'codestral-2405': {'input': 0.20 / 1_000_000, 'output': 0.60 / 1_000_000},
    # ministral 3b
    'ministral-3b-latest': {'input': 0.04 / 1_000_000, 'output': 0.04 / 1_000_000},
    'ministral-3b-2410': {'input': 0.04 / 1_000_000, 'output': 0.04 / 1_000_000},
    # ministral 8b
    'ministral-8b-latest': {'input': 0.10 / 1_000_000, 'output': 0.10 / 1_000_000},
    'ministral-8b-2410': {'input': 0.10 / 1_000_000, 'output': 0.10 / 1_000_000},
    # pixtral
    'pixtral-latest': {'input': 0.15 / 1_000_000, 'output': 0.15 / 1_000_000},
    'pixtral-12b-2409': {'input': 0.15 / 1_000_000, 'output': 0.15 / 1_000_000},
    # mistral-nemo
    'mistral-nemo': {'input': 0.15 / 1_000_000, 'output': 0.15 / 1_000_000},
    # open-mistral
    'open-mistral-7b': {'input': 0.25 / 1_000_000, 'output': 0.25 / 1_000_000},
    # open-mixtral
    'open-mixtral-8x7b': {'input': 0.70 / 1_000_000, 'output': 0.70 / 1_000_000},
    # open-mixtral
    'open-mixtral-8x22b': {'input': 2.00 / 1_000_000, 'output': 6.00 / 1_000_000},
}

EMBEDDING_MODEL_COST_PER_TOKEN = {
    'mistral-embed': 0.10 / 1_000_000,
}

MODEL_COST_PER_TOKEN = CHAT_MODEL_COST_PER_TOKEN | EMBEDDING_MODEL_COST_PER_TOKEN


class MistralAIResponse(BaseModel):
    """Base class for MistralAI responses."""

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


class MistralAICompletionResponse(MistralAIResponse):
    """Stores the parsed response/content of an MistralAI chat completion chunk."""

    content: str


class MistralAIToolsResponse(MistralAIResponse):
    """Stores the parsed response/content of an MistralAI tool rseult."""

    tools: list[dict]
    role: str | None = None
    usage: dict | None = None
    duration_seconds: float | None = None


class MistralAIChatResponse(MistralAICompletionResponse):
    """Stores the parsed response/content of an MistralAI chat completion chunk."""

    content: str
    role: str | None = None
    usage: dict | None = None
    duration_seconds: float | None = None


class MistralAICompletionWrapperBase(ABC):
    """
    Wrapper for MistralAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.

    The user can specify the model name, timeout, stream, and other parameters for the API call
    either in the constructor or when calling the object. If the latter, the parameters specified
    when calling the object will override the parameters specified in the constructor.
    """

    def __init__(
        self,
        client: Mistral,
        model: str,
        stream_callback: Callable | None = None,
        **model_kwargs: dict,
    ) -> None:
        self.client = client
        self.model = model
        self.stream_callback = stream_callback
        self.model_parameters = model_kwargs or {}

    @staticmethod
    def _parse_response(
        response: ChatCompletionResponse,
    ) -> MistralAIChatResponse | MistralAICompletionResponse:
        # chat.completion is the latest response type
        # 'chat.completion.chunk' indicates streaming
        if hasattr(response, "data"):
            response = response.data

        if len(response.choices) != 1:
            raise ValueError(
                f"Currently only handling one choice, received {len(response.choices)}",
            )

        is_function_call = (
            response.object == "chat.completion"
            and hasattr(response.choices[0], "message")
            and hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        )

        def _try_parse(value):  # noqa
            if not value:
                return None
            try:
                return json.loads(value)
            except:  # noqa: E722
                return value

        if is_function_call:
            tools = [
                {
                    "type": tool.type,
                    "name": tool.function.name,
                    "arguments": _try_parse(tool.function.arguments),
                }
                for tool in response.choices[0].message.tool_calls
            ]
            return MistralAIToolsResponse(
                object_name="tools",
                model=response.model,
                created=response.created,
                tools=tools,
                finish_reason=response.choices[0].finish_reason,
                role=response.choices[0].message.role,
                usage=dict(response.usage) if response.usage else None,
            )
        is_streaming = response.object == "chat.completion.chunk"
        is_legacy_streaming = (
            (response.object == "text_completion")
            and hasattr(response.choices[0], "delta")
            and response.choices[0].delta
        )
        is_non_streaming = response.object == "chat.completion"
        is_legacy_non_streaming = (
            (response.object == "text_completion")
            and hasattr(response.choices[0], "message")
            and response.choices[0].message
        )

        assert (
            is_streaming
            or is_legacy_streaming
            or is_non_streaming
            or is_legacy_non_streaming
        ), f"Unexpected response object: {response.object}"

        if is_streaming or is_legacy_streaming:
            return MistralAICompletionResponse(
                object_name=response.object,
                model=response.model,
                created=response.created,
                content=response.choices[0].delta.content,
                finish_reason=response.choices[0].finish_reason,
            )
        if is_non_streaming or is_legacy_non_streaming:
            return MistralAIChatResponse(
                object_name=response.object,
                model=response.model,
                created=response.created,
                content=response.choices[0].message.content,
                finish_reason=response.choices[0].finish_reason,
                role=response.choices[0].message.role,
                usage=dict(response.usage) if response.usage else None,
            )
        raise ValueError(f"Unexpected response object: {response.object}")


class MistralAICompletion(MistralAICompletionWrapperBase):
    """
    Non-Async wrapper for MistralAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    def __call__(
        self,
        messages: list[str],
        model: str | None = None,
        stream_callback: Callable | None = None,
        **model_kwargs: dict,
    ) -> MistralAIChatResponse | MistralAICompletionResponse:
        """Non-Async __call__."""
        model = model or self.model
        stream_callback = stream_callback or self.stream_callback
        model_parameters = {**self.model_parameters, **model_kwargs}
        if stream_callback:
            chunks = []
            start_time = time.time()
            response = self.client.chat.stream(
                model=model,
                messages=messages,
                stream=True,
                **model_parameters,
            )
            for chunk in response:
                if chunk.data.choices[0].delta.content:
                    chunk_parsed = MistralAICompletion._parse_response(chunk)
                    stream_callback(chunk_parsed)
                    chunks.append(chunk_parsed)
            # send a final chunk with no content to indicate the end of the stream
            stream_callback(
                MistralAICompletionResponse(
                    object_name=chunk.data.object,
                    model=chunk.data.model,
                    created=chunk.data.created,
                    content="",
                    finish_reason=chunk.data.choices[0].finish_reason,  # last finish reason
                ),
            )
            end_time = time.time()
            if hasattr(chunk, "data"):
                chunk = chunk.data
            return MistralAIChatResponse(
                object_name="chat.stream",
                model=chunk.model,
                created=chunk.created,
                role="assistant",
                duration_seconds=end_time - start_time,
                content="".join([chunk.content for chunk in chunks]),
                finish_reason=chunk.choices[0].finish_reason,
            )
        start_time = time.time()
        response = self.client.chat.complete(
            model=model,
            messages=messages,
            stream=False,
            **model_parameters,
        )
        end_time = time.time()
        response = MistralAICompletion._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response


class AsyncMistralAICompletion(MistralAICompletionWrapperBase):
    """
    Async wrapper for MistralAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    async def __call__(
            self,
            messages: list[str],
            model: str | None = None,
            stream_callback: Callable | None = None,
            **model_kwargs: dict,
            ) -> MistralAIChatResponse | MistralAICompletionResponse:
        """Async __call__."""
        model = model or self.model
        stream_callback = stream_callback or self.stream_callback
        model_parameters = {**self.model_parameters, **model_kwargs}
        if stream_callback:
            chunks = []
            start_time = time.time()
            response = await self.client.chat.complete_async(
                model=model,
                messages=messages,
                stream=True,
                **model_parameters,
            )
            async for chunk in response:
                if chunk.data.choices[0].delta.content:
                    chunk_parsed = MistralAICompletion._parse_response(chunk)
                    await stream_callback(chunk_parsed)
                    chunks.append(chunk_parsed)
            # send a final chunk with no content to indicate the end of the stream
            await stream_callback(MistralAICompletionResponse(
                object_name=chunk.data.object,
                model=chunk.model,
                created=chunk.created,
                content="",
                finish_reason=chunk.data.choices[0].finish_reason,  # last finish reason
            ))
            end_time = time.time()
            if hasattr(chunk, "data"):
                chunk = chunk.data
            return MistralAIChatResponse(
                object_name='chat.completion',
                model=chunk.model,
                created=chunk.created,
                role='assistant',
                duration_seconds=end_time - start_time,
                content="".join([chunk.content for chunk in chunks]),
                finish_reason=chunk.choices[0].finish_reason,
            )
        start_time = time.time()
        response = await self.client.chat.complete_async(
            model=model,
            messages=messages,
            stream=False,
            **model_parameters,
        )
        end_time = time.time()
        response = MistralAICompletion._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response


class MistralAITools(MistralAICompletionWrapperBase):
    """
    Wrapper for MistralAI Tools API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    def __call__(
            self,
            messages: list[str],
            tools: list[dict],
            tool_choice: Literal['none', 'auto', 'any'] | dict[str] = 'any',
            model: str | None = None,
            **model_kwargs: dict,
        ) -> MistralAIToolsResponse | MistralAICompletionResponse:
        """Call the MistralAI Tools API and return the response."""
        model = model or self.model
        model_parameters = {**self.model_parameters, **model_kwargs}
        start_time = time.time()
        response = self.client.chat.complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **model_parameters,
        )
        end_time = time.time()
        response = MistralAICompletion._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response
