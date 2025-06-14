"""Contains helper functions for interacting with Anthropic models."""

from abc import ABC
from collections.abc import Callable
import time
import datetime
from typing import Literal
import json

from anthropic import Anthropic
from anthropic.types import (
    Message,
    ToolUseBlock,
    RawMessageStartEvent,
    RawMessageDeltaEvent,
)
from pydantic import BaseModel, SerializeAsAny


# https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table
CHAT_MODEL_COST_PER_TOKEN = {
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    "claude-3-5-sonnet-20240620": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    # Claude 3 Opus
    "claude-3-opus-20240229": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
    },
    # Claude 3 Haiku
    "claude-3-haiku-20240307": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    # Claude 2.1
    "claude-2.1": {"input": 8.00 / 1_000_000, "output": 24.00 / 1_000_000},
    # Claude 2
    "claude-2.0": {"input": 8.00 / 1_000_000, "output": 24.00 / 1_000_000},
    # Claude Instant 1.2
    "claude-instant-1.2": {"input": 0.80 / 1_000_000, "output": 2.40 / 1_000_000},
}

EMBEDDING_MODEL_COST_PER_TOKEN = {}

MODEL_COST_PER_TOKEN = CHAT_MODEL_COST_PER_TOKEN | EMBEDDING_MODEL_COST_PER_TOKEN

USAGE_ALIASES = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
}


class AnthropicResponse(BaseModel):
    """Base class for Anthropic responses."""

    object_name: str
    model: str
    created: int
    metadata: dict | SerializeAsAny[BaseModel] | None = None
    finish_reason: str | None = None


class AnthropicCompletionResponse(AnthropicResponse):
    """Stores the parsed response/content of an Anthropic chat completion chunk."""

    content: str


class AnthropicToolsResponse(AnthropicResponse):
    """Stores the parsed response/content of an Anthropic tool rseult."""

    tools: list[dict]
    role: str | None = None
    usage: dict | None = None
    duration_seconds: float | None = None


class AnthropicChatResponse(AnthropicCompletionResponse):
    """Stores the parsed response/content of an Anthropic chat completion chunk."""

    content: str
    role: str | None = "assistant"
    usage: dict | None = None
    duration_seconds: float | None = None


class AnthropicCompletionWrapperBase(ABC):
    """
    Wrapper for Anthropic API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.

    The user can specify the model name, timeout, stream, and other parameters for the API call
    either in the constructor or when calling the object. If the latter, the parameters specified
    when calling the object will override the parameters specified in the constructor.
    """

    def __init__(
        self,
        client: Anthropic,
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
        response: Message | RawMessageDeltaEvent,
        is_function_call: bool = False,
        model: str | None = None,
    ) -> AnthropicChatResponse | AnthropicCompletionResponse:
        created = int(datetime.datetime.now().timestamp())

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
                    "name": tool.name,
                    "arguments": _try_parse(tool.input),
                }
                for tool in response.content
                if isinstance(tool, ToolUseBlock)
            ]
            usage = (
                {USAGE_ALIASES.get(k, k): v for k, v in response.usage}
                if response.usage
                else None
            )
            usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get(
                "output_tokens", 0,
            )
            return AnthropicToolsResponse(
                object_name="tools",
                model=model,
                created=created,
                tools=tools,
                finish_reason=response.stop_reason,
                role="assistant",
                usage=usage,
            )
        is_non_streaming = hasattr(response, "content")
        is_streaming = not is_non_streaming

        stop_reason = ""
        if is_streaming:
            if isinstance(response, RawMessageDeltaEvent):
                stop_reason = response.delta.stop_reason
            return AnthropicCompletionResponse(
                object_name="chat.completion.chunk",
                model=model,
                created=created,
                content=response.delta.text if response.delta.text else "",
                finish_reason=stop_reason,
            )
        if is_non_streaming:
            usage = (
                {USAGE_ALIASES.get(k, k): v for k, v in response.usage}
                if response.usage
                else None
            )
            usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get(
                "output_tokens", 0,
            )
            return AnthropicChatResponse(
                object_name="chat.completion",
                model=response.model,
                created=created,
                content=response.content[0].text,
                finish_reason=response.stop_reason,
                role="assistant",
                usage=usage,
            )
        raise ValueError(f"Unexpected response object: {response.object}")


class AnthropicCompletion(AnthropicCompletionWrapperBase):
    """
    Non-Async wrapper for Anthropic API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    def __call__(
        self,
        messages: list[str],
        model: str | None = None,
        stream_callback: Callable | None = None,
        **model_kwargs: dict,
    ) -> AnthropicChatResponse | AnthropicCompletionResponse:
        """Non-Async __call__."""
        created = int(datetime.datetime.now().timestamp())
        model = model or self.model
        stream_callback = stream_callback or self.stream_callback
        model_parameters = {**self.model_parameters, **model_kwargs}
        if stream_callback:
            chunks = []
            start_time = time.perf_counter()
            response = self.client.messages.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=model_parameters.pop("max_tokens", 2048),
                **model_parameters,
            )
            stop_reason = ""
            for chunk in response:
                if isinstance(chunk, RawMessageStartEvent):
                    model = chunk.message.model
                elif isinstance(chunk, RawMessageDeltaEvent):
                    stop_reason = chunk.delta.stop_reason
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text") and chunk.delta.text:
                    chunk_parsed = AnthropicCompletion._parse_response(
                        chunk,
                        is_function_call="tool_use" in stop_reason,
                        model=model,
                    )
                    stream_callback(chunk_parsed)
                    chunks.append(chunk_parsed)
            # send a final chunk with no content to indicate the end of the stream
            stream_callback(
                AnthropicCompletionResponse(
                    object_name="chat.completion.chunk",
                    model=model,
                    created=created,
                    content="",
                    finish_reason=stop_reason,  # last finish reason
                ),
            )
            end_time = time.perf_counter()
            return AnthropicChatResponse(
                object_name="chat.completion",
                model=model,
                created=created,
                duration_seconds=end_time - start_time,
                content="".join(chunk.content for chunk in chunks),
                finish_reason=stop_reason,
            )
        start_time = time.perf_counter()
        response = self.client.messages.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=model_parameters.pop("max_tokens", 2048),
            **model_parameters,
        )
        end_time = time.perf_counter()
        response = AnthropicCompletion._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response


class AsyncAnthropicCompletion(AnthropicCompletionWrapperBase):
    """
    Async wrapper for Anthropic API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    async def __call__(
        self,
        messages: list[str],
        model: str | None = None,
        stream_callback: Callable | None = None,
        **model_kwargs: dict,
    ) -> AnthropicChatResponse | AnthropicCompletionResponse:
        """Async __call__."""
        created = datetime.datetime.now().timestamp()
        model = model or self.model
        stream_callback = stream_callback or self.stream_callback
        model_parameters = {**self.model_parameters, **model_kwargs}
        if stream_callback:
            chunks = []
            start_time = time.perf_counter()
            response = self.client.messages.create(
                model=model,
                messages=messages,
                stream=True,
                **model_parameters,
            )
            stop_reason = ""
            for chunk in response:
                if isinstance(chunk, RawMessageStartEvent):
                    model = chunk.message.model
                elif isinstance(chunk, RawMessageDeltaEvent):
                    stop_reason = chunk.delta.stop_reason

                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text") and chunk.delta.text:
                    chunk_parsed = AnthropicCompletion._parse_response(
                        chunk,
                        is_function_call="tool_use" in chunk.delta.stop_reason,
                        model=model,
                    )
                    stream_callback(chunk_parsed)
                    chunks.append(chunk_parsed)
            # send a final chunk with no content to indicate the end of the stream
            stream_callback(
                AnthropicCompletionResponse(
                    object_name="chat.completion.chunk",
                    model=model,
                    created=created,
                    content="",
                    finish_reason=stop_reason,  # last finish reason
                ),
            )
            end_time = time.perf_counter()
            return AnthropicChatResponse(
                object_name="chat.completion",
                model=model,
                created=created,
                duration_seconds=end_time - start_time,
                content="".join(chunk.content for chunk in chunks),
                finish_reason=stop_reason,
            )
        start_time = time.perf_counter()
        response = self.client.messages.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=model_parameters.pop("max_tokens", 2048),
            **model_parameters,
        )
        end_time = time.perf_counter()
        response = AnthropicCompletion._parse_response(
            response,
            is_function_call="tool_use" in response.stop_reason,
            model=model,
        )
        response.duration_seconds = end_time - start_time
        return response


class AnthropicTools(AnthropicCompletionWrapperBase):
    """
    Wrapper for Anthropic Tools API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    def __call__(
        self,
        messages: list[str],
        tools: list[dict],
        tool_choice: Literal["tool", "auto", "any"] | dict[str] = "any",
        model: str | None = None,
        **model_kwargs: dict,
    ) -> AnthropicToolsResponse | AnthropicCompletionResponse:
        """Call the Anthropic Tools API and return the response."""
        model = model or self.model
        model_parameters = {**self.model_parameters, **model_kwargs}
        start_time = time.perf_counter()
        response = self.client.messages.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=model_parameters.pop("max_tokens", 2048),
            **model_parameters,
        )
        end_time = time.perf_counter()
        response = AnthropicCompletion._parse_response(
            response,
            is_function_call="tool_use" in response.stop_reason,
            model=model,
        )
        response.duration_seconds = end_time - start_time
        return response
