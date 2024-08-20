"""Contains helper functions for interacting with OpenAI models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Callable, Literal
from functools import cache
import json
from openai import OpenAI
from pydantic import BaseModel, SerializeAsAny
import tiktoken
from tiktoken import Encoding


CHAT_MODEL_COST_PER_TOKEN = {
    # LATEST MODELS
    'gpt-4o': {'input': 5.00 / 1_000_000, 'output': 15.00 / 1_000_000},
    'gpt-4o-2024-05-13': {'input': 5.00 / 1_000_000, 'output': 15.00 / 1_000_000},

    'gpt-4o-mini':  {'input': 0.15 / 1_000_000, 'output': 0.60 / 1_000_000},
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

def user_message(message: str) -> dict:
    """Returns a user message."""
    return {'role': 'user', 'content': message}

def assistant_message(message: str) -> dict:
    """Returns an assistant message."""
    return {'role': 'assistant', 'content': message}

def system_message(message: str) -> dict:
    """Returns a system message."""
    return {'role': 'system', 'content': message}


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


class OpenAICompletionResponse(OpenAIResponse):
    """Stores the parsed response/content of an OpenAI chat completion chunk."""

    content: str


class OpenAIToolsResponse(OpenAIResponse):
    """Stores the parsed response/content of an OpenAI tool rseult."""

    tools: list[dict]
    role: str | None = None
    usage: dict | None = None
    duration_seconds: float | None = None


class OpenAIChatResponse(OpenAICompletionResponse):
    """Stores the parsed response/content of an OpenAI chat completion chunk."""

    content: str
    role: str | None = None
    usage: dict | None = None
    duration_seconds: float | None = None
    logprobs_tokens: list[str] | None = None
    logprobs: list[float] | None = None


class OpenAICompletionWrapperBase(ABC):
    """
    Wrapper for OpenAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.

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
    def _parse_response(response) -> OpenAIChatResponse | OpenAICompletionResponse:  # noqa: ANN001
        # chat.completion is the latest response type
        # 'chat.completion.chunk' indicates streaming
        if len(response.choices) != 1:
            raise ValueError(f"Currently only handling one choice, received {len(response.choices)}")  # noqa: E501

        is_function_call = (
            response.object == 'chat.completion'
            and hasattr(response.choices[0], 'message')
            and hasattr(response.choices[0].message, 'tool_calls')
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
            return OpenAIToolsResponse(
                object_name='tools',
                model=response.model,
                created=response.created,
                tools=tools,
                finish_reason=response.choices[0].finish_reason,
                role=response.choices[0].message.role,
                usage=dict(response.usage) if response.usage else None,
            )
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
            return OpenAICompletionResponse(
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
            ) -> OpenAIChatResponse | OpenAICompletionResponse:
        """
        Calls the client's chat.completions.create method. Returns the parsed response. If any
        of the parameters are specified when calling the object, they will override the parameters
        specified in the constructor.
        """


class OpenAICompletion(OpenAICompletionWrapperBase):
    """
    Non-Async wrapper for OpenAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    def __call__(
            self,
            messages: list[str],
            model: str | None = None,
            stream_callback: Callable | None = None,
            **model_kwargs: dict,
        ) -> OpenAIChatResponse | OpenAICompletionResponse:
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
                    chunk_parsed = OpenAICompletion._parse_response(chunk)
                    stream_callback(chunk_parsed)
                    chunks.append(chunk_parsed)
            # send a final chunk with no content to indicate the end of the stream
            stream_callback(OpenAICompletionResponse(
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
        response = OpenAICompletion._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response


class AsyncOpenAICompletion(OpenAICompletionWrapperBase):
    """
    Async wrapper for OpenAI API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    async def __call__(
            self,
            messages: list[str],
            model: str | None = None,
            stream_callback: Callable | None = None,
            **model_kwargs: dict,
            ) -> OpenAIChatResponse | OpenAICompletionResponse:
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
                    chunk_parsed = OpenAICompletion._parse_response(chunk)
                    await stream_callback(chunk_parsed)
                    chunks.append(chunk_parsed)
            # send a final chunk with no content to indicate the end of the stream
            await stream_callback(OpenAICompletionResponse(
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
        response = OpenAICompletion._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response


_VALID_PARAM_TYPES = {"string", "number", "integer", "object", "array", "boolean", "null"}

@dataclass
class FunctionParameter:
    """
    The Function and FunctionParameter classes are used to generate "functions" in the OpenAI
    "tools" API. These classes can be used to to convienently define build a list of tools without
    having to manually construct the JSON schema for each tool. Note that the `tools` parameter in
    the OpenAITools class takes a list of dictionaries (not Function/FunctionParameter objects).
    Using these classes are optional, but can be helpful for defining tools with many parameters.
    To use these classes, you can call the `to_dict` method on a Function object to get the
    dictionary representation of the tool.

    https://platform.openai.com/docs/api-reference/chat/create
    https://platform.openai.com/docs/guides/function-calling

    The FunctionParameter class represents a single parameter for a function in the OpenAI API.
    """

    name: str
    type: Literal["string", "number", "integer", "object", "array", "boolean", "null"]
    description: str | None = None
    valid_values: list[str] | None = None
    required: bool = False

    def __post_init__(self):
        if self.type not in _VALID_PARAM_TYPES:
            raise ValueError(f"Invalid type '{self.type}'. Must be one of {_VALID_PARAM_TYPES}.")

    def to_dict(self) -> dict:  # noqa: D102
        param_dict = {"type": self.type}
        if self.description:
            param_dict["description"] = self.description
        if self.valid_values:
            param_dict["enum"] = self.valid_values
        return param_dict


@dataclass
class Function:
    """
    The Function and FunctionParameter classes are used to generate "functions" in the OpenAI
    "tools" API. These classes can be used to to convienently define build a list of tools without
    having to manually construct the JSON schema for each tool. Note that the `tools` parameter in
    the OpenAITools class takes a list of dictionaries (not Function/FunctionParameter objects).
    Using these classes are optional, but can be helpful for defining tools with many parameters.
    To use these classes, you can call the `to_dict` method on a Function object to get the
    dictionary representation of the tool.

    https://platform.openai.com/docs/api-reference/chat/create
    https://platform.openai.com/docs/guides/function-calling

    The FunctionParameter class represents a single parameter for a function in the OpenAI API.
    """

    name: str
    parameters: list[FunctionParameter]
    description: str | None = None
    strict: bool = False

    def to_dict(self) -> dict:  # noqa: D102
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)

        tool_dict = {"name": self.name}
        if self.description:
            tool_dict["description"] = self.description
        tool_dict["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
        if self.strict:
            tool_dict["strict"] = self.strict
        return {"type": "function", "function": tool_dict}


class OpenAITools(OpenAICompletionWrapperBase):
    """
    Wrapper for OpenAI Tools API which provides a simple interface for calling the
    chat.completions.create method and parsing the response.
    """

    def __call__(
            self,
            messages: list[str],
            tools: list[dict],
            tool_choice: Literal['none', 'auto', 'required'] | dict[str] = 'required',
            model: str | None = None,
            **model_kwargs: dict,
        ) -> OpenAIToolsResponse | OpenAICompletionResponse:
        """
        TODO.
        For example, OpenAICompletionResponse can be returned if `auto` and unrelated question.
        """
        model = model or self.model
        model_parameters = model_kwargs or self.model_parameters
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **model_parameters,
        )
        end_time = time.time()
        response = OpenAICompletion._parse_response(response)
        response.duration_seconds = end_time - start_time
        return response
