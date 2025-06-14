"""
Defines classes for different types of built-in Candidates and a corresponding registry system for
custom Candidates.

A Candidate is a callable object that takes input (e.g. OpenAI-style messages) and returns a
response. The input and response can be any type of object that can be serialized/de-serialized
(e.g. string, dictionary, list, etc.).

IMPORTANT: Candidate objects should be stateless, meaning the objects should not retain any
state between calls (`__call__` invocations). This is because the Candidate objects are reused
between different evals and tests, and the state of the object should not be retained/used between
different evals/tests.

The purpose of a Candidate is to provide a standard interface that A) allows the user to define
evals in a consistent manner and B) allows the results of the evals to be evaluated (checked) in a
consistent manner. A Candidate is essentially an adapter that takes the input from the Eval and
converts it to the input format that the underlying LLM (e.g. OpenAI) expects, and then converts
the response from the LLM to the response format that the Eval/Checks expects.

Candidates can be created from a dictionary using the `Candidate.from_dict(...)` method. The
dictionary must have a `candidate_type` field that matches the type name of the registered
Candidate class. This allows the user to, for example, have a directory of yaml files that define
different Candidates and load them in bulk to use in the EvalHarness (evals.py). The EvalHarness
will instantiate the correct (registered) Candidate class based on the `candidate_type` field in
the yaml file.

Candidates can also be passed to the EvalHarness (or Eval object) directory as an Candidate
(subclass) object or simply as a function. The benefit of using a Candidate object is that it can
can be serialized into a dictionary and the information can be saved in the EvalResult object
(evals.py).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, UTC
from inspect import iscoroutinefunction
from copy import deepcopy
from abc import ABC, abstractmethod
from enum import Enum, auto
from textwrap import dedent
from collections.abc import Callable
from typing import Any, Literal
from pydantic import BaseModel, Field
# from openai import OpenAI
from sik_llms import OpenAI, OpenAITools, Anthropic, AnthropicTools, Parameter, TextResponse, Tool, ToolChoice, ToolPredictionResponse
from sik_llm_eval.openai import (
    MODEL_COST_PER_TOKEN as OPENAI_MODEL_COST_PER_TOKEN,
    OpenAICompletion,
    OpenAIToolsResponse,
    OpenAICompletionResponse,
)
from sik_llm_eval.internal_utilities import (
    DictionaryEqualsMixin,
    EnumMixin,
    Registry,
    SerializationMixin,
)
from sik_llm_eval.anthropic import (
    AnthropicCompletion,
    AnthropicCompletionResponse,
    AnthropicToolsResponse,
)


@staticmethod
def is_async_candidate(candidate: Callable | Candidate) -> bool:
    """Tests if the Candidate object or callable is an async function."""
    if iscoroutinefunction(candidate):
        return True
    if hasattr(candidate, "__call__"):
        return iscoroutinefunction(candidate.__call__)
    return False


class CandidateType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of Candidates."""

    ANTHROPIC = auto()
    ANTROPIC_TOOLS = auto()
    OPENAI = auto()
    OPENAI_TOOLS = auto()


class CandidateResponse(BaseModel):
    """
    Provides a standard response object for Candidates so that the Eval/TestHarness can
    consistently evaluate the response and store any metadata (e.g. cost, usage, etc.) for the
    response.

    Content is the text/dict/etc. from the LLM that is meant to be evaluated (via Check objects).
    Metadata is a dictionary of metadata about the response (e.g. cost, usage, etc.).
    """

    response: object
    metadata: dict | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Candidate(SerializationMixin, DictionaryEqualsMixin, ABC):
    """
    A Candidate describes an LLM and the client for interfacing with the LLM (or specific
    implementation of an LLM interface (e.g. history/context management)) along with optional
    model parameters.

    A Candidate is a callable object that takes an input and returns a CandidateResponse.
    """

    registry = Registry()

    def __init__(
        self,
        metadata: dict | None = None,
        parameters: dict | None = None,
    ) -> None:
        """
        Initialize a Candidate object.

        Args:
            metadata: A dictionary of metadata about the Candidate. This information will be
                included in the CandidateResponse and can be used to store information about the
                Candidate (e.g. the model name, the model version, the model provider, etc.).
            parameters: A dictionary of parameters for the Candidate (most likely, the model
                parameters passed to the LLM).
        """
        self.metadata = deepcopy(metadata) or {}
        self.parameters = deepcopy(parameters)

    @abstractmethod
    def __call__(self, input: Any) -> CandidateResponse:  # noqa: A002, ANN401
        """Invokes the underlying model with the input and returns the response."""

    @classmethod
    def register(cls, candidate_type: str | Enum):
        """Register a subclass of Candidate."""

        def decorator(subclass: type[Candidate]) -> type[Candidate]:
            assert issubclass(
                subclass,
                Candidate,
            ), f"Candidate '{candidate_type}' ({subclass.__name__}) must extend Candidate"
            cls.registry.register(type_name=candidate_type, item=subclass)
            return subclass

        return decorator

    @classmethod
    def is_registered(cls, candidate_type: str | Enum) -> bool:
        """Check if a candidate type is registered."""
        return candidate_type in cls.registry

    @classmethod
    def from_dict(
        cls: type[Candidate],
        data: dict,
    ) -> Candidate | list[Candidate]:
        """
        Creates a Candidate object.

        This method requires that the Candidate subclass has been registered with the `register`
        decorator before calling this method. It also requires that the dictionary has a
        `candidate_type` field that matches the type name of the registered Candidate subclass.
        """
        data = deepcopy(data)
        candidate_type = data.pop("candidate_type", "")
        if cls.is_registered(candidate_type):
            return cls.registry.create_instance(type_name=candidate_type, **data)
        raise ValueError(f"Unknown Candidate type `{candidate_type}`")

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        # value = self.model_dump(exclude_defaults=True, exclude_none=True)
        value = {}
        if self.metadata:
            value["metadata"] = deepcopy(self.metadata)
        if self.parameters:
            value["parameters"] = deepcopy(self.parameters)
        if self.candidate_type:
            value["candidate_type"] = self.candidate_type
        return value

    @property
    def candidate_type(self) -> str | None:
        """The type of Candidate."""
        # check that self.__class__ has _type_name attribute
        if hasattr(self.__class__, "_type_name"):
            return self.__class__._type_name.upper()
        return self.__class__.__name__

    def __str__(self) -> str:
        """Returns a string representation of the Candidate."""
        parameters = (
            ""
            if not self.parameters
            else f"\n            parameters={self.parameters},"
        )
        return dedent(
            f"""
        {self.__class__.__name__}(
            metadata={self.metadata},
            {parameters}
        )
        """,
        ).strip()


class BuiltinCandidate(Candidate, ABC):
    """Wrapper around classes that expose builtin Candidates (uses `sik-lmms` package)."""

    def __init__(  # noqa: D417
        self,
        model_name: str | None = None,
        endpoint_url: str | None = None,
        metadata: dict | None = None,
        parameters: dict | None = None,
    ) -> None:
        """
        Initialize a Service object.

        Args:
            model:
                The name of the model to use.
            endpoint_url:
                This parameter is used when running against a local service-compatible API endpoint.
            metadata:
                A dictionary of metadata about the Candidate.
            parameters:
                A dictionary of model-specific parameters (e.g. `temperature`).
        """  # noqa
        super().__init__(metadata=metadata, parameters=parameters)
        assert model_name or endpoint_url, "model or endpoint_url must be provided"
        self.model = model_name
        self.endpoint_url = endpoint_url

    @property
    @abstractmethod
    def client_callable(self) -> Any:  # noqa: ANN401
        """Return the client for the service."""

    @abstractmethod
    def _parse_response(self, response: Any) -> str | dict:  # noqa: ANN401
        """Get the response content from the response object."""
        pass

    def __call__(self, input: list[dict[str, str]]) -> CandidateResponse:  # noqa: A002
        """Invokes the underlying model with the input and returns the response."""
        start = time.perf_counter()
        summary = self._invoke_client_callable(input)
        duration = time.perf_counter() - start
        input_tokens = summary.input_tokens
        output_tokens = summary.output_tokens
        total_tokens = summary.total_tokens
        input_cost = summary.input_cost
        output_cost = summary.output_cost
        total_cost = summary.total_cost

        response = self._parse_response(summary)
        output_characters = (
            len(response) if isinstance(response, str) else None
        )
        return CandidateResponse(
            response=response,
            metadata={
                "type": (
                    "tools"
                    if isinstance(summary, OpenAIToolsResponse)
                    else "completion"
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "output_characters": output_characters,
                "duration_seconds": duration,
            },
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        value = super().to_dict()
        if self.model:
            value["model_name"] = self.model
        if self.endpoint_url:
            value["endpoint_url"] = self.endpoint_url
        return value


@Candidate.register(CandidateType.OPENAI)
class OpenAICandidate(BuiltinCandidate):
    """
    Wrapper around the OpenAI API that allows the user to create an OpenAI candidate from a
    dictionary.

    NOTE: the `OPENAI_API_KEY` environment variable must be set to use this class.
    """

    # @property
    # def model_cost_per_token(self) -> float | None:
    #     """
    #     Return the cost per token for the model. This is used to calculate the cost of the
    #     completion.
    #     """
    #     return OPENAI_MODEL_COST_PER_TOKEN.get(self.model)

    def _parse_response(self, response: TextResponse) -> str:
        return response.response

    @property
    def client_callable(self) -> OpenAICompletion:
        """Return the client for the OpenAI service."""
        return OpenAI(
            server_url=self.endpoint_url,
            model_name=self.model,
            **self.parameters or {},
        )

    def _invoke_client_callable(
        self,
        input: list[dict[str, str]],  # noqa: A002
    ) -> OpenAICompletionResponse:
        """Invoke the client with the input and return the response."""
        return self.client_callable(input)


@Candidate.register(CandidateType.OPENAI_TOOLS)
class OpenAIToolsCandidate(OpenAICandidate):
    """
    Wrapper around the OpenAI Tools API that allows the user to create an OpenAI candidate from a
    dictionary.

    NOTE: the `OPENAI_API_KEY` environment variable must be set to use this class.
    """

    def __init__(  # noqa: D417
        self,
        tools: list[dict] | list[Tool],
        tool_choice: str | ToolChoice,
        model_name: str | None = None,
        endpoint_url: str | None = None,
        metadata: dict | None = None,
        parameters: dict | None = None,
    ) -> None:
        """
        Initialize a OpenAIToolsCandidate object.

        Args:
            tools:
                A list of tools to use with the OpenAI model. See https://platform.openai.com/docs/api-reference/chat/create
                for more information.

                The Function and FunctionParameter classes in `openai.py` can be used to create the
                tools list. See `openai.py` for more information.
            tool_choice:
                See https://platform.openai.com/docs/guides/function-calling/configuring-function-calling-behavior-using-the-tool_choice-parameter
            model:
                The name of the OpenAI model to use (e.g. 'gpt-4o-mini').
            endpoint_url:
                This parameter is used when running against a local OpenAI-compatible API endpoint.
            metadata:
                A dictionary of metadata about the Candidate.
            parameters:
                A dictionary of model-specific parameters (e.g. `temperature`).
        """  # noqa
        super().__init__(
            model_name=model_name,
            endpoint_url=endpoint_url,
            metadata=metadata,
            parameters=parameters,
        )
        if isinstance(tools[0], dict):
            tools = [Tool(**tool) for tool in tools]

        if not all(isinstance(tool, Tool) for tool in tools):
            raise ValueError(
                "All tools must be instances of Tool or a dictionary that can be converted to Tool.",  # noqa: E501
            )
        self.tools = tools
        self.tool_choice = tool_choice

    @property
    def client_callable(self) -> OpenAICompletion:
        """Return the client for the OpenAI service."""
        tool_choice = self.tool_choice
        if isinstance(tool_choice, str):
            tool_choice = ToolChoice[tool_choice.upper()]
        return OpenAITools(
            server_url=self.endpoint_url,
            model_name=self.model,
            tools=self.tools,
            tool_choice=tool_choice,
            **self.parameters or {},
        )

    def _invoke_client_callable(
        self,
        input: list[dict[str, str]],  # noqa: A002
    ) -> OpenAIToolsResponse | OpenAICompletionResponse:
        """Invoke the client with the input and return the response."""
        return self.client_callable(
            messages=input,
        )

    def _parse_response(
        self,
        response: ToolPredictionResponse,
    ) -> str | dict:
        """Get the desired attribute from the response object."""
        return (
            response.tool_prediction
            if response.tool_prediction
            else response.message
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        value = super().to_dict()
        tools = []
        # convert parameter types to strings for serialization
        # e.g. <class 'str'> to 'str'
        for t in self.tools:
            if not isinstance(t, Tool):
                raise ValueError(
                    "Tools must be instances of Tool or a dictionary that can be converted to Tool.",  # noqa: E501
                )
            tool = t.model_dump(exclude_defaults=True, exclude_none=True)
            parameters = tool.get("parameters", [])
            for param in parameters:
                param['param_type'] = param['param_type'].__name__
            tools.append(tool)
        value["tools"] = tools
        value["tool_choice"] = self.tool_choice
        return value


@Candidate.register(CandidateType.ANTROPIC)
class AnthropicCandidate(BuiltinCandidate):
    """
    Wrapper around the Anthropic API that allows the user to create an Anthropic candidate from
    a dictionary.

    NOTE: the `ANTHROPIC_API_KEY` environment variable must be set to use this class.
    """

    @property
    def client_callable(self) -> AnthropicCompletion:
        """Return the client for the Anthropic service."""
        from anthropic import Anthropic
        from sik_llm_eval.anthropic import AnthropicCompletion

        parameters = self.parameters or {}
        api_key = parameters.pop("api_key", os.getenv("ANTHROPIC_API_KEY"))
        return AnthropicCompletion(
            client=Anthropic(api_key=api_key),
            model=self.model,
            max_tokens=parameters.pop("max_tokens", 2048),
            **parameters,
        )

    @property
    def model_cost_per_token(self) -> float | None:
        """
        Return the cost per token for the model. This is used to calculate the cost of the
        completion.
        """
        from sik_llm_eval.anthropic import (
            MODEL_COST_PER_TOKEN as ANTHROPIC_MODEL_COST_PER_TOKEN,
        )

        return ANTHROPIC_MODEL_COST_PER_TOKEN.get(self.model)

    def _invoke_client_callable(
        self,
        input: list[dict[str, str]],  # noqa: A002
    ) -> AnthropicCompletionResponse | AnthropicToolsResponse:
        """Invoke the client with the input and return the response."""
        return self.client_callable(messages=input)


@Candidate.register(CandidateType.ANTROPIC_TOOLS)
class AnthropicToolsCandidate(AnthropicCandidate):
    """
    Wrapper around the Anthropic Tools API that allows the user to create an Anthropic candidate
    from a dictionary.

    NOTE: the `ANTHROPIC_API_KEY` environment variable must be set to use this class.
    """

    def __init__(  # noqa: D417
        self,
        tools: list[dict],
        tool_choice: Literal["auto", "any", "none"] | dict[str] = "auto",
        model: str | None = None,
        endpoint_url: str | None = None,
        metadata: dict | None = None,
        parameters: dict | None = None,
    ) -> None:
        """
        Initialize an AnthropicToolsCandidate object.

        Args:
            tools:
                A list of tools to use with the Anthropic model. See
                https://docs.anthropic.com/en/docs/tools-overview for more information.
            tool_choice:
                Select from "auto", "any", "none"; for more information, see
                https://docs.anthropic.com/en/docs/tools-overview
            model:
                The name of the Anthropic model to use (e.g. 'gpt-4o-mini').
            endpoint_url:
                This parameter is used when running against a local Anthropic-compatible API
                endpoint.
            metadata:
                A dictionary of metadata about the Candidate.

        Parameters
                A dictionary of model-specific parameters (e.g. `temperature`).
        """
        super().__init__(
            model_name=model,
            endpoint_url=endpoint_url,
            metadata=metadata,
            parameters=parameters,
        )
        self.tools = tools
        self.tool_choice = tool_choice

    def _invoke_client_callable(
        self,
        input: list[dict[str, str]],  # noqa: A002
    ) -> AnthropicCompletionResponse | AnthropicToolsResponse:
        """Invoke the client with the input and return the response."""
        return self.client_callable(
            messages=input,
            tools=self.tools,
            tool_choice=(
                {"type": self.tool_choice}
                if isinstance(self.tool_choice, str)
                else self.tool_choice
            ),
        )

    def _parse_response(
        self,
        response: AnthropicCompletionResponse | AnthropicToolsResponse,
    ) -> str | dict:
        """Get the desired attribute from the response object."""
        from sik_llm_eval.anthropic import AnthropicToolsResponse

        return (
            response.tools
            if isinstance(response, AnthropicToolsResponse)
            else response.content
        )
