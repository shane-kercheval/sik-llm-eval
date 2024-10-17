"""
Defines classes for different types of built-in Candidates and a corresponding registry system for
custom Candidates.

A Candidate is a callable object that takes input (e.g. OpenAI-style messages) and returns a
response. The input and response can be any type of object that can be serialized/de-serialized
(e.g. string, dictionary, list, etc.).

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

import os
import yaml
from inspect import iscoroutinefunction
from copy import deepcopy
from abc import ABC, abstractmethod
from enum import Enum, auto
from textwrap import dedent
from typing import Any, Callable, List, Literal, Type, Union
from pydantic import BaseModel
from openai import OpenAI
from llm_eval.openai import (
    MODEL_COST_PER_TOKEN as OPENAI_MODEL_COST_PER_TOKEN,
    OpenAICompletion,
    OpenAICompletionResponse,
    OpenAIToolsResponse,
)
from llm_eval.internal_utilities import (
    DictionaryEqualsMixin,
    EnumMixin,
    Registry,
)


@staticmethod
def is_async_candidate(candidate: Callable | "Candidate") -> bool:
    """Tests if the Candidate object or callable is an async function."""
    if iscoroutinefunction(candidate):
        return True
    if hasattr(candidate, "__call__"):
        return iscoroutinefunction(candidate.__call__)
    return False


class CandidateType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of Candidates."""

    OPENAI = auto()
    OPENAI_TOOLS = auto()
    MISTRALAI = auto()
    MISTRALAI_TOOLS = auto()


class CandidateResponse(BaseModel):
    """
    Provides a standard response object for Candidates so that the Eval/TestHarness can
    consistently evaluate the response and store the metadata (e.g. cost, usage, etc.) for the
    response.

    Content is the text/dict/etc. from the LLM that is meant to be evaluated (via Check objects).
    Metadata is a dictionary of metadata about the response (e.g. cost, usage, etc.).
    """

    response: Any
    metadata: dict | None = None


class Candidate(DictionaryEqualsMixin, ABC):
    """
    A Candidate describes an LLM and the client for interfacing with the LLM (or specific
    implementation of an LLM interface (e.g. history/context management)) along with optional
    model parameters.

    A Candidate is a callable object that takes an input and returns a CandidateResponse.
    """

    registry = Registry()

    def __init__(
        self, metadata: dict | None = None, parameters: dict | None = None
    ) -> None:  # noqa: D417
        """
        Initialize a Candidate object.

        Args:
            metadata:
                A dictionary of metadata about the Candidate.
            parameters:
                A dictionary of parameters for the Candidate (most likely, the model parameters
                passed to the LLM).
        """  # noqa
        self.metadata = deepcopy(metadata) or {}
        self.parameters = deepcopy(parameters)

    @abstractmethod
    def __call__(self, input: Any) -> CandidateResponse:  # noqa: A002, ANN401
        """Invokes the underlying model with the input and returns the response."""

    @classmethod
    def register(cls, candidate_type: str | Enum):  # noqa: ANN102
        """Register a subclass of Candidate."""

        def decorator(subclass: Type[Candidate]) -> Type[Candidate]:
            assert issubclass(
                subclass, Candidate
            ), f"Candidate '{candidate_type}' ({subclass.__name__}) must extend Candidate"
            cls.registry.register(type_name=candidate_type, item=subclass)
            return subclass

        return decorator

    @classmethod
    def from_dict(
        cls, data: dict
    ) -> Union["Candidate", List["Candidate"]]:  # noqa: ANN102
        """
        Creates a Candidate object.

        This method requires that the Candidate subclass has been registered with the `register`
        decorator before calling this method. It also requires that the dictionary has a
        `candidate_type` field that matches the type name of the registered Candidate subclass.
        """
        data = deepcopy(data)
        candidate_type = data.pop("candidate_type", "")
        if candidate_type in cls.registry:
            return cls.registry.create_instance(type_name=candidate_type, **data)
        raise ValueError(f"Unknown type {candidate_type}")

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

    @classmethod
    def from_yaml(
        cls, path: str
    ) -> Union["Candidate", List["Candidate"]]:  # noqa: ANN102
        """
        Creates a Candidate object from a YAML file. This method requires the Candidate subclass to
        be registered via `Candidate.register(...)` before calling this method. It also requires
        that the YAML file has a `candidate_type` field that matches the type name of the
        registered Candidate subclass.
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

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
        """
        ).strip()


class ServiceCandidate(Candidate, ABC):
    """
    Wrapper around a service API that allows the user to create a service candidate from a
    dictionary.
    """

    def __init__(  # noqa: D417
        self,
        model: str | None = None,
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
        assert model or endpoint_url, "model or endpoint_url must be provided"
        self.model = model
        self.endpoint_url = endpoint_url

    @property
    @abstractmethod
    def client(self):
        """Return the client for the service."""

    @property
    @abstractmethod
    def model_cost_per_token(self) -> float | None:
        """
        Return the cost per token for the model. This is used to calculate the cost of the
        completion.
        """

    @abstractmethod
    def _invoke_client(self, input: Any) -> Any:
        """
        Invoke the client with the input and return the response.
        """

    @abstractmethod
    def _parse_response(self, response):
        """
        Get the desired attribute from the response object.
        """

    def __call__(self, input: list[dict[str, str]]) -> CandidateResponse:  # noqa: A002
        """Invokes the underlying model with the input and returns the response."""
        response = self._invoke_client(input)
        prompt_tokens = response.usage.get("prompt_tokens")
        completion_tokens = response.usage.get("completion_tokens")
        total_tokens = response.usage.get("total_tokens")

        cost_per_token = self.model_cost_per_token
        if cost_per_token and prompt_tokens and completion_tokens:
            prompt_cost = cost_per_token["input"] * prompt_tokens
            completion_cost = cost_per_token["output"] * completion_tokens
            total_cost = prompt_cost + completion_cost
        else:
            prompt_cost = None
            completion_cost = None
            total_cost = None

        parsed_response = self._parse_response(response)
        completion_characters = len(parsed_response) if isinstance(parsed_response, str) else None
        return CandidateResponse(
            response=parsed_response,
            metadata={
                "type": (
                    "tools"
                    if isinstance(response, OpenAIToolsResponse)
                    else "completion"
                ),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": total_cost,
                "completion_characters": completion_characters,
            },
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        value = super().to_dict()
        if self.model:
            value["model"] = self.model
        if self.endpoint_url:
            value["endpoint_url"] = self.endpoint_url
        return value


@Candidate.register(CandidateType.OPENAI)
class OpenAICandidate(ServiceCandidate):
    """
    Wrapper around the OpenAI API that allows the user to create an OpenAI candidate from a
    dictionary.

    NOTE: the `OPENAI_API_KEY` environment variable must be set to use this class.
    """

    def model_cost_per_token(self) -> float | None:
        """
        Return the cost per token for the model. This is used to calculate the cost of the
        completion.
        """
        return OPENAI_MODEL_COST_PER_TOKEN.get(self.model)

    def client(self) -> OpenAICompletion:
        """Return the client for the OpenAI service."""
        return OpenAICompletion(
            client=OpenAI(base_url=self.endpoint_url),
            model=self.model,
            **self.parameters or {},
        )

    def _invoke_client(self, input: Any) -> Any:
        """
        Invoke the client with the input and return the response.
        """
        return self.client(input)


@Candidate.register(CandidateType.OPENAI_TOOLS)
class OpenAIToolsCandidate(OpenAICandidate):
    """
    Wrapper around the OpenAI Tools API that allows the user to create an OpenAI candidate from a
    dictionary.

    NOTE: the `OPENAI_API_KEY` environment variable must be set to use this class.
    """

    def __init__(  # noqa: D417
        self,
        tools: list[dict],
        tool_choice: Literal["none", "auto", "required"] | dict[str] = "required",
        model: str | None = None,
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
            model=model,
            endpoint_url=endpoint_url,
            metadata=metadata,
            parameters=parameters,
        )
        self.tools = tools
        self.tool_choice = tool_choice

    def _invoke_client(self, input: Any) -> Any:
        """
        Invoke the client with the input and return the response.
        """
        return self.client(
            messages=input, tools=self.tools, tool_choice=self.tool_choice
        )

    def _parse_response(self, response):
        """
        Get the desired attribute from the response object.
        """
        return (
            response.tools
            if isinstance(response, OpenAIToolsResponse)
            else response.content
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        value = super().to_dict()
        value["tools"] = self.tools
        value["tool_choice"] = self.tool_choice
        return value


@Candidate.register(CandidateType.MISTRALAI)
class MistralAICandidate(ServiceCandidate):
    """
    Wrapper around the MistralAI API that allows the user to create an MistralAI candidate from a
    dictionary.

    NOTE: the `MISTRAL_API_KEY` environment variable must be set to use this class.
    """

    @property
    def client(self):
        """Return the client for the OpenAI service."""
        from mistralai import Mistral
        from llm_eval.mistralai import MistralAICompletion

        parameters = self.parameters or {}
        api_key = parameters.pop("api_key", os.getenv("MISTRAL_API_KEY"))
        return MistralAICompletion(
            client=Mistral(api_key=api_key, server_url=self.endpoint_url),
            model=self.model,
            **parameters,
        )

    @property
    def model_cost_per_token(self):
        from llm_eval.mistralai import (
            MODEL_COST_PER_TOKEN as MISTRAL_MODEL_COST_PER_TOKEN,
        )

        return MISTRAL_MODEL_COST_PER_TOKEN.get(self.model)

    def _invoke_client(self, input: Any) -> Any:
        """
        Invoke the client with the input and return the response.
        """
        return self.client(messages=input)


@Candidate.register(CandidateType.MISTRALAI_TOOLS)
class MistralAIToolsCandidate(MistralAICandidate):
    """
    Wrapper around the MistralAI Tools API that allows the user to create an MistralAI candidate
    from a dictionary.

    NOTE: the `MISTRAL_API_KEY` environment variable must be set to use this class.
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
        Initialize a MistralAIToolsCandidate object.

        Args:
            tools:
                A list of tools to use with the MistralAI model. See
                https://docs.mistral.ai/capabilities/function_calling/ for more
                information.
            tool_choice:
                Select from "auto", "any", "none"; for more information, see
                https://docs.mistral.ai/capabilities/function_calling/#tool_choice.
            model:
                The name of the MistralAI model to use (e.g. 'mistral-large-latest').
            endpoint_url:
                This parameter is used when running against a local MistralAI-compatible API
                endpoint.
            metadata:
                A dictionary of metadata about the Candidate.
            parameters:
                A dictionary of model-specific parameters (e.g. `temperature`).
        """
        super().__init__(
            model=model,
            endpoint_url=endpoint_url,
            metadata=metadata,
            parameters=parameters,
        )
        self.tools = tools
        self.tool_choice = tool_choice

    def _invoke_client(self, input: Any) -> Any:
        """
        Invoke the client with the input and return the response.
        """
        return self.client(
            messages=input, tools=self.tools, tool_choice=self.tool_choice
        )

    def _parse_response(self, response):
        """
        Get the desired attribute from the response object.
        """
        from llm_eval.mistralai import MistralAIToolsResponse
        return (
            response.tools
            if isinstance(response, MistralAIToolsResponse)
            else response.content
        )
