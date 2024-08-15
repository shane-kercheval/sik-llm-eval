"""
Defines classes for different types of built-in Candidates and a corresponding registry system for
custom Candidates.

The purpose of a Candidate is to provide a standard interface for the the underlying LLM and client
(e.g. ChatpGPT via OpenAI() client, Lamma3 via LM Studio, etc.) so that the user can define evals
for any type of LLM/agent/etc and client.

A Candidate is a callable object that takes input (e.g. OpenAI-style messages) and returns a
response. The input and response can be any type of object that can be serialized/de-serialized
(e.g. string, dictionary, list, etc.).

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
from openai import OpenAI
import yaml
from inspect import iscoroutinefunction
from copy import deepcopy
from abc import ABC, abstractmethod
from enum import Enum, auto
from textwrap import dedent
from typing import Any, Callable, List, Type, Union
from llm_eval.openai import OpenAIChatResponse, OpenAICompletionWrapper
from llm_eval.internal_utilities import (
    DictionaryEqualsMixin,
    EnumMixin,
    Registry,
)

@staticmethod
def is_async_candidate(candidate: Callable | 'Candidate') -> bool:
    """Tests if the Candidate object or callable is an async function."""
    if iscoroutinefunction(candidate):
        return True
    if hasattr(candidate, '__call__'):
        return iscoroutinefunction(candidate.__call__)
    return False

class CandidateType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of Candidates."""

    OPENAI = auto()
    OPENAI_TOOLS = auto()
    CALLABLE_NO_SERIALIZE = auto()
    OPENAI_SERVER = auto()


class Candidate(DictionaryEqualsMixin, ABC):
    """
    A Candidate describes an LLM and the client for interfacing with the LLM (or specific
    implementation of an LLM interface (e.g. history/context management)) along with optional
    model parameters.

    A Candidate is a callable object that takes an input and returns a response.
    """

    registry = Registry()

    def __init__(self, metadata: dict | None = None, parameters: dict | None = None) -> None:
        """
        Initialize a Candidate object.

        Args:
            metadata: A dictionary of metadata about the Candidate.
            parameters: A dictionary of parameters for the Candidate.
        """
        self.metadata = deepcopy(metadata) or {}
        self.parameters = deepcopy(parameters)

    @abstractmethod
    def __call__(self, input: Any) -> Any:  # noqa: A002, ANN401
        """Invokes the underlying model with the input and returns the response."""

    # @abstractmethod
    # def clone(self) -> 'Candidate':
    #     """
    #     Returns a copy of the Candidate with the same state but with a different instance of the
    #     underlying model (e.g. same parameters but reset history/context).

    #     This is needed because the same Candidate object should not be reused across multiple Eval
    #     objects. This method allows the Eval object to create a new Candidate object and ensure
    #     the original Candidate object is not modified.
    #     """

    @classmethod
    def register(cls, candidate_type: str | Enum):  # noqa: ANN102
        """Register a subclass of Candidate."""
        def decorator(subclass: Type[Candidate]) -> Type[Candidate]:
            assert issubclass(subclass, Candidate), \
                f"Candidate '{candidate_type}' ({subclass.__name__}) must extend Candidate"
            cls.registry.register(type_name=candidate_type, item=subclass)
            return subclass
        return decorator

    @classmethod
    def from_dict(cls, data: dict) -> Union['Candidate', List['Candidate']]:  # noqa: ANN102
        """
        Creates a Candidate object.

        This method requires that the Candidate subclass has been registered with the `register`
        decorator before calling this method. It also requires that the dictionary has a
        `candidate_type` field that matches the type name of the registered Candidate subclass.
        """
        data = deepcopy(data)
        candidate_type = data.pop('candidate_type', '')
        if candidate_type in cls.registry:
            return cls.registry.create_instance(type_name=candidate_type, **data)
        raise ValueError(f"Unknown type {candidate_type}")

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        # value = self.model_dump(exclude_defaults=True, exclude_none=True)
        value = {}
        if self.metadata:
            value['metadata'] = deepcopy(self.metadata)
        if self.parameters:
            value['parameters'] = deepcopy(self.parameters)
        if self.candidate_type:
            value['candidate_type'] = self.candidate_type
        return value

    @property
    def candidate_type(self) -> str | None:
        """The type of Candidate."""
        # check that self.__class__ has _type_name attribute
        if hasattr(self.__class__, '_type_name'):
            return self.__class__._type_name.upper()
        return self.__class__.__name__

    @classmethod
    def from_yaml(cls, path: str) -> Union['Candidate', List['Candidate']]:  # noqa: ANN102
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
        parameters = '' if not self.parameters else f'\n            parameters={self.parameters},'
        return dedent(f"""
        {self.__class__.__name__}(
            metadata={self.metadata},
            {parameters}
        )
        """).strip()


@Candidate.register(CandidateType.OPENAI)
class OpenAICandidate(Candidate):
    """
    Wrapper around the OpenAI API that allows the user to create an OpenAI candidate from a
    dictionary.

    NOTE: the `OPENAI_API_KEY` environment variable must be set to use this class.
    """

    def __init__(  # noqa: D417
            self,
            model_name: str | None = None,
            endpoint_url: str | None = None,
            metadata: dict | None = None,
            parameters: dict | None = None) -> None:
        """
        Initialize a OpenAICandidate object.

        Args:
            metadata:
                A dictionary of metadata about the Candidate.
            parameters:
                A dictionary of parameters passed to OpenAI. `model_name` (e.g.
                'gpt-3.5-turbo-1106') is the only required parameter. However, other parameters
                such as `model_name` and model-specific parameters (e.g. `temperature`) can be
                passed.
        """  # noqa
        assert model_name or endpoint_url, "model_name or endpoint_url must be provided"
        self.model_name = model_name
        self.endpoint_url = endpoint_url
        super().__init__(metadata=metadata, parameters=parameters)
        self.client = OpenAICompletionWrapper(
            client=OpenAI(base_url=self.endpoint_url),
            model=self.model_name or self.endpoint_url,
            **self.parameters or {},
        )

    def __call__(self, input: list[dict[str, str]]) -> str:  # noqa: A002
        """Invokes the underlying model with the input and returns the response."""
        return self.client(input).content

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        value = super().to_dict()
        if self.model_name:
            value['model_name'] = self.model_name
        if self.endpoint_url:
            value['endpoint_url'] = self.endpoint_url
        return value

    # def clone(self) -> 'Candidate':
    #     """
    #     Returns a copy of the Candidate with the same state but with a different instance of the
    #     underlying model (e.g. same parameters but reset history/context).

    #     Reques
    #     """
    #     return Candidate.from_dict(deepcopy(self.to_dict()))


# @Candidate.register(CandidateType.OPENAI_TOOLS)
# class OpenAIToolsCandidate(ChatModelCandidate):
#     """
#     Wrapper around the OpenAI API that allows the user to create an OpenAI candidate with tools
#     from a dictionary. The client is a callable object that takes a prompt and returns a response.
#     It will also track the history/messages, supporting stateful conversations, which is needed
#     to evaluate multiple prompts in a single Eval object.

#     NOTE: the `OPENAI_API_KEY` environment variable must be set to use this class.
#     """

#     def __init__(  # noqa: D417
#             self,
#             metadata: dict | None = None,
#             parameters: dict | None = None) -> None:
#         """
#         Initialize a OpenAICandidate object.

#         Args:
#             metadata:
#                 A dictionary of metadata about the Candidate.
#             parameters:
#                 A dictionary of parameters passed to OpenAI. `model_name` (e.g.
#                 'gpt-3.5-turbo-1106') is the only required parameter. However, other parameters
#                 such as `model_name` and model-specific parameters (e.g. `temperature`) can be
#                 passed.
#         """  # noqa
#         if parameters is None:
#             parameters = {}
#         super().__init__(
#             model=OpenAITools(**deepcopy(parameters)),
#             metadata=metadata,
#             parameters=parameters,
#         )


