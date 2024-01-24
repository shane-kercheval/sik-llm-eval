"""Defines a registration system for Candidate models."""
from abc import ABC, abstractmethod
from enum import Enum, auto
from textwrap import dedent
from typing import Callable, ForwardRef, Type
import yaml
from llm_evals.llms.hugging_face import HuggingFaceEndpointChat
from llm_evals.llms.openai import OpenAIChat

from llm_evals.utilities.internal_utilities import EnumMixin, Registry


Candidate = ForwardRef('Candidate')


class CandidateType(EnumMixin, Enum):
    """
    Defines the types of Candidates. This could be a specific LLM or a specific implementation
    of an LLM interface (e.g. history/context management).
    """

    OPENAI = auto()
    HUGGING_FACE_ENDPOINT = auto()
    LLAMA_CPP_SERVER = auto()
    API = auto()
    CALLABLE_NO_SERIALIZE = auto()


class Candidate(ABC):
    """
    A Candidate describes an LLM (or specific implementation of an LLM interface (e.g.
    history/context management)) along wiht optional parameters or hardware.

    NOTE: If a candidate is being evaluated against multiple prompts (i.e. multiple PromptTest
    objects) in the same Eval, the assumption those prompts are testing a conversation (i.e.
    sequentially building on each other). This means that the candidate should be able to
    maintain state between prompts (e.g. history/context) and a single Candidate object should
    be created for a single Eval object, and not reused across multiple Eval objects.
    """

    registry = Registry()

    def __init__(
        self,
        uuid: str | None = None,
        metadata: dict | None = None,
        parameters: dict | None = None,
        system_info: dict | None = None) -> None:
        """
        Initialize a Candidate object.

        Args:
            uuid: A unique identifier for the Candidate.
            metadata: A dictionary of metadata about the Candidate.
            parameters: A dictionary of parameters for the Candidate.
            system_info: A dictionary of system information about the Candidate.
        """
        self.uuid = uuid
        self.metadata = metadata
        self.parameters = parameters
        self.system_info = system_info

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""

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
    def from_dict(cls, data: dict):  # noqa: ANN102
        """
        Create a Candidate object from a dictionary. This method requires that the Candidate
        subclass has been registered with the `register` decorator.
        """
        data_copy = data.copy()
        candidate_type = data_copy.pop('candidate_type', '')
        if candidate_type in cls.registry:
            return cls.registry.create_instance(type_name=candidate_type, **data_copy)
        raise ValueError(f"Unknown type {candidate_type}")

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        # value = self.model_dump(exclude_defaults=True, exclude_none=True)
        value = {}
        if self.uuid:
            value['uuid'] = self.uuid
        if self.metadata:
            value['metadata'] = self.metadata
        if self.parameters:
            value['parameters'] = self.parameters
        if self.system_info:
            value['system_info'] = self.system_info
        if self.candidate_type:
            value['candidate_type'] = self.candidate_type.upper()
        return value

    @property
    def candidate_type(self) -> str | None:
        """The type of Candidate."""
        # check that self.__class__ has _type_name attribute
        if hasattr(self.__class__, '_type_name'):
            return self.__class__._type_name.upper()
        return None

    @classmethod
    def from_yaml(cls, path: str) -> Candidate:  # noqa: ANN102
        """
        Creates a Candidate object from a YAML file. This method requires the Candidate subclass to
        be registered via `Candidate.register(...)` before calling this method. It also requires
        that the YAML file has a `candidate_type` field that matches the type name of the
        registered Candidate subclass.
        """
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(**config)

    def __str__(self) -> str:
        """Returns a string representation of the Candidate."""
        parameters = '' if not self.parameters else f'\n            parameters={self.parameters},'
        system_info = '' if not self.system_info else f'\n            system_info={self.system_info},'  # noqa
        return dedent(f"""
        {self.__class__.__name__}(
            uuid={self.uuid},
            metadata={self.metadata},
            {parameters}{system_info}
        )
        """).strip()

    def __eq__(self, other: object) -> bool:
        """Returns True if the two Candidates are equal."""
        if not isinstance(other, Candidate):
            return False
        return self.to_dict() == other.to_dict()

    def clone(self) -> Candidate:
        """
        Returns a copy of the Candidate with the same state but with a different instance of the
        underlying model (e.g. same parameters but reset history/context).

        Reques
        """
        return self.from_dict(self.to_dict().copy())


@Candidate.register(CandidateType.CALLABLE_NO_SERIALIZE)
class CallableCandidate(Candidate):
    """
    Candidate for a simple callable model. This is useful for simple use-cases, stateless models,
    and for testing.

    NOTE: This class is with the Candidate registry and can be created from a dictionary using
    `Candidate.from_dict(...)`. However, since the model is a callable, it cannot be serialized
    and is not included in the dict representation of the Candidate. When creating a
    CallableCandidate from a dictionary, the `model` field will be `None`. Therefore, it can
    be reloded from a dictionary but will not be able to run evaluations.
    """

    def __init__(
            self,
            model: Callable | None = None,
            uuid: str | None = None,
            metadata: dict | None = None,
            parameters: dict | None = None,
            system_info: dict | None = None) -> None:
        """
        Initialize a CallableCandidate object.

        Args:
            model: The callable model.
            uuid: A unique identifier for the Candidate.
            metadata: A dictionary of metadata about the Candidate.
            parameters: A dictionary of parameters for the Candidate.
            system_info: A dictionary of system information about the Candidate.
        """
        super().__init__(uuid, metadata, parameters, system_info)
        self.model = model

    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""
        return self.model(prompt)


@Candidate.register(CandidateType.OPENAI)
class OpenAICandidate(Candidate):
    """
    Wrapper around OpenAI API.

    NOTE: the `OPENAI_API_KEY` environment variable must be set to use this class.
    """

    def __init__(self,
        uuid: str | None = None,
        metadata: dict | None = None,
        parameters: dict | None = None) -> None:
        """
        Initialize a OpenAICandidate object.

        Args:
            uuid: A unique identifier for the Candidate.
            metadata: A dictionary of metadata about the Candidate.
            parameters: A dictionary of parameters passed to OpenAI.
        """
        super().__init__(uuid=uuid, metadata=metadata, parameters=parameters, system_info=None)
        if parameters is None:
            parameters = {}
        self.model = OpenAIChat(**parameters)

    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""
        return self.model(prompt)

    @property
    def total_tokens(self) -> int:
        """Returns the total number of tokens processed by the model."""
        return self.model.total_tokens

    @property
    def input_tokens(self) -> int:
        """Returns the total number of input tokens processed by the model."""
        return self.model.input_tokens

    @property
    def response_tokens(self) -> int:
        """Returns the total number of response tokens returned by the model."""
        return self.model.response_tokens

    @property
    def cost(self) -> float:
        """Returns the total cost of using the model."""
        return self.model.cost


@Candidate.register(CandidateType.HUGGING_FACE_ENDPOINT)
class HuggingFaceEndpointCandidate(Candidate):
    """
    Wrapper around Hugging Face Inference API.

    NOTE: the `HUGGING_FACE_API_KEY` environment variable must be set to use this class.
    """

    def __init__(self,
        uuid: str | None = None,
        metadata: dict | None = None,
        parameters: dict | None = None,
        system_info: dict | None = None) -> None:
        """
        Initialize a HuggingFaceEndpointCandidate object.

        Args:
            uuid: A unique identifier for the Candidate.
            metadata: A dictionary of metadata about the Candidate.
            parameters: A dictionary of parameters passed to Hugging Face.
            system_info: A dictionary of system information about the Candidate.
        """
        super().__init__(
            uuid=uuid, metadata=metadata,
            parameters=parameters, system_info=system_info,
        )
        if parameters is None:
            parameters = {}
        self.model = HuggingFaceEndpointChat(**parameters)

    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""
        return self.model(prompt)
