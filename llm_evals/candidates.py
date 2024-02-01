"""
Defines classes for different types of built-in Candidates and a corresponding registry system for
custom Candidates.

A Candidate encapsulates the underlying LLM and corresponding client the user is interested in
evaluating the prompts (Evals) against. Examples of candidates are ChatGPT 4.0 (LLM & client/API
are synonymous), Llama-2-7b-Chat (LLM) running on Hugging Face Endpoints with Nvidia 10G (client),
Llama-2-7b-Chat Q6_K.gguf (LLM) running locally on LM Studio (client). The latter two are examples
of the same underlying model running on different hardware. They are likely to have very similar
quality of responses (but this is also determined by the quantization) but may have very different
performance (e.g. characters per second).
"""
import yaml
from copy import deepcopy
from abc import ABC, abstractmethod
from enum import Enum, auto
from textwrap import dedent
from typing import Callable, List, Type, Union
from llm_evals.llms.hugging_face import HuggingFaceEndpointChat
from llm_evals.llms.message_formatters import create_message_formatter
from llm_evals.llms.openai import OpenAIChat
from llm_evals.utilities.internal_utilities import (
    DictionaryEqualsMixin,
    EnumMixin,
    Registry,
    generate_dict_combinations,
)


class CandidateType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of Candidates."""

    OPENAI = auto()
    HUGGING_FACE_ENDPOINT = auto()
    CALLABLE_NO_SERIALIZE = auto()


class Candidate(DictionaryEqualsMixin, ABC):
    """
    A Candidate describes an LLM and the client for interfacing with the LLM (or specific
    implementation of an LLM interface (e.g. history/context management)) along with optional
    model parameters.

    A Candidate is a callable object that takes a prompt and returns a response.

    NOTE: If a candidate is being evaluated against multiple prompts (i.e. multiple PromptTest
    objects) in the same Eval, the assumption is that those prompts are testing a conversation
    (i.e. sequential prompt/response exchanges). This means that the candidate/client should be
    able to maintain state between prompts (e.g. history/context) and a single Candidate object
    should be created for a single Eval object, and not reused across multiple Eval objects.
    """

    registry = Registry()

    def __init__(self, metadata: dict | None = None, model_parameters: dict | None = None) -> None:
        """
        Initialize a Candidate object.

        Args:
            metadata: A dictionary of metadata about the Candidate.
            model_parameters: A dictionary of parameters for the Candidate.
        """
        self.metadata = deepcopy(metadata)
        self.model_parameters = deepcopy(model_parameters)

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
    def from_dict(cls, data: dict) -> Union['Candidate', List['Candidate']]:  # noqa: ANN102
        """
        Creates a Candidate object (or multiple objects) from a dictionary. If any of the values
        within the `model_parameters` dict is a list (i.e. multiple parameters to evaluate
        against), this method will return a list of Candidates corresponding to all combinations of
        model parameters.

        This method requires that the Candidate subclass has been registered with the `register`
        decorator before calling this method. It also requires that the dictionary has a
        `candidate_type` field that matches the type name of the registered Candidate subclass.
        """
        data = deepcopy(data)
        candidate_type = data.pop('candidate_type', '')
        if candidate_type in cls.registry:
            # check if any of the model parameters (values) are lists
            params_are_lists = 'model_parameters' in data \
                and any(isinstance(v, list) for v in data['model_parameters'].values())
            if params_are_lists:
                    # create a list of all combinations of the model parameters
                    # `data` will potentially have `metadata` or other fields
                    model_parameters = data.pop('model_parameters')
                    model_parameters = generate_dict_combinations(deepcopy(model_parameters))
                    return [
                        cls.registry.create_instance(
                            type_name=candidate_type,
                            # merge the original data with the model parameters
                            **(data | {'model_parameters': p}),
                        )
                        for p in model_parameters
                    ]
            return cls.registry.create_instance(type_name=candidate_type, **data)
        raise ValueError(f"Unknown type {candidate_type}")

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        # value = self.model_dump(exclude_defaults=True, exclude_none=True)
        value = {}
        if self.metadata:
            value['metadata'] = deepcopy(self.metadata)
        if self.model_parameters:
            value['model_parameters'] = deepcopy(self.model_parameters)
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
        parameters = '' if not self.model_parameters else f'\n            model_parameters={self.model_parameters},'  # noqa
        return dedent(f"""
        {self.__class__.__name__}(
            metadata={self.metadata},
            {parameters}
        )
        """).strip()

    def clone(self) -> 'Candidate':
        """
        Returns a copy of the Candidate with the same state but with a different instance of the
        underlying model (e.g. same parameters but reset history/context).

        Reques
        """
        return Candidate.from_dict(deepcopy(self.to_dict()))


@Candidate.register(CandidateType.CALLABLE_NO_SERIALIZE)
class CallableCandidate(Candidate):
    """
    Candidate for a simple callable model. This is useful for simple use-cases, stateless models,
    and for testing.

    NOTE: This class is registered with the Candidate registry and can be created from a dictionary
    using `Candidate.from_dict(...)`. However, since the model is a callable defined at runtime,
    it cannot be serialized (to/from dict) and is not included in the dict representation of the
    Candidate. When creating a CallableCandidate from a dictionary, the `model` field will be
    `None`. Therefore, it can be reloded from a dictionary but will not be able to run evaluations.
    """

    def __init__(
            self,
            model: Callable | None = None,
            metadata: dict | None = None,
            parameters: dict | None = None) -> None:
        """
        Initialize a CallableCandidate object.

        Args:
            model: The callable model.
            metadata: A dictionary of metadata about the Candidate.
            parameters: A dictionary of parameters for the Candidate.
        """
        super().__init__(metadata, parameters)
        self.model = model

    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""
        return self.model(prompt)


@Candidate.register(CandidateType.OPENAI)
class OpenAICandidate(Candidate):
    """
    Wrapper around the OpenAI API that allows the user to create an OpenAI candidate from a
    dictionary. The client is a callable object that takes a prompt and returns a response. It will
    also track the history/messages, supporting stateful conversations, which is needed to evaluate
    multiple prompts in a single Eval object.

    NOTE: the `OPENAI_API_KEY` environment variable must be set to use this class.
    """

    def __init__(self,
        metadata: dict | None = None,
        model_parameters: dict | None = None) -> None:
        """
        Initialize a OpenAICandidate object.

        Args:
            metadata:
                A dictionary of metadata about the Candidate.
            model_parameters:
                A dictionary of parameters passed to OpenAI. `model_name` (e.g.
                'gpt-3.5-turbo-1106') is the only required parameter. However, other parameters
                such as `system_message` and model-specific parameters (e.g. `temperature`) can be
                passed.
        """
        super().__init__(metadata=metadata, model_parameters=model_parameters)
        if model_parameters is None:
            model_parameters = {}
        self.model = OpenAIChat(**model_parameters)

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
    Wrapper around the Hugging Face Endpoint API that allows the user to create the candidate from
    a dictionary. The client is a callable object that takes a prompt and returns a response. It
    will also track the history/messages, supporting stateful conversations, which is needed to
    evaluate multiple prompts in a single Eval object.

    NOTE: the `HUGGING_FACE_API_KEY` environment variable must be set to use this class.
    """

    def __init__(self,
        model_parameters: dict,
        metadata: dict | None = None) -> None:
        r"""
        Initialize a HuggingFaceEndpointCandidate object.

        Args:
            metadata:
                A dictionary of metadata about the Candidate.
            model_parameters:
                A dictionary of parameters passed to OpenAI. `endpoint_url`, `system_format`,
                `prompt_format`, and `response_format` are required parameters. Other parameters
                such as `system_message` and model-specific parameters (e.g. `temperature`) can be
                passed.

                An example of system/prompt/response formats is:

                ```
                system_format: '[INST] <<SYS>> {system_message} <</SYS>> [/INST]\n'
                prompt_format: '[INST] {prompt} [/INST]\n'
                response_format: '{response}\n'
                ```
        """
        model_parameters = deepcopy(model_parameters)
        self.system_format = model_parameters.pop('system_format')
        self.prompt_format = model_parameters.pop('prompt_format')
        self.response_format = model_parameters.pop('response_format')
        super().__init__(metadata=metadata, model_parameters=model_parameters)
        message_formatter = create_message_formatter(
            system_format=self.system_format,
            prompt_format=self.prompt_format,
            response_format=self.response_format,
        )
        self.model = HuggingFaceEndpointChat(
            message_formatter=message_formatter,
            **model_parameters,
        )

    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""
        return self.model(prompt)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        # value = self.model_dump(exclude_defaults=True, exclude_none=True)
        value = super().to_dict()
        if self.system_format:
            value['model_parameters']['system_format'] = self.system_format
        if self.prompt_format:
            value['model_parameters']['prompt_format'] = self.prompt_format
        if self.response_format:
            value['model_parameters']['response_format'] = self.response_format
        return value

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
