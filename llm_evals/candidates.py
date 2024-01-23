"""Defines a registration system for Candidate models."""
from abc import ABC, abstractmethod
from enum import Enum, auto
from textwrap import dedent
from typing import Callable, ClassVar, ForwardRef, Type
from pydantic import BaseModel
import yaml
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


class Candidate(BaseModel):
    """
    A Candidate describes an LLM (or specific implementation of an LLM interface (e.g.
    history/context management)) along wiht optional parameters or hardware.

    NOTE: If a candidate is being evaluated against multiple prompts (i.e. multiple PromptTest
    objects) in the same Eval, the assumption those prompts are testing a conversation (i.e.
    sequentially building on each other). This means that the candidate should be able to
    maintain state between prompts (e.g. history/context) and a single Candidate object should
    be created for a single Eval object, and not reused across multiple Eval objects.

    NOTE: The `model` field is not included in the dict representation of the Candidate. This is
    because the model is a callable and cannot be serialized. The `model` field is also not
    included in the equality check between two Candidates.

    The `model` should be created with the `root_validator` decorator. For example:

    @Candidate.register('MOCK_MODEL')
    class MockCandidate(Candidate):
        ''''Mock class representing a Candidate.''''

        _model: Callable | None = None

        @root_validator(pre=True)
        def create_model(cls, values: dict) -> dict:  # noqa: N805
            '''Creates the model from the parameters.'''
            parameters = values.get('parameters')
            if parameters is not None:
                values['_model'] = MockLMM(**parameters)
            return values
    """

    registry: ClassVar[Registry] = Registry()

    uuid: str | None = None
    metadata: dict | None = None
    parameters: dict | None = None
    system_info: dict | None = None
    model: Callable | None = None

    def __call__(self, prompt: str) -> str:
        """Invokes the underlying model with the prompt and returns the response."""
        return self.model(prompt)

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
        candidate_type = data.get('candidate_type', '')
        if candidate_type in cls.registry:
            return cls.registry.create_instance(type_name=candidate_type, **data)
        raise ValueError(f"Unknown type {candidate_type}")

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Candidate."""
        value = self.model_dump(exclude_defaults=True, exclude_none=True)
        if self.candidate_type:
            value['candidate_type'] = self.candidate_type.upper()
        value.pop('model', None)  # do not include model object in dict, if it exists
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
        """
        return self.from_dict(self.to_dict().copy())


Candidate.model_rebuild()

# @Candidate.register(CandidateType.OPENAI)
# class OpenAICandidate(Candidate):
#     """Wrapper around OpenAI API."""
    
#     model: OpenAIChat = Field(default_factory=) 

#     def __call__(self, prompt: str) -> str:
#         """Invokes the underlying model with the prompt and returns the response."""
        








    # # override equals operator to ignore model (callable) when comparing candidates
    # def __eq__(self, other: object) -> bool:
    #     """Returns True if the two Candidates are equal."""
    #     if not isinstance(other, Candidate):
    #         return False
    #     return self.to_dict() == other.to_dict()

    # def to_dict(self) -> dict:
    #     """Return a dictionary representation of the Candidate."""
    #     value = {}
    #     if self.uuid:
    #         value['uuid'] = self.uuid
    #     if self.candidate_type:
    #         value['candidate_type'] = self.candidate_type
    #     if self.metadata:
    #         value['metadata'] = self.metadata
    #     if self.parameters:
    #         value['parameters'] = self.parameters
    #     if self.system_info:
    #         value['system_info'] = self.system_info
    #     return value

