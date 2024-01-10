"""
Defines classes for different types  of checks corresponding registry system.


A "check" is a single test/check defined in an Eval (an Eval can have multiple checks). The check
is responsible for evaluating the responses to the prompts. The intent of the check can range from
simple matching (i.e. does the LLM response exactly match the expected value provided)  regex check to using an LLM to evaluate the responses.
"""
from abc import ABC, abstractmethod
from enum import Enum, auto
import re
from typing import Type
from pydantic import BaseModel


class CheckType(Enum):
    """TODO document."""

    MATCH = auto()
    MATCH_CONTAINS = auto()
    MATCH_REGEX = auto()
    PYTHON_FUNCTION = auto()
    PYTHON_CODE_BLOCKS = auto()

    @staticmethod
    def to_enum(name: str) -> 'CheckType':
        """Get a CheckType from its name."""
        if isinstance(name, CheckType):
            return name
        try:
            return CheckType[name.upper()]
        except KeyError:
            raise ValueError(f"{name.upper()} is not a valid name for a CheckType member")


class EvalCheck(ABC):
    """
    An EvalCheck corresponds to a single test/check defined in an Eval (an Eval can have multiple
    checks). The EvalCheck is responsible for evaluating the responses to the prompts.
    """

    # TODO: not sure if i need eval_uuid since it's in the EvalResult object
    def __init__(self, eval_uuid: str, metadata: dict | None = None) -> None:
        super().__init__()
        self.eval_uuid = eval_uuid
        self.metadata = metadata or {}
        self.result = None

    @abstractmethod
    def __call__(self, responses: list[str]) -> None:
        """TODO document."""


class CheckResult(BaseModel):
    """TODO document."""

    result: bool | int | float | object
    description: str
    metadata: dict | None


class CheckRegistery:
    """Registry for types (subclasses) of EvalCheck."""

    # TODO: test string registration type

    def __init__(self):
        self._registry: dict[str, str] = {}

    def register(self, key: str | CheckType, check_type: Type[EvalCheck]) -> None:
        """Register an EvalCheck with the registry."""
        if isinstance(key, CheckType):
            key = key.name
        if key in self._registry:
            raise ValueError(f"An EvalCheck with name '{key}' is already registered.")
        self._registry[key] = check_type

    def create_check(self, check_type: CheckType | str, params: dict) -> EvalCheck:
        """Create a test from a config."""
        if isinstance(check_type, CheckType):
            check_type = check_type.name
        if check_type not in self._registry:
            raise ValueError(f"CheckType '{check_type}' not found in registry.")
        return self._registry[check_type](**params)

    def registered(self) -> dict[str, Type[EvalCheck]]:
        """List all registered EvalChecks."""
        return self._registry

    def __contains__(self, key: CheckType | str) -> bool:
        """Check if a EvalCheck is registered."""
        if isinstance(key, CheckType):
            key = key.name
        return key in self._registry


def register_check(test_type: CheckType) -> EvalCheck:
    """Decorator to register an EvalCheck."""
    def decorator(cls: EvalCheck) -> EvalCheck:
        assert issubclass(cls, EvalCheck), \
            f"Test '{test_type}' ({cls.__name__}) must extend CheckType"
        assert (test_type not in CHECK_REGISTRY), \
            f"Test '{test_type}' already registered."
        CHECK_REGISTRY.register(test_type, cls)
        return cls
    return decorator


CHECK_REGISTRY = CheckRegistery()


# TODO: __call__ should return a (list?) of CheckResult object(s) in addition to caching the
# results in self.results


@register_check(CheckType.MATCH)
class MatchCheck(EvalCheck):
    """TODO document."""

    def __init__(self,
            eval_uuid: str,
            values: list[str],
            metadata: dict | None = None) -> None:
        super().__init__(eval_uuid=eval_uuid, metadata=metadata)
        self.values = values

    def __call__(self, responses: list[str]) -> None:
        """TODO: document."""
        assert len(responses) == len(self.values), \
            f"Number of responses ({len(responses)}) does not equal number of match values " \
            f"({len(self.values)})"
        self.results = []
        for r, v in zip(responses, self.values):
            if v is None:
                self.results.append(CheckResult(result=None, description="TODO", metadata={}))
            else:
                self.results.append(CheckResult(result=r == v, description="TODO", metadata={}))

@register_check(CheckType.MATCH_CONTAINS)
class MatchContainsCheck(EvalCheck):
    """
    Checks if the LLM response (string) contains the provided value (i.e. the value/string is found
    anywhere in the response).

    If multiple prompts/responses are provided, a list of values must be provided that is the same
    length as the number of prompts/responses.
    """

    def __init__(self,
            eval_uuid: str,
            values: list[str],
            metadata: dict | None = None) -> None:
        super().__init__(eval_uuid=eval_uuid, metadata=metadata)
        self.values = values

    def __call__(self, responses: list[str]) -> None:
        """TODO document."""
        assert len(responses) == len(self.values), \
            f"Number of responses ({len(responses)}) does not equal number of match values " \
            f"({len(self.values)})"
        self.results = []
        for r, v in zip(responses, self.values):
            if v is None:
                self.results.append(CheckResult(result=None, description="TODO", metadata={}))
            else:
                self.results.append(CheckResult(
                    result=bool(v in r),
                    description="TODO",
                    metadata={'value': v}),
                )

@register_check(CheckType.MATCH_REGEX)
class RegexMatchCheck(EvalCheck):
    """
    Checks if the LLM response (string) matches the provided regular expression.

    If multiple prompts/responses are provided, a list of regex values must be provided that is
    the same length as the number of prompts/responses.
    """

    def __init__(self,
            eval_uuid: str,
            patterns: list[str],
            metadata: dict | None = None) -> None:
        """
        Args:
            eval_uuid: TODO document.
            patterns:
                The regular expression(s) to match the LLM response(s) against. If multiple
                prompts/responses are provided, a list of regex values must be provided that is
                the same length as the number of prompts/responses.
            metadata: TODO document.
        """
        super().__init__(eval_uuid=eval_uuid, metadata=metadata)
        if isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = patterns

    def __call__(self, responses: list[str]) -> None:
        """TODO document."""
        assert len(responses) == len(self.patterns), \
            f"Number of responses ({len(responses)}) does not equal number of regex values " \
            f"({len(self.patterns)})"
        # patterns = [re.compile(r) if r is not None else None for r in self.regex]
        self.results = []
        for r, p in zip(responses, self.patterns):
            # TODO: should i return None if regex is None? Or should I Return a CheckResult object?
            # TODO: need to make sure this is consistent with other Check types
            if p is None:
                self.results.append(CheckResult(result=None, description="TODO", metadata={}))
            else:
                self.results.append(CheckResult(
                    result=bool(re.compile(p).match(r)),
                    description="TODO",
                    metadata={'regex': p}),
                )

@register_check(CheckType.PYTHON_FUNCTION)
class PythonFunctionCheck(EvalCheck):
    """
    Runs a Python function (using the LLM responses as input. A Python function is either
    provided as a string, or the name of the function and the file path containing the function.
    A Python function test could be used for anything from a simple regex check to using an LLM
    to evaluate the responses.
    """

    def __init__(self,
            eval_uuid: str,
            function: str | None = None,
            function_name: str | None = None,
            function_file: str | None = None,
            metadata: dict | None = None) -> None:
        """
        TODO document.

        Args:
            eval_uuid: TODO document.
            function: function definition as string value. If not provided, function_name and
                function_file must be provided.
            function_name: The name of the function to import from `function_file`.
            function_file: The file containing the function to import.
            metadata: TODO document.
        """
        super().__init__(eval_uuid=eval_uuid, metadata=metadata)
        if function is None:
            assert function_name is not None and function_file is not None, \
                "Either function or function_name and function_file must be provided."  # noqa: PT018
        self._function_str = function
        self._function_name = function_name
        self._function_file = function_file

    def __call__(self, responses: list[str]) -> None:
        """TODO document."""
        return responses
        # A slightly different requirement is that I have a python file and the name of a function
        # in that file. I need to dynamically import everything in that file and execute the
        # provided function, while passing in arguments. I don't want anything imported to affect
        # the environment that is running it.


@register_check(CheckType.PYTHON_CODE_BLOCKS)
class PythonCodeBlocksCheck(EvalCheck):
    """
    This class is responsible for executing Python code blocks returned by the LLM and then
    running the python function(s) defined in the test in the same environment as code blocks.
    For example, if the code blocks define a pandas DataFrame, the function could be used to
    check that the shape or data of the DataFrame matches expectations.

    The difference between this class and PythonFunctionTest is that this class is responsible
    for running tests against the code blocks returned by the LLM, whereas PythonFunctionTest
    is responsible for running tests against the (string) responses returned by the LLM.
    """  # noqa: D404

    def __init__(self,
            eval_uuid: str,
            code_setup: str | None = None,
            checks: list[dict] | None = None,
            # function: str | None = None,
            # function_name: str | None = None,
            # function_file: str | None = None,
            metadata: dict | None = None) -> None:
        super().__init__(eval_uuid=eval_uuid, metadata=metadata)
        # if function is None:
        #     assert function_name is not None and function_file is not None, \
        #         "Either function or function_name and function_file must be provided."
        # self._function_str = function
        # self._function_name = function_name
        # self._function_file = function_file
        self._checks = checks
        self._code_setup = code_setup

    def __call__(self, responses: list[str]) -> None:
        """TODO document."""
        # extract code blocks
        # run code setup if provided
        # run code blocks
        # run function in same environent as code blocks
        pass
