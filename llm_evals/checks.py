"""
Defines classes for different types  of checks corresponding registry system.

A "check" is a single test/check defined in an Eval (an Eval can have multiple checks). The check
is responsible for evaluating the response to the prompts. The intent of the check can range from
simple matching (i.e. does the LLM response exactly match the expected value provided) to using an
LLM to evaluate the response.
"""
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum, auto
import re
from textwrap import dedent
from typing import Callable, Type


class CheckType(Enum):
    """TODO document."""

    MATCH_EXACT = auto()
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


class Result(ABC):
    """
    The result of an individual check. There can be multiple results per check and various Check
    objects can have different types of results, making large-scale summarization difficult if
    results are not standardized. The Result class is responsible for standardizing the results
    across all checks.
    """

    def __init__(self, value: bool | int | float | object, metadata: dict | None = None) -> None:
        self.value = value
        self.metadata = metadata

    @abstractproperty
    @property
    def success(self) -> bool:
        """
        Regardless if the result is a boolean (pass/fail) or int/float (score), the definition of
        success should be defined.
        """

    def __str__(self) -> str:
        """TODO document."""
        return dedent(f"""
            {self.__class__.__name__}(
                success={self.success},
                value={self.value},
                metadata={self.metadata}
            )
        """).strip()

class PassFailResult(Result):
    """TODO document."""

    @property
    def success(self) -> bool:
        """TODO document."""
        return self.value


class ScoreResult(Result):
    """TODO document."""

    def __init__(
            self,
            value: bool | int | float | object,
            success_threshold: float | None = None,
            metadata: dict | None = None) -> None:
        super().__init__(value, metadata)
        self.success_threshold = success_threshold

    @property
    def success(self) -> bool:
        """TODO document."""
        if self.success_threshold is None:
            return None
        return self.value >= self.success_threshold

    def __str__(self) -> str:
        """TODO document."""
        return dedent(f"""
            {self.__class__.__name__}(
                success={self.success},
                value={self.value},
                success_threshold={self.success_threshold},
                metadata={self.metadata}
            )
        """).strip()


class Check(ABC):
    """TODO."""

    def __init__(self, metadata: dict | None = None) -> None:
        super().__init__()
        self.metadata = metadata or {}

    @abstractmethod
    def __call__(self, response: str) -> list[Result]:
        """A check can have multiple sub-checks/results."""

    def __str__(self) -> str:
        """TODO document."""
        return f"{self.__class__.__name__}(metadata={self.metadata})"


class CheckRegistery:
    """Registry for types (subclasses) of Check."""

    # TODO: test string registration type

    def __init__(self):
        self._registry: dict[str, str] = {}

    def register(self, key: str | CheckType, check_type: Type[Check]) -> None:
        """Register an Check with the registry."""
        if isinstance(key, CheckType):
            key = key.name
        if key in self._registry:
            raise ValueError(f"An Check with name '{key}' is already registered.")
        self._registry[key] = check_type

    def create_check(self, check_type: CheckType | str, params: dict) -> Check:
        """Create a test from a config."""
        if isinstance(check_type, CheckType):
            check_type = check_type.name
        if check_type not in self._registry:
            raise ValueError(f"CheckType '{check_type}' not found in registry.")
        return self._registry[check_type](**params)

    def registered(self) -> dict[str, Type[Check]]:
        """List all registered Checks."""
        return self._registry

    def __contains__(self, key: CheckType | str) -> bool:
        """Check if a Check is registered."""
        if isinstance(key, CheckType):
            key = key.name
        return key in self._registry


def register_check(test_type: CheckType) -> Check:
    """Decorator to register an Check."""
    def decorator(cls: Check) -> Check:
        assert issubclass(cls, Check), \
            f"Test '{test_type}' ({cls.__name__}) must extend CheckType"
        assert (test_type not in CHECK_REGISTRY), \
            f"Test '{test_type}' already registered."
        CHECK_REGISTRY.register(test_type, cls)
        return cls
    return decorator


CHECK_REGISTRY = CheckRegistery()


# TODO: __call__ should return a (list?) of CheckResult object(s) in addition to caching the
# results in self.results


@register_check(CheckType.MATCH_EXACT)
class MatchExactCheck(Check):
    """TODO document."""

    def __init__(self,
            values: list[str],
            metadata: dict | None = None) -> None:
        super().__init__(metadata=metadata)
        self.values = values

    def __call__(self, response: str) -> list[Result]:
        """TODO: document."""
        return [
            PassFailResult(
                value=response == value,
                metadata={'type': CheckType.MATCH_EXACT.name, 'value': value},
            )
            for value in self.values
        ]

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(values={self.values}, metadata={self.metadata})"

@register_check(CheckType.MATCH_CONTAINS)
class MatchContainsCheck(Check):
    """
    Checks if the LLM response (string) contains the provided value(s) (i.e. the value/string is
    found anywhere in the response).
    """

    def __init__(self,
            values: list[str],
            metadata: dict | None = None) -> None:
        super().__init__(metadata=metadata)
        self.values = values

    def __call__(self, response: str) -> list[Result]:
        """TODO: document."""
        return [
            PassFailResult(
                value=value in response,
                metadata={'type': CheckType.MATCH_CONTAINS.name, 'value': value},
            )
            for value in self.values
        ]

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(values={self.values}, metadata={self.metadata})"


@register_check(CheckType.MATCH_REGEX)
class MatchRegexCheck(Check):
    """Checks if the LLM response (string) matches the provided regular expression."""

    def __init__(self,
            patterns: list[str],
            metadata: dict | None = None) -> None:
        """
        Args:
            patterns:
                The regular expression(s) to match the LLM response(s) against.
            metadata: TODO document.
        """
        super().__init__(metadata=metadata)
        if isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = patterns

    def __call__(self, response: str) -> list[Result]:
        """TODO document."""
        return [
            PassFailResult(
                value=re.compile(pattern).match(response) is not None,
                metadata={'type': CheckType.MATCH_REGEX.name, 'pattern': pattern},
            )
            for pattern in self.patterns
        ]

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(patterns={self.patterns}, metadata={self.metadata})"


@register_check(CheckType.PYTHON_FUNCTION)
class PythonFunctionCheck(Check):
    """
    Runs a Python function (using the LLM response as input). A Python function is either
    provided as a string, or the name of the function and the file path containing the function.
    A Python function test could be used for anything from a simple regex check to using an LLM
    to evaluate the response.
    """

    def __init__(self,
            function: Callable[[str, str], list[Result]],
            metadata: dict | None = None) -> None:
        """
        TODO document.

        Args:
            function: function definition as string value. If not provided, function_name and
                function_file must be provided.
            function_name: The name of the function to import from `function_file`.
            function_file: The file containing the function to import.
            metadata: TODO document.
        """
        super().__init__(metadata=metadata)
        self._function = function

    def __call__(self, prompt: str, response: str, ideal_response: str) -> Result:
        """TODO document."""
        return None
        # A slightly different requirement is that I have a python file and the name of a function
        # in that file. I need to dynamically import everything in that file and execute the
        # provided function, while passing in arguments. I don't want anything imported to affect
        # the environment that is running it.


@register_check(CheckType.PYTHON_CODE_BLOCKS)
class PythonCodeBlocksCheck(Check):
    """
    This class is responsible for executing Python code blocks returned by the LLM and then
    running the python function(s) defined in the test in the same environment as code blocks.
    For example, if the code blocks define a pandas DataFrame, the function could be used to
    check that the shape or data of the DataFrame matches expectations.

    The difference between this class and PythonFunctionTest is that this class is responsible
    for running tests against the code blocks returned by the LLM, whereas PythonFunctionTest
    is responsible for running tests against the (string) response returned by the LLM.
    """  # noqa: D404

    def __init__(self,
            assert_code_blocks: bool = True,
            code_setup: str | None = None,
            functions: list[Callable[[list[str], str, dict], list[Result]]] | None = None,
            metadata: dict | None = None) -> None:
        """
        args:
            assert_code_blocks:
                If True, ensure that the response contains code blocks. The code blocks do not
                necessary need to run successfully, but they must be present.
                If False, 
            functions:
                A list of callables. Each callable is passed the list of code blocks that were
                extracted from the response. The functions are executed in the same environment
                that the code blocks were executed in. The code blocks may or may not have executed
                successfully. The functions can test the enviroment or the code blocks.
        """
        super().__init__(metadata=metadata)
        self._assert_code_blocks = assert_code_blocks
        self._functions = functions
        self._code_setup = code_setup

    def __call__(self, response: str, environment: dict) -> list[Result]:
        """TODO document."""
        check_results = []
        # extract code blocks
        code_blocks = []
        if self._assert_code_blocks:
            check_results.append(PassFailResult(
                value=len(code_blocks) >= 1,
                metadata={
                    'type': CheckType.PYTHON_CODE_BLOCKS.name,
                    'subtype': 'assert_code_blocks'
                },
            ))
        # run code setup if provided
        if self._code_setup is not None:
            exec(self._code_setup, environment)
        # run code blocks; capture if the code blocks run successfully
        for code in code_blocks:
            try:
                exec(code, environment)
                check_results.append(PassFailResult(
                    value=True,
                    metadata={
                        'type': CheckType.PYTHON_CODE_BLOCKS.name,
                        'subtype': 'code_block',
                        'code': code,
                    },
                ))
            except Exception as e:
                check_results.append(PassFailResult(
                    value=False,
                    metadata={
                        'type': CheckType.PYTHON_CODE_BLOCKS.name,
                        'subtype': 'code_block',
                        'code': code,
                        'exception': e,
                    },
                ))
        for func in self.functions:
            # run function in same environent as code blocks
            # check_results.append(func(code_blocks, response, environment))
            pass

        return check_results

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(functions={self._functions}, metadata={self.metadata})"
