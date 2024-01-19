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
    """Provides a typesafe representation of the built-in types of Checks."""

    MATCH_EXACT = auto()
    MATCH_CONTAINS = auto()
    MATCH_REGEX = auto()
    PYTHON_FUNCTION = auto()
    PYTHON_CODE_BLOCKS_PRESENT = auto()
    PYTHON_CODE_BLOCKS_RUN = auto()

    @staticmethod
    def to_enum(name: str) -> 'CheckType':
        """Get a CheckType from its string name (case-insensitive)."""
        if isinstance(name, CheckType):
            return name
        try:
            return CheckType[name.upper()]
        except KeyError:
            raise ValueError(f"{name.upper()} is not a valid name for a CheckType member")

    def __eq__(self, other: str) -> bool:
        """Check if the CheckType is equal to a string (case-insensitive)."""
        if isinstance(other, CheckType):
            return super().__eq__(other)
        if isinstance(other, str):
            return other.upper() == self.name.upper()
        return NotImplemented


class CheckResult(ABC):
    """
    Encapsulates the result of an individual Check. There are different types of
    checks and corresponding results (e.g. pass/fail, integer/float scores with different
    thresholds of success, etc.), making large-scale summarization difficult if results are not
    standardized. The CheckResult class is a mechanism to standardize the results of checks.

    Each subclass should define the `success` property, which is used to determine if the check
    should be considered successful or not.

    As a general rule, the `value` property should be a simple type (e.g. bool, int, float, etc.)
    that represents the underlying result, for which success is based on.
    The `metadata` property can be used to store additional information about the result.
    """

    def __init__(self, value: bool | int | float | object, metadata: dict | None = None) -> None:
        """
        Args:
            value:
                The underlying value/result of the check. In general, this value will be used in
                the `success` property to determine if the check was successful or not.
            metadata:
                Additional information about the result.
        """
        self.value = value
        self.metadata = metadata or {}

    @abstractproperty
    @property
    def success(self) -> bool:
        """Indicates wehther the result was successful or not."""

    def __str__(self) -> str:
        """TODO document."""
        return dedent(f"""
            {self.__class__.__name__}(
                success={self.success},
                value={self.value},
                metadata={self.metadata}
            )
        """).strip()


class PassFailResult(CheckResult):
    """Simple class representing a pass/fail (True/False) result."""

    @property
    def success(self) -> bool:
        """Indicates wehther the result was successful or not."""
        return self.value


class ScoreResult(CheckResult):
    """
    Represents a result that has a score (e.g. int/float) and, optionally, a threshold for success.

    If the `success_threshold` is not provided, the `success` property will be None.
    """

    def __init__(
            self,
            value: int | float,
            success_threshold: int | float | None = None,
            metadata: dict | None = None) -> None:
        super().__init__(value, metadata)
        self.success_threshold = success_threshold

    @property
    def success(self) -> bool:
        """
        Indicates wehther the result was successful or not, based on the success_threshold. If the
        success_threshold is not provided, the `success` property will be None.
        """
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
    """
    Represents a single check/test in an Eval. Each Eval can test multiple/sequential prompts, and
    each prompt can have multiple checks. The check is responsible for evaluating the response to
    the prompt. The intent of the check can range from simple matching (i.e. does the LLM response
    exactly match the expected value provided) to using custom logic (e.g. using an LLM to evaluate
    the response).
    """

    def __init__(self, metadata: dict | None = None) -> None:
        super().__init__()
        self.metadata = metadata or {}

    @abstractmethod
    def __call__(self, response: str) -> CheckResult:
        """Invokes the check on the response and returns a single result."""

    def __str__(self) -> str:
        """String representation of the Check."""
        return f"{self.__class__.__name__}(metadata={self.metadata})"


class CheckRegistery:
    """
    Registry sytem of 'checks' i.e. (subclasses) of Check. The registry system is used to
    dynamically create checks from a config (dictionary or yaml). Any user can register a new
    check by decorating a class with the `register_check` decorator. A Check object can then be
    created from a dictionary by calling `create_instance` with the name of the check (registered
    with the decorator) and any parameters for the Check.

    The CHECK_REGISTRY is a global instance of CheckRegistery that is used to create checks.
    """

    def __init__(self):
        self.registered: dict[str, Type[Check]] = {}

    def register(self, check_type: str | CheckType, check_class: Type[Check]) -> None:
        """Register an Check with the registry."""
        if isinstance(check_type, CheckType):
            check_type = check_type.name
        check_type = check_type.upper()
        if check_type in self.registered:
            raise ValueError(f"An Check with name '{check_type}' is already registered.")
        self.registered[check_type] = check_class

    def create_instance(self, check_type: CheckType | str, params: dict | None = None) -> Check:
        """Create a test from a config."""
        if isinstance(check_type, CheckType):
            check_type = check_type.name
        check_type = check_type.upper()
        if check_type not in self.registered:
            raise ValueError(f"CheckType '{check_type}' not found in registry.")
        if params is None:
            params = {}
        obj = self.registered[check_type](**params)
        obj.type = check_type
        return obj

    def __contains__(self, check_type: CheckType | str) -> bool:
        """Return true if the CheckType is registered."""
        if isinstance(check_type, CheckType):
            check_type = check_type.name
        return check_type.upper() in self.registered


def register_check(check_type: CheckType | str) -> Check:
    """Decorator to register a Check with CHECK_REGISTRY."""
    def decorator(cls: Check) -> Check:
        assert issubclass(cls, Check), \
            f"Test '{check_type}' ({cls.__name__}) must extend CheckType"
        assert (check_type not in CHECK_REGISTRY), \
            f"Test '{check_type}' already registered."
        CHECK_REGISTRY.register(check_type, cls)
        return cls
    return decorator


CHECK_REGISTRY = CheckRegistery()


@register_check(CheckType.MATCH_EXACT)
class MatchExactCheck(Check):
    """Checks if the LLM response exactly matches the provided value."""

    def __init__(self,
            value: str,
            metadata: dict | None = None) -> None:
        """
        Args:
            value:
                The value to match the LLM response against. If the response exactly matches the
                value, the check is considered successful.
            metadata: Any additional metadata to store with the check.
        """
        super().__init__(metadata=metadata)
        self.value = value

    def __call__(self, response: str) -> CheckResult:
        """Executes the check on the response and returns a PassFailResult."""
        return PassFailResult(
            value=response == self.value,
            metadata={
                'check_type': CheckType.MATCH_EXACT.name,
                'check_value': self.value,
                'check_metadata': self.metadata,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(value={self.value}, metadata={self.metadata})"


@register_check(CheckType.MATCH_CONTAINS)
class MatchContainsCheck(Check):
    """
    Checks if the LLM response contains the provided value (i.e. the value is found anywhere in the
    response).
    """

    def __init__(self,
            value: str,
            metadata: dict | None = None) -> None:
        """
        Args:
            value:
                The value to match the LLM response against. If the response contains the value,
                the check is considered successful.
            metadata: Any additional metadata to store with the check.
        """
        super().__init__(metadata=metadata)
        self.value = value

    def __call__(self, response: str) -> CheckResult:
        """Executes the check on the response and returns a PassFailResult."""
        return PassFailResult(
            value=self.value in response,
            metadata={
                'check_type': CheckType.MATCH_CONTAINS.name,
                'check_value': self.value,
                'check_metadata': self.metadata,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(value='{self.value}', metadata={self.metadata})"


@register_check(CheckType.MATCH_REGEX)
class MatchRegexCheck(Check):
    """Checks if the LLM response (string) matches the provided regular expression."""

    def __init__(self,
            pattern: list[str],
            metadata: dict | None = None) -> None:
        """
        Args:
            pattern:
                The regular expression(s) to match the LLM response(s) against.
            metadata: TODO document.
        """
        super().__init__(metadata=metadata)
        self.pattern = pattern

    def __call__(self, response: str) -> CheckResult:
        """TODO document."""
        return PassFailResult(
            value=re.compile(self.pattern).match(response) is not None,
            metadata={'type': CheckType.MATCH_REGEX.name, 'pattern': self.pattern},
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(pattern='{self.pattern}', metadata={self.metadata})"


@register_check(CheckType.PYTHON_FUNCTION)
class PythonFunctionCheck(Check):
    """
    Runs a Python function (using the LLM response as input). A Python function is either
    provided as a string, or the name of the function and the file path containing the function.
    A Python function test could be used for anything from a simple regex check to using an LLM
    to evaluate the response.

    TODO: document, named parameters are required for the function so we can pass in correct
    parameters.
    """

    def __init__(self,
            function: Callable[[str, str, str], list[CheckResult]],
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

    def __call__(
            self,
            **kwargs) -> CheckResult:  # noqa: ANN003
        """
        Calls the function based on the named parameters of the function supplied during object
        creation.

        The named parameters of the function can be kwargs (which passes all options below to the
        function) or any combination of the following:
            - prompt
            - ideal_response
            - response
            - code_blocks
        """
        return self._function(**kwargs)


@register_check(CheckType.PYTHON_CODE_BLOCKS_PRESENT)
class PythonCodeBlocksPresent(Check):
    """
    Checks that the response contains code blocks. The code blocks do not necessary need to run
    successfully, but they must be present.
    """

    def __init__(self,
            min_code_blocks: int = 1,
            metadata: dict | None = None) -> None:
        super().__init__(metadata=metadata)
        self._min_code_blocks = min_code_blocks

    def __call__(self, code_blocks: str) -> CheckResult:
        """TODO document."""
        return PassFailResult(
            value=len(code_blocks) >= self._min_code_blocks,
            metadata={
                'type': CheckType.PYTHON_CODE_BLOCKS_PRESENT.name,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(min_code_blocks={self._min_code_blocks}, metadata={self.metadata})"  # noqa


@register_check(CheckType.PYTHON_CODE_BLOCKS_RUN)
class PythonCodeBlocksRun(Check):
    """
    Unlike other checks, this check aggregates several metrics into a single result.
        - how many code blocks were generated
        - how many code blocks ran successfully
        - how custom functions ran successfully

    NOTE: this check will run all code blocks. If you have multiple PythonCodeBlocksRun (e.g. one
    for each PromptTest, then the code blocks will be run multiple times. It's recommended to
    only have one PythonCodeBlocksRun which is ran on the last PromptTest.)

    # This class is responsible for executing Python code blocks returned by the LLM and then
    # running the python function(s) defined in the test in the same environment as code blocks.
    # For example, if the code blocks define a pandas DataFrame, the function could be used to
    # check that the shape or data of the DataFrame matches expectations.

    # The difference between this class and PythonFunctionTest is that this class is responsible
    # for running tests against the code blocks returned by the LLM, whereas PythonFunctionTest
    # is responsible for running tests against the (string) response returned by the LLM.
    """  # noqa: D404

    def __init__(self,
            code_setup: str | None = None,
            functions: list[Callable[[list[str], str, dict], list[CheckResult]]] | None = None,
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
        self._functions = functions or []
        self._code_setup = code_setup

    def __call__(self, code_blocks: list[str]) -> list[CheckResult]:
        """TODO document."""

        # run code blocks and any setup code
        code_block_errors = []
        if code_blocks:
            environment = {}
            if self._code_setup is not None:
                exec(self._code_setup, environment)
            # run code blocks; capture if the code blocks run successfully
            for code in code_blocks:
                try:
                    exec(code, environment)
                    code_block_errors.append(None)
                except Exception as e:
                    code_block_errors.append(e)
            for func in self._functions:
                # run function in same environent as code blocks
                # check_results.append(func(code_blocks, response, environment))
                pass
            # return CodeBlocksRunResult(
                
            # )
        num_code_blocks = len(code_blocks)
        num_code_blocks_successful = len([e for e in code_block_errors if e is None])

        return ScoreResult(
            value=num_code_blocks_successful / num_code_blocks if num_code_blocks > 0 else 0.0,
            success_threshold=1.0,
            metadata={
                'type': CheckType.PYTHON_CODE_BLOCKS_RUN.name,
                'num_code_blocks': num_code_blocks,
                'num_code_blocks_successful': num_code_blocks_successful,
                'code_blocks': code_blocks,
                'code_block_errors': code_block_errors,
                'function_check_results': None, # TODO;; these are individual PassFailResults
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(functions={self._functions}, metadata={self.metadata})"
