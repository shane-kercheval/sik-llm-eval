"""
Defines classes for different types of checks and corresponding registry system.

A "check" is a single test/check defined in an Eval (an Eval can have multiple checks). The check
is responsible for evaluating the response to the prompts. The intent of the check can range from
simple matching (i.e. does the LLM response exactly match the expected value provided) to using an
LLM to evaluate the response.
"""
from abc import ABC, abstractmethod
from enum import Enum, auto
import re
from textwrap import dedent
from typing import Any, Callable, ClassVar, Type
from pydantic import BaseModel, Field

from llm_evals.utilities.internal_utilities import EnumMixin, Registry


class CheckType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of Check classes."""

    MATCH = auto()
    CONTAINS = auto()
    REGEX = auto()
    PYTHON_FUNCTION = auto()
    PYTHON_CODE_BLOCKS_PRESENT = auto()
    PYTHON_CODE_BLOCKS_RUN = auto()


class CheckResultsType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of CheckReturn classes."""

    PASS_FAIL = auto()
    SCORE = auto()


class CheckResult(BaseModel, ABC):
    """
    Encapsulates the result of an individual Check. There are different types of
    checks and corresponding results, making large-scale summarization difficult if results are not
    standardized. The CheckResult class is a mechanism to standardize the results of checks.

    Each subclass should define the `success` property, which is used to determine if the check
    should be considered successful or not.

    The `value` property should be a simple type that represents the underlying result (which
    "success" is based on). The `metadata` property can be used to store additional information
    about the result.
    """

    registry: ClassVar[Registry] = Registry()

    value: bool | int | float | Any
    success: bool | None = None
    metadata: dict[str, Any] = {}

    def __str__(self) -> str:
        return dedent(f"""
            {self.__class__.__name__}(
                success={self.success},
                value={self.value},
                metadata={self.metadata}
            )
        """).strip()

    @classmethod
    def register(cls, result_type: str | Enum):  # noqa: ANN102
        """Register a subclass of Check."""
        def decorator(subclass: Type[CheckResult]) -> Type[CheckResult]:
            assert issubclass(subclass, CheckResult), \
                f"CheckResult '{result_type}' ({subclass.__name__}) must extend CheckResult"
            cls.registry.register(type_name=result_type, item=subclass)
            return subclass
        return decorator

    @classmethod
    def from_dict(cls, data: dict):  # noqa: ANN102
        """
        Create a Checkresult object from a dictionary. This method requires that the Checkresult
        subclass has been registered with the `register` decorator.
        """
        result_type = data.get('result_type', '')
        if result_type in cls.registry:
            return cls.registry.create_instance(type_name=result_type, **data)
        raise ValueError(f"Unknown type {result_type}")

    def to_dict(self) -> dict:
        """Return a dictionary representation of the CheckResult."""
        result_dict = self.model_dump(exclude_defaults=True, exclude_none=True)
        if self.result_type:
            result_dict['result_type'] = self.result_type.upper()
        if 'success' not in result_dict:
            result_dict['success'] = self.success
        return result_dict

    @property
    def result_type(self) -> str:
        """The type of check."""
        return self.__class__._type_name.upper()


@CheckResult.register(CheckResultsType.PASS_FAIL)
class PassFailResult(CheckResult):
    """Represents a pass/fail (True/False) result."""

    def __init__(self, **data):  # noqa: ANN003
        super().__init__(**data)
        # definition of success is simply the value
        self.success = self.value


@CheckResult.register(CheckResultsType.SCORE)
class ScoreResult(CheckResult):
    """
    Represents a result that has a score (e.g. int/float) and, optionally, a threshold for success.

    If the `success_threshold` is not provided, the `success` property will be None.
    """

    success_threshold: int | float | None = None
    # result_type: str = Field(default='SCORE')

    def __init__(self, **data):  # noqa: ANN003
        super().__init__(**data)
        # definition of success is whether the value is greater than the success_threshold
        if self.success_threshold is not None:
            self.success = self.value >= self.success_threshold

    def __str__(self) -> str:
        return dedent(f"""
            {self.__class__.__name__}(
                success={self.success},
                value={self.value},
                success_threshold={self.success_threshold},
                metadata={self.metadata}
            )
        """).strip()


class Check(BaseModel, ABC):
    """
    Represents a single check/test in an Eval. Each Eval can test multiple/sequential prompts, and
    each prompt can have multiple checks. The check is responsible for evaluating the response to
    the prompt. The intent of the check can range from simple matching (i.e. does the LLM response
    exactly match the expected value provided) to using custom logic (e.g. using an LLM to evaluate
    the response).

    The Check class defines a registry system that allows users to register custom checks. The
    registry system is used to dynamically create checks from a config (dictionary or yaml). Any
    user can register a new check by decorating a class with the `register` decorator. A Check
    object can then be created from a dictionary by calling `from_dict` with the name of the check
    in the dictionary with key `check_type` (registered with the decorator) and any parameters for
    the Check.
    """

    registry: ClassVar[Registry] = Registry()
    metadata: dict[str, Any] = {}

    @abstractmethod
    def __call__(self, response: str) -> CheckResult:
        """Invokes the check on the response and returns a single result."""

    @classmethod
    def register(cls, check_type: str | Enum):  # noqa: ANN102
        """Register a subclass of Check."""
        def decorator(subclass: Type[Check]) -> Type[Check]:
            assert issubclass(subclass, Check), \
                f"Check '{check_type}' ({subclass.__name__}) must extend Check"
            cls.registry.register(type_name=check_type, item=subclass)
            return subclass
        return decorator

    @classmethod
    def from_dict(cls, data: dict):  # noqa: ANN102
        """
        Create a Check object from a dictionary. This method requires that the Check subclass has
        been registered with the `register` decorator.
        """
        check_type = data.get('check_type', '')
        if check_type in cls.registry:
            return cls.registry.create_instance(type_name=check_type, **data)
        raise ValueError(f"Unknown type {check_type}")

    def to_dict(self) -> dict:
        """Return a dictionary representation of the Check."""
        value = self.model_dump(exclude_defaults=True, exclude_none=True)
        if self.check_type:
            value['check_type'] = self.check_type.upper()
        return value

    @property
    def check_type(self) -> str:
        """The type of check."""
        return self.__class__._type_name.upper()

    def __str__(self) -> str:
        """String representation of the Check."""
        return f"{self.__class__.__name__}(metadata={self.metadata})"

    # def to_dict(self) -> dict:
    #     """Return a dictionary representation of the Check."""
    #     return self.model_dump(exclude_defaults=True, exclude_none=True)


@Check.register(CheckType.MATCH)
class MatchCheck(Check):
    """Checks if the LLM response exactly matches the provided value."""

    value: str = Field(description="The value to match the LLM response against.")

    def __call__(self, response: str) -> CheckResult:
        """Executes the check on the response and returns a PassFailResult."""
        return PassFailResult(
            value=response == self.value,
            metadata={
                'check_type': self.check_type,
                'check_value': self.value,
                'check_metadata': self.metadata,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(value={self.value}, metadata={self.metadata})"


@Check.register(CheckType.CONTAINS)
class ContainsCheck(Check):
    """
    Checks if the LLM response contains the provided value (i.e. the value is found anywhere in the
    response).
    """

    value: str = Field(description="The value to match the LLM response against. If the response contains the value, the check is considered successful.")  # noqa

    def __call__(self, response: str) -> CheckResult:
        """Executes the check on the response and returns a PassFailResult."""
        return PassFailResult(
            value=self.value in response,
            metadata={
                'check_type': self.check_type,
                'check_value': self.value,
                'check_metadata': self.metadata,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(value='{self.value}', metadata={self.metadata})"


@Check.register(CheckType.REGEX)
class RegexCheck(Check):
    """Checks if the a given regular expression matches the LLM response."""

    pattern: str = Field(description="The regular expression to match the LLM response against.")

    def __call__(self, response: str) -> CheckResult:
        """Executes the check on the response and returns a PassFailResult."""
        return PassFailResult(
            value=re.compile(self.pattern).match(response) is not None,
            metadata={
                'check_type': self.check_type,
                'check_pattern': self.pattern,
                'check_metadata': self.metadata,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(pattern='{self.pattern}', metadata={self.metadata})"


@Check.register(CheckType.PYTHON_FUNCTION)
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


@Check.register(CheckType.PYTHON_CODE_BLOCKS_PRESENT)
class PythonCodeBlocksPresent(Check):
    """
    Checks that the response contains code blocks. The code blocks do not necessary need to run
    successfully, but they must be present.
    """

    min_code_blocks: int = Field(
        default=1,
        description="The minimum number of code blocks that must be present in the response.",
    )

    def __call__(self, code_blocks: str) -> CheckResult:
        """TODO document."""
        # We are currently assuming any code blocks are Python code blocks.
        # We could either check for "```python" or we could check for "```" and then check if the
        # code blocks run, but a) we'd be running the code blocks twice if there is a 
        # PythonCodeBlocksRun check and b) just because the code blocks fail doesn't mean they
        # aren't Python code blocks.
        return PassFailResult(
            value=len(code_blocks) >= self.min_code_blocks,
            metadata={
                'check_type': CheckType.PYTHON_CODE_BLOCKS_PRESENT.name,
                'num_code_blocks': len(code_blocks),
                'min_code_blocks': self.min_code_blocks,
                'code_blocks': code_blocks,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(min_code_blocks={self.min_code_blocks}, metadata={self.metadata})"  # noqa


@Check.register(CheckType.PYTHON_CODE_BLOCKS_RUN)
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

    code_setup: str | None = Field(
        default=None,
        description="Python code that is executed before the code blocks are executed.",
    )
    functions: list[
        str |
        Callable[[list[str], str, dict], list[CheckResult]]
        ] | None = Field(
        default=None,
        description="A list of callables. Each callable is passed the list of code blocks that were extracted from the response. The functions are executed in the same environment that the code blocks were executed in. The code blocks may or may not have executed successfully. The functions can test the enviroment or the code blocks.",  # noqa
    )

    def __call__(self, code_blocks: list[str]) -> ScoreResult:
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
            for func in self.functions:
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
                'check_type': CheckType.PYTHON_CODE_BLOCKS_RUN.name,
                'num_code_blocks': num_code_blocks,
                'num_code_blocks_successful': num_code_blocks_successful,
                'code_blocks': code_blocks,
                'code_block_errors': code_block_errors,
                'function_check_results': None, # TODO;; these are individual PassFailResults
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(functions={self.functions}, metadata={self.metadata})"
