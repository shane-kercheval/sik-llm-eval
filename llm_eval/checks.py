"""
Defines classes for different types of checks and corresponding registry system.

A "check" is a single test defined within an Eval corresponding to a specific prompt. The goal of a
check is to test various aspects of the LLMs response to the prompt. (An Eval can have multiple
prompts; each prompt can have multiple checks.) The intent of the check can range from simple
matching (i.e. does the LLM response exactly match the expected value provided) to using an
LLM to evaluate the response.
"""
from abc import ABC, abstractmethod
from enum import Enum, auto
from inspect import getsource
import re
from textwrap import dedent
from typing import Any, Callable, ClassVar, Type
from pydantic import BaseModel, Field, model_validator
from llm_eval.utilities.internal_utilities import EnumMixin, Registry, execute_code_blocks


class CheckType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of Check classes."""

    MATCH = auto()
    CONTAINS = auto()
    REGEX = auto()
    PYTHON_FUNCTION = auto()
    PYTHON_CODE_BLOCKS_PRESENT = auto()
    PYTHON_CODE_BLOCK_TESTS = auto()


class CheckResultsType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of CheckResult classes."""

    PASS_FAIL = auto()
    SCORE = auto()


class CheckResult(BaseModel, ABC):
    """
    Encapsulates the result and metadata of an individual Check. There are different types of
    checks and corresponding results, making large-scale summarization difficult if results are not
    standardized. The CheckResult class is a mechanism to standardize the results of checks.

    Each subclass should define the `success` property, which is used to determine if the check
    should be considered successful or not.

    The `value` property should be a simple type that represents the underlying result (which
    "success" is based on). The `metadata` property can be used to store additional information
    about the result.

    CheckResult objects can be saved to and loaded from a dictionary (e.g. from an underlying yaml
    file). If the user wants to load the CheckResult into memory and into the original subclass
    (either directly or by saving/loading an EvalResult which contains all checks associated with
    an Eval) the CheckResult subclass must be registered with the `register` decorator. This allows
    the CheckResult to be created from a dictionary by calling `from_dict` with the name of the
    check in the dictionary with key `result_type` (registered with the decorator) and any
    parameters for the CheckResult.
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
        # definition of success is simply the value in the case of a pass/fail result
        self.success = self.value


@CheckResult.register(CheckResultsType.SCORE)
class ScoreResult(CheckResult):
    """
    Represents a result that has a score (e.g. int/float) and, optionally, a threshold for success.

    If the `success_threshold` is not provided, the `success` property will be None.
    """

    success_threshold: int | float | None = None

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
    Represents a single check in an Eval. Each Eval can test multiple/sequential prompts, and
    each prompt can have multiple checks. The check is responsible for evaluating the response to
    the prompt. The intent of the check can range from simple matching (i.e. does the LLM response
    exactly match the expected value provided) to using custom logic (e.g. using an LLM to evaluate
    the response).

    A Check can be saved to and loaded from a dictionary (e.g. from an underlying yaml file). If
    the user wants to load the Check into memory and into the original subclass (either directly or
    by saving/loading an Eval or EvalResult which contains all checks associated with an Eval) the
    Check subclass must be registered with the `register` decorator. This allows the Check to be
    created from a dictionary by calling `from_dict` with the name of the check in the dictionary
    with key `check_type` (registered with the decorator) and any parameters for the Check.
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
            value=re.search(self.pattern, response, re.MULTILINE) is not None,
            metadata={
                'check_type': self.check_type,
                'check_pattern': self.pattern,
                'check_metadata': self.metadata,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(pattern='{self.pattern}', metadata={self.metadata})"


@Check.register(CheckType.PYTHON_CODE_BLOCKS_PRESENT)
class PythonCodeBlocksPresent(Check):
    """
    Checks that the response contains code blocks. The code blocks do not necessary need to run
    successfully (this check does not run the code blocks), but they must be present.
    """

    min_code_blocks: int = Field(
        default=1,
        description="The minimum number of code blocks that must be present in the response.",
    )

    def __call__(self, code_blocks: list[str]) -> PassFailResult:
        """
        Returns a PassFailResult based on the number of code blocks present.

        NOTE: We are currently assuming any code blocks are Python code blocks.
        We could either check for "```python" or we could check for "```" and then check if the
        code blocks run, but a) we'd be running the code blocks twice if there is a
        PythonCodeBlockTests check and b) just because the code blocks fail doesn't mean they
        aren't Python code blocks.
        """
        code_blocks = code_blocks or []
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


@Check.register(CheckType.PYTHON_CODE_BLOCK_TESTS)
class PythonCodeBlockTests(Check):
    """
    Tests that the code blocks contained within the response run successfully, and allows users to
    define custom tests that can be used to test the code blocks and the environment that the code
    blocks are executed in. These custom tests are functions that are executed in the same
    environment as the code blocks. They return a boolean value indicating whether or not the test/
    function had a successful result. The functions are also passed the code blocks as a parameter,
    so the functions can either test the environment and/or the code blocks directly.

    The user can define the `code_setup` which is a string containing a block of python code that
    will be executed before the code blocks are executed. Both the setup code and the code blocks
    are ran in an isolated environment. The setup code can be used to set up the environment for
    the code blocks (e.g. importing libraries, defining variables, etc.). The intent of the setup
    code is to prevent the checks from failing due to errors in the code blocks that are not
    related to the LLM response.

    NOTE: If the code within the `code_setup` raises an exception, the exception will be raised to
    the main environment and execution of the Eval will stop. This is because the setup code is
    assumed to work and if it doesn't, the check is not valid. If the code blocks raise any errors,
    the errors will be captured and returned as part of the check result, but the Eval will
    continue to run.

    Unlike other checks, this check aggregates several metrics into a single result.

        - number of code blocks were generated
        - number of code blocks that ran successfully
        - number of custom checks that ran successfully

    The `success_threshold` is the minimum **percent** of successfully executed code blocks *and*
    custom tests (if `code_tests` is used) required for the check to be considered successful.
    
    NOTE: this check will run all code blocks generated across all responses for a given Eval.
    Therefore, you cannot define multiple PythonCodeBlockTests checks within a single Eval (in
    order to avoid running the same code blocks multiple times). It is recommended to define the
    PythonCodeBlockTests check at the end of the test sequence for a given Eval. If a
    PythonCodeBlockTests check is defined in the middle of the test sequence, the code blocks
    generated from subsequent responses will not have executed yet (and corresponding values, 
    functions, etc., defined in those code blocks will not be available to the functions/tests).
    """  # noqa

    success_threshold: float = Field(
        default=1.0,
        description="""
        The minimum **percent** of successfully executed code blocks and custom tests (if
        `code_tests` is used) required for the check to be considered successful. Defaulted to 1.0
        (i.e. 100% of code blocks must run successfully).
        """,
    )
    code_setup: str | None = Field(
        default=None,
        description="Python code that is executed before the code blocks are executed.",
    )
    code_tests: list[str | Callable[[list[str]], bool]] | None = Field(
        default=None,
        description="""
        code_tests can either be a list of functions (or strings representing functions), or string
        values containing single assertion statement, or string values containing a single
        statement that results in a boolean value, or some combination of the three.

        All statements (i.e. functions, assertions, or boolean statements) are executed in the same
        environment that the code blocks were executed in. Therefore, if the code blocks were
        executed successfully, the functions will have access to the environment (e.g. function
        definitions, variables, etc.) that was created by the code blocks.

        If `code_tests` is a list of functions (or strings representing functions), the functions
        will take the code blocks (generated/extracted from the response) as input and return a
        boolean indicating if the test was successful. The functions are executed in the same
        environment that the code blocks were executed in. The code blocks may or may not have
        executed successfully. The functions can test the enviroment or the code blocks (that we
        passed into the function).

        If an item in `code_tests` is a string value and that value doesn't contain a function or
        assertion statement, then it is assumed to be a boolean statement.
        """,
    )

    @model_validator(mode='before')
    def strip_code_tests(cls, values: dict) -> dict:  # noqa: N805
        """Strip whitespace from code_tests."""
        code_tests = values.get('code_tests')
        if code_tests is not None:
            stripped_code_tests = [
                dedent(test.strip()) if isinstance(test, str) else test
                for test in code_tests
            ]
            values['code_tests'] = stripped_code_tests
        return values

    def __call__(self, code_blocks: list[str]) -> ScoreResult:
        """
        Executes the check on the response and returns a ScoreResult containing the success rate of
        the code blocks and function checks (if `functions` is used), along with additional
        metadata (e.g. the code blocks, errors, etc.).
        """
        code_blocks = code_blocks or []
        code_block_errors = []
        test_results = []
        test_errors = []
        code_tests = self.code_tests or []

        num_code_blocks = len(code_blocks)
        num_code_tests = 0
        num_code_tests_successful = 0

        if code_blocks:
            code_blocks = code_blocks.copy()
            env_namespace = {}

            if self.code_setup:
                # execute code setup; if there are errors, raise an exception and fail the check
                setup_errors = execute_code_blocks(
                    [dedent(self.code_setup)],
                    env_namespace=env_namespace,
                )
                assert all(e is None for e in setup_errors), \
                    f"Errors executing code setup in PythonCodeBlockTests: \n`{setup_errors}`"

            def _errors_to_dict(errors: list[Exception | None]) -> list[dict[str, str] | None]:
                return [
                    {'error': type(e).__name__, 'message': str(e)} if e else None
                    for e in errors
                ]

            # run the primary code blocks
            code_block_errors = execute_code_blocks(code_blocks, env_namespace=env_namespace)
            code_block_errors = _errors_to_dict(code_block_errors)
            # add code blocks to the environment; the functions will take the code blocks
            # as input
            env_namespace['__code_blocks__'] = code_blocks
            # run the custom/user functions with contain additional tests (they functions should
            # return boolean success/fail)
            for test in code_tests:
                num_code_tests += 1
                # we need to reset `__result__` to False in case one of the functions fails to
                # execute (which means `__result__` will not be set) in order to avoid grabbing
                # the result from the previous function check
                env_namespace['__result__'] = False
                if isinstance(test, Callable):
                    func_name = test.__name__
                    test = dedent(getsource(test))  # noqa: PLW2901
                else:
                    assert isinstance(test, str), \
                        f"Function must be callable or string, got {type(test)}"
                    match = re.search(r'def (\w+)\(', test)
                    if match:
                        # if the test is a string and contains a function definition, then
                        # extract the function name, but we don't need to set test because
                        # the function is already defined in test
                        func_name = match.group(1)
                    else:
                        # we are only expecting a single statement
                        test = test.strip()  # noqa: PLW2901
                        assert '\n' not in test, \
                            "Only a single statement is allowed if the value is a string."
                        # if the string value in `test` is not a function; we need to wrap it in
                        # function
                        # We will assume it is either an assertion statement or a statement that
                        # resolves to a boolean
                        # if it is an assertion statement, then we don't actually need the assert
                        # we can just remove it and return a boolean value
                        # this has the added benefit of not adding AssertionError to the list
                        # of errors returned (we will only return add the Error if the statement
                        # errors for some other reason which will reduce the noise; we already
                        # return False for unsuccessful tests)
                        if test.startswith('assert '):
                            test = test[7:]  # noqa: PLW2901
                        func_name = '__code_test__'
                        test = dedent(f"""
                        def {func_name}(code_blocks: list[str]) -> bool:
                            return {test}
                        """).strip()  # noqa: PLW2901
                # add function to environment; ignore errors, we will capture and return the errors
                # associated when we execute the function, which will fail if added the function
                # to the environment fails
                _ = execute_code_blocks([test], env_namespace=env_namespace)
                function_call = f"__result__ = {func_name}(__code_blocks__)"
                # execute the function
                # if there are errors, we will capture and return the errors
                # Errors could be caused by the LLM response (e.g. if the LLM response doesn't
                # contain the expected function name) so we don't want to fail out of the entire
                # check
                func_errors = execute_code_blocks([function_call], env_namespace=env_namespace)
                test_errors.extend(_errors_to_dict(func_errors))
                # get the result of the function from the environment
                func_result = env_namespace['__result__']
                assert isinstance(func_result, bool), f"Test must return a boolean value:\n{test}"
                if func_result:
                    num_code_tests_successful += 1
                test_results.append(func_result)

        num_code_blocks_successful = len([e for e in code_block_errors if e is None])

        if num_code_blocks > 0:
            score = (num_code_blocks_successful + num_code_tests_successful) \
                / (num_code_blocks + num_code_tests)
        else:
            score = 0.0
        return ScoreResult(
            value=score,
            success_threshold=self.success_threshold,
            metadata={
                'check_type': CheckType.PYTHON_CODE_BLOCK_TESTS.name,
                'num_code_blocks': num_code_blocks,
                'num_code_blocks_successful': num_code_blocks_successful,
                'code_blocks': code_blocks,
                'code_block_errors': code_block_errors,
                'code_tests': code_tests,
                'num_code_tests': num_code_tests,
                'num_code_tests_successful': num_code_tests_successful,
                'code_test_results': test_results,
                'code_test_errors': test_errors,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(functions={self.code_tests}, metadata={self.metadata})"
