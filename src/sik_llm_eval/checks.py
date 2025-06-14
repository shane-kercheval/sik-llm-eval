"""
Defines classes for Check objects.

A "check" is a single test defined within an Eval which correspondes to a specific prompt/input.
The goal of a check is to test various aspects of the LLMs response. The intent of the check can
range from simple matching (e.g. does the LLM response match the expected value provided?)
to calculating a score (e.g. F1 score). Users can create custom Check classes.

A registry system is used to allow the user to save and load checks and check results to/from a
dictionary (e.g. from an underlying yaml or json file). This is useful for defining/storing/running
large amounts of checks/results.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from inspect import getsource
import re
from textwrap import dedent
from collections.abc import Callable
from typing import Any, ClassVar
from pydantic import BaseModel, Field
from sik_llm_eval.internal_utilities import (
    EnumMixin,
    Registry,
    execute_code_blocks,
    extract_code_blocks,
    get_value_from_path,
)
from sik_llms import OpenAI
from sik_llm_eval.utilities import (
    f1_score,
    f1_score_tokens,
    precision_score_tokens,
    recall_score_tokens,
    simple_tokenizer,
)

class CheckType(EnumMixin, Enum):
    """Provides a typesafe representation of the built-in types of Check classes."""

    MATCH = auto()
    CONTAINS = auto()
    REGEX = auto()
    LAMBDA = auto()
    PYTHON_FUNCTION = auto()
    PYTHON_CODE_BLOCKS_PRESENT = auto()
    PYTHON_CODE_BLOCK_TESTS = auto()
    LLM = auto()
    TOXICITY = auto()
    TOOL_CALL = auto()
    PRECISION_SCORE = auto()
    RECALL_SCORE = auto()
    F1_SCORE = auto()
    MAX_F1_SCORE = auto()


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
    def register(cls, result_type: str | Enum):
        """Register a subclass of Check."""
        def decorator(subclass: type[CheckResult]) -> type[CheckResult]:
            assert issubclass(subclass, CheckResult), \
                f"CheckResult '{result_type}' ({subclass.__name__}) must extend CheckResult"
            cls.registry.register(type_name=result_type, item=subclass)
            return subclass
        return decorator

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a Checkresult object from a dictionary. This method requires that the Checkresult
        subclass has been registered with the `register` decorator.
        """
        data = deepcopy(data)
        result_type = data.get('result_type', '')
        if result_type in cls.registry:
            return cls.registry.create_instance(type_name=result_type, **data)
        raise ValueError(f"Unknown type {result_type}")

    def to_dict(self) -> dict:
        """Return a dictionary representation of the CheckResult."""
        result_dict = self.model_dump(exclude_defaults=True, exclude_none=True)
        if self.result_type:
            result_dict['result_type'] = self.result_type
        if 'success' not in result_dict:
            result_dict['success'] = self.success
        return result_dict

    @property
    def result_type(self) -> str:
        """The type of check."""
        if hasattr(self, '_type_name'):
            return self.__class__._type_name.upper()
        return self.__class__.__name__


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


@dataclass
class ResponseModel:
    """
    Stores the data associated with a request/response. This object is created by, for example, the
    Eval/EvalHarness and passed to the Check objects' __call__ function to evaluate the response,
    potentially using additional information like input, metadata, or ideal_response.
    """

    input: object | None = None
    response: object | None = None
    ideal_response: object | None = None
    metadata: dict[str, object] | None = None

    def extract_values(self, path: str | list | dict | None) -> object:
        """
        Extracts the values from the ResponseData object based on the path.

        If the path is None, the entire ResponseData object is returned.

        If the path is a string, then a single value is extracted from the ResponseData object
        according to the specified path.

        For example, if the path is `response['content']`, then the response property is assumed
        to be a dictionary and the value associated with the key 'content' is extracted.

        If the path is a dictionary, the dictionary keys should correspond to the names of the
        keys of the dictionary returned, and the values should be the paths to the values in the
        ResponseData object.

        If the path is a list, the list should contain the paths to the values in the ResponseData
        object. A list of the extracted values will be returned.
        """
        if not path:
            return self
        # the most common case is 'response'; it's faster to simply check for this value
        # then call get_value_from_path every time
        if path == 'response':
            return self.response
        return get_value_from_path(path, self)


class Check(BaseModel, ABC):
    """
    Represents a single check in an Eval.

    A Check is a single test defined within an Eval which correspondes to a specific prompt/input.
    The goal of a check is to test various aspects of the LLMs response. The intent of the check
    can range from simple matching (e.g. does the LLM response match the expected value provided?)
    to calculating a score (e.g. F1 score). Users can create custom Check classes.

    A Check can be saved to and loaded from a dictionary (e.g. to/from an underlying yaml or json
    file). If the user wants to load the Check into memory as a Check object the, then
    corresponding Check subclass must be registered with the `register` decorator (e.g.
    `@Check.register(<check name>)`). This allows the Check to be created from a dictionary by
    calling `from_dict`.
    """

    registry: ClassVar[Registry] = Registry()
    data_path: str | dict = Field(
        default=None,
        description="""
        `data_path` is a string, list, or dictionary specifying where to extract the value from
        the ResponseModel. When the Check is ran from an Eval, the check is called via
        `run_on_model`, and the ResponseModel object is passed to the check so that the object has
        access to all data associated with the eval (e.g. response, original input, the ideal
        response, etc.).

        If the value is a string, then a single value is extracted from the ResponseModel object
        according to the specified path. The string path value specifies the attribute access,
        dictionary key access, or list index access to the value in the ResponseModel object.

        For example, if the path is `response['content']`, then the response property is assumed
        to be a dictionary and the value associated with the key 'content' is extracted.

        The default `data_path` is 'response', which extracts the `response` attribute from
        the ResponseModel object. Therefore, by default, all Check objects will be passed the
        `response` attribute from the ResponseModel object, unless the class overrides the
        `default_data_path` property, or the `data_path` is set to a different value.

        If `data_path` is set to an empty string, the entire ResponseModel object will be
        passed to the check. This is useful if the check needs to access multiple fields in the
        ResponseModel object.

        Syntax:
        1. Attribute access: "attribute_name"
        2. Dictionary key access: "['key_name']"
        3. List index access: "[index]"

        Basic Usage:
        - Attribute access: "response.content"
        - Dictionary key access: "response.metadata['sentiment']"
        - List index access: "response.content[0]"
        - Chaining operations: "response.metadata[0]['sentiment']"

        NOTE: Integer values used in brackets (e.g. [0]) will be converted to integers for support
        of list indexing. This means that dictionaries with integer keys will work as expected,
        but dictionaries with string keys that are digits will not index correctly.

        If the path is a list, then each item should be the path to a value in the data object, and
        a list of values is returned.

        If the data_path is a dictionary, the dictionary dictionary keys should correspond to
        the names of the parameters in the Check subclass `__call__` method. The values in the
        dictionary should be the paths to the values in the ResponseModel object (in the same
        format as the string data_path, described above). The values extracted from the
        ResponseModel object will be passed to the Check subclass `__call__` method as keyword
        arguments.

        Example:

        ```
        data_path = {
            'response': 'response['content']',
            'metadata': 'response.metadata',
        }
        class MyCheck(Check):
            def __call__(self, response: str, metadata: dict[str, Any]) -> CheckResult:
                # response will be the value at response['content']
                # metadata will be the value at response.metadata
                pass
        ```
        """,
    )
    metadata: dict[str, Any] = {}

    def __init__(self, **data: dict):
        super().__init__(**data)
        if self.data_path is None:
            self.data_path = self.default_data_path

    @property
    def default_data_path(self) -> str | dict:
        """
        Returns the default value extractor for the check. The most common value is 'response',
        which returns the `response` attribute from the ResponseModel object. If the check needs to
        access multiple fields in the ResponseData object, the data_path should be set to an
        empty string and the entire ResponseData object will be passed to the check.
        """
        return 'response'

    @abstractmethod
    def __call__(self, **kwargs: dict[str, Any]) -> CheckResult:
        """
        Invokes the check on the value extracted from the ResponseData object, based on the
        `data_path`.

        NOTE: if there is an error extracting the value from the ResponseData object based on the
        `data_path` (within __call__, which calls this method), the value passed to the check
        will be `None` and the Check subclass should handle accordingly and return a CheckResult
        with the appropriate metadata.
        """

    def run_on_model(self, data: ResponseModel) -> CheckResult:
        """
        Invokes the check on the ResponseData object returned. Rather than running the check
        directly by calling the object and passing in the value(s) to check, this method extracts
        the value(s) from the ResponseData object based on the `data_path` and then calls the
        check with the extracted value(s).

        This is useful for running a large number of checks where the entity responsible for
        running the checks (e.g. in TestHarness) isn't aware of the the required parameters
        for each check. The entity simply passes the ResponseData object to the check, and the
        check extracts the required values from the ResponseData object and runs the check.
        """
        check_data = None
        error = None
        try:
            check_data = data.extract_values(self.data_path)
        except Exception as e:
            # if there is an error extracting the value from the ResponseModel object based on the
            # data_path, the value passed to the check will be None
            # we still need to do the check, but the check will fail
            error = str(e)
            if isinstance(self.data_path, dict):
                check_data = dict.fromkeys(self.data_path)
        # if the data_path is a dictionary, the keys correspond to the names of the
        # parameters in the Check subclass `_call` method and the values are the paths to the
        # values in the ResponseData object; so we will pass as keyword arguments
        result = self(**check_data) if isinstance(self.data_path, dict) else self(check_data)
        if error or (self.data_path != self.default_data_path):
            result.metadata['data_path'] = self.data_path
            result.metadata['value_extracted'] = check_data
        if error:
            result.metadata['data_path_error'] = error
        return result

    @classmethod
    def register(cls, check_type: str | Enum):
        """Register a subclass of Check."""
        def decorator(subclass: type[Check]) -> type[Check]:
            assert issubclass(subclass, Check), \
                f"Check '{check_type}' ({subclass.__name__}) must extend Check"
            cls.registry.register(type_name=check_type, item=subclass)
            return subclass
        return decorator

    @classmethod
    def from_dict(cls, data: dict):
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
        value = {}
        if self.check_type:
            value['check_type'] = self.check_type
        value.update(self.model_dump(exclude_defaults=True, exclude_none=True))
        if 'data_path' in value and value['data_path'] == self.default_data_path:
            del value['data_path']
        return value

    @property
    def check_type(self) -> str:
        """The type of check."""
        if hasattr(self, '_type_name'):
            return self.__class__._type_name.upper()
        return self.__class__.__name__

    def __str__(self) -> str:
        """String representation of the Check."""
        return f"{self.__class__.__name__}(metadata={self.metadata})"


@Check.register(CheckType.MATCH)
class MatchCheck(Check):
    """Checks if the LLM response exactly matches the provided value."""

    value: str = Field(description="The value to match the LLM response against.")
    negate: bool = Field(
        default=False,
        description="If True, the check will pass if the response does not match the value.",
    )

    def __call__(self, value: str | None) -> PassFailResult:
        """Executes the check on the response and returns a PassFailResult."""
        if value is None:
            result = False
        else:
            result = self.value == value if not self.negate else self.value != value
        return PassFailResult(
            value=result,
            metadata={
                'check_type': self.check_type,
                'check_value': self.value,
                'check_negate': self.negate,
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
    negate: bool = Field(
        default=False,
        description="If True, the check will pass if the response does not contain the value.",
    )

    def __call__(self, value: str | None) -> PassFailResult:
        """Executes the check on the response and returns a PassFailResult."""
        if value is None:
            result = False
        else:
            result = self.value in value if not self.negate else self.value not in value
        return PassFailResult(
            value=result,
            metadata={
                'check_type': self.check_type,
                'check_value': self.value,
                'check_negate': self.negate,
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
    negate: bool = Field(
        default=False,
        description="If True, the check will pass if the response does not match the regular expression.",  # noqa
    )

    def __call__(self, value: str | None) -> PassFailResult:
        """Executes the check on the response and returns a PassFailResult."""
        if value is None:
            result = False
        else:
            found = re.search(self.pattern, value, re.MULTILINE) is not None
            result = found if not self.negate else not found
        return PassFailResult(
            value=result,
            metadata={
                'check_type': self.check_type,
                'check_pattern': self.pattern,
                'check_negate': self.negate,
                'check_metadata': self.metadata,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(pattern='{self.pattern}', metadata={self.metadata})"


@Check.register(CheckType.LAMBDA)
class LambdaCheck(Check):
    """
    Check that runs a Python lambda function against the response. The lambda function is passed
    in as a string so that the class is serializable. The lambda function should take a single
    argument and return a boolean value indicating whether the check passes or fails.
    """

    lambda_str: str = Field(description="The lambda function to run against the response.")

    def __call__(self, value: str | None) -> PassFailResult:
        """Executes the check on the response and returns a PassFailResult."""
        try:
            result = eval(self.lambda_str)(value)
            error = None
        except Exception as e:
            result = False
            error = str(e)
        result = PassFailResult(
            value=result,
            metadata={
                'check_type': self.check_type,
                'check_metadata': self.metadata,
                'lambda_str': self.lambda_str,
            },
        )
        if error:
            result.metadata['lambda_error'] = error
        return result

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(value={self.lambda_str}, metadata={self.metadata})"


@Check.register(CheckType.PYTHON_CODE_BLOCKS_PRESENT)
class PythonCodeBlocksPresent(Check):
    """
    Checks that the response contains code blocks. The code blocks do not necessary need to run
    successfully (this check does not run the code blocks), but they must be present.

    The full response from the LLM is passed to the __call__ method of the Check, and the code
    blocks are extracted from the response and executed in the order they are found in the
    response.
    """

    min_code_blocks: int = Field(
        default=1,
        description="The minimum number of code blocks that must be present in the response.",
    )

    def __call__(self, value: str) -> PassFailResult:
        """
        Returns a PassFailResult based on the number of code blocks present.

        NOTE: We are currently assuming any code blocks are Python code blocks.
        We could either check for "```python" or we could check for "```" and then check if the
        code blocks run, but a) we'd be running the code blocks twice if there is a
        PythonCodeBlockTests check and b) just because the code blocks fail doesn't mean they
        aren't Python code blocks.
        """
        value = value or ''
        if not isinstance(value, str):
            raise ValueError(f"Expected value to be a string, got {type(value)}")
        code_blocks = extract_code_blocks(value)
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
    This Check tests that the code blocks contained within the response run successfully, and
    allows users to define custom tests that can be used to test the code blocks and the
    environment that the code blocks are executed in. The full response from the LLM is passed to
    the __call__ method of the Check, and the code blocks are extracted from the response and
    executed in the order they are found in the response.

    Unlike other checks, this check aggregates several metrics into a single result.

        - number of code blocks were generated
        - number of code blocks that ran successfully
        - number of custom checks that ran successfully

    The `success_threshold` is the minimum **percent** of successfully executed code blocks *and*
    custom tests (if `code_tests` is used) required for the check to be considered successful.
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
        description="""
        Python code that is executed before the code blocks are executed.

        NOTE: If the code within the `code_setup` raises an exception, the exception will be raised
        to the main environment and execution of the Eval will stop. This is because the setup code
        is assumed to work and if it doesn't, the check is not valid. If the code blocks raise any
        errors, the errors will be captured and returned as part of the check result, but the Eval
        will continue to run.
        """,
    )
    env_namespace: dict[str, Any] | None = Field(
        default=None,
        description="""
        The environment namespace to use when executing the code blocks and custom tests. This
        allows the user to define variables, functions, etc. that can be used by the code blocks
        and custom tests. If `env_namespace` is not provided, the code blocks and custom tests will
        be executed in a clean environment.

        The namespace is a dictionary where the keys define environment variables and the values
        define the values of those variables).

        The following is an example of a dictionary that can be passed to `env_namespace` to define
        a pandas DataFrame called `df`, which can be used by the code blocks and custom tests:

        `{'df': pd.DataFrame({'col_1': [20, 5, 50], 'col_2': ['a', 'a', 'b']})}`
        """,
    )
    code_block_timeout: int | None = Field(
        default=None,
        description="""
        The maximum time (in seconds) to allow each/individual code block to run. If the code block
        takes longer than the timeout, that code block will be marked as failed and a TimeoutError
        will be included in the `code_block_errors` list in the metadata of the ScoreResult object
        returned (in the corresponding index of the code-block).
        """,
    )
    code_test_timeout: int | None = Field(
        default=None,
        description="""
        The maximum time (in seconds) to allow each/individual code test (in `code_tests` list) to
        run. If the code test takes longer than the timeout, that test will be marked as failed and
        a TimeoutError will be included in the `code_test_errors` list in the metadata of the
        ScoreResult object returned (in the corresponding index of the code-test).
        """,
    )
    code_tests: list[str | Callable[[list[str]], bool]] | None = Field(
        default=None,
        description="""
        code_tests can either be a list of functions (or strings representing functions), or string
        values containing single assertion statement, or string values containing a single
        statement that results in a boolean value.

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

        Example yaml:
        - check_type: PYTHON_CODE_BLOCK_TESTS
            code_block_timeout: 5
            code_test_timeout: 5
            code_tests:
                - |
                def verify_function_exists_and_runs_correctly(code_blocks: list[str]) -> bool:
                    # should pass
                    return sum_two_numbers(2, 3) == 5

        Example yaml:
        - check_type: PYTHON_CODE_BLOCK_TESTS
            code_tests:
                - assert mask_email('john.doe@example.com') == '[MASKED]@example.com'
                - assert mask_email('jane.smith@example.com') == '[MASKED]@example.com'
                - assert mask_email('foo@bar.com') != '[MASKED]@bar.com'
        """,
    )

    @staticmethod
    def _strip_code_tests(code_tests: list[str | Callable] | None) -> list[str] | None:
        """Strip whitespace from code_tests."""
        if code_tests is None:
            return None
        return [
            dedent(test.strip()) if isinstance(test, str) else test
            for test in code_tests
        ]

    def __call__(self, value: str) -> ScoreResult:  # noqa: PLR0915
        """
        Executes the check on the response and returns a ScoreResult containing the success rate of
        the code blocks and function checks (if `functions` is used), along with additional
        metadata (e.g. the code blocks, errors, etc.).
        """
        value = value or ''
        if not isinstance(value, str):
            raise ValueError(f"Expected value to be a string, got {type(value)}")
        env_namespace = self.env_namespace or {}
        code_blocks = extract_code_blocks(value)
        code_block_errors = []
        test_results = []
        test_errors = []
        code_tests = self._strip_code_tests(self.code_tests) if self.code_tests else []

        num_code_blocks = len(code_blocks)
        num_code_tests = len(code_tests)
        num_code_tests_successful = 0

        if code_blocks:
            code_blocks = code_blocks.copy()
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
            code_block_errors = execute_code_blocks(
                code_blocks=code_blocks,
                env_namespace=env_namespace,
                timeout=self.code_block_timeout,
            )
            code_block_errors = _errors_to_dict(code_block_errors)
            # add code blocks to the environment; the functions will take the code blocks
            # as input
            env_namespace['__code_blocks__'] = code_blocks
            # run the custom/user functions with contain additional tests (they functions should
            # return boolean success/fail)
            for test in code_tests:
                # __result__ is used to capture the result of the test
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
                test_exceptions = execute_code_blocks(
                    code_blocks=[function_call],
                    env_namespace=env_namespace,
                    timeout=self.code_test_timeout,
                )
                test_errors.extend(_errors_to_dict(test_exceptions))
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
        return f"{self.__class__.__name__}(tests={self.code_tests}, metadata={self.metadata})"


@Check.register(CheckType.LLM)
class LLMCheck(Check):
    """
    LLMCheck is designed to take a pydantic model that represents the response of the evaluator LLM
    and evaluate the response using the pydantic model as a schema.
    """

    eval_prompt: str = Field(description="The prompt to use by the evaluator to evaluate the response.")  # noqa: E501
    response_format: type[BaseModel] = Field("A pydantic model that represents the response of the evaluator LLM.")  # noqa: E501
    openai_model_name: str = Field(description="The OpenAI model to use for the evaluator. Structured Outputs is used to generate the response and so OpenAI is required at this time.")  # noqa: E501
    openai_model_params: dict[str, Any] | None = Field(default=None, description="The OpenAI model configuration to use for the evaluator. Structured Outputs is used to generate the response and so OpenAI is required at this time.")  # noqa: E501

    @property
    def default_data_path(self) -> str:
        """Default value extractor for the check."""
        # return entire ResponseData object so the evaluator has access to both input/response
        return None

    # TODO should/can i make __call__ async? then would have to modify TestHarness to handle async
    # checks
    def __call__(self, data: ResponseModel) -> CheckResult:
        """Executes the check on the response and returns the response of the evaluator LLM."""
        evaluator = OpenAI(
                model_name=self.openai_model_name,
                response_format=self.response_format,
                **(self.openai_model_params or {}),
        )
        messages = [{
            'role': 'user',
            'content': f"[INSTRUCTIONS]\n\n{self.eval_prompt}\n\n[USER QUESTION/REQUEST]: {data.input}\n\n[ANSWER/RESPONSE]: {data.response}",  # noqa: E501
        }]
        response = evaluator(messages)
        if not response.parsed:
            raise ValueError(f"Evaluator response is empty: {response}")

        input_cost = response.input_cost
        output_cost = response.output_cost
        total_cost = response.total_cost

        return CheckResult(
            value={
                'parsed': response.parsed,
                'refusal': response.refusal,
            },
            success=None,
            metadata={
                'check_type': self.check_type,
                'response_format': str(self.response_format),
                'check_metadata': self.metadata,
                'usage': {
                    'input_tokens': response.input_tokens,
                    'output_tokens': response.output_tokens,
                    'total_tokens': response.total_tokens,
                    'input_cost': input_cost,
                    'output_cost': output_cost,
                    'total_cost': total_cost,
                },
                'duration_seconds': response.duration_seconds,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class ActualIdealResponseMixin:
    """
    Mixin class is used to provide a common default_data_path for checks that compare the
    actual and ideal responses and take `actual_response` and `ideal_response` as arguments.
    """

    @property
    def default_data_path(self) -> dict:
        """
        The value extractor needs to extract both the actual response and the ideal response.
        Users can override this value and set `data_path` to a set of custom paths.
        It must be a dictionary with keys `actual_response` and `ideal_response` and corresponding
        paths.

        For example:
        ```
        {
            'actual_response': "response["generated_text"]",
            'ideal_response': "ideal_response",  # this will not change unless the user uses a
            # different key in the Eval
        }
        ```
        """
        return {'actual_response': 'response', 'ideal_response': 'ideal_response'}


@Check.register(CheckType.PRECISION_SCORE)
class PrecisionScore(ActualIdealResponseMixin, Check):
    """
    Calculate the precision score for token comparison.

    Precision measures the accuracy of the generated tokens. It answers the question:
    "Of the tokens we generated, what fraction were found in the expected (ideal) tokens?"

    A high precision score indicates that when the model generates tokens, they are
    often correct, but it doesn't tell us about tokens the model might have missed.
    """

    success_threshold: int | float | None = Field(
        default=None,
        description="""
        In high-stakes applications (e.g., medical diagnosis, fraud detection), precision should be
        very high, ideally above 0.9. This ensures that positive predictions are highly reliable.
        In less critical applications, a precision score above 0.7 or 0.8 may be acceptable.
        """,
    )

    def __call__(self, actual_response: str, ideal_response: str) -> ScoreResult:
        """
        Args:
            actual_response: The response generated by the LLM.
            ideal_response: The ideal response that the LLM should have generated.
        """
        generated_tokens = simple_tokenizer(actual_response)
        expected_tokens = simple_tokenizer(ideal_response)
        precision_score = precision_score_tokens(
            expected_tokens=expected_tokens,
            actual_tokens=generated_tokens,
        )
        return ScoreResult(
            value=precision_score,
            success_threshold=self.success_threshold,
            metadata={
                'check_type': self.check_type,
                'check_metadata': self.metadata,
            },
        )


@Check.register(CheckType.RECALL_SCORE)
class RecallScore(ActualIdealResponseMixin, Check):
    """
    Calculate the recall score for token comparison.

    Recall measures the completeness of the generated tokens. It answers the question:
    "Of the tokens that should have been generated, what fraction did we actually generate?"

    A high recall score indicates that the model is good at finding all the correct tokens,
    but it doesn't tell us if it also included incorrect tokens.
    """

    success_threshold: int | float | None = Field(
        default=None,
        description="""
        For applications where catching all positives is crucial (e.g., identifying spam, detecting
        potential security threats), recall should ideally be above 0.9. If false negatives are
        less critical, a recall of 0.7 or 0.8 might be considered sufficient.
        """,
    )

    def __call__(self, actual_response: str, ideal_response: str) -> ScoreResult:
        """
        Args:
            actual_response: The response generated by the LLM.
            ideal_response: The ideal response that the LLM should have generated.
        """
        generated_tokens = simple_tokenizer(actual_response)
        expected_tokens = simple_tokenizer(ideal_response)
        recall_score = recall_score_tokens(
            expected_tokens=expected_tokens,
            actual_tokens=generated_tokens,
        )
        return ScoreResult(
            value=recall_score,
            success_threshold=self.success_threshold,
            metadata={
                'check_type': self.check_type,
                'check_metadata': self.metadata,
            },
        )


@Check.register(CheckType.F1_SCORE)
class F1Score(ActualIdealResponseMixin, Check):
    """
    Calculate the F1 score for token comparison.

    The F1 score is the harmonic mean of precision and recall, providing a single score
    that balances both metrics. It answers the question:
    "What is the overall quality of the generated tokens, considering both accuracy and
    completeness?"

    A high F1 score indicates that the model has both good precision and good recall.
    It's particularly useful when you need a balance between precision and recall.
    """

    success_threshold: int | float | None = Field(
        default=None,
        description="""
        An F1-score above 0.8 is typically good, indicating a strong balance between precision and
        recall. For less critical applications, an F1-score of 0.7 or 0.75 may be acceptable.
        """,
    )
    return_precision_recall: bool = Field(
        default=False,
        description="""
        If True, the precision and recall scores will be returned in the metadata of the
        ScoreResult object.
        """,
    )

    def __call__(self, actual_response: str, ideal_response: str | list[str]) -> ScoreResult:
        """
        Args:
            actual_response: The response generated by the LLM.
            ideal_response: The ideal response that the LLM should have generated.
        """
        generated_tokens = simple_tokenizer(actual_response)
        expected_tokens = simple_tokenizer(ideal_response)
        metadata={
            'check_type': self.check_type,
            'check_metadata': self.metadata,
        }

        if self.return_precision_recall:
            precision_score = precision_score_tokens(
                expected_tokens=expected_tokens,
                actual_tokens=generated_tokens,
            )
            recall_score = recall_score_tokens(
                expected_tokens=expected_tokens,
                actual_tokens=generated_tokens,
            )
            _f1_score = f1_score(
                precision=precision_score,
                recall=recall_score,
            )
            metadata['precision'] = precision_score
            metadata['recall'] = recall_score
        else:
            _f1_score = f1_score_tokens(
                expected_tokens=expected_tokens,
                actual_tokens=generated_tokens,
            )
        return ScoreResult(
            value=_f1_score,
            success_threshold=self.success_threshold,
            metadata=metadata,
        )


@Check.register(CheckType.MAX_F1_SCORE)
class MaxF1Score(ActualIdealResponseMixin, Check):
    """
    Similar to F1Score but calculates the maximum F1 score for a list of ideal responses. Returns
    the maximum maximum F1 score.
    """

    success_threshold: int | float | None = Field(
        default=None,
        description="""
        In high-stakes applications (e.g., medical diagnosis, fraud detection), precision should be
        very high, ideally above 0.9. This ensures that positive predictions are highly reliable.
        In less critical applications, a precision score above 0.7 or 0.8 may be acceptable.
        """,
    )
    return_precision_recall: bool = Field(
        default=False,
        description="""
        If True, the precision and recall scores will be returned in the metadata of the
        ScoreResult object.
        """,
    )

    def __call__(self, actual_response: str, ideal_response: list[str]) -> ScoreResult:
        """
        Args:
            actual_response: The response generated by the LLM.
            ideal_response: The ideal response that the LLM should have generated.
        """
        generated_tokens = simple_tokenizer(actual_response)
        max_f1_score = -1
        max_metadata = {}
        for expected in ideal_response:
            expected_tokens = simple_tokenizer(expected)
            if self.return_precision_recall:
                precision_score = precision_score_tokens(
                    expected_tokens=expected_tokens,
                    actual_tokens=generated_tokens,
                )
                recall_score = recall_score_tokens(
                    expected_tokens=expected_tokens,
                    actual_tokens=generated_tokens,
                )
                _f1_score = f1_score(
                    precision=precision_score,
                    recall=recall_score,
                )
                metadata = {
                    'precision': precision_score,
                    'recall': recall_score,
                }
            else:
                _f1_score = f1_score_tokens(
                    expected_tokens=expected_tokens,
                    actual_tokens=generated_tokens,
                )
                metadata = {}
            if _f1_score > max_f1_score:
                max_f1_score = _f1_score
                max_metadata = metadata

        max_metadata.update({
            'check_type': self.check_type,
            'check_metadata': self.metadata,
        })
        return ScoreResult(
            value=max_f1_score,
            success_threshold=self.success_threshold,
            metadata=max_metadata,
        )
