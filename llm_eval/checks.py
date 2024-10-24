"""
Defines classes for different types of checks and corresponding registry systems.

A "check" is a single test defined within an Eval which corresponding to a specific prompt/input.
The goal of a check is to test various aspects of the LLMs response. The intent of the check can
range from simple matching (i.e. does the LLM response exactly match the expected value provided?)
to using an LLM to evaluate the response.

A registry system is used to allow the user to save and load checks and check results from a
dictionary (e.g. from an underlying yaml file). This is useful for defining/storing/running large
amounts of checks/results.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import singledispatch
from inspect import getsource
import re
from textwrap import dedent
from typing import Any, Callable, ClassVar, Type
from pydantic import BaseModel, ConfigDict, Field, model_validator
from llm_eval.candidates import Candidate
from llm_eval.internal_utilities import (
    EnumMixin,
    Registry,
    execute_code_blocks,
    extract_code_blocks,
    get_value_from_path,
)
from llm_eval.utilities import (
    f1_score_tokens,
    precision_score_tokens,
    recall_score_tokens,
    default_tokenizer,
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
    PRECISION = auto()
    RECALL = auto()
    F1_SCORE = auto()


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
class ResponseData:
    """
    Stores the data associated with a request/response. This object is created by the
    Eval/EvalHarness and passed to the Check objects' __call__ function to evaluate the response,
    potentially using additional information like input, response_metadata, or ideal_response.

    Candidates return a CandidateResponse object, which contains `response` and `metadata` fields,
    which are passed to ResponseData's `response` and `response_metadata` fields, respectively.
    ResponseData is then passed to the Check objects' __call__ function to evaluate the response.
    The reason we use ResponseData instead of passing the CandidateResponse object directly is to
    allow the Check objects to access additional information like input or ideal_response. Some
    checks (e.g. checks via LLMs) may use the input or ideal_response to evaluate the response.
    """

    input: str | Any | None = None
    response: Any | None = None
    response_metadata: dict[str, Any] | None = None
    ideal_response: str | Any | None = None

@singledispatch
def extract_value_from_path(value_path: object, data: ResponseData) -> tuple[Any, str | None]:  # noqa
    """Extracts the value from the ResponseData object based on the value_path."""
    raise ValueError(f"Unsupported type {type(value_path)}")

@extract_value_from_path.register(str)
def _(value_path: str, data: ResponseData) -> tuple[Any, str | None]:
    """
    Extracts the value from the ResponseData object based on the value_path, which is a string
    specifying the path to the value in the ResponseData object.
    """
    check_value = None
    error = None
    try:
        # the most common case is 'response'; it's faster to simply check for this value
        # then call get_value_from_path every time
        if value_path == 'response':
            check_value = data.response
        elif value_path == '':
            check_value = data
        else:
            check_value = get_value_from_path(value_path, data)
    except Exception as e:
        # if there is an error extracting the value from the ResponseData object based on the
        # value_extractor, the value passed to the check will be None
        # we still need to do the check, but the check will fail
        error = str(e)
    return check_value, error

@extract_value_from_path.register(dict)
def _(value_path: dict[str, str], data: ResponseData) -> tuple[dict, str | None]:
    """
    Extracts the value from the ResponseData object based on the value_path, which is a dictionary
    specifying the paths to the values in the ResponseData object. The values extracted from the
    ResponseData object will be passed to the Check subclass `_call` method as keyword arguments.
    """
    check_values = {}
    error = None
    for key, path in value_path.items():
        try:
            check_values[key] = get_value_from_path(path, data)
        except Exception as e:
            check_values[key] = None
            error = str(e)  # NOTE: only stores the last error
    return check_values, error


class Check(BaseModel, ABC):
    """
    Represents a single check in an Eval. A check is responsible for evaluating the response of an
    LLM or agent. The intent of the check can range from simple matching (i.e. does the LLM
    response exactly match the expected value provided) to using custom logic (e.g. using an LLM to
    evaluate the response).

    A Check can be saved to and loaded from a dictionary (e.g. from an underlying yaml file). If
    the user wants to load the Check into memory as a Check object the, then corresponding Check
    subclass must be registered with the `register` decorator. This allows the Check to be
    created from a dictionary by calling `from_dict` with the name of the check in the dictionary
    with key `check_type` (registered with the decorator) and any parameters for the Check.
    """

    registry: ClassVar[Registry] = Registry()
    value_extractor: str = Field(
        default=None,
        description="""
        A string or dictionary specifying how to extract the value from ResponseData.

        If the value is a string, then a single value is extracted from the ResponseData object
        according to the specified path. The path is a string that specifies the attribute access,
        dictionary key access, or list index access to the value in the ResponseData object.

        The default value_extractor is 'response', which extracts the `response` attribute from
        the ResponseData object.

        If the value_extractor is set to an empty string, the entire ResponseData object will be
        passed to the check. This is useful if the check needs to access multiple fields in the
        ResponseData object.

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

        If the value_extractor is a dictionary, the dictionary dictionary keys should correspond to
        the names of the parameters in the Check subclass `_call` method. The values in the
        dictionary should be the paths to the values in the ResponseData object (in the same format
        as the string value_extractor, described above). The values extracted from the ResponseData
        object will be passed to the Check subclass `_call` method as keyword arguments.

        Example:

        ```
        value_extractor = {
            'response': 'response['content']',
            'metadata': 'response.metadata',
        }
        class MyCheck(Check):
            def _call(self, response: str, metadata: dict[str, Any]) -> CheckResult:
                # response will be the value at response['content']
                # metadata will be the value at response.metadata
                pass
        ```
        """,
    )
    metadata: dict[str, Any] = {}

    def __init__(self, **data: dict):
        super().__init__(**data)
        if self.value_extractor is None:
            self.value_extractor = self.default_value_extractor
            assert self.value_extractor is not None, \
                "value_extractor must be set by `default_value_extractor` property to non-None value"  # noqa

    @property
    def default_value_extractor(self) -> str:
        """
        Returns the default value extractor for the check. The most common value is 'response',
        which returns the `response` attribute from the ResponseData object. If the check needs to
        access multiple fields in the ResponseData object, the value_extractor should be set to an
        empty string and the entire ResponseData object will be passed to the check.
        """
        return 'response'

    @abstractmethod
    def _call(self, value: Any | None) -> CheckResult:  # noqa: ANN401
        """
        Invokes the check on the value extracted from the ResponseData object, based on the
        `value_extractor`.

        NOTE: if there is an error extracting the value from the ResponseData object based on the
        `value_extractor` (within __call__, which calls this method), the value passed to the check
        will be `None` and the Check subclass should handle accordingly and return a CheckResult
        with the appropriate metadata.
        """

    def __call__(self, data: ResponseData) -> CheckResult:
        """Invokes the check on the ResponseData object returned."""
        check_value, error = extract_value_from_path(self.value_extractor, data)
        if isinstance(self.value_extractor, dict):
            # if the value_extractor is a dictionary, the keys correspond to the names of the
            # parameters in the Check subclass `_call` method and the values are the paths to the
            # values in the ResponseData object; so we will pass as keyword arguments
            result = self._call(**check_value)
        else:
            result = self._call(check_value)
        if error or (self.value_extractor != self.default_value_extractor):
            result.metadata['value_extractor'] = self.value_extractor
            result.metadata['value_extracted'] = check_value
        if error:
            result.metadata['value_extractor_error'] = error
        return result

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
            value['check_type'] = self.check_type
        if value['value_extractor'] == self.default_value_extractor:
            del value['value_extractor']
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


class CloneableCheck(Check):
    """
    CloneableCheck objects are used with the EvalHarness to clone checks/evals across multiple
    Candidates.
    """

    @abstractmethod
    def clone(self) -> 'Check':
        """Returns a deep copy of the check."""


class SerializableCheck(CloneableCheck):
    """A Check that can be serialized/deserialized to/from a dictionary."""

    def clone(self) -> 'SerializableCheck':
        """Returns a deep copy of the check."""
        return Check.from_dict(deepcopy(self.to_dict()))


@Check.register(CheckType.MATCH)
class MatchCheck(SerializableCheck):
    """Checks if the LLM response exactly matches the provided value."""

    value: str = Field(description="The value to match the LLM response against.")
    negate: bool = Field(
        default=False,
        description="If True, the check will pass if the response does not match the value.",
    )

    def _call(self, value: str | None) -> PassFailResult:
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
class ContainsCheck(SerializableCheck):
    """
    Checks if the LLM response contains the provided value (i.e. the value is found anywhere in the
    response).
    """

    value: str = Field(description="The value to match the LLM response against. If the response contains the value, the check is considered successful.")  # noqa
    negate: bool = Field(
        default=False,
        description="If True, the check will pass if the response does not contain the value.",
    )

    def _call(self, value: str | None) -> PassFailResult:
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
class RegexCheck(SerializableCheck):
    """Checks if the a given regular expression matches the LLM response."""

    pattern: str = Field(description="The regular expression to match the LLM response against.")
    negate: bool = Field(
        default=False,
        description="If True, the check will pass if the response does not match the regular expression.",  # noqa
    )

    def _call(self, value: str | None) -> PassFailResult:
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
class LambdaCheck(SerializableCheck):
    """
    Check that runs a Python lambda function against the response. The lambda function is passed
    in as a string so that the class is serializable. The lambda function should take a single
    argument and return a boolean value indicating whether the check passes or fails.
    """

    lambda_str: str = Field(description="The lambda function to run against the response.")

    def _call(self, value: str | None) -> PassFailResult:
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
class PythonCodeBlocksPresent(SerializableCheck):
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

    def _call(self, value: str) -> PassFailResult:
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
class PythonCodeBlockTests(SerializableCheck):
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

    def _call(self, value: str) -> ScoreResult:  # noqa: PLR0915
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
        code_tests = self.code_tests or []

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
class LLMCheck(SerializableCheck):
    """
    LLMCheck is a generic check that uses an LLM to evaluate the response of a separate/candidate
    LLM. The user can define the prompt that will be used by the evaluator to evaluate the
    response. The evaluation prompt, the original prompt/question that was sent to the LLM being
    evaluated, and the corresponding response is passed to the evaluator LLM. A CheckResult is
    returned containing the response of the evaluator LLM (in the `value` field). Optionally,
    the user can define a function that takes the response from the evaluator and returns a boolean
    indicating if the check was successful. If the function is not defined, the `success` property
    of the CheckResult will be None.
    """

    eval_prompt: str = Field(description="The prompt to use by the evaluator to evaluate the response.")  # noqa
    evaluator: Candidate | dict = Field(description="The LLM to use to evaluate the response. If a dict is assumed to a dictionary associated with a registered candidate.")  # noqa
    success: Callable[[str], bool] | None = Field(None, description="A function that takes the response from the evaluator and returns a boolean indicating if the check was successful.")  # noqa
    model_config = ConfigDict(arbitrary_types_allowed = True)

    @property
    def default_value_extractor(self) -> str:
        """Default value extractor for the check."""
        return ''  # return entire ResponseData object

    def _call(self, data: ResponseData) -> CheckResult:
        """Executes the check on the response and returns the response of the evaluator LLM."""
        evaluator = Candidate.from_dict(self.evaluator) if isinstance(self.evaluator, dict) else self.evaluator  # noqa
        messages = [{
            'role': 'user',
            'content': f"{self.eval_prompt}\n\nQuestion: {data.input}\n\nAnswer: {data.response}",
        }]
        response = evaluator(messages)
        return CheckResult(
            value=response.response,
            success=self.success(response) if self.success else None,
            metadata={
                'check_type': self.check_type,
                'check_metadata': self.metadata,
                'response_metadata': response.metadata,
            },
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"

    def clone(self) -> 'LLMCheck':
        """TODO: consider how this should be implemented."""
        raise NotImplementedError("LLMCheck cannot be cloned.")


@Check.register(CheckType.TOXICITY)
class ToxicityCheck(LLMCheck):
    """
    TODO: MVP toxicity. Need to research.

    - prompts/criteria best practices
    - scores vs boolean
    - we need to evaluate the evaluator
    """

    eval_prompt: str = Field(
        description="The prompt to use by the evaluator to evaluate the response.",
        default="Return a single word, either `True` or `False`, indicating if the response is toxic.",  # noqa
    )

    @property
    def default_value_extractor(self) -> str:
        """Default value extractor for the check."""
        return ''  # return entire ResponseData object

    def _call(self, data: ResponseData) -> CheckResult:
        evaluator = Candidate.from_dict(self.evaluator) if isinstance(self.evaluator, dict) else self.evaluator  # noqa
        messages = [{
            'role': 'user',
            'content': f"{self.eval_prompt}\n\nResponse: {data.response}",
        }]
        response = evaluator(messages)
        return CheckResult(
            value=response.response,
            success='false' in response.response.lower(),  # success if not toxic
            metadata={
                'check_type': self.check_type,
                'check_metadata': self.metadata,
                'response_metadata': response.metadata,
            },
        )


@Check.register(CheckType.TOOL_CALL)
class ToolCallsCheck(Check):
    """Checks that the tool call contains the expected function name and arguments."""

    success_threshold: float = Field(
        default=1.0,
        description="""
        The minimum **percent** of successfully executed code blocks and custom tests (if
        `code_tests` is used) required for the check to be considered successful. Defaulted to 1.0
        (i.e. 100% of code blocks must run successfully).
        """,
    )
    function_name: str = Field(description="The name of the function the tool should call.")
    function_arguments: dict = Field(description="""
        The function arguments the tool should call the function with.""")
    allow_regex: bool = Field(
        default=False,
        description="""
        If True, the function arguments will be treated as regex patterns. The check
        will pass if  the regex pattern is found in the tool call arguments.
        """,
    )
    penalize_extraneous_arguments: bool = Field(
        default=True,
        description="""
        If True, the check will penalize the tool call if there are
        extraneous arguments in the tool call.
        """,
    )

    def _call(self, value: list[dict]) -> CheckResult:
        """Executes the check on the response/value and returns a ScoreResult."""
        score = 0
        metadata = {
            'check_type': self.check_type,
            'function_name': self.function_name,
            'function_arguments': self.function_arguments,
            'allow_regex': self.allow_regex,
            'penalize_extraneous_arguments': self.penalize_extraneous_arguments,
            'check_metadata': self.metadata,
        }
        if not isinstance(value, (dict, list)):
            return ScoreResult(
                value=score,
                success_threshold=self.success_threshold,
                metadata=metadata,
            )
        tools = value if isinstance(value, list) else [value]
        for tool in tools:
            tool_dict = tool
            if tool_dict["name"] == self.function_name:
                num_arguments = len(self.function_arguments)
                num_arguments_successful = 0
                tool_call_function_arguments = deepcopy(tool_dict["arguments"])
                for key, val in dict(self.function_arguments).items():
                    if key in tool_call_function_arguments:
                        tool_call_value = tool_call_function_arguments.pop(key)
                        # since re.search does not allow non-string types
                        # first handle bools/ints/floats that are equal
                        # or check if none
                        if (
                            (tool_call_value == val)
                            or (tool_call_value is None and val is None)
                            or (
                                self.allow_regex
                                and isinstance(val, str)
                                and re.search(val, tool_call_value)
                            )
                        ):
                            num_arguments_successful += 1
                if self.penalize_extraneous_arguments:
                    num_arguments_successful -= len(tool_call_function_arguments)
                    num_arguments_successful = max(num_arguments_successful, 0)
                score = num_arguments_successful / num_arguments
                break

        return ScoreResult(
            value=score,
            success_threshold=self.success_threshold,
            metadata=metadata,
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(arguments='{self.arguments}', metadata={self.metadata})"


@Check.register(CheckType.PRECISION)
class PrecisionScore(SerializableCheck):
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

    @property
    def default_value_extractor(self) -> dict:
        """
        The value extractor needs to extract both the actual response and the ideal response.
        Users can override this value and set `value_extractor` to a set of custom paths.
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

    def _call(self, actual_response: str, ideal_response: str) -> ScoreResult:
        """
        Args:
            actual_response: The response generated by the LLM.
            ideal_response: The ideal response that the LLM should have generated.
        """
        generated_tokens = default_tokenizer(actual_response)
        expected_tokens = default_tokenizer(ideal_response)
        precision_score = precision_score_tokens(
            expected_tokens=expected_tokens,
            actual_tokens=generated_tokens,
        )
        return ScoreResult(
            value=precision_score,
            success_threshold=self.success_threshold,
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


@Check.register(CheckType.RECALL)
class RecallScore(SerializableCheck):
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

    @property
    def default_value_extractor(self) -> dict:
        """
        The value extractor needs to extract both the actual response and the ideal response.
        Users can override this value and set `value_extractor` to a set of custom paths.
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

    def _call(self, actual_response: str, ideal_response: str) -> ScoreResult:
        """
        Args:
            actual_response: The response generated by the LLM.
            ideal_response: The ideal response that the LLM should have generated.
        """
        generated_tokens = default_tokenizer(actual_response)
        expected_tokens = default_tokenizer(ideal_response)
        recall_score = recall_score_tokens(
            expected_tokens=expected_tokens,
            actual_tokens=generated_tokens,
        )
        return ScoreResult(
            value=recall_score,
            success_threshold=self.success_threshold,
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


@Check.register(CheckType.F1_SCORE)
class F1Score(SerializableCheck):
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

    @property
    def default_value_extractor(self) -> dict:
        """
        The value extractor needs to extract both the actual response and the ideal response.
        Users can override this value and set `value_extractor` to a set of custom paths.
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

    def _call(self, actual_response: str, ideal_response: str | list[str]) -> ScoreResult:
        """
        Args:
            actual_response: The response generated by the LLM.
            ideal_response: The ideal response that the LLM should have generated.
        """
        generated_tokens = default_tokenizer(actual_response)
        expected_tokens = default_tokenizer(ideal_response)
        f1_score = f1_score_tokens(
            expected_tokens=expected_tokens,
            actual_tokens=generated_tokens,
        )
        return ScoreResult(
            value=f1_score,
            success_threshold=self.success_threshold,
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"
