"""Defines classes for type of tests and a registry system."""
from abc import ABC, abstractmethod
from enum import Enum, auto
from pydantic import BaseModel


class TestType(Enum):
    """TODO document."""

    MATCH = auto()
    PYTHON_FUNCTION = auto()
    PYTHON_CODE_BLOCKS = auto()

    @staticmethod
    def to_enum(name: str) -> 'TestType':
        """Get a TestType from its name."""
        if isinstance(name, TestType):
            return name
        try:
            return TestType[name.upper()]
        except KeyError:
            raise ValueError(f"{name.upper()} is not a valid name for a TestType member")


class EvalTest(ABC):
    """
    An EvalTest corresponds to a single test defined in an Eval (an Eval can have multiple tests).
    The EvalTest is responsible for evaluating the responses to the prompts.
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


class TestResult(BaseModel):
    """TODO document."""

    result: bool | int | float | object
    description: str
    metadata: dict | None


class TestRegistry:
    """Registry for models."""

    def __init__(self):
        self._registry: dict[str, TestType] = {}

    def register(self, name: str, cls: TestType) -> None:
        """Register a model with the registry."""
        if name in self._registry:
            raise ValueError(f"A model with name '{name}' is already registered.")
        self._registry[name] = cls

    def create_test(self, test_type: TestType, params: dict) -> EvalTest:
        """Create a test from a config."""
        if test_type not in self._registry:
            raise ValueError(f"TestType '{test_type}' not found in registry.")
        return self._registry[test_type](**params)

    def registered(self) -> dict[str, TestType]:
        """List all registered models."""
        return self._registry

    def __contains__(self, value: str) -> bool:
        """Check if a model is registered."""
        return value in self._registry


def register_test(test_type: TestType) -> EvalTest:
    """Decorator to register an EvalTest."""
    def decorator(cls: EvalTest) -> EvalTest:
        assert issubclass(cls, EvalTest), \
            f"Test '{test_type}' ({cls.__name__}) must extend TestType"
        assert (test_type not in TEST_REGISTRY), \
            f"Test '{test_type}' already registered."
        TEST_REGISTRY.register(test_type, cls)
        return cls
    return decorator


TEST_REGISTRY = TestRegistry()


@register_test(TestType.MATCH)
class MatchTest(EvalTest):
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
                self.results.append(TestResult(result=None, description="TODO", metadata={}))
            else:
                self.results.append(TestResult(result=r == v, description="TODO", metadata={}))


@register_test(TestType.PYTHON_FUNCTION)
class PythonFunctionTest(EvalTest):
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


@register_test(TestType.PYTHON_CODE_BLOCKS)
class PythonCodeBlocksTest(EvalTest):
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
