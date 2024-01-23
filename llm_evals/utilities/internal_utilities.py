
"""Helper functions and classes that are not intended to be used externally."""

from enum import Enum
from inspect import isclass, ismethod, signature, isfunction
import datetime
from types import FunctionType
import hashlib
from collections.abc import Callable
import re
import io
import contextlib
from textwrap import dedent
from typing import Type, TypeVar
import tenacity


class Timer:
    """Provides way to time the duration of code within the context manager."""

    def __enter__(self):
        self._start = datetime.datetime.now()
        return self

    def __exit__(self, *args):  # noqa
        self._end = datetime.datetime.now()
        self.interval = self._end - self._start

    def __str__(self):
        return self.formatted(units='seconds', decimal_places=2)

    def formatted(self, units: str = 'seconds', decimal_places: int = 2) -> str:
        """
        Returns a string with the number of seconds that elapsed on the timer. Displays out to
        `decimal_places`.

        Args:
            units:
                format the elapsed time in terms of seconds, minutes, hours
                (currently only supports seconds)
            decimal_places:
                the number of decimal places to display
        """
        if units == 'seconds':
            return f"{self.interval.total_seconds():.{decimal_places}f} seconds"

        raise ValueError("Only suppports seconds.")


def create_hash(value: str) -> str:
    """Based on `value`, returns a hash."""
    # Create a new SHA-256 hash object
    hash_object = hashlib.sha256()
    # Convert the string value to bytes and update the hash object
    hash_object.update(value.encode('utf-8'))
    # Get the hexadecimal representation of the hash
    return hash_object.hexdigest()


def retry_handler(num_retries: int = 3, wait_fixed: int = 1) -> Callable:
    """
    Returns a tenacity callable object that can be used for retrying a function call.

    ```
    r = retry_handler()
    r(
        openai.Completion.create,
        model="text-davinci-003",
        prompt="Once upon a time,"
    )
    ```
    """
    return tenacity.Retrying(
        stop=tenacity.stop_after_attempt(num_retries),
        wait=tenacity.wait_fixed(wait_fixed),
        reraise=True,
    )


def has_property(obj: object, property_name: str) -> bool:
    """
    Returns True if the object has a property (or instance variable) with the name
    `property_name`.
    """
    # if `obj` is itself a function, it will not have any properties
    if isfunction(obj):
        return False

    return hasattr(obj, property_name) and \
        not callable(getattr(obj.__class__, property_name, None))


def has_method(obj: object, method_name: str) -> bool:
    """Returns True if the object has a method with the name `property_name`."""
    # if `obj` is itself a function, it will not have any properties
    if isfunction(obj):
        return False
    return hasattr(obj, method_name) and callable(getattr(obj.__class__, method_name, None))


def extract_code_blocks(markdown_text: str) -> list[str]:
    """Extract code blocks from Markdown text (e.g. llm response)."""
    pattern = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)
    matches = pattern.findall(markdown_text)
    return [match.strip() for match in matches]


def execute_code_blocks(
        code_blocks: list[str],
        global_namespace: dict | None = None) -> list[Exception]:
    """
    Execute code blocks and determine if the code blocks run successfully.

    For code blocks that run successfully, None is returned. For code blocks that fail, the
    exception is returned.

    Args:
        code_blocks:
            A list of code blocks to be executed.
        global_namespace:
            A dictionary containing the global namespace for the code blocks. If None, an empty
            dictionary is used. This is useful for passing in variables that are needed for the
            code blocks to run. This is also useful if you need to keep track of the state of the
            global namespace after the code blocks have been executed (e.g. if you want to use
            variables or functions that were created during a previous call to this function.)
        global_namespace:
            A dictionary containing the global namespace for the code blocks. If None, an empty
            dictionary is used.
    """
    block_results = []
    if global_namespace is None:
        global_namespace = {}
    for code in code_blocks:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _ = exec(code, global_namespace)
            block_results.append(None)
        except Exception as e:
            block_results.append(e)
    return block_results


def extract_variables(value: str) -> set[str]:
    """Extract variables in the format of "@variable" from a string."""
    # The regex pattern looks for @ followed by word characters or underscores and
    # ensures that it is not preceded by a word character or dot.
    return set(re.findall(r'(?<![\w.])@([a-zA-Z0-9_]+)(?![a-zA-Z0-9_]*\.[a-zA-Z0-9_])', value))


def extract_valid_parameters(func: callable, parameters: dict) -> dict:
    """
    Given a dictionary of possible parameters to pass a function `func`, returns a dictionary
    containing only the parameters that are valid for `func`.

    If `func` has a parameter named `kwargs`, then all parameters are valid and the original
    dictionary is returned.
    """
    valid_parameters = list(signature(func).parameters.keys())
    if 'kwargs' in valid_parameters:  # all parameters are valid
        return parameters
    return {p: parameters[p] for p in valid_parameters if p in parameters}


def create_function(func_str: str, func_name: str | None = None) -> callable:
    r"""
    Create a function from a string containing a Python function definition.

    NOTE: lambda functions are not directly supported. However, a lambda function assigned to a
    variable can be used.

    Example:
        func_string = "def multiply(x, y):\n    return x * y"
        func = create_function(func_string)
        result = func(2, 3)
        assert result == 6

        func_string = '''
        my_value = 5
        my_lambda = lambda x: x + my_value
        another_lambda = lambda x: x * my_value
        '''
        func = create_function(func_string, func_name='my_lambda')
        result = func(10)
        assert result == 15
        func = create_function(func_string, func_name='another_lambda')
        result = func(10)
        assert result == 50

    Args:
        func_str:
            A string containing the full definition of a Python function.
        func_name:
            The name of the function to return. If None, the first function defined in the string
            is returned.
    """
    # Dictionary to hold local scope which will contain the function definition
    local_scope = {}
    # Execute the function string within the local scope
    exec(dedent(func_str), local_scope, local_scope)
    # Filter out non-user-defined functions
    user_defined_functions = {k: v for k, v in local_scope.items() if isinstance(v, FunctionType)}
    # Retrieve the function object from the user defined functions
    if func_name:
        return user_defined_functions.get(func_name)
    return next(iter(user_defined_functions.values()), None)


def get_callable_info(callable_obj: Callable) -> str:
    """Takes a callable object and returns a string containing the signature."""
    if isfunction(callable_obj) or ismethod(callable_obj):
        # Function or method
        name = callable_obj.__name__
        params = str(signature(callable_obj))
        return f"def {name}{params}"
    if isclass(callable_obj):
        # Class
        name = callable_obj.__name__
        constructor = callable_obj.__init__
        params = str(signature(constructor))
        return f"class {name}{params}"
    if callable(callable_obj):
        # Lambda or other callable objects without a __name__ attribute
        params = str(signature(callable_obj))
        return f"lambda {params}"
    raise ValueError(f"Unsupported callable object: {callable_obj}")



T = TypeVar('T')


class Registry:
    """
    A registry for managing different types of classes.
    Allows for registering classes with a type name and creating instances of these classes.
    The registry is case-insensitive for type names.
    """

    def __init__(self):
        """Initialize the registry with an empty dictionary."""
        self._registry: dict[str, Type[T]] = {}

    def register(self, type_name: str | Enum, item: Type[T]) -> None:
        """
        Register a class with a specified type name.

        Args:
            type_name: The type name to be associated with the class.
            item: The class to be registered.
        """
        if isinstance(type_name, Enum):
            type_name = type_name.name
        type_name = type_name.upper()
        assert type_name not in self._registry, f"Type '{type_name}' already registered."
        item._type_name = type_name
        self._registry[type_name] = item

    def get(self, type_name: str) -> Type[T]:
        """
        Get the class associated with the given type name.

        Args:
            type_name: The type name of the class to retrieve.
        """
        return self._registry[type_name.upper()]

    def __contains__(self, type_name: str | Enum) -> bool:
        """
        Check if a type name is registered in the registry (case insensitive).

        Args:
            type_name: The type name to check.
        """
        if isinstance(type_name, Enum):
            type_name = type_name.name
        return type_name.upper() in self._registry

    def create_instance(self, type_name: str | Enum, **data: dict) -> T:
        """
        Create an instance of the class associated with the given type name.

        Args:
            type_name (str): The type name of the class to instantiate.
            data: Keyword arguments to be passed to the class constructor.

        Raises:
            ValueError: If the type name is not registered in the registry.
        """
        if isinstance(type_name, Enum):
            type_name = type_name.name
        if type_name.upper() not in self._registry:
            raise ValueError(f"Unknown type {type_name}")
        return self._registry[type_name.upper()](**data)
