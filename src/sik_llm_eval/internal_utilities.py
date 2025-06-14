
"""Helper functions and classes that are not intended to be used externally."""

from abc import abstractmethod
from enum import Enum
from functools import singledispatch
from inspect import isclass, ismethod, signature, isfunction
import datetime
from itertools import product
import json
import os
import signal
from types import FunctionType
import hashlib
from collections.abc import Callable
import re
import io
import contextlib
from textwrap import dedent
from typing import Any, TypeVar
import tenacity
import yaml


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
    if not markdown_text:
        return []
    pattern = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)
    matches = pattern.findall(markdown_text)
    return [match.strip() for match in matches]


def __exec_timeout_handler(signum, frame):  # noqa
            raise TimeoutError()


def execute_code_blocks(
        code_blocks: list[str],
        env_namespace: dict | None = None,
        timeout: int | None = None) -> list[Exception]:
    """
    Execute code blocks and determine if the code blocks run successfully. Any values, functions,
    classes, etc, that are created during the execution of the code blocks are stored in the
    `env_namespace` dictionary.

    A list is returned of length `len(code_blocks)` where the items correspond to the exceptions
    raised for each code block (or a value of `None` for the code blocks that ran successfully).

    Args:
        code_blocks:
            A list of code blocks to be executed.
        env_namespace:
            A dictionary containing the global namespace for the code blocks. If None, an empty
            dictionary is used. This is useful for passing in variables that are needed for the
            code blocks to run. This is also useful if you need to keep track of the state of the
            global namespace after the code blocks have been executed (e.g. if you want to use
            variables or functions that were created during a previous call to this function.)
        timeout:
            The maximum number of seconds to wait for each/individual code block to execute.
            If None, the code blocks will not be interrupted. If the running code block take longer
            than `timeout` seconds, a TimeoutError will be added to the list of exceptions for that
            particular code block.
    """
    code_errors = []
    if env_namespace is None:
        env_namespace = {}
    for code in code_blocks:
        if timeout:
            # start the alarm/timeout
            signal.signal(signal.SIGALRM, __exec_timeout_handler)
            signal.alarm(timeout)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _ = exec(code, env_namespace)
            code_errors.append(None)
        except Exception as e:
            code_errors.append(e)
        finally:
            if timeout:
                # cancel the alarm
                signal.alarm(0)
    return code_errors


def generate_dict_combinations(value: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate all possible combinations of values from a dictionary where values can be either a
    single value or a list of values.

    Args:
        value: A dictionary where values are either a single value or a list of values.
    """
    params_lists = {
        key: value if isinstance(value, list) else [value]
        for key, value in value.items()
    }
    # Generate all combinations of parameter values
    combinations = product(*params_lists.values())
    # Convert each combination back into a dictionary format
    return [dict(zip(params_lists.keys(), combination)) for combination in combinations]


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


@singledispatch
def get_value_from_path(path: object, data: object) -> object:  # noqa: ARG001
    """
    Retrieves the value from the data object based on the specified path.

    If a string is passed as the path, the function will interpret the path as a series of
    object properties, dictionary keys, or list indices.

    If the path is a dictionary key, a dictionary is returned, and each key in the path dictionary
    corresponds to the key that is returned in the dictionary, and each value should corresponds to
    the value that is returned in the dictionary.

    If the path is a list, then each item should be the path to a value in the data object, and a
    list of values is returned.

    For example:

    ```
    data = {
        'a': {
            'b': {
                'c': 5
            }
        }
    }
    value = get_value_from_path("['a']['b']['c']", data)
    assert value == 5

    data = [10, 20, 30]
    value = get_value_from_path("[1]", data)
    assert value == 20
    ```

    This function also works with a lambda that is passed as a string:

    ```
    data = {
        'a': {
            'b': {
                'c': 'foo',
            }
        }
    }
    value = get_value_from_path("lambda x: x['a']['b']['c'].upper()", data)
    assert value(data) == 'FOO'
    ```
    """
    raise ValueError(f"Unsupported path type: `{type(path)}`")

@get_value_from_path.register(str)
def _(path: str, data: object) -> object:
    """See the main `get_value_from_path` function for details."""
    if path.startswith('lambda'):
        return eval(path)(data)
    current = data
    parts = re.findall(r'\[.*?\]|\w+', path)
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            # Dictionary or list access
            key = part[1:-1].strip("'\"")
            # Check if key is a digit (including negative numbers)
            if key.lstrip('-').isdigit():
                key = int(key)  # Convert key to int if it's a digit
            current = current[key]
        else:
            current = getattr(current, part)
    return current

@get_value_from_path.register(dict)
def _(path: dict, data: object) -> dict:
    """See the main `get_value_from_path` function for details."""
    return {key: get_value_from_path(value, data) for key, value in path.items()}

@get_value_from_path.register(list)
def _(path: list, data: object) -> list:
    """See the main `get_value_from_path` function for details."""
    return [get_value_from_path(item, data) for item in path]

T = TypeVar('T')

class Registry:
    """
    A registry for managing different types of classes.
    Allows for registering classes with a type name and creating instances of these classes.
    The registry is case-insensitive for type names.
    """

    def __init__(self):
        """Initialize the registry with an empty dictionary."""
        self._registry: dict[str, type[T]] = {}

    @staticmethod
    def _clean_type_name(type_name: str | Enum) -> str:
        """Convert the type name to uppercase."""
        if isinstance(type_name, Enum):
            return type_name.name.upper()
        return type_name.upper()

    def register(self, type_name: str | Enum, item: type[T]) -> None:
        """
        Register a class with a specified type name (case-insensitive).

        If the type name is already registered, an assertion error is raised.

        Args:
            type_name:
                The type name to be associated with the class.

                The type_name is case-insensitive. If the type_name is an Enum, the name
                (`type_name.name`) is used.
            item: The class to be registered.
        """
        type_name = self._clean_type_name(type_name)
        assert type_name not in self._registry, f"Type '{type_name}' already registered."
        item._type_name = type_name
        self._registry[type_name] = item

    def get(self, type_name: str | Enum) -> type[T]:
        """
        Get the class associated with the given type name.

        Args:
            type_name: The type name of the class to retrieve.
        """
        type_name = self._clean_type_name(type_name)
        return self._registry[type_name]

    def __contains__(self, type_name: str | Enum) -> bool:
        """
        Check if a type name is registered in the registry (case insensitive).

        Args:
            type_name: The type name to check.
        """
        type_name = self._clean_type_name(type_name)
        return type_name in self._registry

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
            raise ValueError(f"Unknown type `{type_name}`")
        return self._registry[type_name.upper()](**data)


class EnumMixin:
    """
    Mixin class for enums that provides a string-to-enum (`to_enum`) method and string (case
    insensitive) equality operater.
    """

    @classmethod
    def to_enum(cls, name: str) -> 'Enum':
        """Get an Enum member from its string name (case-insensitive)."""
        if isinstance(name, cls):
            return name
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"{name.upper()} is not a valid name for a {cls.__name__} member")

    def __eq__(self, other: Enum | str | object) -> bool:
        """Check if the Enum is equal to a string (case-insensitive)."""
        if isinstance(other, self.__class__):
            return super(Enum, self).__eq__(other)
        if isinstance(other, str):
            return other.upper() == self.name.upper()
        return False


class DictionaryEqualsMixin:
    """Mixin to compare dictionaries."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Returns a dictionary representation of the object."""

    def __eq__(self, other: object) -> bool:
        """Returns True if the dictionaries are equal."""
        if not isinstance(other, self.__class__):
            return False
        other = other if isinstance(other, dict) else other.to_dict()
        return self.to_dict() == other


class SerializationMixin:
    """Mixin for serializing objects."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Returns a dictionary representation of the object."""

    @classmethod
    @abstractmethod
    def from_dict(cls: 'SerializationMixin', data: dict) -> 'SerializationMixin':
        """Creates an object from a dictionary."""

    def to_yaml(self, file_path: str) -> None:
        """Saves the object to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls: 'SerializationMixin', path: str) -> 'SerializationMixin':
        """Creates an object from a YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    def to_json(self, file_path: str) -> None:
        """Saves the object to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls: 'SerializationMixin', path: str) -> 'SerializationMixin':
        """Creates an object from a JSON file."""
        with open(path) as f:
            config = json.load(f)
        return cls.from_dict(config)

    @classmethod
    def from_file(cls: 'SerializationMixin', path: str) -> 'SerializationMixin':
        """Creates an object from a file, detecting JSON or YAML format based on the extension."""
        _, ext = os.path.splitext(path)
        if ext.lower() == '.json':
            return cls.from_json(path)
        if ext.lower() in {'.yaml', '.yml'}:
            return cls.from_yaml(path)
        raise ValueError(f"Unsupported file extension: {ext}. Use '.json', '.yaml', or '.yml'.")
