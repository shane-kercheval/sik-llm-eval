"""Tests the utilities.py file."""
from enum import Enum, auto
from time import sleep
import re
import pytest
from textwrap import dedent
from sik_llm_eval.internal_utilities import (
    Registry,
    create_function,
    create_hash,
    execute_code_blocks,
    extract_code_blocks,
    extract_valid_parameters,
    extract_variables,
    generate_dict_combinations,
    get_value_from_path,
    has_method,
    has_property,
    retry_handler,
    Timer,
)
from sik_llm_eval.utilities import (
    f1_score_tokens,
    precision_score,
    precision_score2,
    recall_score,
    recall_score2,
    f_score,
    f1_score,
    precision_score_tokens,
    recall_score_tokens,
    simple_tokenizer,
)


def test__timer_seconds():
    with Timer() as timer:
        sleep(1.1)

    assert timer.interval
    assert re.match(pattern=r'1\.\d+ seconds', string=timer.formatted())
    assert str(timer) == timer.formatted()

    with pytest.raises(ValueError):  # noqa
        timer.formatted(units='days')

def test__create_hash():
    value_a = create_hash('Test value 1')
    assert value_a
    value_b = create_hash('Test value 2')
    assert value_b
    assert value_a != value_b
    value_c = create_hash('Test value 1')
    assert value_c == value_a

def test__retry_handler():
    r = retry_handler()
    actual_value = r(
        lambda x, y: (x, y),
        x='A',
        y='B',
    )
    assert actual_value == ('A', 'B')

def test__has_method_has_property():
    class Fake:
        def __init__(self) -> None:
            self.variable_c = 'c'

        def method_a(self) -> str:
            """Not Needed."""

        @property
        def property_b(self) -> str:
            """Not Needed."""

    assert has_method(Fake(), 'method_a')
    assert not has_method(Fake(), 'property_b')
    assert not has_method(Fake(), 'variable_c')
    assert not has_method(lambda x: x, 'test')

    assert not has_property(Fake(), 'method_a')
    assert has_property(Fake(), 'property_b')
    assert has_property(Fake(), 'variable_c')
    assert not has_property(lambda x: x, 'test')

def test__extract_code_blocks__no_code_blocks():
    result = extract_code_blocks(None)
    assert result == []
    result = extract_code_blocks("")
    assert result == []
    result = extract_code_blocks("This is a test")
    assert result == []
    result = extract_code_blocks("This is a test ```")
    assert result == []
    result = extract_code_blocks("This is a test ```python")
    assert result == []

def test__extract_code_blocks__conversation_sum(conversation_sum: dict):
    extracted_code_blocks = extract_code_blocks(conversation_sum['model_1']['responses'][0])
    assert len(extracted_code_blocks) == 2
    assert extracted_code_blocks[0] == dedent("""
        def sum_numbers(num1, num2):
            return num1 + num2
        """).strip()
    assert extracted_code_blocks[1] == dedent("""
        result = sum_numbers(5, 3)
        print(result)  # Output: 8
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_sum['model_1']['responses'][1])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        assert sum_numbers(5, 3) == 8
        assert sum_numbers(-10, 10) == 0
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_sum['model_2']['responses'][0])
    assert len(extracted_code_blocks) == 2
    assert extracted_code_blocks[0] == dedent("""
        def sum_two_numbers(num1, num2):
            return num1 + num2
        """).strip()
    assert extracted_code_blocks[1] == dedent("""
        result = sum_two_numbers(5, 3)
        print(result)  # Outputs: 8
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_sum['model_2']['responses'][1])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        assert sum_two_numbers(5, 3) == 8, "Should be 8"
        assert sum_two_numbers(-1, 1) == 0, "Should be 0"
        assert sum_two_numbers(0, 0) == 0, "Should be 0"
        assert sum_two_numbers(100, 200) == 300, "Should be 300"
        """).strip()

def test__extract_code_blocks__conversation_mask_emails(conversation_mask_email: str):
    extracted_code_blocks = extract_code_blocks(conversation_mask_email['model_1']['responses'][0])
    assert len(extracted_code_blocks) == 2
    assert extracted_code_blocks[0] == dedent("""
        def mask_email(email):
            local_part, domain = email.split('@')
            masked_local_part = '*' * len(local_part)
            masked_email = masked_local_part + '@' + domain
            return masked_email
        """).strip()
    assert extracted_code_blocks[1] == dedent("""
        email = 'example@example.com'
        masked_email = mask_email(email)
        print(masked_email)  # Output: ********@example.com
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_mask_email['model_1']['responses'][1])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        # Test case 1: Masking email with alphanumeric local part
        email1 = 'example123@example.com'
        assert mask_email(email1) == '***********@example.com'

        # Test case 2: Masking email with special characters in local part
        email2 = 'ex@mple@example.com'
        assert mask_email(email2) == '******@example.com'
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_mask_email['model_2']['responses'][0])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        def mask_email(email):
            try:
                email_parts = email.split('@')
                # Mask first part
                masked_part = email_parts[0][0] + "****" + email_parts[0][-1]
                # Combine masked part and domain
                masked_email = masked_part + '@' + email_parts[1]
                return masked_email
            except Exception as e:
                print("An error occurred: ", e)
                return None
        """).strip()

    extracted_code_blocks = extract_code_blocks(conversation_mask_email['model_2']['responses'][1])
    assert len(extracted_code_blocks) == 1
    assert extracted_code_blocks[0] == dedent("""
        assert mask_email("john.doe@example.com") == "j****e@example.com"
        assert mask_email("jane_doe@example.com") == "j****e@example.com"
        assert mask_email("test@test.com") == "t****t@test.com"
        """).strip()

def test__extract_code_blocks__llama_response():
    # output from llama response that doesn't contain `python` or multiple new lines between code
    with open('tests/fake_data/fake_llama_response_with_code_block.txt') as f:
        response = f.read()
    extracted_code_blocks = extract_code_blocks(response)
    assert len(extracted_code_blocks) == 3
    assert extracted_code_blocks[0].startswith('from typing import Generator')
    assert extracted_code_blocks[0].endswith('yield (x, y)')
    assert extracted_code_blocks[1] == 'random_walk(10)'
    assert extracted_code_blocks[2].startswith('(0, 0)')
    assert extracted_code_blocks[2].endswith('(5, -4)')

def test__execute_code_blocks__without_env_namespace(conversation_sum: dict):
    code_blocks = extract_code_blocks(conversation_sum['model_1']['responses'][0])
    code_blocks.append('assert sum_numbers(5, 3) == 8')
    code_blocks.append('assert sum_numbers(5, 3) != 8')
    assert len(code_blocks) == 4
    results = execute_code_blocks(code_blocks)
    assert len(results) == 4
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert isinstance(results[3], AssertionError)

    # this will fail because env_namespace was not reused so sum_numbers is not defined during
    # a subsequent call to execute_code_blocks
    results = execute_code_blocks(code_blocks=['assert sum_numbers(5, 3) == 8'])
    assert len(results) == 1
    assert isinstance(results[0], NameError)
    assert str(results[0]) == "name 'sum_numbers' is not defined"

def test__execute_code_blocks__with_env_namespace(conversation_sum: dict):
    code_blocks = extract_code_blocks(conversation_sum['model_1']['responses'][0])
    code_blocks.append('assert sum_numbers(5, 3) == 8')
    code_blocks.append('assert sum_numbers(5, 3) != 8')
    assert len(code_blocks) == 4
    env_namespace = {}
    results = execute_code_blocks(code_blocks, env_namespace)
    assert len(results) == 4
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert isinstance(results[3], AssertionError)
    assert 'sum_numbers' in env_namespace
    assert 'result' in env_namespace

    # this will NOT fail because env_namespace was reused so the state is carried over to
    # a subsequent call to execute_code_blocks
    results = execute_code_blocks(
        code_blocks=['assert sum_numbers(5, 3) == 8', 'assert sum_numbers(5, 3) != 8'],
        env_namespace=env_namespace,
    )
    assert len(results) == 2
    assert results[0] is None
    assert isinstance(results[1], AssertionError)

def test__execute_code_blocks__with_env_namespace__test_dependencies():
    code_blocks = [
        'my_value = 10',
        'def add_my_value(num1):\n    return num1 + my_value',
        'result = add_my_value(5)',
        'assert result != 15',  # execute error before last code block to ensure if it runs
        'assert result == 15',
    ]
    env_namespace = {}
    errors = execute_code_blocks(code_blocks, env_namespace=env_namespace)
    assert len(errors) == 5
    assert errors[0] is None
    assert errors[1] is None
    assert errors[2] is None
    assert isinstance(errors[3], AssertionError)
    assert errors[4] is None
    assert 'add_my_value' in env_namespace
    assert 'result' in env_namespace

    # this will NOT fail because env_namespace was reused so the state is carried over to
    # a subsequent call to execute_code_blocks
    errors = execute_code_blocks(
        code_blocks=['assert result != 15', 'assert result == 15'],
        env_namespace=env_namespace,
    )
    assert len(errors) == 2
    assert isinstance(errors[0], AssertionError)
    assert errors[1] is None

def test__execute_code_blocks__with_env_namespace__test_import_dependencies():
    code_blocks = [
        'import itertools',
        "s=[[ 'a', 'b', 'c'], ['d'], ['e', 'f']]",
        'combinations = list(itertools.product(*s))',
        'num_combinations = len(combinations)',
        'assert num_combinations != 6',
        'assert num_combinations == 6',
    ]
    env_namespace = {}
    errors = execute_code_blocks(code_blocks, env_namespace=env_namespace)
    assert len(errors) == 6
    assert errors[0] is None
    assert errors[1] is None
    assert errors[2] is None
    assert errors[3] is None
    assert isinstance(errors[4], AssertionError)
    assert errors[5] is None
    assert 'num_combinations' in env_namespace
    assert env_namespace['num_combinations'] == 6

def test__execute_code_blocks__with_env_namespace__inject_objects_into_namespace():
    func = dedent("""
    def get_combinations(list_of_lists: list[list[str]]):
        import itertools
        combinations = list(itertools.product(*list_of_lists))
        return len(combinations)
    """)
    code_blocks = [
        func,
        'num_combinations = get_combinations(__list_of_lists__)',
        'assert num_combinations != 6',
        'assert num_combinations == 6',
    ]
    list_to_inject = [[ 'a', 'b', 'c'], ['d'], ['e', 'f']]
    env_namespace = {
        '__list_of_lists__': list_to_inject,
    }
    errors = execute_code_blocks(code_blocks, env_namespace=env_namespace)

    assert len(errors) == 4
    assert errors[0] is None
    assert errors[1] is None
    assert isinstance(errors[2], AssertionError)
    assert errors[3] is None
    assert 'num_combinations' in env_namespace
    assert env_namespace['num_combinations'] == 6
    assert '__list_of_lists__' in env_namespace
    assert env_namespace['__list_of_lists__'] == list_to_inject

def test__execute_code_blocks__timeout():
    code_blocks = [
        """
        import time
        my_value_1 = 'test1'
        time.sleep(1.5)
        my_value_2 = 'test2'
        """,
        """
        raise ValueError("This is a test")
        """,
        """
        my_value_3 = 'test3'
        """,
    ]
    code_blocks = [dedent(c).strip() for c in code_blocks]
    # test instance where timeout is not needed
    namespace = {}
    errors = execute_code_blocks(code_blocks, env_namespace=namespace, timeout=2)
    assert len(errors) == len(code_blocks)
    assert errors[0] is None
    assert isinstance(errors[1], ValueError)
    assert errors[2] is None
    assert namespace['my_value_1'] == 'test1'
    assert namespace['my_value_2'] == 'test2'
    assert namespace['my_value_3'] == 'test3'

    # test that the timeout is enforced
    namespace = {}
    errors = execute_code_blocks(code_blocks, env_namespace=namespace, timeout=1)
    assert len(errors) == len(code_blocks)
    assert isinstance(errors[0], TimeoutError)
    assert isinstance(errors[1], ValueError)
    assert errors[2] is None

    assert namespace['my_value_1'] == 'test1'
    assert 'my_value_2' not in namespace
    assert namespace['my_value_3'] == 'test3'

def test__generate_dict_combinations():
    # test all single parameters; should return a list with one dict
    test_params = {
        'param_1': 'param_a',
        'param_2': 'param_b',
        'param_3': 'param_c',
    }
    expected_output = [
        {'param_1': 'param_a', 'param_2': 'param_b', 'param_3': 'param_c'},
    ]
    generated_combinations = generate_dict_combinations(test_params)
    assert isinstance(generated_combinations, list)
    assert all(isinstance(comb, dict) for comb in generated_combinations)
    assert len(generated_combinations) == len(expected_output)
    assert generated_combinations == expected_output

    # test single list parameter
    test_params = {
        'param_1': 'param_a',
        'param_2': 'param_b',
        'param_3': ['param_c', 'param_d'],
    }
    expected_output = [
        {'param_1': 'param_a', 'param_2': 'param_b', 'param_3': 'param_c'},
        {'param_1': 'param_a', 'param_2': 'param_b', 'param_3': 'param_d'},
    ]
    generated_combinations = generate_dict_combinations(test_params)
    assert isinstance(generated_combinations, list)
    assert all(isinstance(comb, dict) for comb in generated_combinations)
    assert len(generated_combinations) == len(expected_output)
    assert generated_combinations == expected_output

    # test all list parameters
    test_params = {
        'param_1': ['param_a', 'param_b'],
        'param_2': ['param_c', 'param_d'],
        'param_3': ['param_e', 'param_f'],
    }
    expected_output = [
        {'param_1': 'param_a', 'param_2': 'param_c', 'param_3': 'param_e'},
        {'param_1': 'param_a', 'param_2': 'param_c', 'param_3': 'param_f'},
        {'param_1': 'param_a', 'param_2': 'param_d', 'param_3': 'param_e'},
        {'param_1': 'param_a', 'param_2': 'param_d', 'param_3': 'param_f'},
        {'param_1': 'param_b', 'param_2': 'param_c', 'param_3': 'param_e'},
        {'param_1': 'param_b', 'param_2': 'param_c', 'param_3': 'param_f'},
        {'param_1': 'param_b', 'param_2': 'param_d', 'param_3': 'param_e'},
        {'param_1': 'param_b', 'param_2': 'param_d', 'param_3': 'param_f'},
    ]
    # test that no dictionaries are duplicated in expected_output
    assert len(expected_output) == len({tuple(sorted(d.items())) for d in expected_output})
    generated_combinations = generate_dict_combinations(test_params)
    assert isinstance(generated_combinations, list)
    assert all(isinstance(comb, dict) for comb in generated_combinations)
    assert len(generated_combinations) == len(expected_output)
    assert generated_combinations == expected_output

def test__extract_variables():
    assert extract_variables('') == set()
    assert extract_variables('This is an email shane@email.com not a variable.') == set()
    assert extract_variables('This is not an email shane@email and not a variable.') == set()
    assert extract_variables('shane@email.com') == set()
    assert extract_variables('.@email.com') == set()
    assert extract_variables('@email.com') == set()
    assert extract_variables('@email') == {'email'}
    text = 'This is a variable @my_variable and should be extracted'
    results = extract_variables(text)
    assert results == {'my_variable'}
    text = 'This variable is at the end of a sentence @my_variable!'
    results = extract_variables(text)
    assert results == {'my_variable'}
    text = 'This has two @my_variable and another @my_variable.'
    results = extract_variables(text)
    assert results == {'my_variable'}
    text = 'This has three @my_variable and another @my_variable and @my_variable.'
    results = extract_variables(text)
    assert results == {'my_variable'}
    text = '@_my_var_1 and @_my_var_2_.'
    results = extract_variables(text)
    assert results == {'_my_var_1', '_my_var_2_'}
    text = '@_my_var_1 and @_my_var_2_. This is another sentence'
    results = extract_variables(text)
    assert results == {'_my_var_1', '_my_var_2_'}
    text = '@_my_var_1 and @_my_var_2_; this is some more text.'
    results = extract_variables(text)
    assert results == {'_my_var_1', '_my_var_2_'}
    text = 'A variable with number @var1234 should match.'
    results = extract_variables(text)
    assert results == {'var1234'}
    text = 'A variable with underscore @var_name should match.'
    results = extract_variables(text)
    assert results == {'var_name'}
    text = 'Multiple @@ signs should not confuse @@var.'
    results = extract_variables(text)
    assert results == {'var'}
    text = 'Variable at the end of a line @end_of_line\n'
    results = extract_variables(text)
    assert results == {'end_of_line'}
    # text = 'Variables next to each other @var1@var2'
    # results = extract_variables(text)
    # assert results == {'var1', 'var2'}
    text = 'Variable in parentheses (@var_in_paren).'
    results = extract_variables(text)
    assert results == {'var_in_paren'}
    text = 'Variable in brackets [@var_in_brackets].'
    results = extract_variables(text)
    assert results == {'var_in_brackets'}
    text = 'Variable with punctuation @var_punc!'
    results = extract_variables(text)
    assert results == {'var_punc'}
    text = 'Variable with comma, @var_comma, should match.'
    results = extract_variables(text)
    assert results == {'var_comma'}
    text = 'Variables with leading underscores @_underscore_var should match.'
    results = extract_variables(text)
    assert results == {'_underscore_var'}
    text = 'Variable followed by a special character @special$ should match.'
    results = extract_variables(text)
    assert results == {'special'}
    text = 'Variable inside quotes "@quoted_var" should match.'
    results = extract_variables(text)
    assert results == {'quoted_var'}
    text = 'A tricky case with email-like pattern @not_an_email@domain.com'
    results = extract_variables(text)
    assert results == {'not_an_email'}
    text = 'Multiple variables separated by comma @var1, @var2, and @var3.'
    results = extract_variables(text)
    assert results == {'var1', 'var2', 'var3'}
    text = 'Multiple variables separated by comma and backtick `@var1`, `@var2`, and `@var3`.'
    results = extract_variables(text)
    assert results == {'var1', 'var2', 'var3'}
    text = 'Variable followed by a period and space @var_period. should match.'
    results = extract_variables(text)
    assert results == {'var_period'}
    text = 'Variable followed by other symbols @var_symbols?! should match.'
    results = extract_variables(text)
    assert results == {'var_symbols'}

def test__extract_valid_parameters():
    def my_func(x, y):  # noqa
        return x + y

    possible_parameters = {'a': 1, 'b': 2, 'c': 3, 'x': 10, 'y': 20}
    assert extract_valid_parameters(my_func, possible_parameters) == {'x': 10, 'y': 20}

    possible_parameters = {'a': 1, 'b': 2, 'c': 3, 'x': 10}
    assert extract_valid_parameters(my_func, possible_parameters) == {'x': 10}

    possible_parameters = {'a': 1, 'b': 2, 'c': 3}
    assert extract_valid_parameters(my_func, possible_parameters) == {}

    possible_parameters = {}
    assert extract_valid_parameters(my_func, possible_parameters) == {}

    def my_func(**kwargs):  # noqa
        print(kwargs)

    # if kwargs is used, then all possible parameters are valid
    possible_parameters = {'a': 1, 'b': 2}
    assert extract_valid_parameters(my_func, possible_parameters) == {'a': 1, 'b': 2}
    possible_parameters = {}
    assert extract_valid_parameters(my_func, possible_parameters) == {}

def test__create_function_from_string():
    # basic case
    func = """
    def sum_numbers(num1, num2):
        return num1 + num2
    """
    func = create_function(func)
    assert func(5, 3) == 8

    func_string = "def multiply(x, y):\n    return x * y"
    func = create_function(func_string)
    result = func(2, 3)
    assert result == 6

    # test lambda (has to be assigned to a variable)
    func_string = """
    my_value = 5
    my_lambda = lambda x: x + my_value
    """
    func = create_function(func_string)
    result = func(10)
    assert result == 15

    # test lambda (has to be assigned to a variable)
    func_string = """
    my_value = 5
    my_lambda = lambda x: x + my_value
    another_lambda = lambda x: x * my_value
    """
    func = create_function(func_string, func_name='my_lambda')
    result = func(10)
    assert result == 15
    func = create_function(func_string, func_name='another_lambda')
    result = func(10)
    assert result == 50

    # testing dependency on another function
    func_str = """
    def my_func(x):
        return x
    def sum_numbers(num1, num2):
        return my_func(num1) + num2
    """
    sum_func = create_function(func_str, func_name='sum_numbers')
    assert sum_func(5, 3) == 8

    # testing dependency on another function and value
    func_str = """
    my_value = 5
    def add_my_value(x):
        return x + my_value
    def sum_numbers(num1, num2):
        return add_my_value(num1) + num2
    """
    sum_func = create_function(func_str, func_name='sum_numbers')
    assert sum_func(5, 3) == 8 + 5

    # What happens if the function and/or value is already defined in the global namespace?
    my_value = 100
    def add_my_value(x):  # noqa
        return x + my_value * 100
    def sum_numbers(num1, num2):  # noqa
        return add_my_value(num1) + num2 * 100
    # now retest the same function string as above
    func_str = """
    my_value = 5
    def add_my_value(x):
        return x + my_value
    def sum_numbers(num1, num2):
        return add_my_value(num1) + num2
    """
    sum_func = create_function(func_str, func_name='sum_numbers')
    assert sum_func(5, 3) == 8 + 5

def test__get_value_from_path_dict():
    data = {
        'a': {
            'b': {
                'c': 5,
            },
        },
    }
    assert get_value_from_path("['a']['b']['c']", data) == 5

def test__get_value_from_path__str__non_existent_key():
    data = {'a': {'b': 10}}
    with pytest.raises(KeyError):
        get_value_from_path("['a']['c']", data)

def test__get_value_from_path__str__list():
    data = [10, 20, 30]
    assert get_value_from_path("[1]", data) == 20

def test__get_value_from_path__str__list_out_of_range():
    data = [10, 20, 30]
    with pytest.raises(IndexError):
        get_value_from_path("[5]", data)

def test__get_value_from_path__str__tuple():
    data = (100, 200, 300)
    assert get_value_from_path("[2]", data) == 300

def test__get_value_from_path__str__tuple_out_of_range():
    data = (100, 200, 300)
    with pytest.raises(IndexError):
        get_value_from_path("[5]", data)

def test__get_value_from_path__str__mixed():
    data = {
        'x': [1, 2, {'y': 10}],
    }
    assert get_value_from_path("['x'][2]['y']", data) == 10

def test__get_value_from_path__str__custom_object():
    class CustomObject:
        def __init__(self, name, age):  # noqa
            self.name = name
            self.age = age

    obj = CustomObject(name="Alice", age=30)
    assert get_value_from_path(".name", obj) == "Alice"
    assert get_value_from_path(".age", obj) == 30

def test__get_value_from_path__str__non_existent_attribute():
    class CustomObject:
        def __init__(self, name, age):  # noqa
            self.name = name
            self.age = age

    obj = CustomObject(name="Bob", age=40)
    with pytest.raises(AttributeError):
        get_value_from_path(".address", obj)

def test__get_value_from_path__str__invalid_path():
    data = {'a': {'b': 10}}
    with pytest.raises(KeyError):
        get_value_from_path("['x']['b'].x", data)

def test__get_value_from_path__str__negative_index():
    data = [10, 20, 30]
    assert get_value_from_path("[-1]", data) == 30
    assert get_value_from_path("[-2]", data) == 20

def test__get_value_from_path__str__nested_list_and_dict():
    data = [{'a': [1, 2, {'b': 3}]}]
    assert get_value_from_path("[0]['a'][2]['b']", data) == 3

def test__get_value_from_path__str__with_none():
    data = {'a': None}
    assert get_value_from_path("['a']", data) is None

def test__get_value_from_path__str__dict_with_integer_key():
    data = {1: 'one', 2: 'two'}
    assert get_value_from_path("[1]", data) == 'one'
    assert get_value_from_path("[2]", data) == 'two'

def test__get_value_from_path__str__deeply_nested():
    data = {'a': {'b': {'c': {'d': {'e': {'f': 100}}}}}}
    assert get_value_from_path("['a']['b']['c']['d']['e']['f']", data) == 100

def test__get_value_from_path__str__empty_data():
    data = {}
    with pytest.raises(KeyError):
        get_value_from_path("['a']", data)

def test__get_value_from_path__str__non_string_dict_keys():
    data = {1: {'a': 10}, 2: {'b': 20}}
    assert get_value_from_path("[1]['a']", data) == 10
    assert get_value_from_path("[2]['b']", data) == 20

def test__get_value_from_path__str__list_of_lists():
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_value_from_path("[1][2]", data) == 6
    assert get_value_from_path("[0][0]", data) == 1

def test__get_value_from_path__str__using_lamda():
    data = {'a': {'b': {'c': 5}}}
    assert get_value_from_path("lambda x: x['a']['b']['c']", data) == 5

    data = {'a': [1, 2, {'b': 'foo'}]}
    assert get_value_from_path("lambda x: x['a'][2]['b'].upper()", data) == 'FOO'

def test__get_value_from_path__dict__simple_dict():
    path = {'key1': "['a']['b']", 'key2': "['x']['y']"}
    data = {
        'a': {'b': 10},
        'x': {'y': 20},
    }
    result = get_value_from_path(path, data)
    assert result == {'key1': 10, 'key2': 20}

def test__get_value_from_path__dict__nested_dict():
    path = {'nested': {'key1': "['a'][0]", 'key2': "['b']['c']"}}
    data = {
        'a': [10, 20],
        'b': {'c': 30},
    }
    result = get_value_from_path(path, data)
    assert result == {'nested': {'key1': 10, 'key2': 30}}

def test__get_value_from_path__dict__non_existent_key_in_dict():
    path = {'key1': "['a']['non_existent_key']"}
    data = {'a': {'b': 10}}
    with pytest.raises(KeyError):
        get_value_from_path(path, data)

def test__get_value_from_path__list__simple_list():
    path = ["['a']['b']", "['x']['y']"]
    data = {
        'a': {'b': 10},
        'x': {'y': 20},
    }
    result = get_value_from_path(path, data)
    assert result == [10, 20]

def test__get_value_from_path__list__nested_list():
    path = ["['a'][0]", "['b']['c']"]
    data = {
        'a': [10, 20],
        'b': {'c': 30},
    }
    result = get_value_from_path(path, data)
    assert result == [10, 30]

def test__get_value_from_path__list__mixed():
    path = ["['a'][0]", "['b'].name"]
    class CustomObject:
        def __init__(self, name):  # noqa: ANN001
            self.name = name

    data = {
        'a': [10, 20],
        'b': CustomObject("TestName"),
    }
    result = get_value_from_path(path, data)
    assert result == [10, "TestName"]

def test__get_value_from_path__dict__lambda_in_dict():
    path = {'value': "lambda x: x['a'][0] + x['b']['c']"}
    data = {
        'a': [10, 20],
        'b': {'c': 30},
    }
    result = get_value_from_path(path, data)
    assert result == {'value': 40}

def test__get_value_from_path__list__with_lambda():
    path = ["lambda x: x['a']['b'] * 2", "lambda x: x['x']['y'] + 5"]
    data = {
        'a': {'b': 10},
        'x': {'y': 20},
    }
    result = get_value_from_path(path, data)
    assert result == [20, 25]

def test__get_value_from_path__list__list_indexing_in_list():
    path = ["[0][1]", "[1][0]"]
    data = [[1, 2, 3], [4, 5, 6]]
    result = get_value_from_path(path, data)
    assert result == [2, 4]

def test__get_value_from_path__dict__empty_dict_path():
    path = {}
    data = {'a': 10}
    result = get_value_from_path(path, data)
    assert result == {}

def test__get_value_from_path__list__empty_list_path():
    path = []
    data = {'a': 10}
    result = get_value_from_path(path, data)
    assert result == []

def test__get_value_from_path__dict__mixed_keys_and_lambdas():
    path = {
        'key1': "['a'][1]",
        'key2': "['b']['c']",
        'lambda_key': "lambda x: x['a'][0] + x['b']['c']",
    }
    data = {
        'a': [10, 20],
        'b': {'c': 30},
    }
    result = get_value_from_path(path, data)
    assert result == {'key1': 20, 'key2': 30, 'lambda_key': 40}

def test__get_value_from_path__dict__tuple_access_in_dict():
    path = {
        'key1': "[0]",
        'key2': "[2]",
    }
    data = (100, 200, 300)
    result = get_value_from_path(path, data)
    assert result == {'key1': 100, 'key2': 300}

def test__get_value_from_path__list__dictionary_keys_in_list():
    path = ["[0]['a']", "[1]['b']"]
    data = [{'a': 1}, {'b': 2}]
    result = get_value_from_path(path, data)
    assert result == [1, 2]

def test__get_value_from_path__list__tuple_keys_in_list():
    path = ["[1]", "[2]"]
    data = (10, 20, 30)
    result = get_value_from_path(path, data)
    assert result == [20, 30]

def test__get_value_from_path__list__unsupported_path_type():
    path = 12
    data = {'a': 10}
    with pytest.raises(ValueError):  # noqa: PT011
        get_value_from_path(path, data)

class SampleEnum(Enum):
    """Enum for testing Registry."""

    TYPE1 = auto()
    TYPE2 = auto()

class SampleClass1:
    """Sample class for testing Registry."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class SampleClass2:
    """Sample class for testing Registry."""

    def __init__(self, name: str):
        self.name = name

def test__Registry__register_and_get_with_string():
    """Test registering and retrieving a class using a string type name."""
    registry = Registry()
    registry.register("sample_class_1", SampleClass1)
    assert "sample_class_1" in registry
    assert "SAMPLE_CLASS_1" in registry
    assert registry.get("sample_class_1") == SampleClass1
    assert registry.get("SAMPLE_CLASS_1") == SampleClass1

def test__Registry__register_and_get_with_enum():
    """Test registering and retrieving a class using an Enum type name."""
    registry = Registry()
    registry.register(SampleEnum.TYPE1, SampleClass1)
    assert SampleEnum.TYPE1 in registry
    assert "TYPE1" in registry
    assert "type1" in registry
    assert registry.get("TYPE1") == SampleClass1
    assert registry.get(SampleEnum.TYPE1) == SampleClass1

def test__Registry__duplicate_registration_raises_error():
    """Test that registering a type name twice raises an assertion error."""
    registry = Registry()
    registry.register("duplicate", SampleClass1)
    with pytest.raises(AssertionError, match="Type 'DUPLICATE' already registered."):
        registry.register("duplicate", SampleClass2)

def test__Registry__create_instance_with_string():
    registry = Registry()
    """Test creating an instance using a string type name."""
    registry.register("sample_class_1", SampleClass1)
    instance = registry.create_instance("sample_class_1", x=10, y=20)
    assert isinstance(instance, SampleClass1)
    assert instance.x == 10
    assert instance.y == 20

    registry.register("sample_class_2", SampleClass2)
    instance = registry.create_instance("sample_class_2", name="test")
    assert isinstance(instance, SampleClass2)
    assert instance.name == "test"

def test__Registry__create_instance_with_enum():
    """Test creating an instance using an Enum type name."""
    registry = Registry()
    registry.register(SampleEnum.TYPE1, SampleClass1)
    instance = registry.create_instance(SampleEnum.TYPE1, x=30, y=40)
    assert isinstance(instance, SampleClass1)
    assert instance.x == 30
    assert instance.y == 40

    instance = registry.create_instance("TYPE1", x=50, y=60)
    assert isinstance(instance, SampleClass1)
    assert instance.x == 50
    assert instance.y == 60

    registry.register(SampleEnum.TYPE2, SampleClass2)
    instance = registry.create_instance(SampleEnum.TYPE2, name="test")
    assert isinstance(instance, SampleClass2)
    assert instance.name == "test"

def test__Registry__create_instance_unregistered_type_raises_error():
    registry = Registry()
    """Test that trying to create an instance of an unregistered type raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown type `sample_class_2`"):
        registry.create_instance("sample_class_2", name="test")

def test__Registry__clean_type_name_with_string():
    """Test the _clean_type_name method with a string input."""
    assert Registry._clean_type_name("sample") == "SAMPLE"

def test__Registry__clean_type_name_with_enum():
    """Test the _clean_type_name method with an Enum input."""
    assert Registry._clean_type_name(SampleEnum.TYPE1) == "TYPE1"

def test__precision_score():
    assert precision_score(true_pos=10, false_pos=0) == 1.0
    assert precision_score2(true_pos=10, pred_pos=10) == precision_score(true_pos=10, false_pos=0)
    assert precision_score(true_pos=0, false_pos=10) == 0.0
    assert precision_score2(true_pos=0, pred_pos=10) == precision_score(true_pos=0, false_pos=10)
    assert precision_score(true_pos=10,false_pos=5) == pytest.approx(0.6666667, 0.0000001)
    assert precision_score2(true_pos=10, pred_pos=15) == precision_score(true_pos=10, false_pos=5)
    assert precision_score(true_pos=0, false_pos=0) == 0.0
    assert precision_score2(true_pos=0, pred_pos=0) == precision_score(true_pos=0, false_pos=0)
    assert precision_score2(true_pos=10, pred_pos=0) == 0.0

def test__recall_score():
    assert recall_score(true_pos=10,false_neg=0) == 1.0
    assert recall_score2(true_pos=10, actual_pos=10) == recall_score(true_pos=10, false_neg=0)
    assert recall_score(true_pos=0, false_neg=10) == 0.0
    assert recall_score2(true_pos=0, actual_pos=10) == recall_score(true_pos=0, false_neg=10)
    assert recall_score(true_pos=10,false_neg=5) == pytest.approx(0.6666667, 0.0000001)
    assert recall_score2(true_pos=10, actual_pos=15) == recall_score(true_pos=10, false_neg=5)
    assert recall_score(true_pos=0, false_neg=0) == 0.0
    assert recall_score2(true_pos=0, actual_pos=0) == recall_score(true_pos=0, false_neg=0)
    assert recall_score2(true_pos=10, actual_pos=0) == 0.0

def test__f_score():
    assert f_score(precision=0.0, recall=0.0, beta=1) == 0.0
    assert f_score(precision=0.0, recall=1.0, beta=0) == 0.0
    assert f_score(precision=1.0, recall=0.1, beta=0) == 1.0
    assert f_score(precision=1.0, recall=1.0, beta=1) == 1.0
    assert f_score(precision=1.0, recall=1.0, beta=1) == f1_score(precision=1.0, recall=1.0)
    assert f_score(precision=1.0, recall=0.0, beta=1) == 0.0
    assert f_score(precision=1.0, recall=0.0, beta=1) == f1_score(precision=1.0, recall=0.0)
    assert f_score(precision=0.0, recall=1.0, beta=1) == 0.0
    assert f_score(precision=0.0, recall=1.0, beta=1) == f1_score(precision=0.0, recall=1.0)
    assert f_score(precision=0.5, recall=0.5, beta=1) == 0.5
    assert f_score(precision=0.5, recall=0.5, beta=1) == f1_score(precision=0.5, recall=0.5)
    assert f_score(precision=0.5, recall=0.5, beta=2) == 0.5

    # weights recall higher than precision
    true_pos = 25
    # true_neg = 50
    false_pos = 10
    false_neg = 15
    precision = precision_score(true_pos=true_pos, false_pos=false_pos)
    assert precision == pytest.approx(0.7143, 0.0001)
    assert precision == precision_score2(true_pos=true_pos, pred_pos=true_pos + false_pos)

    recall = recall_score(true_pos=true_pos, false_neg=false_neg)
    assert recall == pytest.approx(0.625, 0.0001)
    assert recall == recall_score2(true_pos=true_pos, actual_pos=true_pos + false_neg)

    f1 = f_score(precision=precision, recall=recall, beta=1)
    assert f1 == pytest.approx(0.6667, 0.0001)
    assert f1 == f1_score(precision=precision, recall=recall)

    # weighs recall higher than precision
    f2 = f_score(precision=precision, recall=recall, beta=2)
    assert f2 == pytest.approx(0.6410, 0.0001)
    #  weighs precision higher than recall
    f05 = f_score(precision=precision, recall=recall, beta=0.5)
    assert f05 == pytest.approx(0.6944, 0.0001)

def test__precision_score_tokens():
    assert precision_score_tokens(expected_tokens=['a', 'b'], actual_tokens=['a', 'b']) == 1.0
    assert precision_score_tokens(expected_tokens=['a', 'b', 'c'], actual_tokens=['a', 'b']) == 1.0
    assert precision_score_tokens(expected_tokens=['c', 'd'], actual_tokens=['a', 'b']) == 0.0
    score = precision_score_tokens(expected_tokens=['a', 'b'], actual_tokens=['a', 'b', 'c'])
    assert score == pytest.approx(0.6666667, 0.0000001)
    assert score == precision_score_tokens(expected_tokens=['a', 'b', 'a'], actual_tokens=['a', 'b', 'c', 'c'])  # noqa

def test__recall_score_tokens():
    assert recall_score_tokens(expected_tokens=['a', 'b'], actual_tokens=['a', 'b']) == 1.0
    assert recall_score_tokens(expected_tokens=['a', 'a', 'b'], actual_tokens=['a', 'b', 'b']) == 1.0  # noqa
    assert recall_score_tokens(expected_tokens=['a', 'b'], actual_tokens=['a', 'b', 'c', 'd']) == 1.0  # noqa
    assert recall_score_tokens(expected_tokens=['c', 'd'], actual_tokens=['a', 'b']) == 0.0
    score = recall_score_tokens(expected_tokens=['a', 'b', 'c'], actual_tokens=['a', 'b'])
    assert score == pytest.approx(0.6666667, 0.0000001)
    assert score == recall_score_tokens(expected_tokens=['a', 'b', 'c', 'c', 'b'], actual_tokens=['a', 'b', 'b'])  # noqa

def test__f1_score_tokens():
    assert f1_score_tokens(expected_tokens=['a', 'b'], actual_tokens=['a', 'b']) == 1.0
    assert f1_score_tokens(expected_tokens=['a', 'b', 'c'], actual_tokens=['f', 'd']) == 0.0
    precision = precision_score_tokens(expected_tokens=['a', 'b', 'c'], actual_tokens=['a', 'b'])
    assert precision == 1
    recall = recall_score_tokens(expected_tokens=['a', 'b', 'c'], actual_tokens=['a', 'b'])
    assert recall == pytest.approx(0.6666667, 0.0000001)
    expected_f1 = f_score(precision=precision, recall=recall, beta=1)
    assert expected_f1 == f1_score_tokens(expected_tokens=['a', 'b', 'c'], actual_tokens=['a', 'b'])  # noqa: E501

def test__simple_tokenizer__empty_string():
    assert simple_tokenizer('') == []
    assert simple_tokenizer(' ') == []
    assert simple_tokenizer(None) == []

def test__simple_tokenizer__only_stop_words():
    assert simple_tokenizer('the and in') == []

def test__simple_tokenizer__with_punctuation():
    assert simple_tokenizer('Hello, world!') == ['hello', 'world']

def test__simple_tokenizer__with_apostrophe_stop_word():
    assert simple_tokenizer("Wouldn't and shouldn't SHOULDN'T show up") == ['show']

def test__simple_tokenizer__mixed_content():
    assert simple_tokenizer('This is a test, with punctuation!') == ['test', 'punctuation']

def test__simple_tokenizer__multiple_spaces():
    assert simple_tokenizer('This    is  a   test') == ['test']

def test__simple_tokenizer__no_stop_words():
    assert simple_tokenizer('Python programming language') == ['python', 'programming', 'language']

def test__simple_tokenizer__uppercase_words():
    assert simple_tokenizer('HELLO WORLD') == ['hello', 'world']

def test__simple_tokenizer__with_numbers():
    assert simple_tokenizer('Python 3.8 is awesome') == ['python', '38', 'awesome']

def test__simple_tokenizer__mixed_case_stop_words():
    assert simple_tokenizer('The quick brown fox') == ['quick', 'brown', 'fox']

def test__simple_tokenizer__hyphenated_words():
    assert simple_tokenizer('state-of-the-art technology') == ['stateoftheart', 'technology']

def test__simple_tokenizer__non_ascii_characters():
    assert simple_tokenizer('Café müßig') == ['café', 'müßig']
