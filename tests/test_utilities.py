"""Tests the utilities.py file."""
from time import sleep
import re
import pytest
from textwrap import dedent
from llm_eval.internal_utilities import (
    create_function,
    create_hash,
    execute_code_blocks,
    extract_code_blocks,
    extract_valid_parameters,
    extract_variables,
    generate_dict_combinations,
    has_method,
    has_property,
    retry_handler,
    Timer,
)


def test_timer_seconds():  # noqa
    with Timer() as timer:
        sleep(1.1)

    assert timer.interval
    assert re.match(pattern=r'1\.\d+ seconds', string=timer.formatted())
    assert str(timer) == timer.formatted()

    with pytest.raises(ValueError):  # noqa
        timer.formatted(units='days')

def test_create_hash():  # noqa
    value_a = create_hash('Test value 1')
    assert value_a
    value_b = create_hash('Test value 2')
    assert value_b
    assert value_a != value_b
    value_c = create_hash('Test value 1')
    assert value_c == value_a

def test_retry_handler():  # noqa
    r = retry_handler()
    actual_value = r(
        lambda x, y: (x, y),
        x='A',
        y='B',
    )
    assert actual_value == ('A', 'B')

def test_has_method_has_property():  # noqa
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

def test__extract_code_blocks__no_code_blocks():  # noqa
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

def test__extract_code_blocks__conversation_sum(conversation_sum):  # noqa
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

def test__extract_code_blocks__conversation_mask_emails(conversation_mask_email):  # noqa
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

def test__extract_code_blocks__llama_response():  # noqa
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

def test__execute_code_blocks__without_env_namespace(conversation_sum):  # noqa
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

def test__execute_code_blocks__with_env_namespace(conversation_sum):  # noqa
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

def test__execute_code_blocks__with_env_namespace__test_dependencies():  # noqa
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

def test__execute_code_blocks__with_env_namespace__test_import_dependencies():  # noqa
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

def test__execute_code_blocks__with_env_namespace__inject_objects_into_namespace():  # noqa
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

def test__execute_code_blocks__timeout():  # noqa
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

def test__generate_dict_combinations():  # noqa
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

def test__extract_variables():  # noqa
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

def test__extract_valid_parameters():  # noqa
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

def test__create_function_from_string():  # noqa
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
