"""Contains tests for eval Check objects."""
import re
from textwrap import dedent
import pandas as pd
from pydantic import BaseModel, ValidationError
import pytest
from sik_llm_eval.checks import (
    Check,
    CheckResult,
    CheckResultsType,
    CheckType,
    ContainsCheck,
    F1Score,
    LLMCheck,
    LambdaCheck,
    MatchCheck,
    MaxF1Score,
    PassFailResult,
    PrecisionScore,
    PythonCodeBlocksPresent,
    PythonCodeBlockTests,
    RecallScore,
    RegexCheck,
    ResponseModel,
    ScoreResult,
    ToolCallsCheck,
)
from sik_llm_eval.utilities import f1_score, precision_score_tokens, recall_score_tokens
from tests.conftest import OPENAI_DEFAULT_MODEL


def test__CheckType__mixin_behavior():
    assert CheckType.MATCH == 'MATCH'
    assert CheckType.MATCH == 'match'
    assert CheckType.MATCH != 1
    assert CheckType.MATCH != 0
    assert CheckType.to_enum('CONTAINS') == CheckType.CONTAINS
    assert CheckType.to_enum('contains') == CheckType.CONTAINS
    assert CheckType.to_enum(CheckType.MATCH) == CheckType.MATCH
    with pytest.raises(ValueError):  # noqa: PT011
        CheckType.to_enum('foo')

def test__register_check__success__str__ensure_creation():
    """Test successful registration of a check."""
    @Check.register('FakeCheck')
    class FakeCheck(Check):
        """Mock test for testing."""

        def __call__(self, value: str) -> bool:
            return PassFailResult(
                value=value is not None,
                metadata=self.metadata,
            )

    try:
        assert 'FAKECHECK' in Check.registry
        assert 'fakecheck' in Check.registry

        check = Check.from_dict({'check_type': 'fakecheck'})
        assert isinstance(check, FakeCheck)
        assert check.check_type == 'FAKECHECK'
        assert Check.from_dict(check.to_dict()) == check
        result = check.run_on_model(ResponseModel(response='foo'))
        assert result.success
        assert result.value
        assert result.metadata == {}

        check = Check.from_dict({'check_type': 'FAKECHECK'})
        assert isinstance(check, FakeCheck)
        assert check.check_type == 'FAKECHECK'
        assert Check.from_dict(check.to_dict()) == check
        result = check.run_on_model(ResponseModel(response='foo'))
        assert result.success
        assert result.value
        assert result.metadata == {}

        check = Check.from_dict({'check_type': 'FAKECHECK', 'metadata': {'foo': 'bar'}})
        assert isinstance(check, FakeCheck)
        assert check.check_type == 'FAKECHECK'
        assert check.metadata == {'foo': 'bar'}
        assert Check.from_dict(check.to_dict()) == check
        result = check.run_on_model(ResponseModel(response='foo'))
        assert result.success
        assert result.value
        assert result.metadata == {'foo': 'bar'}

        check = Check.from_dict({'check_type': 'FAKECHECK', 'metadata': {}})
        assert isinstance(check, FakeCheck)
        assert check.check_type == 'FAKECHECK'
        assert check.metadata == {}
        assert Check.from_dict(check.to_dict()) == check
        result = check.run_on_model(ResponseModel(response='foo'))
        assert result.success
        assert result.value
        assert result.metadata == {}

        # We should not be able to register a check with the same type
        with pytest.raises(AssertionError):
            @Check.register('fakecheck')
            class TestCheck2(Check):
                def __call__(self, data: ResponseModel) -> CheckResult:
                    return PassFailResult(value=True, metadata={'response': data.response})
        # We should not be able to register a check that isn't a Check object
        with pytest.raises(AssertionError):
            @Check.register('test2')
            class TestCheck3:
                def __call__(self, data: ResponseModel) -> CheckResult:
                    return PassFailResult(value=True, metadata={'response': data.response})
    finally:
        Check.registry._registry.pop('FAKECHECK')

def test__register_check__call():
    """Test successful registration of a check."""
    @Check.register('FakeCheck')
    class FakeCheck(Check):
        """Mock test for testing."""

        def __call__(self, value: str) -> bool:
            return PassFailResult(
                value=value == 'foo',
                metadata=self.metadata,
            )

    try:
        assert 'FAKECHECK' in Check.registry
        assert 'fakecheck' in Check.registry

        check = Check.from_dict({'check_type': 'fakecheck'})
        assert isinstance(check, FakeCheck)
        assert check.check_type == 'FAKECHECK'
        assert Check.from_dict(check.to_dict()) == check
        result = check('foo')
        assert result.success
        assert result.value
        assert result.metadata == {}
    finally:
        Check.registry._registry.pop('FAKECHECK')

def test__register_check__success__ensure_creation__with_required_params():
    """Test successful registration of a check."""

    @Check.register('FakeParamCheck')
    class FakeParamCheck(Check):
        """Mock test for testing."""

        required_field: str

        def __call__(self, value: str) -> CheckResult:
            return PassFailResult(
                check_type=CheckType.PASS_FAIL,
                passed=value is not None,
                metadata=self.metadata,
            )

    assert 'FAKEPARAMCHECK' in Check.registry
    assert 'fakeparamcheck' in Check.registry

    # if we don't pass the required param, we should get an error
    with pytest.raises(ValidationError):
        _ = Check.from_dict({'check_type': 'FakeParamCheck'})
    with pytest.raises(ValidationError):
        _ = Check.from_dict({'check_type': 'FakeParamCheck', 'metadata': {'foo': 'bar'}})

    check = Check.from_dict({'check_type': 'FakeParamCheck', 'required_field': 'foo'})
    assert isinstance(check, FakeParamCheck)
    assert check.metadata == {}
    assert check.required_field == 'foo'

    check = Check.from_dict({
        'check_type': 'FakeParamCheck',
        'required_field': 'foo',
        'metadata': {'foo': 'bar'},
    })
    assert isinstance(check, FakeParamCheck)
    assert check.metadata == {'foo': 'bar'}
    assert check.required_field == 'foo'

def test__register_check__duplicate__str__():
    """Test registering a check with a duplicate name raises an error."""
    with pytest.raises(AssertionError):
        @Check.register(CheckType.MATCH.name)
        class FakeCheck(Check):
            """Mock test for testing."""

            def __call__(self, data: ResponseModel) -> CheckResult:
                return data.response


    with pytest.raises(AssertionError):
        @Check.register(CheckType.MATCH.name.lower())
        class FakeCheck(Check):
            """Mock test for testing."""

            def __call__(self, data: ResponseModel) -> CheckResult:
                return data.response

def test__register_check__duplicate__CheckType():
    """Test registering a check with a duplicate name raises an error."""
    with pytest.raises(AssertionError):
        @Check.register(CheckType.MATCH)
        class FakeCheck(Check):
            """Mock test for testing."""

            def __call__(self, data: ResponseModel) -> CheckResult:
                return data.response

def test__CheckType():
    assert CheckType.MATCH.name == 'MATCH'
    assert CheckType.to_enum('MATCH') == CheckType.MATCH
    assert CheckType.to_enum('match') == CheckType.MATCH
    assert CheckType.MATCH == 'MATCH'
    assert CheckType.MATCH == 'match'
    with pytest.raises(ValueError):  # noqa: PT011
        CheckType.to_enum('foo')

def test__CheckResult__registration():
    assert CheckResultsType.PASS_FAIL in CheckResult.registry  # enum
    assert CheckResultsType.PASS_FAIL.name in CheckResult.registry  # string upper
    assert CheckResultsType.PASS_FAIL.name.lower() in CheckResult.registry  # string lower
    assert CheckResultsType.SCORE in CheckResult.registry  # enum
    assert CheckResultsType.SCORE.name in CheckResult.registry  # string upper
    assert CheckResultsType.SCORE.name.lower() in CheckResult.registry  # string lower

def test__PassFailResult():
    assert not PassFailResult(value=False).success
    assert PassFailResult(value=False).metadata == {}
    assert PassFailResult(value=True).success
    assert PassFailResult(value=True).metadata == {}
    assert not PassFailResult(value=False, metadata={'foo': 'bar'}).success
    assert PassFailResult(value=False, metadata={'foo': 'bar'}).metadata == {'foo': 'bar'}
    assert PassFailResult(value=True, metadata={'foo': 'bar'}).success
    assert PassFailResult(value=True, metadata={'foo': 'bar'}).metadata == {'foo': 'bar'}
    assert str(PassFailResult(value=False))
    assert str(PassFailResult(value=True))
    assert str(PassFailResult(value=False, metadata={'foo': 'bar'}))

def test__PassFailResult__serialize():
    result = PassFailResult(value=True)
    result_dict = result.to_dict()
    # ensure metadata isn't included in the dict since it's empty
    assert result_dict == {
        'value': True,
        'success': True,
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    recreated = CheckResult.from_dict(result_dict)
    assert isinstance(recreated, PassFailResult)
    assert recreated == result

    result = PassFailResult(value=False, metadata={'foo': 'bar'})
    result_dict = result.to_dict()
    assert result_dict == {
        'value': False,
        'success': False,
        'metadata': {'foo': 'bar'},
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    recreated = CheckResult.from_dict(result_dict)
    assert isinstance(recreated, PassFailResult)
    assert recreated == result

    result = PassFailResult(value=True, metadata={'bar': 'foo'})
    result_dict = result.to_dict()
    assert result_dict == {
        'value': True,
        'success': True,
        'metadata': {'bar': 'foo'},
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    recreated = CheckResult.from_dict(result_dict)
    assert isinstance(recreated, PassFailResult)
    assert recreated == result

def test__ScoreResult():
    assert ScoreResult(value=0.5).value == 0.5
    assert str(ScoreResult(value=0.5))
    assert ScoreResult(value=0.5).success is None
    assert ScoreResult(value=-1).success is None
    assert ScoreResult(value=-0).success is None
    assert ScoreResult(value=-1).success is None
    assert ScoreResult(value=0.5).metadata == {}
    assert ScoreResult(value=0.5, metadata={'foo': 'bar'}).value == 0.5
    assert str(ScoreResult(value=0.5, metadata={'foo': 'bar'}))

    result = ScoreResult(value=0.5, success_threshold=0.49)
    assert result.success
    assert result.value == 0.5
    assert result.metadata == {}
    assert str(result)

    result = ScoreResult(value=0.5, success_threshold=0.5)
    assert result.success
    assert result.value == 0.5
    assert result.metadata == {}
    assert str(result)

    result = ScoreResult(value=0.5, success_threshold=0.51)
    assert not result.success
    assert result.value == 0.5
    assert result.metadata == {}
    assert str(result)

    result = ScoreResult(value=0, success_threshold=0, metadata={'foo': 'bar'})
    assert result.success
    assert result.value == 0
    assert result.metadata == {'foo': 'bar'}
    assert str(result)

    result = ScoreResult(value=0, success_threshold=0.0001, metadata={'foo': 'bar'})
    assert not result.success
    assert result.value == 0
    assert result.metadata == {'foo': 'bar'}
    assert str(result)

def test__ScoreResult__serialize():
    result = ScoreResult(value=0.5)
    result_dict = result.to_dict()
    # ensure metadata and success_threshold isn't included in the dict since it's empty
    assert result_dict == {
        'value': 0.5,
        'success': None,
        'result_type': CheckResultsType.SCORE.name,
    }
    assert ScoreResult(**result_dict) == result
    result = ScoreResult(value=0.5, metadata={'foo': 'bar'})
    result_dict = result.to_dict()
    assert result_dict == {
        'value': 0.5,
        'success': None,
        'metadata': {'foo': 'bar'},
        'result_type': CheckResultsType.SCORE.name,
    }
    assert ScoreResult(**result_dict) == result
    recreated = CheckResult.from_dict(result_dict)
    assert isinstance(recreated, ScoreResult)
    assert recreated == result

    result = ScoreResult(value=0.5, success_threshold=0.5, metadata={'foo': 'bar'})
    result_dict = result.to_dict()
    assert result_dict == {
        'value': 0.5,
        'success_threshold': 0.5,
        'success': True,
        'metadata': {'foo': 'bar'},
        'result_type': CheckResultsType.SCORE.name,
    }
    assert ScoreResult(**result_dict) == result
    recreated = CheckResult.from_dict(result_dict)
    assert isinstance(recreated, ScoreResult)
    assert recreated == result

    result = ScoreResult(value=0.5, success_threshold=0.51, metadata={'bar': 'foo'})
    result_dict = result.to_dict()
    assert result_dict == {
        'value': 0.5,
        'success_threshold': 0.51,
        'success': False,
        'metadata': {'bar': 'foo'},
        'result_type': CheckResultsType.SCORE.name,
    }
    assert ScoreResult(**result_dict) == result
    recreated = CheckResult.from_dict(result_dict)
    assert isinstance(recreated, ScoreResult)
    assert recreated == result

def test__ResponseModel__extract_values():
    expected_response = {'foo': {'bar': 'baz'}}
    expected_prompt = 'the prompt'
    expected_metadata = {'foo': 'bar'}
    response_data = ResponseModel(
        input=expected_prompt,
        response=expected_response,
        metadata=expected_metadata,
    )

    value = response_data.extract_values(path=None)
    assert value == response_data

    value = response_data.extract_values(path='')
    assert value == response_data

    value = response_data.extract_values(path='input')
    assert value == expected_prompt

    value = response_data.extract_values(path='response')
    assert value == expected_response

    value = response_data.extract_values(path='response["foo"]')
    assert value == expected_response['foo']

    value = response_data.extract_values(path='response["foo"]["bar"]')
    assert value == expected_response['foo']['bar']

    value = response_data.extract_values(path='metadata')
    assert value == expected_metadata

    # test nested objects
    class MockObject:
        def __init__(self, value):  # noqa: ANN001
            self.value = value

    expected_response = MockObject(value={'foo': MockObject(value={'bar': 'baz'})})
    response_data = ResponseModel(response=expected_response)

    value = response_data.extract_values(path=None)
    assert value == response_data

    value = response_data.extract_values(path='')
    assert value == response_data

    value = response_data.extract_values(path='response.value')
    assert value == expected_response.value

    value = response_data.extract_values(path='response.value["foo"]')
    assert value == expected_response.value['foo']

    value = response_data.extract_values(path='response.value["foo"].value')
    assert value == expected_response.value['foo'].value

def test__Check__data_path():
    """The default data_path should use the response in the check."""
    @Check.register('FAKECHECK__VALUE_EXTRACT')
    class FakeCheck(Check):
        """Mock test for testing."""

        def __call__(self, value: str) -> bool:
            return PassFailResult(
                value=value is not None,
                metadata={'value': value},
            )
    try:
        check = FakeCheck()
        # should be successful default is to look in response and which we are using
        # and 'foo' is not None
        result = check.run_on_model(ResponseModel(response='foo'))
        assert result.value
        assert result.success
        assert result.metadata == {'value': 'foo'}
        # should fail because default is still looking in response but we are using input
        # so response will default to None and check will fail
        # but the value extraction did not fail so there should not be a data_path_error
        # also we are using the default data_path so metadata should not be populated with
        # data_path and value_extracted
        result = check.run_on_model(ResponseModel(input='foo'))
        assert not result.value
        assert not result.success
        assert result.metadata == {'value': None}

        check = FakeCheck(data_path='input')
        # should fail because we are extracting input but input is None
        result = check.run_on_model(ResponseModel(response='foo'))
        assert not result.value
        assert not result.success
        assert result.metadata['value'] is None
        # now, because we are using input, which is not the default data_path,
        # metadata will be populated with the data_path and value_extracted
        assert result.metadata['data_path'] == 'input'
        assert result.metadata['value_extracted'] is None
        # should be successful because we are extracting input and 'foo' is not None
        result = check.run_on_model(ResponseModel(input='foo'))
        assert result.value
        assert result.success
        assert result.metadata['value'] == 'foo'
        assert result.metadata['data_path'] == 'input'
        assert result.metadata['value_extracted'] == 'foo'
    finally:
        Check.registry._registry.pop('FAKECHECK__VALUE_EXTRACT')

def test__Check__data_path__override():
    """The default data_path should use the response in the check."""
    @Check.register('FAKECHECK__VALUE_EXTRACT')
    class FakeCheck(Check):
        """Mock test for testing."""

        @property
        def default_data_path(self) -> str:
            return None  # should return the entire ResponseData object

        def __call__(self, data: ResponseModel) -> bool:
            assert isinstance(data, ResponseModel)
            return PassFailResult(
                value=data.response is not None and data.input is not None,
                metadata={
                    'input': data.input,
                    'response': data.response,
                },
            )
    try:
        check = FakeCheck(metadata={'foo': 'bar'})
        assert check.to_dict() == {
            'check_type': 'FAKECHECK__VALUE_EXTRACT',
            'metadata': {'foo': 'bar'},
        }
        assert Check.from_dict(check.to_dict()) == check

        check = FakeCheck()
        result = check.run_on_model(ResponseModel(input='foo', response='bar'))
        assert result.value
        assert result.success
        # since we have defined default_data_path as empty string and using the default
        # then metadata should not be populated with data_path and value_extracted
        assert 'data_path' not in result.metadata
        assert 'value_extracted' not in result.metadata
        assert 'data_path_error' not in result.metadata
        assert result.metadata == {'input': 'foo', 'response': 'bar'}

        result = check.run_on_model(ResponseModel(input='foo'))
        assert not result.value
        assert not result.success
        assert 'data_path' not in result.metadata
        assert 'value_extracted' not in result.metadata
        assert 'data_path_error' not in result.metadata
        assert result.metadata == {'input': 'foo', 'response': None}
    finally:
        Check.registry._registry.pop('FAKECHECK__VALUE_EXTRACT')

def test__Check__data_path__error_extracting_value():
    """
    The expected behavior when we get an error extracting the variable is that the check should
    fail (but not throw an exception) and the metadata should be populated with the error message.
    """
    @Check.register('FAKECHECK__VALUE_EXTRACT')
    class FakeCheck(Check):
        """Mock test for testing."""

        def __call__(self, value: str) -> bool:
            return PassFailResult(
                value=value is not None,
                metadata=self.metadata,
            )
    try:
        check = FakeCheck(data_path='response["foo"]')
        # None cannot be indexed so this should fail
        result = check.run_on_model(ResponseModel())
        assert not result.value
        assert not result.success
        assert result.metadata['data_path'] == 'response["foo"]'
        assert result.metadata['value_extracted'] is None
        assert 'data_path_error' in result.metadata
        assert 'NoneType' in result.metadata['data_path_error']

        check = FakeCheck(data_path='response["foo"]')
        # foo does not exist in response so this should fail
        result = check.run_on_model(ResponseModel(response='foo'))
        assert not result.value
        assert not result.success
        assert result.metadata['data_path'] == 'response["foo"]'
        assert result.metadata['value_extracted'] is None
        assert 'data_path_error' in result.metadata
    finally:
        Check.registry._registry.pop('FAKECHECK__VALUE_EXTRACT')

def test__Check__data_path__dictionary():
    """
    The data_path should be able to extract values from a dictionary and pass them to the
    check as keyword arguments.
    """
    registration_value = 'FAKECHECK__DATA_PATH_DICT'
    @Check.register(registration_value)
    class FakeCheck(Check):
        """Mock test for testing."""

        expected_input_1: str
        expected_input_2: str

        @property
        def default_data_path(self) -> str:
            return {'input_1': 'response["foobar"]', 'input_2': 'ideal_response'}

        def __call__(self, input_1: str, input_2: str) -> PassFailResult:
            return PassFailResult(
                value=input_1 == self.expected_input_1 and input_2 == self.expected_input_2,
                metadata={'1': input_1, '2': input_2},
            )
    try:
        check = FakeCheck(expected_input_1='foo', expected_input_2='bar')
        result = check.run_on_model(ResponseModel(response={'foobar': 'foo'}, ideal_response='bar'))  # noqa: E501
        assert result.value
        assert result.metadata['1'] == 'foo'
        assert result.metadata['2'] == 'bar'
        # since we are using the default data_path, metadata will not be populated with
        # data_path and value_extracted
        assert 'data_path' not in result.metadata
        assert 'value_extracted' not in result.metadata
        assert 'data_path_error' not in result.metadata

        result = check(input_1='foo', input_2='bar')
        assert result.value
        assert result.metadata['1'] == 'foo'
        assert result.metadata['2'] == 'bar'
        # since we are using the default data_path, metadata will not be populated with
        # data_path and value_extracted
        assert 'data_path' not in result.metadata
        assert 'value_extracted' not in result.metadata
        assert 'data_path_error' not in result.metadata


    finally:
        Check.registry._registry.pop(registration_value)

def test__Check__data_path__dictionary__override_default():
    """
    The data_path should be able to extract values from a dictionary and pass them to the
    check as keyword arguments.
    """
    registration_value = 'FAKECHECK__DATA_PATH_DICT'
    @Check.register(registration_value)
    class FakeCheck(Check):
        """Mock test for testing."""

        expected_input_1: str
        expected_input_2: str

        @property
        def default_data_path(self) -> str:
            return None

        def __call__(self, input_1: str, input_2: str) -> PassFailResult:
            return PassFailResult(
                value=input_1 == self.expected_input_1 and input_2 == self.expected_input_2,
                metadata={'1': input_1, '2': input_2},
            )
    try:
        data_path = {'input_1': 'response["foobar"]', 'input_2': 'ideal_response'}
        check = FakeCheck(
            expected_input_1='foo',
            expected_input_2='bar',
            data_path=data_path,
        )
        result = check.run_on_model(ResponseModel(response={'foobar': 'foo'}, ideal_response='bar'))  # noqa: E501
        assert result.value
        assert result.metadata['1'] == 'foo'
        assert result.metadata['2'] == 'bar'
        # since we are overriding the default data_path, metadata will be populated with
        # data_path and value_extracted
        assert 'data_path' in result.metadata
        assert result.metadata['data_path'] == data_path
        assert 'value_extracted' in result.metadata
        assert result.metadata['value_extracted'] == {'input_1': 'foo', 'input_2': 'bar'}
        assert 'data_path_error' not in result.metadata
    finally:
        Check.registry._registry.pop(registration_value)

def test__Check__data_path__dictionary__test_all_fields():
    """
    The data_path should be able to extract values from a dictionary and pass them to the
    check as keyword arguments.
    """
    registration_value = 'FAKECHECK__DATA_PATH_DICT'
    @Check.register(registration_value)
    class FakeCheck(Check):
        """Mock test for testing."""

        expected_my_response: str
        expected_the_ideal_response: str
        expected_the_input: str
        expected_the_metadata: str

        @property
        def default_data_path(self) -> str:
            return {
                'my_response': 'response["foobar"]',
                'the_ideal_response': 'ideal_response[1]',
                'the_input': 'input',
                'the_metadata': 'metadata["foo"]',

            }

        def __call__(
                self,
                my_response: str,
                the_ideal_response: str,
                the_input: str,
                the_metadata: str) -> PassFailResult:
            return PassFailResult(value=(
                self.expected_my_response == my_response
                and self.expected_the_ideal_response == the_ideal_response
                and self.expected_the_input == the_input
                and self.expected_the_metadata == the_metadata
            ))
    try:
        check = FakeCheck(
            expected_my_response='foo',
            expected_the_ideal_response='bar',
            expected_the_input='baz',
            expected_the_metadata='qux',
        )
        result = check.run_on_model(ResponseModel(
            response={'foobar': 'foo'},
            ideal_response=['wrong', 'bar'],
            input='baz',
            metadata={'foo': 'qux'},
        ))
        assert result.value
        assert 'data_path' not in result.metadata
        assert 'value_extracted' not in result.metadata
        assert 'data_path_error' not in result.metadata
    finally:
        Check.registry._registry.pop(registration_value)

def test__Check__data_path__dictionary__error():
    registration_value = 'FAKECHECK__DATA_PATH_DICT'
    @Check.register(registration_value)
    class FakeCheck(Check):
        """Mock test for testing."""

        expected_input_1: str
        expected_input_2: str

        @property
        def default_data_path(self) -> str:
            return {'input_1': 'response["foobar"]', 'input_2': 'ideal_response'}

        def __call__(self, input_1: str, input_2: str) -> PassFailResult:
            return PassFailResult(
                value=input_1 == self.expected_input_1 and input_2 == self.expected_input_2,
                metadata={'1': input_1, '2': input_2},
            )
    try:
        check = FakeCheck(expected_input_1='foo', expected_input_2='bar')
        # this will fail because input_1's value path is response["foobar"] but response does not
        # have a key 'foobar'
        # however, input_2 should be extracted correctly
        result = check.run_on_model(ResponseModel(response={'does_not_exist': 'foo'}, ideal_response='bar'))  # noqa: E501
        assert not result.value
        assert result.metadata['1'] is None
        assert result.metadata['2'] is None
        assert result.metadata['data_path'] == check.default_data_path
        assert result.metadata['value_extracted'] == {'input_1': None, 'input_2': None}
        assert 'data_path_error' in result.metadata
    finally:
        Check.registry._registry.pop(registration_value)

def test__Check__data_path__lambda_string():
    registration_value = 'FAKECHECK__DATA_PATH_LAMBDA'
    @Check.register(registration_value)
    class FakeCheck(Check):
        """Mock test for testing."""

        expected_input: str

        @property
        def default_data_path(self) -> str:
            return 'lambda data: data.response["foobar"].upper()'

        def __call__(self, input: str) -> PassFailResult:  # noqa
            return PassFailResult(
                value=input == self.expected_input,
                metadata={'input': input},
            )
    try:
        check = FakeCheck(expected_input='FOO')
        result = check.run_on_model(ResponseModel(response={'foobar': 'foo'}))
        assert result.value
        assert result.metadata['input'] == 'FOO'
        assert 'data_path' not in result.metadata
        assert 'value_extracted' not in result.metadata
        assert 'data_path_error' not in result.metadata
    finally:
        Check.registry._registry.pop(registration_value)

def test__Check__data_path__lambda_string__override_default():
    registration_value = 'FAKECHECK__DATA_PATH_LAMBDA'
    @Check.register(registration_value)
    class FakeCheck(Check):
        """Mock test for testing."""

        expected_input: str
        # @property
        # def default_data_path(self) -> str:
        #     return 'lambda data: data.response["foobar"].upper()'

        def __call__(self, input: str) -> PassFailResult:  # noqa
            return PassFailResult(
                value=input == self.expected_input,
                metadata={'input': input},
            )
    try:
        data_path = 'lambda data: data.response["foobar"].upper()'
        check = FakeCheck(
            expected_input='FOO',
            data_path=data_path,
        )
        result = check.run_on_model(ResponseModel(response={'foobar': 'foo'}))
        assert result.value
        assert result.metadata['input'] == 'FOO'
        # since we are overriding the default data_path, metadata will be populated with
        # data_path and value_extracted
        assert 'data_path' in result.metadata
        assert result.metadata['data_path'] == data_path
        assert 'value_extracted' in result.metadata
        assert result.metadata['value_extracted'] == 'FOO'
        assert 'data_path_error' not in result.metadata
    finally:
        Check.registry._registry.pop(registration_value)

def test__Check__data_path__lambda_string__error():
    registration_value = 'FAKECHECK__DATA_PATH_LAMBDA'
    @Check.register(registration_value)
    class FakeCheck(Check):
        """Mock test for testing."""

        expected_input: str

        @property
        def default_data_path(self) -> str:
            return 'lambda data: data.response["bar"].function_does_not_exist()'

        def __call__(self, input: str) -> PassFailResult:  # noqa: A002
            return PassFailResult(
                value=input == self.expected_input,
                metadata={'input': input},
            )
    try:
        check = FakeCheck(expected_input='FOO')
        result = check.run_on_model(ResponseModel(response={'bar': 'foo'}))
        assert not result.value
        assert result.metadata['input'] is None
        assert 'data_path' in result.metadata
        assert 'value_extracted' in result.metadata
        assert 'data_path_error' in result.metadata
        assert 'function_does_not_exist' in result.metadata['data_path_error']
    finally:
        Check.registry._registry.pop(registration_value)

def test__MatchCheck__has_check_type():
    """
    Test that the check has a check_type upon object creation (without using create_instance from
    the registry).
    """
    check = MatchCheck(value='foo')
    assert check.check_type == CheckType.MATCH.name
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.MATCH.name,
        'value': 'foo',
    }
    assert MatchCheck(**check_dict) == check
    assert Check.from_dict(check_dict) == check

def test__MatchCheck__to_from_dict():
    check = MatchCheck(value='foo')
    assert Check.from_dict(check.to_dict()) == check
    check = MatchCheck(value='foo', negate=True, metadata={'foo': 'bar'})
    assert Check.from_dict(check.to_dict()) == check
    check = MatchCheck(value='foo', negate=True, data_path='response["foo"]')
    assert Check.from_dict(check.to_dict()) == check

@pytest.mark.parametrize(
    'negate',
    [True, False],
)
def test__MatchCheck(negate: bool):
    assert CheckType.MATCH.name in Check.registry
    assert CheckType.MATCH in Check.registry

    # this should fail because we didn't pass the required param
    with pytest.raises(ValidationError):
        Check.from_dict({'check_type': CheckType.MATCH})

    # check check_type with Enum
    original_dict = {'check_type': CheckType.MATCH, 'value': 'foo'}
    if negate:
        original_dict['negate'] = negate
    check = Check.from_dict(original_dict)
    assert check.value == 'foo'
    assert check.check_type == CheckType.MATCH.name
    assert check.negate == negate
    assert check.metadata == {}
    assert str(check)
    actual_dict = check.to_dict()
    expected_dict = {
        'check_type': CheckType.MATCH.name,
        'value': 'foo',
        # 'metadata': {},
    }
    if negate:
        expected_dict['negate'] = negate

    assert actual_dict == expected_dict
    assert MatchCheck(**actual_dict) == check

    # check check_type with lower case
    original_dict = {'check_type': CheckType.MATCH.name.lower(), 'value': 'foo'}
    if negate:
        original_dict['negate'] = negate
    check = Check.from_dict(original_dict)
    assert check.value == 'foo'
    assert check.check_type == CheckType.MATCH.name
    assert check.negate == negate
    assert check.metadata == {}
    assert str(check)
    actual_dict = check.to_dict()
    expected_dict = {
        'check_type': CheckType.MATCH.name,
        'value': 'foo',
    }
    if negate:
        expected_dict['negate'] = negate
    assert actual_dict == expected_dict
    assert MatchCheck(**actual_dict) == check

    result = check.run_on_model(ResponseModel(response='foo'))  # passing in matching value which should match  # noqa: E501
    assert result.success == (not negate)
    assert result.value == (not negate)
    assert result.metadata['check_type'] == CheckType.MATCH.name
    assert result.metadata['check_value'] == 'foo'
    assert result.metadata['check_negate'] == negate
    assert result.metadata['check_metadata'] == {}
    assert str(result)
    result_dict = result.to_dict()
    expected_dict = {
        'value': not negate,
        'success': not negate,
        'metadata': {
            'check_type': CheckType.MATCH.name,
            'check_value': 'foo',
            'check_negate': negate,
            'check_metadata': {},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert result_dict == expected_dict
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result

    # should get same results when calling the check directly
    result_2 = check('foo')
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

    check = Check.from_dict({
        'check_type': CheckType.MATCH.name.lower(),
        'value': 'bar',
        'negate': negate,
        'metadata': {'bar': 'foo'},
    })
    assert check.value == 'bar'
    assert check.check_type == CheckType.MATCH.name
    assert check.negate == negate
    assert check.metadata == {'bar': 'foo'}
    assert str(check)
    expected_dict = {
        'check_type': CheckType.MATCH.name,
        'value': 'bar',
        'metadata': {'bar': 'foo'},
    }
    if negate:
        expected_dict['negate'] = negate
    actual_dict = check.to_dict()
    assert actual_dict == expected_dict
    assert MatchCheck(**actual_dict) == check

    result = check.run_on_model(ResponseModel(response='foo'))  # should not match
    assert result.success == negate
    assert result.value == negate
    assert result.metadata['check_type'] == CheckType.MATCH.name
    assert result.metadata['check_value'] == 'bar'
    assert result.metadata['check_negate'] == negate
    assert result.metadata['check_metadata'] == {'bar': 'foo'}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': negate,
        'success': negate,
        'metadata': {
            'check_type': CheckType.MATCH.name,
            'check_value': 'bar',
            'check_negate': negate,
            'check_metadata': {'bar': 'foo'},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result
    # should get same results when calling the check directly
    result_2 = check('foo')
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

def test__MatchCheck__data_path():
    response = {'foo': 'bar'}
    check = MatchCheck(value='bar', data_path='response["foo"]')
    result = check.run_on_model(ResponseModel(response=response))
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.MATCH.name
    assert result.metadata['check_value'] == 'bar'
    assert result.metadata['check_negate'] is False
    assert result.metadata['check_metadata'] == {}
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == 'bar'

    assert 'data_path' in check.to_dict()
    assert check == Check.from_dict(check.to_dict())

    check = MatchCheck(value='bar', data_path='response["foo"]', negate=True)
    result = check.run_on_model(ResponseModel(response=response))
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.MATCH.name
    assert result.metadata['check_value'] == 'bar'
    assert result.metadata['check_negate'] is True
    assert result.metadata['check_metadata'] == {}
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == 'bar'

def test__ContainsCheck__has_check_type():
    """
    Test that the check has a check_type upon object creation (without using create_instance from
    the registry).
    """
    check = ContainsCheck(value='foo')
    assert check.check_type == CheckType.CONTAINS.name
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.CONTAINS.name,
        'value': 'foo',
    }
    assert ContainsCheck(**check_dict) == check
    assert Check.from_dict(check_dict) == check

@pytest.mark.parametrize(
    'negate',
    [True, False],
)
def test__ContainsCheck(negate: bool):
    assert CheckType.CONTAINS.name in Check.registry
    assert CheckType.CONTAINS in Check.registry

    # this should fail because we didn't pass the required param
    with pytest.raises(ValidationError):
        Check.from_dict({'check_type': CheckType.CONTAINS})

    # check check_type with Enum
    check_dict = {'check_type': CheckType.CONTAINS, 'value': 'o ba'}
    if negate:
        check_dict['negate'] = negate
    check = Check.from_dict(check_dict)
    assert check.value == 'o ba'
    assert check.check_type == CheckType.CONTAINS.name
    assert check.negate == negate
    assert check.metadata == {}
    assert str(check)
    check_dict = check.to_dict()
    expected_value = {
        'check_type': CheckType.CONTAINS.name,
        'value': 'o ba',
        # 'metadata': {},
    }
    if negate:
        expected_value['negate'] = negate
    assert check_dict == expected_value
    assert ContainsCheck(**check_dict) == check

    # check check_type with lower case
    check_dict = {'check_type': CheckType.CONTAINS.name.lower(), 'value': 'o ba'}
    if negate:
        check_dict['negate'] = negate
    check = Check.from_dict(check_dict)
    assert check.value == 'o ba'
    assert check.check_type == CheckType.CONTAINS.name
    assert check.negate == negate
    assert check.metadata == {}
    assert str(check)
    check_dict = check.to_dict()
    expected_value = {
        'check_type': CheckType.CONTAINS.name,
        'value': 'o ba',
        # 'metadata': {},
    }
    if negate:
        expected_value['negate'] = negate
    assert check_dict == expected_value
    assert ContainsCheck(**check_dict) == check

    # passing in str that is contained within value which should match
    result = check.run_on_model(ResponseModel(response='foo bar'))
    assert result.success == (not negate)
    assert result.value == (not negate)
    assert result.metadata['check_type'] == CheckType.CONTAINS.name
    assert result.metadata['check_value'] == 'o ba'
    assert result.metadata['check_negate'] == negate
    assert result.metadata['check_metadata'] == {}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': not negate,
        'success': not negate,
        'metadata': {
            'check_type': CheckType.CONTAINS.name,
            'check_value': 'o ba',
            'check_negate': negate,
            'check_metadata': {},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result
    # should get same results when calling the check directly
    result_2 = check('foo bar')
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

    check = Check.from_dict({
        'check_type': CheckType.CONTAINS.name.lower(),
        'value': 'o ba',
        'negate': negate,
        'metadata': {'bar': 'foo'},
    })
    assert check.value == 'o ba'
    assert check.check_type == CheckType.CONTAINS.name
    assert check.negate == negate
    assert check.metadata == {'bar': 'foo'}
    assert str(check)
    check_dict = check.to_dict()
    expected_value = {
        'check_type': CheckType.CONTAINS.name,
        'value': 'o ba',
        'metadata': {'bar': 'foo'},
    }
    if negate:
        expected_value['negate'] = negate
    assert check_dict == expected_value
    assert ContainsCheck(**check_dict) == check

    # passing in str that is not contained within value which should not match
    result = check.run_on_model(ResponseModel(response='bar foo'))
    assert result.success == negate
    assert result.value == negate
    assert result.metadata['check_type'] == CheckType.CONTAINS.name
    assert result.metadata['check_value'] == 'o ba'
    assert result.metadata['check_negate'] == negate
    assert result.metadata['check_metadata'] == {'bar': 'foo'}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': negate,
        'success': negate,
        'metadata': {
            'check_type': CheckType.CONTAINS.name,
            'check_value': 'o ba',
            'check_negate': negate,
            'check_metadata': {'bar': 'foo'},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result
    # should get same results when calling the check directly
    result_2 = check('bar foo')
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

def test__ContainsCheck__data_path():
    response = {'foo': 'the bar'}
    check = ContainsCheck(value='bar', data_path='response["foo"]')
    result = check.run_on_model(ResponseModel(response=response))
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.CONTAINS.name
    assert result.metadata['check_value'] == 'bar'
    assert result.metadata['check_negate'] is False
    assert result.metadata['check_metadata'] == {}
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == 'the bar'

    assert 'data_path' in check.to_dict()
    assert check == Check.from_dict(check.to_dict())

    check = ContainsCheck(value='bar', data_path='response["foo"]', negate=True)
    result = check.run_on_model(ResponseModel(response=response))
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.CONTAINS.name
    assert result.metadata['check_value'] == 'bar'
    assert result.metadata['check_negate'] is True
    assert result.metadata['check_metadata'] == {}
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == 'the bar'

def test__RegexCheck__has_check_type():
    """
    Test that the check has a check_type upon object creation (without using create_instance from
    the registry).
    """
    check = RegexCheck(pattern='foo')
    assert check.check_type == CheckType.REGEX.name
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.REGEX.name,
        'pattern': 'foo',
    }
    assert RegexCheck(**check_dict) == check
    assert Check.from_dict(check_dict) == check

@pytest.mark.parametrize(
    'negate',
    [True, False],
)
def test__RegexCheck(negate: bool):
    assert CheckType.REGEX.name in Check.registry
    assert CheckType.REGEX in Check.registry

    # this should fail because we didn't pass the required param
    with pytest.raises(ValidationError):
        Check.from_dict({'check_type': CheckType.REGEX})

    # example of regex to test
    regex = r'^[a-z]+$'  # this regex matches any string that is all lowercase letters
    check_dict = {'check_type': CheckType.REGEX, 'pattern': regex}
    if negate:
        check_dict['negate'] = negate
    check = Check.from_dict(check_dict)
    assert check.pattern == regex
    assert check.check_type == CheckType.REGEX.name
    assert check.negate == negate
    assert check.metadata == {}
    assert str(check)
    check_dict = check.to_dict()
    expected_value = {
        'check_type': CheckType.REGEX.name,
        'pattern': regex,
        # 'metadata': {},
    }
    if negate:
        expected_value['negate'] = negate
    assert check_dict == expected_value
    assert RegexCheck(**check_dict) == check

    # passing in str that matches the regex which should match
    result = check.run_on_model(ResponseModel(response='foo'))
    assert result.success == (not negate)
    assert result.value == (not negate)
    assert result.metadata['check_type'] == CheckType.REGEX.name
    assert result.metadata['check_pattern'] == regex
    assert result.metadata['check_negate'] == negate
    assert result.metadata['check_metadata'] == {}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': not negate,
        'success': not negate,
        'metadata': {
            'check_type': CheckType.REGEX.name,
            'check_pattern': regex,
            'check_negate': negate,
            'check_metadata': {},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result
    # should get same results when calling the check directly
    result_2 = check('foo')
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

    check_dict = {
        'check_type': CheckType.REGEX.name.lower(),
        'pattern': regex,
        'metadata': {'bar': 'foo'},
    }
    if negate:
        check_dict['negate'] = negate
    check = Check.from_dict(check_dict)
    assert check.pattern == regex
    assert check.check_type == CheckType.REGEX.name
    assert check.negate == negate
    assert check.metadata == {'bar': 'foo'}
    assert str(check)
    check_dict = check.to_dict()
    expected_value = {
        'check_type': CheckType.REGEX.name,
        'pattern': regex,
        'metadata': {'bar': 'foo'},
    }
    if negate:
        expected_value['negate'] = negate
    assert check_dict == expected_value
    assert RegexCheck(**check_dict) == check

    # passing in str that does not match the regex which should not match
    result = check.run_on_model(ResponseModel(response='Foo'))
    assert result.success == negate
    assert result.value == negate
    assert result.metadata['check_type'] == CheckType.REGEX.name
    assert result.metadata['check_pattern'] == regex
    assert result.metadata['check_negate'] == negate
    assert result.metadata['check_metadata'] == {'bar': 'foo'}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': negate,
        'success': negate,
        'metadata': {
            'check_type': CheckType.REGEX.name,
            'check_pattern': regex,
            'check_negate': negate,
            'check_metadata': {'bar': 'foo'},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result
    # should get same results when calling the check directly
    result_2 = check('Foo')
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

    assert check.run_on_model(ResponseModel(response='foo')).success == (not negate)
    assert check.run_on_model(ResponseModel(response='foo123')).success == negate
    assert check.run_on_model(ResponseModel(response='123foo')).success == negate
    assert check.run_on_model(ResponseModel(response='Foo')).success == negate

    check.negate = not negate
    assert check.run_on_model(ResponseModel(response='foo')).success == negate
    assert check.run_on_model(ResponseModel(response='foo123')).success == (not negate)
    assert check.run_on_model(ResponseModel(response='123foo')).success == (not negate)
    assert check.run_on_model(ResponseModel(response='Foo')).success == (not negate)

def test__RegexCheck__multiline_response():
    response = r"""
    Here's a Python function called mask_emails that uses regex to mask all emails:

    ```
    import re

    def mask_emails(text: str) -> str:
        '''
        Mask all emails in the given text.

        Args:
        text (str): The input text containing emails to be masked.

        Returns:
        str: The text with masked emails.
        '''
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        masked_text = re.sub(email_pattern, '[MASKED]@[MASKED]', text)
        return masked_text
    ```
    """
    check = RegexCheck(pattern='def mask_emails\\([a-zA-Z_]+\\: str\\) -> str\\:')
    result = check.run_on_model(ResponseModel(response=response))
    assert result.success

    # check without type hints; should fail
    response = """
    Here's a Python function called mask_emails that uses regex to mask all emails:

    ```
    import re

    def mask_emails(text):
        '''
        Mask all emails in the given text.

        Args:
        text (str): The input text containing emails to be masked.

        Returns:
        str: The text with masked emails.
        '''
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        masked_text = re.sub(email_pattern, '[MASKED]@[MASKED]', text)
        return masked_text
    ```
    """  # noqa: W605
    check = RegexCheck(pattern='def mask_emails\\([a-zA-Z_]+\\: str\\) -> str\\:')
    result = check.run_on_model(ResponseModel(response=response))
    assert not result.success

def test__RegexCheck__data_path():
    response = {'foo': 'the bar'}
    check = RegexCheck(pattern='bar', data_path='response["foo"]')
    result = check.run_on_model(ResponseModel(response=response))
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.REGEX.name
    assert result.metadata['check_pattern'] == 'bar'
    assert result.metadata['check_negate'] is False
    assert result.metadata['check_metadata'] == {}
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == 'the bar'

    assert 'data_path' in check.to_dict()
    assert check == Check.from_dict(check.to_dict())

    check = RegexCheck(pattern='bar', data_path='response["foo"]', negate=True)
    result = check.run_on_model(ResponseModel(response=response))
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.REGEX.name
    assert result.metadata['check_pattern'] == 'bar'
    assert result.metadata['check_negate'] is True
    assert result.metadata['check_metadata'] == {}
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == 'the bar'

def test__LambdaCheck__has_check_type():
    check = LambdaCheck(lambda_str='lambda x: x == 1')
    assert check.check_type == CheckType.LAMBDA.name
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.LAMBDA.name,
        'lambda_str': 'lambda x: x == 1',
    }
    assert LambdaCheck(**check_dict) == check
    assert Check.from_dict(check_dict) == check

def test__LambdaCheck():
    check = LambdaCheck(lambda_str='lambda x: x == 1', metadata={'foo': 'bar'})
    result = check.run_on_model(ResponseModel(response=1))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {'foo': 'bar'}
    assert 'lambda_error' not in result.metadata
    assert str(result)

    result_2 = check(1)
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

    new_check = Check.from_dict(check.to_dict())
    assert new_check == check
    result = new_check.run_on_model(ResponseModel(response=1))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {'foo': 'bar'}
    assert 'lambda_error' not in result.metadata
    assert str(result)

    result = check.run_on_model(ResponseModel(response=2))
    assert not result.success
    assert not result.value
    assert result.metadata['lambda_str'] == 'lambda x: x == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {'foo': 'bar'}
    assert 'lambda_error' not in result.metadata

    result_2 = check(2)
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

    result = new_check.run_on_model(ResponseModel(response=2))
    assert not result.success
    assert not result.value
    assert result.metadata['lambda_str'] == 'lambda x: x == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {'foo': 'bar'}
    assert 'lambda_error' not in result.metadata

def test__LambdaCheck__data_path__double_quotes_key():
    check = LambdaCheck(lambda_str='lambda x: x[0] == 1', data_path='response["foo"]')
    result = check.run_on_model(ResponseModel(response={'foo': [1, 2, 3]}))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x[0] == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == [1, 2, 3]

    result = check.run_on_model(ResponseModel(response={'foo': [2, 3]}))
    assert not result.success
    assert not result.value
    assert result.metadata['lambda_str'] == 'lambda x: x[0] == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == [2, 3]

    check = LambdaCheck(lambda_str='lambda x: len(x) == 3', data_path='response["foo"]')
    result = check.run_on_model(ResponseModel(response={'foo': [1, 2, 3]}))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: len(x) == 3'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == [1, 2, 3]

def test__LambdaCheck__data_path__single_quotes_key():
    check = LambdaCheck(lambda_str='lambda x: x[0] == 1', data_path="response['foo']")
    result = check.run_on_model(ResponseModel(response={'foo': [1, 2, 3]}))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x[0] == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata

    result = check.run_on_model(ResponseModel(response={'foo': [2, 3]}))
    assert not result.success
    assert not result.value
    assert result.metadata['lambda_str'] == 'lambda x: x[0] == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata

    check = LambdaCheck(lambda_str='lambda x: len(x) == 3', data_path="response['foo']")
    result = check.run_on_model(ResponseModel(response={'foo': [1, 2, 3]}))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: len(x) == 3'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata

def test__LambdaCheck__data_path__with_numeric_indexes__dict():
    # test list
    # test dictionary
    check = LambdaCheck(lambda_str='lambda x: x[0] == 1', data_path='response[100]')
    result = check.run_on_model(ResponseModel(response={100: [1, 2, 3]}))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x[0] == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response[100]'
    assert result.metadata['value_extracted'] == [1, 2, 3]

    result = check.run_on_model(ResponseModel(response={100: [2, 3]}))
    assert not result.success
    assert not result.value
    assert result.metadata['lambda_str'] == 'lambda x: x[0] == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response[100]'
    assert result.metadata['value_extracted'] == [2, 3]

    check = LambdaCheck(lambda_str='lambda x: len(x) == 3', data_path='response[100]')
    result = check.run_on_model(ResponseModel(response={100: [1, 2, 3]}))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: len(x) == 3'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response[100]'
    assert result.metadata['value_extracted'] == [1, 2, 3]

def test__LambdaCheck__data_path__with_numeric_indexes__list():
    # test list
    # test dictionary
    response = [0, 1, 2, 3, 4, 5]
    check = LambdaCheck(lambda_str='lambda x: x == 5', data_path='response[-1]')
    result = check.run_on_model(ResponseModel(response=response))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x == 5'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response[-1]'
    assert result.metadata['value_extracted'] == 5

    result = check.run_on_model(ResponseModel(response=[0, 1]))
    assert not result.success
    assert not result.value
    assert result.metadata['lambda_str'] == 'lambda x: x == 5'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response[-1]'
    assert result.metadata['value_extracted'] == 1

    response = [0, 1, 2, 3, 4, {'foo': 'bar'}]
    check = LambdaCheck(lambda_str='lambda x: x == "bar"', data_path='response[-1]["foo"]')
    result = check.run_on_model(ResponseModel(response=response))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x == "bar"'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response[-1]["foo"]'
    assert result.metadata['value_extracted'] == 'bar'

    check = LambdaCheck(lambda_str='lambda x: x == 3', data_path='response[3]')
    result = check.run_on_model(ResponseModel(response=response))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x == 3'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {}
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response[3]'
    assert result.metadata['value_extracted'] == 3

def test__LambdaCheck__data_path__from_dict():
    check_dict = {
        'check_type': CheckType.LAMBDA.name,
        'lambda_str': 'lambda x: x[0] == 1',
        'data_path': 'response["foo"]',
    }
    check = Check.from_dict(check_dict)
    result = check.run_on_model(ResponseModel(response={'foo': [1, 2, 3]}))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: x[0] == 1'
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == [1, 2, 3]

    result = check.run_on_model(ResponseModel(response={'foo': [2, 3]}))
    assert not result.success
    assert not result.value
    assert result.metadata['lambda_str'] == 'lambda x: x[0] == 1'
    assert 'lambda_error' not in result.metadata

    check_dict = {
        'check_type': CheckType.LAMBDA.name,
        'lambda_str': 'lambda x: len(x) == 3',
        'data_path': 'response["foo"]',
    }
    check = Check.from_dict(check_dict)
    result = check.run_on_model(ResponseModel(response={'foo': [1, 2, 3]}))
    assert result.success
    assert result.value
    assert result.metadata['lambda_str'] == 'lambda x: len(x) == 3'
    assert 'lambda_error' not in result.metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == 'response["foo"]'
    assert result.metadata['value_extracted'] == [1, 2, 3]

def test__LambdaCheck__error_handling__lambda():
    check = LambdaCheck(lambda_str='lambda x: y == 1', metadata={'foo': 'bar'})
    result = check.run_on_model(ResponseModel(response=1))
    assert not result.success
    assert not result.value
    assert result.metadata['lambda_str'] == 'lambda x: y == 1'
    assert result.metadata['check_type'] == CheckType.LAMBDA.name
    assert result.metadata['check_metadata'] == {'foo': 'bar'}
    assert 'lambda_error' in result.metadata

def test__PythonCodeBlocksPresent__has_check_type():
    """
    Test that the check has a check_type upon object creation (without using create_instance from
    the registry).
    """
    check = PythonCodeBlocksPresent()
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    check_dict = check.to_dict()
    assert check_dict == {'check_type': CheckType.PYTHON_CODE_BLOCKS_PRESENT.name}
    assert PythonCodeBlocksPresent(**check_dict) == check
    assert Check.from_dict(check_dict) == check

def test__test__PythonCodeBlocksPresent():
    check = PythonCodeBlocksPresent(min_code_blocks=1)
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 1
    assert check.metadata == {}
    assert str(check)
    result = check.run_on_model(ResponseModel(response=''))
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['min_code_blocks'] == 1
    assert result.metadata['code_blocks'] == []

    check = PythonCodeBlocksPresent(min_code_blocks=1, data_path='metadata')
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 1
    assert check.metadata == {}
    assert str(check)
    result = check.run_on_model(ResponseModel(metadata=''))
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['min_code_blocks'] == 1
    assert result.metadata['code_blocks'] == []

    assert 'data_path' in check.to_dict()
    assert check == Check.from_dict(check.to_dict())

    check = PythonCodeBlocksPresent(min_code_blocks=1)
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 1
    assert check.metadata == {}
    assert str(check)
    result = check.run_on_model(ResponseModel(response=None))
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['min_code_blocks'] == 1
    assert result.metadata['code_blocks'] == []

    assert check == Check.from_dict(check.to_dict())

    check = PythonCodeBlocksPresent(
        min_code_blocks=1,
        metadata={'foo': 'bar'},
    )
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 1
    assert check.metadata == {'foo': 'bar'}
    assert str(check)

    code_block_str = 'print("hello world")'
    expected_code_blocks = [code_block_str]
    result = check.run_on_model(ResponseModel(response=f'```\n{code_block_str}\n```'))
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['min_code_blocks'] == 1
    assert result.metadata['code_blocks'] == expected_code_blocks

    check = PythonCodeBlocksPresent(
        min_code_blocks=1,
        metadata={'foo': 'bar'},
        data_path='metadata["response"]',
    )
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 1
    assert check.metadata == {'foo': 'bar'}
    assert str(check)
    response = f'This is a response: ```python\n{code_block_str}\n```\n'
    result = check.run_on_model(ResponseModel(metadata={'response': response}))
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['min_code_blocks'] == 1
    assert result.metadata['code_blocks'] ==expected_code_blocks

    check = PythonCodeBlocksPresent(
        min_code_blocks=2,
        metadata={'foo': 'bar'},
    )
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 2
    assert check.metadata == {'foo': 'bar'}
    assert str(check)
    response = f'This is a response: ```python\n{code_block_str}\n```\n'
    result = check.run_on_model(ResponseModel(response=response))
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['min_code_blocks'] == 2
    assert result.metadata['code_blocks'] ==expected_code_blocks

def test__PythonCodeBlockTests__has_check_type():
    """
    Test that the check has a check_type upon object creation (without using create_instance from
    the registry).
    """
    check = PythonCodeBlockTests()
    assert check.check_type == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    check_dict = check.to_dict()
    assert check_dict == {'check_type': CheckType.PYTHON_CODE_BLOCK_TESTS.name}
    assert PythonCodeBlockTests(**check_dict) == check
    assert Check.from_dict(check_dict) == check

def _code_blocks_to_response(code_blocks: list[str]) -> str:
    """
    Takes code blocks and returns a response that will generate the code blocks when extracting
    them from within the check.
    """
    return '\n\n'.join(f"```\n{dedent(c).strip()}\n```" for c in code_blocks)

def test__PythonCodeBlockTests__no_code_blocks():
    check = PythonCodeBlockTests()
    assert check.success_threshold == 1
    assert check.code_setup is None
    assert not check.code_tests
    assert check.metadata == {}
    assert str(check)

    result = check.run_on_model(ResponseModel(response=''))
    assert result.value == 0
    assert not result.success
    assert result.success_threshold == 1
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['num_code_blocks_successful'] == 0
    assert result.metadata['code_blocks'] == []
    assert result.metadata['code_block_errors'] == []
    assert not result.metadata['code_tests']
    assert result.metadata['num_code_tests'] == 0
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == []
    assert result.metadata['code_test_errors'] == []

    result = check.run_on_model(ResponseModel(response=None))
    assert result.value == 0
    assert not result.success
    assert result.success_threshold == 1
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['num_code_blocks_successful'] == 0
    assert result.metadata['code_blocks'] == []
    assert result.metadata['code_block_errors'] == []
    assert result.metadata['code_tests'] == []
    assert result.metadata['num_code_tests'] == 0
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == []
    assert result.metadata['code_test_errors'] == []

def test__PythonCodeBlockTests__no_code_blocks__with_code_tests():
    code_test = [
        'assert True',
        'assert my_value == 1',
        'assert my_value != 1',
    ]
    check = PythonCodeBlockTests(code_tests=code_test)
    assert check.success_threshold == 1
    assert check.code_setup is None
    assert check.code_tests == code_test
    assert check.metadata == {}
    assert str(check)

    result = check.run_on_model(ResponseModel(response=''))
    assert result.value == 0
    assert not result.success
    assert result.success_threshold == 1
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['num_code_blocks_successful'] == 0
    assert result.metadata['code_blocks'] == []
    assert result.metadata['code_block_errors'] == []
    assert result.metadata['code_tests'] == code_test
    assert result.metadata['num_code_tests'] == len(code_test)
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == []
    assert result.metadata['code_test_errors'] == []

    result = check.run_on_model(ResponseModel(response=None))
    assert result.value == 0
    assert not result.success
    assert result.success_threshold == 1
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['num_code_blocks_successful'] == 0
    assert result.metadata['code_blocks'] == []
    assert result.metadata['code_block_errors'] == []
    assert result.metadata['code_tests'] == code_test
    assert result.metadata['num_code_tests'] == len(code_test)
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == []
    assert result.metadata['code_test_errors'] == []

def test__PythonCodeBlockTests__no_setup__no_functions():
    check = PythonCodeBlockTests()
    assert check.success_threshold == 1
    assert check.code_setup is None
    assert check.code_tests is None
    assert check.metadata == {}
    assert str(check)
    expected_code_blocks = [
        'my_value = 1',
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    response = _code_blocks_to_response(expected_code_blocks)
    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == 2/3
    assert not result.success
    assert result.success_threshold == 1
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 3
    assert result.metadata['num_code_blocks_successful'] == 2
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert result.metadata['code_block_errors'][0] is None
    assert result.metadata['code_block_errors'][1] == {'error': 'AssertionError', 'message': ''}
    assert result.metadata['code_block_errors'][2] is None
    assert result.metadata['code_tests'] == []
    assert result.metadata['num_code_tests'] == 0
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == []
    assert result.metadata['code_test_errors'] == []

def test__PythonCodeBlockTests__with_setup():
    check = PythonCodeBlockTests(
        success_threshold=0.5,
        code_setup='my_value = 1',  # my_value is depended on the code_blocks
    )
    assert check.success_threshold == 0.5
    assert check.code_setup == 'my_value = 1'
    assert check.code_tests is None
    assert check.metadata == {}
    assert str(check)
    expected_code_blocks = [
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    response = _code_blocks_to_response(expected_code_blocks)
    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == 0.5
    assert result.success
    assert result.success_threshold == 0.5
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert result.metadata['code_block_errors'][0] == {'error': 'AssertionError', 'message': ''}
    assert result.metadata['code_block_errors'][1] is None
    assert result.metadata['code_tests'] == []
    assert result.metadata['num_code_tests'] == 0
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == []
    assert result.metadata['code_test_errors'] == []
    assert Check.from_dict(check.to_dict()) == check
    assert CheckResult.from_dict(result.to_dict()) == result
    result_2 = check(response)
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

def test__PythonCodeBlockTests__with_code_tests():
    def check_code_blocks(code_blocks):  # noqa
        return code_blocks == ['assert my_value != 1', 'assert my_value == 1']
    def my_value_equals_1(code_blocks):  # noqa
        return my_value == 1  # noqa
    def non_existant_value_should_fail(code_blocks):  # noqa
        raise does_not_exist  # noqa
    def my_value_not_equals(code_blocks):  # noqa
        return my_value != 1  # noqa
    def raises_error_should_fail(code_blocks):  # noqa
        raise ValueError('This should fail')

    code_tests = [
        check_code_blocks,  # expect success
        my_value_equals_1,  # expect success
        # this needs to follow a successful function check because `__results__` (which is returned
        # but the function) will be set to True after the previous function check, but a function
        # check that fails won't set `__results__`, so we need to make sure `__results__` is reset
        # to False before this function check is run so we don't grab the previous function check's
        # result
        non_existant_value_should_fail,  # expect failure
        my_value_not_equals,  # expect failure
        raises_error_should_fail,  # expect failure
    ]
    expected_num_code_blocks = 2
    expected_successful_code_blocks = 1
    expected_code_tests = len(code_tests)
    expected_successful_code_tests = 2
    expected_total_checks = expected_num_code_blocks + expected_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests
    threshold = (expected_successful_checks/expected_total_checks) + 0.001
    check = PythonCodeBlockTests(
        success_threshold=threshold,
        code_setup='my_value = 1',  # my_value is depended on the code_blocks
        code_tests=code_tests,
    )
    assert check.success_threshold == threshold
    assert check.code_setup == 'my_value = 1'
    assert len(check.code_tests) == len(code_tests)
    assert check.metadata == {}
    assert str(check)
    expected_code_blocks = [
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    response = _code_blocks_to_response(expected_code_blocks)
    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert result.metadata['code_block_errors'][0] == {'error': 'AssertionError', 'message': ''}
    assert result.metadata['code_block_errors'][1] is None
    assert result.metadata['code_tests'] == code_tests
    assert result.metadata['num_code_tests'] == expected_code_tests
    assert result.metadata['num_code_tests_successful'] == expected_successful_code_tests
    assert result.metadata['code_test_results'] == [True, True, False, False, False]
    assert result.metadata['code_test_errors'][0] is None
    assert result.metadata['code_test_errors'][1] is None
    assert result.metadata['code_test_errors'][2] == {'error': 'NameError', 'message': "name 'does_not_exist' is not defined"}  # noqa
    assert result.metadata['code_test_errors'][3] is None
    assert result.metadata['code_test_errors'][4] == {'error': 'ValueError', 'message': 'This should fail'}  # noqa
    result_2 = check(response)
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

def test__PythonCodeBlockTests__with_code_tests__str():
    # same test as `test__PythonCodeBlockTests__with_code_tests` except the code_tests are strings
    # rather than functions
    check_code_blocks = """
    def check_code_blocks(code_blocks):
        return code_blocks == ['assert my_value != 1', 'assert my_value == 1']
    """
    my_value_equals_1 = """
    def my_value_equals_1(code_blocks):
        return my_value == 1
    """
    non_existant_value_should_fail = """
    def non_existant_value_should_fail(code_blocks):
        raise does_not_exist
    """
    my_value_not_equals = """
    def my_value_not_equals(code_blocks):
        return my_value != 1
    """
    raises_error_should_fail = """
    def raises_error_should_fail(code_blocks):
        raise ValueError('This should fail')
    """

    code_tests = [
        check_code_blocks,  # expect success
        my_value_equals_1,  # expect success
        # this needs to follow a successful function check because `__results__` (which is returned
        # but the function) will be set to True after the previous function check, but a function
        # check that fails won't set `__results__`, so we need to make sure `__results__` is reset
        # to False before this function check is run so we don't grab the previous function check's
        # result
        non_existant_value_should_fail,  # expect failure
        my_value_not_equals,  # expect failure
        raises_error_should_fail,  # expect failure
    ]
    expected_num_code_blocks = 2
    expected_successful_code_blocks = 1
    expected_num_code_tests = len(code_tests)
    expected_successful_code_tests = 2
    expected_total_checks = expected_num_code_blocks + expected_num_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests
    threshold = (expected_successful_checks/expected_total_checks) + 0.001
    check = PythonCodeBlockTests(
        success_threshold=threshold,
        code_setup='my_value = 1',  # my_value is depended on the code_blocks
        code_tests=code_tests,
    )
    assert check.success_threshold == threshold
    assert check.code_setup == 'my_value = 1'
    assert len(check.code_tests) == len(code_tests)
    assert check.metadata == {}
    assert str(check)
    expected_code_blocks = [
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    response = _code_blocks_to_response(expected_code_blocks)
    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert result.metadata['code_block_errors'][0] == {'error': 'AssertionError', 'message': ''}
    assert result.metadata['code_block_errors'][1] is None
    expected_code_tests = [
        dedent(test.strip()) if isinstance(test, str) else test
        for test in code_tests
    ]
    assert result.metadata['code_tests'] == expected_code_tests
    assert result.metadata['num_code_tests'] == expected_num_code_tests
    assert result.metadata['num_code_tests_successful'] == expected_successful_code_tests
    assert result.metadata['code_test_results'] == [True, True, False, False, False]
    assert result.metadata['code_test_errors'][0] is None
    assert result.metadata['code_test_errors'][1] is None
    assert result.metadata['code_test_errors'][2] == {'error': 'NameError', 'message': "name 'does_not_exist' is not defined"}  # noqa
    assert result.metadata['code_test_errors'][3] is None
    assert result.metadata['code_test_errors'][4] == {'error': 'ValueError', 'message': 'This should fail'}  # noqa
    result_2 = check(response)
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

def test__PythonCodeBlockTests__failing_code_setup_raises_error():
    """
    If one of the code_tests (that is checking the results) raises an error, the entire check
    should fail.
    """
    # ensure code block is successfull before we test for failure for a different cause
    response = '```\n1 == 1\n```'
    response_model = ResponseModel(response=response)
    check = PythonCodeBlockTests()
    result = check.run_on_model(response_model)
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['num_code_blocks_successful'] == 1
    result_2 = check(response)
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

    check = PythonCodeBlockTests(
        code_setup='raise ValueError()',
    )
    with pytest.raises(AssertionError):
        check(response)

def test__PythonCodeBlockTests__with_code_tests__failing_function_does_not_raise_error():
    """
    If one of the code_tests (that is checking the results) raises an error, the entire check
    should fail.
    """
    def failing_function(code_blocks):  # noqa
        raise ValueError()
    check = PythonCodeBlockTests(
        code_tests=[
            failing_function,
        ],
    )
    response = '```\n1 == 1\n```'
    response_model = ResponseModel(response=response)
    result = check.run_on_model(response_model)
    assert result.metadata['num_code_tests'] == 1
    assert result.metadata['num_code_tests_successful'] == 0
    assert len(result.metadata['code_test_errors']) == 1
    assert result.metadata['code_test_errors'][0] == {'error': 'ValueError', 'message': ''}
    assert result.metadata['code_test_results'] == [False]
    result_2 = check(response)
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

    assert result.value == 0.5
    assert not result.success
    assert result.success_threshold == 1
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == ['1 == 1']
    assert result.metadata['code_block_errors'][0] is None

def test__PythonCodeBlockTests__with_code_tests__all_code_blocks_fail__test_numbers_are_correct():
    """Make sure the number of code tests is still accurate if all code blocks fail to run."""
    code_tests = ["def failing_function(code_blocks):\n    return variable_does_not_exist == 1"]
    check = PythonCodeBlockTests(code_tests=code_tests)
    response = dedent("""
        ```
        raise ValueError()
        ```

        ```
        raise NameError()
        ```
    """)
    response_model = ResponseModel(response=response)
    expected_code_blocks = ['raise ValueError()', 'raise NameError()']
    result = check.run_on_model(response_model)
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 0
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert result.metadata['code_block_errors'][0] == {'error': 'ValueError', 'message': ''}
    assert result.metadata['code_block_errors'][1] == {'error': 'NameError', 'message': ''}
    assert result.metadata['code_tests'] == code_tests
    assert result.metadata['num_code_tests'] == 1
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == [False]
    assert result.metadata['code_test_errors'][0] == {'error': 'NameError', 'message': "name 'variable_does_not_exist' is not defined"}  # noqa
    result_2 = check(response)
    assert result == result_2
    assert result.to_dict() == result_2.to_dict()

def test__PythonCodeBlockTests__with_code_tests__all_tests_with_same_name():
    # same test as `test__PythonCodeBlockTests__with_code_tests` except all functions are named the
    # same `test_function`
    check_code_blocks = """
    def test_function(code_blocks):
        return code_blocks == ['assert my_value != 1', 'assert my_value == 1']
    """
    my_value_equals_1 = """
    def test_function(code_blocks):
        return my_value == 1
    """
    non_existant_value_should_fail = """
    def test_function(code_blocks):
        raise does_not_exist
    """
    my_value_not_equals = """
    def test_function(code_blocks):
        return my_value != 1
    """
    raises_error_should_fail = """
    def test_function(code_blocks):
        raise ValueError('This should fail')
    """

    code_tests = [
        check_code_blocks,  # expect success
        my_value_equals_1,  # expect success
        # this needs to follow a successful function check because `__results__` (which is returned
        # but the function) will be set to True after the previous function check, but a function
        # check that fails won't set `__results__`, so we need to make sure `__results__` is reset
        # to False before this function check is run so we don't grab the previous function check's
        # result
        non_existant_value_should_fail,  # expect failure
        my_value_not_equals,  # expect failure
        raises_error_should_fail,  # expect failure
    ]
    expected_num_code_blocks = 2
    expected_successful_code_blocks = 1
    expected_num_code_tests = len(code_tests)
    expected_successful_code_tests = 2
    expected_total_checks = expected_num_code_blocks + expected_num_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests
    threshold = (expected_successful_checks/expected_total_checks) + 0.001
    check = PythonCodeBlockTests(
        success_threshold=threshold,
        code_setup='my_value = 1',  # my_value is depended on the code_blocks
        code_tests=code_tests,
    )
    assert check.success_threshold == threshold
    assert check.code_setup == 'my_value = 1'
    assert len(check.code_tests) == len(code_tests)
    assert check.metadata == {}
    assert str(check)
    expected_code_blocks = [
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    response = _code_blocks_to_response(expected_code_blocks)
    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert result.metadata['code_block_errors'][0] == {'error': 'AssertionError', 'message': ''}
    assert result.metadata['code_block_errors'][1] is None
    expected_code_tests = [
        dedent(test.strip()) if isinstance(test, str) else test
        for test in code_tests
    ]
    assert result.metadata['code_tests'] == expected_code_tests
    assert result.metadata['num_code_tests'] == expected_num_code_tests
    assert result.metadata['num_code_tests_successful'] == expected_successful_code_tests
    assert result.metadata['code_test_results'] == [True, True, False, False, False]
    assert result.metadata['code_test_errors'][0] is None
    assert result.metadata['code_test_errors'][1] is None
    assert result.metadata['code_test_errors'][2] == {'error': 'NameError', 'message': "name 'does_not_exist' is not defined"}  # noqa
    assert result.metadata['code_test_errors'][3] is None
    assert result.metadata['code_test_errors'][4] == {'error': 'ValueError', 'message': 'This should fail'}  # noqa

def test__PythonCodeBlockTests__with_code_tests__assertion_boolean_statements():
    # code blocks depend on this setup code
    code_setup = dedent("""
        required_variable = 2
        def raises(expected_exception, code):
            try:
                code()
            except expected_exception:
                return True
            except:
                return False
            return False
    """).strip()
    expected_code_blocks = [
        dedent("""
        def my_function(my_value):
            if my_value <= 0:
               raise ValueError('my_value must be greater than 0')
            return required_variable / my_value == 1
        """).strip(),
        'assert required_variable == 2',
        'assert variable_doesnt_exist == 1',
    ]
    response = _code_blocks_to_response(expected_code_blocks)
    # 1st item is the expected result (pass/fail), 2nd item is expected error, 3rd is the code
    code_tests = [
        (
            True,  # pass/fail
            None,  # error
            'def test(code_blocks: list[str]) -> bool:\n    return required_variable == 2',
        ),
        (
            False,
            None,
            'def test(code_blocks: list[str]) -> bool:\n    return required_variable == 1',
        ),
        (
            False,
            {'error': 'AssertionError', 'message': ''},
            'def test(code_blocks: list[str]) -> bool:\n    assert 1 == 2',
        ),
        (
            False,
            {'error': 'ValueError', 'message': 'test'},
            'def test(code_blocks: list[str]) -> bool:\n    raise ValueError("test")',
        ),
        (
            True,
            None,
            'def test(code_blocks: list[str]) -> bool:\n    return my_function(2)',
        ),
        (
            False,
            None,
            'def test(code_blocks: list[str]) -> bool:\n    return my_function(1)',
        ),
        (
            False,
            {'error': 'ValueError', 'message': 'my_value must be greater than 0'},
            'def test(code_blocks: list[str]) -> bool:\n    return my_function(0)',
        ),
        (
            False,
            None,
            'def test(code_blocks: list[str]) -> bool:\n    return raises(AssertionError, lambda: my_function(0))',  # noqa
        ),
        (
            True,
            None,
            'def test(code_blocks: list[str]) -> bool:\n    return raises(ValueError, lambda: my_function(0))',  # noqa
        ),
        (
            True,
            None,
            'assert required_variable == 2',
        ),
        (
            False,
            None,
            'assert required_variable == 1',
        ),
        (
            # this will result in an error but should fail the test rather than stopping execution
            False,
            {'error': 'NameError', 'message': "name 'this_variable_doesnt_exist' is not defined"},
            'assert this_variable_doesnt_exist',
        ),
        (
            True,
            None,
            'assert my_function(2)',
        ),
        (
            False,
            None,
            'assert my_function(1)',
        ),
        (
            False,
            {'error': 'ValueError', 'message': 'my_value must be greater than 0'},
            'assert my_function(0)',
        ),
        (
            False,
            None,
            'assert raises(AssertionError, lambda: my_function(0))',
        ),
        (
            True,
            None,
            'assert raises(ValueError, lambda: my_function(0))',
        ),

        (
            True,
            None,
            'required_variable == 2',
        ),
        (
            False,
            None,
            'required_variable == 1',
        ),
        (
            # this will result in an error but should fail the test rather than stopping execution
            False,
            {'error': 'NameError', 'message': "name 'this_variable_doesnt_exist' is not defined"},
            'this_variable_doesnt_exist',
        ),
        (
            True,
            None,
            'my_function(2)',
        ),
        (
            False,
            None,
            'my_function(1)',
        ),
        (
            False,
            {'error': 'ValueError', 'message': 'my_value must be greater than 0'},
            'my_function(0)',
        ),
        (
            False,
            None,
            'raises(AssertionError, lambda: my_function(0))',
        ),
        (
            True,
            None,
            'raises(ValueError, lambda: my_function(0))',
        ),
    ]
    expected_test_results = [x[0] for x in code_tests]
    expected_errors = [x[1] for x in code_tests]
    code_tests = [x[2] for x in code_tests]

    expected_num_code_blocks = len(expected_code_blocks)
    expected_successful_code_blocks = 2
    expected_num_code_tests = len(code_tests)
    expected_successful_code_tests = sum(expected_test_results)
    expected_total_checks = expected_num_code_blocks + expected_num_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests
    threshold = (expected_successful_checks/expected_total_checks) + 0.001
    check = PythonCodeBlockTests(
        success_threshold=threshold,
        code_setup=code_setup,
        code_tests=code_tests,
    )
    assert check.success_threshold == threshold
    assert check.code_setup == code_setup
    assert len(check.code_tests) == len(code_tests)
    assert check.metadata == {}
    assert str(check)
    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == len(expected_code_blocks)
    assert result.metadata['num_code_blocks_successful'] == 2
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert result.metadata['code_block_errors'][0] is None
    assert result.metadata['code_block_errors'][1] is None
    assert result.metadata['code_block_errors'][2] == {'error': 'NameError', 'message': "name 'variable_doesnt_exist' is not defined"}  # noqa
    expected_code_tests = [
        dedent(test.strip()) if isinstance(test, str) else test
        for test in code_tests
    ]
    assert result.metadata['code_tests'] == expected_code_tests
    assert result.metadata['num_code_tests'] == expected_num_code_tests
    assert result.metadata['num_code_tests_successful'] == expected_successful_code_tests
    assert result.metadata['code_test_results'] == expected_test_results
    assert result.metadata['code_test_errors'] == expected_errors

def test__PythonCodeBlockTests__with_code_tests__invalid_test_raises_exception():
    # we are only expecting a single line of code for non-functions
    code_tests = ['assert True\nassert True']
    check = PythonCodeBlockTests(code_tests=code_tests)
    with pytest.raises(AssertionError, match='Only a single statement is allowed if the value is a string.'):  # noqa
        check.run_on_model(ResponseModel(response='```\n1 == 1\n```'))

    code_tests = ['def test(code_blocks: list[str]) -> bool:\n    return None']
    check = PythonCodeBlockTests(code_tests=code_tests)
    with pytest.raises(AssertionError, match=re.escape(f"Test must return a boolean value:\n{code_tests[0]}")):  # noqa
        check.run_on_model(ResponseModel(response='```\n1 == 1\n```'))

    code_tests = ['assert None']
    check = PythonCodeBlockTests(code_tests=code_tests)
    with pytest.raises(AssertionError, match=re.escape('Test must return a boolean value:\ndef __code_test__(code_blocks: list[str]) -> bool:\n    return None')):  # noqa
        check.run_on_model(ResponseModel(response='```\n1 == 1\n```'))

    code_tests = ['None']
    check = PythonCodeBlockTests(code_tests=code_tests)
    with pytest.raises(AssertionError, match=re.escape('Test must return a boolean value:\ndef __code_test__(code_blocks: list[str]) -> bool:\n    return None')):  # noqa
        check.run_on_model(ResponseModel(response='```\n1 == 1\n```'))

def test__PythonCodeBlockTests__with_code_tests__timeouts_within_threshold():
    expected_code_blocks = [
        """
        import time
        my_value_1 = 'test1'
        time.sleep(0.5)
        my_value_2 = 'test2'
        """,
        """
        raise ValueError("This is a test")
        """,
        """
        my_value_3 = 'test3'
        """,
    ]
    response = _code_blocks_to_response(expected_code_blocks)
    expected_code_blocks = [dedent(c).strip() for c in expected_code_blocks]
    code_tests = [
        "assert my_value_1 == 'test1'",
        "assert my_value_2 == 'test2'",
        "def test(blocks):\n    time.sleep(1.5)\n    return True",
        "assert my_value_3 == 'test3'",
    ]
    expected_num_code_blocks = len(expected_code_blocks)
    expected_successful_code_blocks = 2
    expected_num_code_tests = len(code_tests)
    expected_successful_code_tests = 4
    expected_total_checks = expected_num_code_blocks + expected_num_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests
    threshold = (expected_successful_checks/expected_total_checks) + 0.001
    code_setup = 'import time'
    check = PythonCodeBlockTests(
        success_threshold=threshold,
        code_setup=code_setup,
        code_block_timeout=1,  # exceeds code block sleep
        code_test_timeout=2,  # exceeds code test sleep
        code_tests=code_tests,
    )
    assert check.success_threshold == threshold
    assert check.code_setup == code_setup
    assert len(check.code_tests) == len(code_tests)
    assert check.metadata == {}
    assert str(check)

    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success  # second code block fails
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == len(expected_code_blocks)
    assert result.metadata['num_code_blocks_successful'] == expected_successful_code_blocks
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert len(result.metadata['code_block_errors']) == len(expected_code_blocks)
    assert result.metadata['code_block_errors'][0] is None
    assert result.metadata['code_block_errors'][1] == {'error': 'ValueError', 'message': 'This is a test'}  # noqa
    assert result.metadata['code_block_errors'][2] is None
    assert result.metadata['code_tests'] == code_tests
    assert result.metadata['num_code_tests'] == expected_num_code_tests
    assert result.metadata['num_code_tests_successful'] == expected_successful_code_tests
    assert result.metadata['code_test_results'] == [True, True, True, True]
    assert result.metadata['code_test_errors'][0] is None
    assert result.metadata['code_test_errors'][1] is None
    assert result.metadata['code_test_errors'][2] is None
    assert result.metadata['code_test_errors'][3] is None

def test__PythonCodeBlockTests__with_code_tests__timeouts_exceed_threshold_code_blocks():
    expected_code_blocks = [
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
    response = _code_blocks_to_response(expected_code_blocks)
    expected_code_blocks = [dedent(c).strip() for c in expected_code_blocks]
    code_tests = [
        "assert my_value_1 == 'test1'",  # should still pass
        "assert my_value_2 == 'test2'",  # should now fail since sleep exceeds timeout
        "def test(blocks):\n    time.sleep(0.5)\n    return True",
        "assert my_value_3 == 'test3'",
    ]
    expected_num_code_blocks = len(expected_code_blocks)
    expected_successful_code_blocks = 1
    expected_num_code_tests = len(code_tests)
    expected_successful_code_tests = 3
    expected_total_checks = expected_num_code_blocks + expected_num_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests
    threshold = (expected_successful_checks/expected_total_checks) + 0.001
    code_setup = 'import time'
    check = PythonCodeBlockTests(
        success_threshold=threshold,
        code_setup=code_setup,
        code_block_timeout=1,  # exceeds code block sleep
        code_test_timeout=1,  # exceeds code test sleep
        code_tests=code_tests,
    )
    assert check.success_threshold == threshold
    assert check.code_setup == code_setup
    assert len(check.code_tests) == len(code_tests)
    assert check.metadata == {}
    assert str(check)

    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success  # second code block fails
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == len(expected_code_blocks)
    assert result.metadata['num_code_blocks_successful'] == expected_successful_code_blocks
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert len(result.metadata['code_block_errors']) == len(expected_code_blocks)
    assert result.metadata['code_block_errors'][0] == {'error': 'TimeoutError', 'message': ''}
    assert result.metadata['code_block_errors'][1] == {'error': 'ValueError', 'message': 'This is a test'}  # noqa
    assert result.metadata['code_block_errors'][2] is None
    assert result.metadata['code_tests'] == code_tests
    assert result.metadata['num_code_tests'] == expected_num_code_tests
    assert result.metadata['num_code_tests_successful'] == expected_successful_code_tests
    assert result.metadata['code_test_results'] == [True, False, True, True]
    assert result.metadata['code_test_errors'][0] is None
    assert result.metadata['code_test_errors'][1] == {'error': 'NameError', 'message': "name 'my_value_2' is not defined"}  # noqa
    assert result.metadata['code_test_errors'][2] is None
    assert result.metadata['code_test_errors'][3] is None

def test__PythonCodeBlockTests__with_code_tests__timeouts_exceed_threshold_code_tests():
    expected_code_blocks = [
        """
        import time
        my_value_1 = 'test1'
        time.sleep(0.5)
        my_value_2 = 'test2'
        """,
        """
        raise ValueError("This is a test")
        """,
        """
        my_value_3 = 'test3'
        """,
    ]
    response = _code_blocks_to_response(expected_code_blocks)
    expected_code_blocks = [dedent(c).strip() for c in expected_code_blocks]
    code_tests = [
        "assert my_value_1 == 'test1'",  # should still pass
        "assert my_value_2 == 'test2'",  # should now pass since sleep < timeout
        "def test(blocks):\n    time.sleep(1.5)\n    return True",  # this should now fail
        "assert my_value_3 == 'test3'",
    ]
    expected_num_code_blocks = len(expected_code_blocks)
    expected_successful_code_blocks = 2
    expected_num_code_tests = len(code_tests)
    expected_successful_code_tests = 3
    expected_total_checks = expected_num_code_blocks + expected_num_code_tests
    expected_successful_checks = expected_successful_code_blocks + \
        expected_successful_code_tests
    threshold = (expected_successful_checks/expected_total_checks) + 0.001
    code_setup = 'import time'
    check = PythonCodeBlockTests(
        success_threshold=threshold,
        code_setup=code_setup,
        code_block_timeout=1,  # exceeds code block sleep
        code_test_timeout=1,  # exceeds code test sleep
        code_tests=code_tests,
    )
    assert check.success_threshold == threshold
    assert check.code_setup == code_setup
    assert len(check.code_tests) == len(code_tests)
    assert check.metadata == {}
    assert str(check)

    result = check.run_on_model(ResponseModel(response=response))
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success  # second code block fails
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == len(expected_code_blocks)
    assert result.metadata['num_code_blocks_successful'] == expected_successful_code_blocks
    assert result.metadata['code_blocks'] == expected_code_blocks
    assert len(result.metadata['code_block_errors']) == len(expected_code_blocks)
    assert result.metadata['code_block_errors'][0] is None
    assert result.metadata['code_block_errors'][1] == {'error': 'ValueError', 'message': 'This is a test'}  # noqa
    assert result.metadata['code_block_errors'][2] is None
    assert result.metadata['code_tests'] == code_tests
    assert result.metadata['num_code_tests'] == expected_num_code_tests
    assert result.metadata['num_code_tests_successful'] == expected_successful_code_tests
    assert result.metadata['code_test_results'] == [True, True, False, True]
    assert result.metadata['code_test_errors'][0] is None
    assert result.metadata['code_test_errors'][1] is None
    assert result.metadata['code_test_errors'][2] == {'error': 'TimeoutError', 'message': ''}
    assert result.metadata['code_test_errors'][3] is None

def test__PythonCodeBlockTests__with_env_namespace():
    code = dedent("""
    import pandas as pd
    # df = pd.DataFrame({'col_1': [20, 5, 50], 'col_2': ['a', 'a', 'b']})
    answer = df.groupby('col_2')['col_1'].sum()
    """).strip()

    # first ensure that without env_namespace the check will fail
    check = PythonCodeBlockTests(
        code_tests=[
            "assert int(answer['a']) == 25",
            "assert int(answer['b']) == 50",
        ],
    )
    result = check.run_on_model(ResponseModel(response=f"```\n{code}\n```"))
    assert result.success is False
    assert result.value == 0
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['num_code_blocks_successful'] == 0
    assert result.metadata['code_blocks'] == [code]
    assert 'error' in result.metadata['code_block_errors'][0]
    assert result.metadata['code_block_errors'][0]['error'] == 'NameError'
    assert 'df' in result.metadata['code_block_errors'][0]['message']
    assert result.metadata['num_code_tests'] == 2
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == [False, False]
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert Check.from_dict(check.to_dict()) == check
    assert CheckResult.from_dict(result.to_dict()) == result

    # now use env_namespace to pass in the DataFrame, which should successfully run the code
    env_namespace = {'df': pd.DataFrame({'col_1': [20, 5, 50], 'col_2': ['a', 'a', 'b']})}
    check = PythonCodeBlockTests(
        code_tests=[
            "assert int(answer['a']) == 25",
            "assert int(answer['b']) == 50",
        ],
        env_namespace=env_namespace,
    )
    result = check.run_on_model(ResponseModel(response=f"```\n{code}\n```"))
    assert result.success is True
    assert result.value == 1
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == [code]
    assert result.metadata['code_block_errors'][0] is None
    assert result.metadata['num_code_tests'] == 2
    assert result.metadata['num_code_tests_successful'] == 2
    assert result.metadata['code_test_results'] == [True, True]
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert Check.from_dict(check.to_dict()) == check
    assert CheckResult.from_dict(result.to_dict()) == result

def test__ToolCallsCheck():
    """Test that the template for an OpenAI candidate works."""
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={'location': 'Boston, MA', 'unit': 'fahrenheit'},
    )
    result = check.run_on_model(ResponseModel(response=[{
        'arguments': {'location':'Boston, MA', 'unit':'fahrenheit'},
        'name': 'get_current_weather',
    }]))
    assert isinstance(result, CheckResult)
    assert result.success is True
    assert result.value == 1
    assert result.metadata['function_name'] == 'get_current_weather'
    assert result.metadata['function_arguments'] == {
        'location': 'Boston, MA', 'unit': 'fahrenheit',
    }
    assert result.metadata['allow_regex'] is False
    assert result.metadata['check_type'] == CheckType.TOOL_CALL

def test__ToolCallsCheck__multiple_functions():
    """Test that the template for an OpenAI candidate works."""
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={'location': 'Boston, MA', 'unit': 'fahrenheit'},
    )
    result = check.run_on_model(ResponseModel(response=[
        {
            "arguments": {"location":"Boston, MA", "unit":"fahrenheit"},
            "name": "get_current_weather",
        },
        {
            "arguments": {"location":"Boston, MA"},
            "name": "get_location",
        },
    ]))
    assert isinstance(result, CheckResult)
    assert result.success is True
    assert result.value == 1
    assert result.metadata['function_name'] == 'get_current_weather'
    assert result.metadata['function_arguments'] == {
        'location': 'Boston, MA', 'unit': 'fahrenheit',
    }
    assert result.metadata['allow_regex'] is False
    assert result.metadata['check_type'] == CheckType.TOOL_CALL

def test__ToolCallsCheck__allow_regex():
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={
            'location': 'Boston',
            'unit': 'fahrenheit',
            'temperature': 70,
            'raining': False,
            'uv_index': None,
        },
        allow_regex=True,
    )
    result = check.run_on_model(ResponseModel(response=[{
        'arguments': {
            'location':'Boston, MA',
            'unit':'fahrenheit',
            'temperature': 70,
            'raining': False,
            'uv_index': None,
        },
        'name': 'get_current_weather',
    }]))
    assert isinstance(result, CheckResult)
    assert result.success is True
    assert result.value == 1
    assert result.metadata['function_name'] == 'get_current_weather'
    assert result.metadata['function_arguments'] == {
        'location': 'Boston',
        'unit': 'fahrenheit',
        'temperature': 70,
        'raining': False,
        'uv_index': None,
    }
    assert result.metadata['allow_regex'] is True
    assert result.metadata['check_type'] == CheckType.TOOL_CALL

def test__ToolCallsCheck__penalize_extraneous_arguments():
    """Test that the template for an OpenAI candidate works."""
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={'location': 'Boston', 'unit': 'fahrenheit'},
    )
    fake_response = [{
        'arguments': {
            'location':'Boston, MA',
            'unit':'fahrenheit',
            'lat': 42.3601,
            'lon': 71.0589,
        },
        'name': 'get_current_weather',
    }]
    result = check.run_on_model(ResponseModel(response=fake_response))
    assert isinstance(result, CheckResult)
    assert result.success is False
    assert result.value == 0
    assert result.metadata['function_name'] == 'get_current_weather'
    assert result.metadata['function_arguments'] == {'location': 'Boston', 'unit': 'fahrenheit'}
    assert result.metadata['allow_regex'] is False
    assert result.metadata['check_type'] == CheckType.TOOL_CALL
    assert result.metadata['penalize_extraneous_arguments']

def test__ToolCallsCheck__incorrect_function_name():
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={'location': 'Boston', 'unit': 'fahrenheit'},
    )
    result = check.run_on_model(ResponseModel(response=[{
        'arguments': {'location':'Boston, MA', 'unit':'fahrenheit'},
        'name': 'get_current_weather_2',
    }]))
    assert isinstance(result, CheckResult)
    assert result.success is False
    assert result.value == 0

def test__ToolCallsCheck__allow_regex_partial_correct():
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={
            'location': 'Boston',
            'unit': 'fahrenheit',
            'temperature': 70,
            'raining': False,
            'uv_index': None,
        },
        allow_regex=True,
    )
    result = check.run_on_model(ResponseModel(response=[{
        'arguments': {
            'location':'Boston, MA',
            'unit':'fahrenheit',
            'temperature': 70,
            'raining': True,
            'uv_index': 0,
        },
        'name': 'get_current_weather',
    }]))
    assert isinstance(result, CheckResult)
    assert result.success is False
    assert result.value == 0.6
    assert result.metadata['function_name'] == 'get_current_weather'
    assert result.metadata['function_arguments'] == {
        'location': 'Boston',
        'unit': 'fahrenheit',
        'temperature': 70,
        'raining': False,
        'uv_index': None,
    }
    assert result.metadata['allow_regex'] is True
    assert result.metadata['check_type'] == CheckType.TOOL_CALL

def test__ToolCallsCheck__empty_string_response():
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={'location': 'Boston', 'unit': 'fahrenheit'},
    )
    result = check.run_on_model(ResponseModel(response=""))
    assert isinstance(result, CheckResult)
    assert result.success is False
    assert result.value == 0

def test__ToolCallsCheck__empty_list_response():
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={'location': 'Boston', 'unit': 'fahrenheit'},
    )
    result = check.run_on_model(ResponseModel(response=[]))
    assert isinstance(result, CheckResult)
    assert result.success is False
    assert result.value == 0

def test__ToolCallsCheck__string_response():
    check = ToolCallsCheck(
        function_name='get_current_weather',
        function_arguments={'location': 'Boston', 'unit': 'fahrenheit'},
    )
    result = check.run_on_model(ResponseModel(response="This is a test."))
    assert isinstance(result, CheckResult)
    assert result.success is False
    assert result.value == 0

def test__LLMCheck__openai():
    """Test that the template for an OpenAI candidate works."""

    class ContainsToxicity(BaseModel):
        contains_toxicity: bool
        toxicity_phrase: str | None = None

    ####
    # Check Toxicity
    ####
    check = LLMCheck(
        eval_prompt="Check if the response contains toxicity.",
        response_format=ContainsToxicity,
        openai_model_name=OPENAI_DEFAULT_MODEL,
    )
    check_to_dict = check.to_dict()
    assert check_to_dict['eval_prompt'] == "Check if the response contains toxicity."
    assert check_to_dict['response_format'] == ContainsToxicity
    assert check_to_dict['openai_model_name'] == OPENAI_DEFAULT_MODEL
    assert check.from_dict(check.to_dict()) == check

    result = check.run_on_model(ResponseModel(response="This is bullshit. I don't understand."))
    assert isinstance(result, CheckResult)
    assert result.metadata['check_type'] == CheckType.LLM
    assert isinstance(result.value['parsed'], ContainsToxicity)
    assert not result.value['refusal']
    assert "ContainsToxicity" in result.metadata['response_format']

    # Check that the response correctly contains toxicity and the phrase
    assert result.value['parsed'].contains_toxicity is True
    assert 'bullshit' in result.value['parsed'].toxicity_phrase.lower()

    assert result.metadata['usage']['prompt_tokens'] > 0
    assert result.metadata['usage']['completion_tokens'] > 0
    assert result.metadata['usage']['total_tokens'] > 0
    assert result.metadata['usage']['prompt_cost'] > 0
    assert result.metadata['usage']['completion_cost'] > 0
    assert result.metadata['usage']['total_cost'] > 0
    assert result.metadata['duration_seconds'] > 0

    ####
    # Check that the response does not contain toxicity
    ####
    check = LLMCheck(
        eval_prompt="Check if the response contains toxicity.",
        response_format=ContainsToxicity,
        openai_model_name=OPENAI_DEFAULT_MODEL,
    )
    result = check.run_on_model(ResponseModel(response="Well hello there."))
    assert isinstance(result, CheckResult)
    assert result.metadata['check_type'] == CheckType.LLM
    assert isinstance(result.value['parsed'], ContainsToxicity)
    assert not result.value['refusal']

    # Check that the response correctly contains toxicity and the phrase
    assert result.value['parsed'].contains_toxicity is False
    assert not result.value['parsed'].toxicity_phrase

    assert result.metadata['usage']['prompt_tokens'] > 0
    assert result.metadata['usage']['completion_tokens'] > 0
    assert result.metadata['usage']['total_tokens'] > 0
    assert result.metadata['usage']['prompt_cost'] > 0
    assert result.metadata['usage']['completion_cost'] > 0
    assert result.metadata['usage']['total_cost'] > 0
    assert result.metadata['duration_seconds'] > 0

def test__PrecisionScore():
    actual_response = "This is the generated TEXT."
    ideal_response = "This is the ideal or correct text."
    score = PrecisionScore()
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == 0.5
    assert result.success_threshold is None
    assert result.success is None
    assert result.metadata['check_type'] == CheckType.PRECISION_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__PrecisionScore__data_path():
    data_path = {
        'actual_response': 'response["generated_response"]',
        'ideal_response': 'ideal_response["correct"]',
    }
    score = PrecisionScore(data_path=data_path)
    result = score.run_on_model(ResponseModel(
        response={'generated_response': "This is the generated TEXT."},
        ideal_response={'correct': "This is the ideal or correct text."},
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == 0.5
    assert result.success_threshold is None
    assert result.success is None
    assert result.metadata['check_type'] == CheckType.PRECISION_SCORE.name
    # since we are overriding the data_path, it should be in the metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == data_path
    assert 'value_extracted' in result.metadata
    assert result.metadata['value_extracted'] == {
        'actual_response': 'This is the generated TEXT.',
        'ideal_response': 'This is the ideal or correct text.',
    }

def test__PrecisionScore__threshold():
    score = PrecisionScore(success_threshold=0.51)
    result = score.run_on_model(ResponseModel(
        response="This is the generated TEXT.",
        ideal_response="This is the ideal or correct text.",
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == 0.5
    assert result.success is False
    assert result.success_threshold == 0.51

    score = PrecisionScore(success_threshold=0.49)
    result = score.run_on_model(ResponseModel(
        response="This is the generated TEXT.",
        ideal_response="This is the ideal or correct text.",
    ))
    assert result.value == 0.5
    assert result.success is True
    assert result.success_threshold == 0.49
    assert result.metadata['check_type'] == CheckType.PRECISION_SCORE.name

def test__PrecisionScore__empty_response():
    ideal_response = "This is the ideal or correct text."
    score = PrecisionScore()
    result = score.run_on_model(ResponseModel(
        response="",
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['check_type'] == CheckType.PRECISION_SCORE.name
    result_2 = score(actual_response="", ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

    score = PrecisionScore()
    result = score.run_on_model(ResponseModel(
        response=None,
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['check_type'] == CheckType.PRECISION_SCORE.name
    result_2 = score(actual_response=None, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__RecallScore():
    actual_response = "This is the generated TEXT."
    ideal_response = "This is the ideal or correct text."
    score = RecallScore()
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == pytest.approx(1/3)
    assert result.success_threshold is None
    assert result.success is None
    assert result.metadata['check_type'] == CheckType.RECALL_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__RecallScore__data_path():
    data_path = {
        'actual_response': 'response["generated_response"]',
        'ideal_response': 'ideal_response["correct"]',
    }
    actual_response = {'generated_response': "This is the generated TEXT."}
    ideal_response = {'correct': "This is the ideal or correct text."}
    score = RecallScore(data_path=data_path)
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == pytest.approx(1/3)
    assert result.success_threshold is None
    assert result.success is None
    assert result.metadata['check_type'] == CheckType.RECALL_SCORE.name
        # since we are overriding the data_path, it should be in the metadata
    assert 'data_path' in result.metadata
    assert result.metadata['data_path'] == data_path
    assert 'value_extracted' in result.metadata
    assert result.metadata['value_extracted'] == {
        'actual_response': 'This is the generated TEXT.',
        'ideal_response': 'This is the ideal or correct text.',
    }

def test__RecallScore__threshold():
    actual_response = "This is the generated TEXT."
    ideal_response = "This is the ideal or correct text."
    score = RecallScore(success_threshold=0.34)
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == pytest.approx(1/3)
    assert result.success is False
    assert result.success_threshold == 0.34
    assert result.metadata['check_type'] == CheckType.RECALL_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

    score = RecallScore(success_threshold=0.32)
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    assert result.value == pytest.approx(1/3)
    assert result.success is True
    assert result.success_threshold == 0.32
    assert result.metadata['check_type'] == CheckType.RECALL_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__RecallScore__empty_response():
    ideal_response = "This is the ideal or correct text."
    score = RecallScore()
    result = score.run_on_model(ResponseModel(
        response="",
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['check_type'] == CheckType.RECALL_SCORE.name
    result_2 = score(actual_response="", ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

    score = RecallScore()
    result = score.run_on_model(ResponseModel(
        response=None,
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['check_type'] == CheckType.RECALL_SCORE.name
    result_2 = score(actual_response=None, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__F1Score():
    actual_response = "This is the generated TEXT."
    ideal_response = "This is the ideal or correct text."
    score = F1Score()
    assert not score.return_precision_recall
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == f1_score(0.5, 1/3)
    assert result.success_threshold is None
    assert result.success is None
    assert result.metadata['check_type'] == CheckType.F1_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__F1Score__return_precision_recall():
    actual_response = "This is the generated TEXT."
    ideal_response = "This is the IDEAL or correct text."
    score = F1Score(return_precision_recall=True)
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    response_tokens = ['generated', 'text']
    ideal_tokens = ['ideal', 'correct', 'text']
    expected_precision = precision_score_tokens(expected_tokens=ideal_tokens, actual_tokens=response_tokens)  # noqa: E501
    expected_recall = recall_score_tokens(expected_tokens=ideal_tokens, actual_tokens=response_tokens)  # noqa: E501
    assert result.value == f1_score(expected_precision, expected_recall)
    assert result.success_threshold is None
    assert result.success is None
    assert 'precision' in result.metadata
    assert 'recall' in result.metadata
    assert result.metadata['precision'] == expected_precision
    assert result.metadata['recall'] == expected_recall
    assert result.metadata['check_type'] == CheckType.F1_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__F1Score__threshold():
    actual_response = "This is the generated TEXT."
    ideal_response = "This is the IDEAL or correct text."
    score = F1Score(success_threshold=0.41)
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == f1_score(0.5, 1/3)
    assert result.success is False
    assert result.success_threshold == 0.41
    assert result.metadata['check_type'] == CheckType.F1_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

    score = F1Score(success_threshold=0.39)
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    assert result.value == f1_score(0.5, 1/3)
    assert result.success is True
    assert result.success_threshold == 0.39
    assert result.metadata['check_type'] == CheckType.F1_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__F1Score__empty_response():
    ideal_response = "This is the IDEAL or correct text."
    score = F1Score()
    result = score.run_on_model(ResponseModel(
        response="",
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['check_type'] == CheckType.F1_SCORE.name
    result_2 = score(actual_response="", ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

    score = F1Score()
    result = score.run_on_model(ResponseModel(
        response=None,
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['check_type'] == CheckType.F1_SCORE.name
    result_2 = score(actual_response=None, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__MaxF1Score():
    actual_response = "This is the generated TEXT."
    ideal_response = [
        "This will have low score because there is not overlap in non-stop words.",
        "This is the IDEAL or correct text.",
    ]
    score = MaxF1Score()
    assert not score.return_precision_recall
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == f1_score(0.5, 1/3)
    assert result.success_threshold is None
    assert result.success is None
    assert result.metadata['check_type'] == CheckType.MAX_F1_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__MaxF1Score__return_precision_recall():
    actual_response = "This is the generated TEXT."
    ideal_response = [
        "This will have low score because there is not overlap in non-stop words.",
        "This is the IDEAL or correct text.",
    ]
    score = MaxF1Score(return_precision_recall=True)
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    response_tokens = ['generated', 'text']
    ideal_tokens = ['ideal', 'correct', 'text']
    expected_precision = precision_score_tokens(expected_tokens=ideal_tokens, actual_tokens=response_tokens)  # noqa: E501
    expected_recall = recall_score_tokens(expected_tokens=ideal_tokens, actual_tokens=response_tokens)  # noqa: E501
    assert result.value == f1_score(expected_precision, expected_recall)
    assert result.success_threshold is None
    assert result.success is None
    assert 'precision' in result.metadata
    assert 'recall' in result.metadata
    assert result.metadata['precision'] == expected_precision
    assert result.metadata['recall'] == expected_recall
    assert result.metadata['check_type'] == CheckType.MAX_F1_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__MaxF1Score__threshold():
    actual_response = "This is the generated TEXT."
    score = MaxF1Score(success_threshold=0.41)
    ideal_response = [
        "This will have low score because there is not overlap in non-stop words.",
        "This is the IDEAL or correct text.",
    ]

    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    # tokens should be ['generated', 'text'] and ['ideal', 'correct', 'text']
    assert result.value == f1_score(0.5, 1/3)
    assert result.success is False
    assert result.success_threshold == 0.41
    assert result.metadata['check_type'] == CheckType.MAX_F1_SCORE.name

    score = MaxF1Score(success_threshold=0.39)
    result = score.run_on_model(ResponseModel(
        response=actual_response,
        ideal_response=ideal_response,
    ))
    assert result.value == f1_score(0.5, 1/3)
    assert result.success is True
    assert result.success_threshold == 0.39
    assert result.metadata['check_type'] == CheckType.MAX_F1_SCORE.name
    result_2 = score(actual_response=actual_response, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

def test__MaxF1Score__empty_response__empty():
    ideal_response = [
        "This will have low score because there is not overlap in non-stop words.",
        "This is the IDEAL or correct text.",
    ]
    score = MaxF1Score(return_precision_recall=True)
    result = score.run_on_model(ResponseModel(
        response=[],
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['precision'] == 0
    assert result.metadata['recall'] == 0
    assert result.metadata['check_type'] == CheckType.MAX_F1_SCORE.name
    result_2 = score(actual_response=[], ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

    score = MaxF1Score(return_precision_recall=True)
    result = score.run_on_model(ResponseModel(
        response=None,
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['precision'] == 0
    assert result.metadata['recall'] == 0
    assert result.metadata['check_type'] == CheckType.MAX_F1_SCORE.name
    result_2 = score(actual_response=None, ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()

    score = MaxF1Score(return_precision_recall=True)
    result = score.run_on_model(ResponseModel(
        response='',
        ideal_response=ideal_response,
    ))
    assert result.value == 0
    assert result.metadata['precision'] == 0
    assert result.metadata['recall'] == 0
    assert result.metadata['check_type'] == CheckType.MAX_F1_SCORE.name
    result_2 = score(actual_response='', ideal_response=ideal_response)
    assert result_2 == result
    assert result_2.to_dict() == result.to_dict()
