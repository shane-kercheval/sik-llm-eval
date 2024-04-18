"""Contains tests for eval Check objects."""
from copy import deepcopy
import os
import re
from textwrap import dedent
from pydantic import ValidationError
import pytest
from llm_eval.candidates import Candidate
from llm_eval.checks import (
    Check,
    CheckResult,
    CheckResultsType,
    CheckType,
    ContainsCheck,
    LLMCheck,
    MatchCheck,
    PassFailResult,
    PythonCodeBlocksPresent,
    PythonCodeBlockTests,
    RegexCheck,
    ScoreResult,
    ToxicityCheck,
)


def test__CheckType__mixin_behaviors():  # noqa
    assert CheckType.MATCH == 'MATCH'
    assert CheckType.MATCH == 'match'
    assert CheckType.MATCH != 1
    assert CheckType.MATCH != 0
    assert CheckType.to_enum('CONTAINS') == CheckType.CONTAINS
    assert CheckType.to_enum('contains') == CheckType.CONTAINS
    assert CheckType.to_enum(CheckType.MATCH) == CheckType.MATCH
    with pytest.raises(ValueError):  # noqa: PT011
        CheckType.to_enum('foo')

def test__register_check__success__str__ensure_creation():  # noqa
    """Test successful registration of a check."""
    @Check.register('FakeCheck')
    class FakeCheck(Check):
        """Mock test for testing."""

        def __call__(self, response: str) -> CheckResult:
            return PassFailResult(
                value=response is not None,
                metadata=self.metadata,
            )

    assert 'FAKECHECK' in Check.registry
    assert 'fakecheck' in Check.registry

    check_instance = Check.from_dict({'check_type': 'fakecheck'})
    assert isinstance(check_instance, FakeCheck)
    assert check_instance.check_type == 'FAKECHECK'
    assert Check.from_dict(check_instance.to_dict()) == check_instance
    result = check_instance(response='foo')
    assert result.success
    assert result.value
    assert result.metadata == {}

    check_instance = Check.from_dict({'check_type': 'FAKECHECK'})
    assert isinstance(check_instance, FakeCheck)
    assert check_instance.check_type == 'FAKECHECK'
    assert Check.from_dict(check_instance.to_dict()) == check_instance
    result = check_instance(response='foo')
    assert result.success
    assert result.value
    assert result.metadata == {}

    check_instance = Check.from_dict({'check_type': 'FAKECHECK', 'metadata': {'foo': 'bar'}})
    assert isinstance(check_instance, FakeCheck)
    assert check_instance.check_type == 'FAKECHECK'
    assert check_instance.metadata == {'foo': 'bar'}
    assert Check.from_dict(check_instance.to_dict()) == check_instance
    result = check_instance(response='foo')
    assert result.success
    assert result.value
    assert result.metadata == {'foo': 'bar'}

    check_instance = Check.from_dict({'check_type': 'FAKECHECK', 'metadata': {}})
    assert isinstance(check_instance, FakeCheck)
    assert check_instance.check_type == 'FAKECHECK'
    assert check_instance.metadata == {}
    assert Check.from_dict(check_instance.to_dict()) == check_instance
    result = check_instance(response='foo')
    assert result.success
    assert result.value
    assert result.metadata == {}

    # We should not be able to register a check with the same type
    with pytest.raises(AssertionError):
        @Check.register('fakecheck')
        class TestCheck2(Check):
            def __call__(self, response: str) -> CheckResult:
                return PassFailResult(value=True, metadata={'response': response})
    # We should not be able to register a check that isn't a Check object
    with pytest.raises(AssertionError):
        @Check.register('test2')
        class TestCheck3:
            def __call__(self, response: str) -> CheckResult:
                return PassFailResult(value=True, metadata={'response': response})

def test__register_check__success__ensure_creation__with_required_params():  # noqa
    """Test successful registration of a check."""

    @Check.register('FakeParamCheck')
    class FakeParamCheck(Check):
        """Mock test for testing."""

        required_field: str

        def __call__(self, response: str) -> CheckResult:
            return PassFailResult(
                check_type=CheckType.PASS_FAIL,
                passed=response is not None,
                metadata=self.metadata,
            )

    assert 'FAKEPARAMCHECK' in Check.registry
    assert 'fakeparamcheck' in Check.registry

    # if we don't pass the required param, we should get an error
    with pytest.raises(ValidationError):
        _ = Check.from_dict({'check_type': 'FakeParamCheck'})
    with pytest.raises(ValidationError):
        _ = Check.from_dict({'check_type': 'FakeParamCheck', 'metadata': {'foo': 'bar'}})

    instance = Check.from_dict({'check_type': 'FakeParamCheck', 'required_field': 'foo'})
    assert isinstance(instance, FakeParamCheck)
    assert instance.metadata == {}
    assert instance.required_field == 'foo'

    instance = Check.from_dict({
        'check_type': 'FakeParamCheck',
        'required_field': 'foo',
        'metadata': {'foo': 'bar'},
    })
    assert isinstance(instance, FakeParamCheck)
    assert instance.metadata == {'foo': 'bar'}
    assert instance.required_field == 'foo'

def test__register_check__duplicate__str__():  # noqa
    """Test registering a check with a duplicate name raises an error."""
    with pytest.raises(AssertionError):
        @Check.register(CheckType.MATCH.name)
        class FakeCheck(Check):
            """Mock test for testing."""

            def __call__(self, response: str) -> CheckResult:
                return response


    with pytest.raises(AssertionError):
        @Check.register(CheckType.MATCH.name.lower())
        class FakeCheck(Check):  # noqa: F811
            """Mock test for testing."""

            def __call__(self, response: str) -> CheckResult:
                return response

def test__register_check__duplicate__CheckType():  # noqa
    """Test registering a check with a duplicate name raises an error."""
    with pytest.raises(AssertionError):
        @Check.register(CheckType.MATCH)
        class FakeCheck(Check):
            """Mock test for testing."""

            def __call__(self, response: str) -> CheckResult:
                return response

def test__CheckType():  # noqa
    assert CheckType.MATCH.name == 'MATCH'
    assert CheckType.to_enum('MATCH') == CheckType.MATCH
    assert CheckType.to_enum('match') == CheckType.MATCH
    assert CheckType.MATCH == 'MATCH'
    assert CheckType.MATCH == 'match'
    with pytest.raises(ValueError):  # noqa: PT011
        CheckType.to_enum('foo')

def test__CheckResult__registration():  # noqa
    assert CheckResultsType.PASS_FAIL in CheckResult.registry  # enum
    assert CheckResultsType.PASS_FAIL.name in CheckResult.registry  # string upper
    assert CheckResultsType.PASS_FAIL.name.lower() in CheckResult.registry  # string lower
    assert CheckResultsType.SCORE in CheckResult.registry  # enum
    assert CheckResultsType.SCORE.name in CheckResult.registry  # string upper
    assert CheckResultsType.SCORE.name.lower() in CheckResult.registry  # string lower

def test__PassFailResult():  # noqa
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

def test__PassFailResult__serialize():  # noqa
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

def test__ScoreResult():  # noqa
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

def test__ScoreResult__serialize():  # noqa
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

def test__MatchCheck__has_check_type():  # noqa
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

def test__MatchCheck():  # noqa
    assert CheckType.MATCH.name in Check.registry
    assert CheckType.MATCH in Check.registry

    # this should fail because we didn't pass the required param
    with pytest.raises(ValidationError):
        Check.from_dict({'check_type': CheckType.MATCH})

    # check check_type with Enum
    check = Check.from_dict({'check_type': CheckType.MATCH, 'value': 'foo'})
    assert check.value == 'foo'
    assert check.check_type == CheckType.MATCH.name
    assert check.metadata == {}
    assert str(check)
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.MATCH.name,
        'value': 'foo',
        # 'metadata': {},
    }
    assert MatchCheck(**check_dict) == check

    # check check_type with lower case
    check = Check.from_dict({'check_type': CheckType.MATCH.name.lower(), 'value': 'foo'})
    assert check.value == 'foo'
    assert check.check_type == CheckType.MATCH.name
    assert check.metadata == {}
    assert str(check)
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.MATCH.name,
        'value': 'foo',
    }
    assert MatchCheck(**check_dict) == check

    result = check(response='foo')  # passing in the matching value which should pass
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.MATCH.name
    assert result.metadata['check_value'] == 'foo'
    assert result.metadata['check_metadata'] == {}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': True,
        'success': True,
        'metadata': {
            'check_type': CheckType.MATCH.name,
            'check_value': 'foo',
            'check_metadata': {},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result

    check = Check.from_dict({
        'check_type': CheckType.MATCH.name.lower(),
        'value': 'bar',
        'metadata': {'bar': 'foo'},
    })
    assert check.value == 'bar'
    assert check.check_type == CheckType.MATCH.name
    assert check.metadata == {'bar': 'foo'}
    assert str(check)
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.MATCH.name,
        'value': 'bar',
        'metadata': {'bar': 'foo'},
    }
    assert MatchCheck(**check_dict) == check

    result = check(response='foo')  # passing in the non-matching value which should fail
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.MATCH.name
    assert result.metadata['check_value'] == 'bar'
    assert result.metadata['check_metadata'] == {'bar': 'foo'}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': False,
        'success': False,
        'metadata': {
            'check_type': CheckType.MATCH.name,
            'check_value': 'bar',
            'check_metadata': {'bar': 'foo'},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result

def test__ContainsCheck__has_check_type():  # noqa
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

def test__ContainsCheck():  # noqa
    assert CheckType.CONTAINS.name in Check.registry
    assert CheckType.CONTAINS in Check.registry

    # this should fail because we didn't pass the required param
    with pytest.raises(ValidationError):
        Check.from_dict({'check_type': CheckType.CONTAINS})

    # check check_type with Enum
    check = Check.from_dict({'check_type': CheckType.CONTAINS, 'value': 'o ba'})
    assert check.value == 'o ba'
    assert check.check_type == CheckType.CONTAINS.name
    assert check.metadata == {}
    assert str(check)
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.CONTAINS.name,
        'value': 'o ba',
        # 'metadata': {},
    }
    assert ContainsCheck(**check_dict) == check

    # check check_type with lower case
    check = Check.from_dict({'check_type': CheckType.CONTAINS.name.lower(), 'value': 'o ba'})
    assert check.value == 'o ba'
    assert check.check_type == CheckType.CONTAINS.name
    assert check.metadata == {}
    assert str(check)
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.CONTAINS.name,
        'value': 'o ba',
        # 'metadata': {},
    }
    assert ContainsCheck(**check_dict) == check

    # passing in str that is contained within value which should pass
    result = check(response='foo bar')
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.CONTAINS.name
    assert result.metadata['check_value'] == 'o ba'
    assert result.metadata['check_metadata'] == {}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': True,
        'success': True,
        'metadata': {
            'check_type': CheckType.CONTAINS.name,
            'check_value': 'o ba',
            'check_metadata': {},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result

    check = Check.from_dict({
        'check_type': CheckType.CONTAINS.name.lower(),
        'value': 'o ba',
        'metadata': {'bar': 'foo'},
    })
    assert check.value == 'o ba'
    assert check.check_type == CheckType.CONTAINS.name
    assert check.metadata == {'bar': 'foo'}
    assert str(check)
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.CONTAINS.name,
        'value': 'o ba',
        'metadata': {'bar': 'foo'},
    }
    assert ContainsCheck(**check_dict) == check

    # passing in str that is not contained within value which should fail
    result = check(response='bar foo')
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.CONTAINS.name
    assert result.metadata['check_value'] == 'o ba'
    assert result.metadata['check_metadata'] == {'bar': 'foo'}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': False,
        'success': False,
        'metadata': {
            'check_type': CheckType.CONTAINS.name,
            'check_value': 'o ba',
            'check_metadata': {'bar': 'foo'},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result

def test__RegexCheck__has_check_type():  # noqa
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

def test__RegexCheck():  # noqa
    assert CheckType.REGEX.name in Check.registry
    assert CheckType.REGEX in Check.registry

    # this should fail because we didn't pass the required param
    with pytest.raises(ValidationError):
        Check.from_dict({'check_type': CheckType.REGEX})

    # example of regex to test
    regex = r'^[a-z]+$'  # this regex matches any string that is all lowercase letters
    check = Check.from_dict({'check_type': CheckType.REGEX, 'pattern': regex})
    assert check.pattern == regex
    assert check.check_type == CheckType.REGEX.name
    assert check.metadata == {}
    assert str(check)
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.REGEX.name,
        'pattern': regex,
        # 'metadata': {},
    }
    assert RegexCheck(**check_dict) == check

    # passing in str that matches the regex which should pass
    result = check(response='foo')
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.REGEX.name
    assert result.metadata['check_pattern'] == regex
    assert result.metadata['check_metadata'] == {}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': True,
        'success': True,
        'metadata': {
            'check_type': CheckType.REGEX.name,
            'check_pattern': regex,
            'check_metadata': {},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result

    check = Check.from_dict({
        'check_type': CheckType.REGEX.name.lower(),
        'pattern': regex,
        'metadata': {'bar': 'foo'},
    })
    assert check.pattern == regex
    assert check.check_type == CheckType.REGEX.name
    assert check.metadata == {'bar': 'foo'}
    assert str(check)
    check_dict = check.to_dict()
    assert check_dict == {
        'check_type': CheckType.REGEX.name,
        'pattern': regex,
        'metadata': {'bar': 'foo'},
    }
    assert RegexCheck(**check_dict) == check

    # passing in str that does not match the regex which should fail
    result = check(response='Foo')
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.REGEX.name
    assert result.metadata['check_pattern'] == regex
    assert result.metadata['check_metadata'] == {'bar': 'foo'}
    assert str(result)
    result_dict = result.to_dict()
    assert result_dict == {
        'value': False,
        'success': False,
        'metadata': {
            'check_type': CheckType.REGEX.name,
            'check_pattern': regex,
            'check_metadata': {'bar': 'foo'},
        },
        'result_type': CheckResultsType.PASS_FAIL.name,
    }
    assert PassFailResult(**result_dict) == result
    assert CheckResult.from_dict(result_dict) == result

    assert check(response='foo').success
    assert not check(response='foo123').success
    assert not check(response='123foo').success
    assert not check(response='Foo').success

def test__RegexCheck__multiline_response():  # noqa
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
    result = check(response=response)
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
    result = check(response=response)
    assert not result.success

def test__PythonCodeBlocksPresent__has_check_type():  # noqa
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

def test__test__PythonCodeBlocksPresent():  # noqa
    check = PythonCodeBlocksPresent(min_code_blocks=1)
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 1
    assert check.metadata == {}
    assert str(check)
    result = check(code_blocks=[])
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['min_code_blocks'] == 1
    assert result.metadata['code_blocks'] == []

    check = PythonCodeBlocksPresent(min_code_blocks=1)
    result = check(code_blocks=None)
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 0
    assert result.metadata['min_code_blocks'] == 1
    assert result.metadata['code_blocks'] == []

    check = PythonCodeBlocksPresent(min_code_blocks=1, metadata={'foo': 'bar'})
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 1
    assert check.metadata == {'foo': 'bar'}
    assert str(check)
    code_blocks = ['```python\nprint("hello world")\n```']
    result = check(code_blocks=code_blocks)
    assert result.success
    assert result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['min_code_blocks'] == 1
    assert result.metadata['code_blocks'] == code_blocks

    check = PythonCodeBlocksPresent(min_code_blocks=2, metadata={'foo': 'bar'})
    assert check.check_type == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert check.min_code_blocks == 2
    assert check.metadata == {'foo': 'bar'}
    assert str(check)
    code_blocks = ['```python\nprint("hello world")\n```']
    result = check(code_blocks=code_blocks)
    assert not result.success
    assert not result.value
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCKS_PRESENT.name
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['min_code_blocks'] == 2
    assert result.metadata['code_blocks'] == code_blocks

def test__PythonCodeBlockTests__has_check_type():  # noqa
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

def test__PythonCodeBlockTests__no_code_blocks():  # noqa
    check = PythonCodeBlockTests()
    assert check.success_threshold == 1
    assert check.code_setup is None
    assert not check.code_tests
    assert check.metadata == {}
    assert str(check)

    result = check(code_blocks=[])
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

    result = check(code_blocks=None)
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

def test__PythonCodeBlockTests__no_code_blocks__with_code_tests():  # noqa
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

    result = check(code_blocks=[])
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

    result = check(code_blocks=None)
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

def test__PythonCodeBlockTests__no_setup__no_functions():  # noqa
    check = PythonCodeBlockTests()
    assert check.success_threshold == 1
    assert check.code_setup is None
    assert check.code_tests is None
    assert check.metadata == {}
    assert str(check)
    code_blocks = [
        'my_value = 1',
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    result = check(code_blocks=code_blocks)
    assert result.value == 2/3
    assert not result.success
    assert result.success_threshold == 1
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 3
    assert result.metadata['num_code_blocks_successful'] == 2
    assert result.metadata['code_blocks'] == code_blocks
    assert result.metadata['code_block_errors'][0] is None
    assert result.metadata['code_block_errors'][1] == {'error': 'AssertionError', 'message': ''}
    assert result.metadata['code_block_errors'][2] is None
    assert result.metadata['code_tests'] == []
    assert result.metadata['num_code_tests'] == 0
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == []
    assert result.metadata['code_test_errors'] == []

def test__PythonCodeBlockTests__with_setup():  # noqa
    check = PythonCodeBlockTests(
        success_threshold=0.5,
        code_setup='my_value = 1',  # my_value is depended on the code_blocks
    )
    assert check.success_threshold == 0.5
    assert check.code_setup == 'my_value = 1'
    assert check.code_tests is None
    assert check.metadata == {}
    assert str(check)
    code_blocks = [
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    result = check(code_blocks=code_blocks)
    assert result.value == 0.5
    assert result.success
    assert result.success_threshold == 0.5
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == code_blocks
    assert result.metadata['code_block_errors'][0] == {'error': 'AssertionError', 'message': ''}
    assert result.metadata['code_block_errors'][1] is None
    assert result.metadata['code_tests'] == []
    assert result.metadata['num_code_tests'] == 0
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == []
    assert result.metadata['code_test_errors'] == []
    assert Check.from_dict(check.to_dict()) == check
    assert CheckResult.from_dict(result.to_dict()) == result

def test__PythonCodeBlockTests__with_code_tests():  # noqa
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
    code_blocks = [
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    result = check(code_blocks=code_blocks)
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == code_blocks
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

def test__PythonCodeBlockTests__with_code_tests__str():  # noqa
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
    code_blocks = [
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    result = check(code_blocks=code_blocks)
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == code_blocks
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

def test__PythonCodeBlockTests__failing_code_setup_raises_error():  # noqa
    """
    If one of the code_tests (that is checking the results) raises an error, the entire check
    should fail.
    """
    check = PythonCodeBlockTests(
        code_setup='raise ValueError()',
    )
    with pytest.raises(AssertionError):
        check(code_blocks=['1 == 1'])

def test__PythonCodeBlockTests__with_code_tests__failing_function_does_not_raise_error():  # noqa
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
    result = check(code_blocks=['1 == 1'])
    assert result.metadata['num_code_tests'] == 1
    assert result.metadata['num_code_tests_successful'] == 0
    assert len(result.metadata['code_test_errors']) == 1
    assert result.metadata['code_test_errors'][0] == {'error': 'ValueError', 'message': ''}
    assert result.metadata['code_test_results'] == [False]

    assert result.value == 0.5
    assert not result.success
    assert result.success_threshold == 1
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 1
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == ['1 == 1']
    assert result.metadata['code_block_errors'][0] is None

def test__PythonCodeBlockTests__with_code_tests__all_code_blocks_fail__test_numbers_are_correct():  # noqa
    """Make sure the number of code tests is still accurate if all code blocks fail to run."""
    code_tests = ["def failing_function(code_blocks):\n    return variable_does_not_exist == 1"]
    check = PythonCodeBlockTests(code_tests=code_tests)
    code_blocks = ['raise ValueError()', 'raise NameError()']
    result = check(code_blocks=code_blocks)
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 0
    assert result.metadata['code_blocks'] == code_blocks
    assert result.metadata['code_block_errors'][0] == {'error': 'ValueError', 'message': ''}
    assert result.metadata['code_block_errors'][1] == {'error': 'NameError', 'message': ''}
    assert result.metadata['code_tests'] == code_tests
    assert result.metadata['num_code_tests'] == 1
    assert result.metadata['num_code_tests_successful'] == 0
    assert result.metadata['code_test_results'] == [False]
    assert result.metadata['code_test_errors'][0] == {'error': 'NameError', 'message': "name 'variable_does_not_exist' is not defined"}  # noqa

def test__PythonCodeBlockTests__with_code_tests__all_tests_with_same_name():  # noqa
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
    code_blocks = [
        'assert my_value != 1',
        'assert my_value == 1',
    ]
    result = check(code_blocks=code_blocks)
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == 2
    assert result.metadata['num_code_blocks_successful'] == 1
    assert result.metadata['code_blocks'] == code_blocks
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

def test__PythonCodeBlockTests__with_code_tests__assertion_boolean_statements():  # noqa
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
    code_blocks = [
        dedent("""
        def my_function(my_value):
            if my_value <= 0:
               raise ValueError('my_value must be greater than 0')
            return required_variable / my_value == 1
        """).strip(),
        'assert required_variable == 2',
        'assert variable_doesnt_exist == 1',
    ]
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

    expected_num_code_blocks = len(code_blocks)
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
    result = check(code_blocks=code_blocks)
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == len(code_blocks)
    assert result.metadata['num_code_blocks_successful'] == 2
    assert result.metadata['code_blocks'] == code_blocks
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

def test__PythonCodeBlockTests__with_code_tests__invalid_test_raises_exception():  # noqa
    # we are only expecting a single line of code for non-functions
    code_tests = ['assert True\nassert True']
    check = PythonCodeBlockTests(code_tests=code_tests)
    with pytest.raises(AssertionError, match='Only a single statement is allowed if the value is a string.'):  # noqa
        check(code_blocks=['1 == 1'])

    code_tests = ['def test(code_blocks: list[str]) -> bool:\n    return None']
    check = PythonCodeBlockTests(code_tests=code_tests)
    with pytest.raises(AssertionError, match=re.escape(f"Test must return a boolean value:\n{code_tests[0]}")):  # noqa
        check(code_blocks=['1 == 1'])

    code_tests = ['assert None']
    check = PythonCodeBlockTests(code_tests=code_tests)
    with pytest.raises(AssertionError, match=re.escape('Test must return a boolean value:\ndef __code_test__(code_blocks: list[str]) -> bool:\n    return None')):  # noqa
        check(code_blocks=['1 == 1'])

    code_tests = ['None']
    check = PythonCodeBlockTests(code_tests=code_tests)
    with pytest.raises(AssertionError, match=re.escape('Test must return a boolean value:\ndef __code_test__(code_blocks: list[str]) -> bool:\n    return None')):  # noqa
        check(code_blocks=['1 == 1'])

def test__PythonCodeBlockTests__with_code_tests__timeouts_within_threshold():  # noqa
    code_blocks = [
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
    code_blocks = [dedent(c).strip() for c in code_blocks]
    code_tests = [
        "assert my_value_1 == 'test1'",
        "assert my_value_2 == 'test2'",
        "def test(blocks):\n    time.sleep(1.5)\n    return True",
        "assert my_value_3 == 'test3'",
    ]
    expected_num_code_blocks = len(code_blocks)
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

    result = check(code_blocks=code_blocks)
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success  # second code block fails
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == len(code_blocks)
    assert result.metadata['num_code_blocks_successful'] == expected_successful_code_blocks
    assert result.metadata['code_blocks'] == code_blocks
    assert len(result.metadata['code_block_errors']) == len(code_blocks)
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

def test__PythonCodeBlockTests__with_code_tests__timeouts_exceed_threshold_code_blocks():  # noqa
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
    code_tests = [
        "assert my_value_1 == 'test1'",  # should still pass
        "assert my_value_2 == 'test2'",  # should now fail since sleep exceeds timeout
        "def test(blocks):\n    time.sleep(0.5)\n    return True",
        "assert my_value_3 == 'test3'",
    ]
    expected_num_code_blocks = len(code_blocks)
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

    result = check(code_blocks=code_blocks)
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success  # second code block fails
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == len(code_blocks)
    assert result.metadata['num_code_blocks_successful'] == expected_successful_code_blocks
    assert result.metadata['code_blocks'] == code_blocks
    assert len(result.metadata['code_block_errors']) == len(code_blocks)
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

def test__PythonCodeBlockTests__with_code_tests__timeouts_exceed_threshold_code_tests():  # noqa
    code_blocks = [
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
    code_blocks = [dedent(c).strip() for c in code_blocks]
    code_tests = [
        "assert my_value_1 == 'test1'",  # should still pass
        "assert my_value_2 == 'test2'",  # should now pass since sleep < timeout
        "def test(blocks):\n    time.sleep(1.5)\n    return True",  # this should now fail
        "assert my_value_3 == 'test3'",
    ]
    expected_num_code_blocks = len(code_blocks)
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

    result = check(code_blocks=code_blocks)
    assert result.value == expected_successful_checks / expected_total_checks
    assert not result.success  # second code block fails
    assert result.success_threshold == threshold
    assert result.metadata['check_type'] == CheckType.PYTHON_CODE_BLOCK_TESTS.name
    assert result.metadata['num_code_blocks'] == len(code_blocks)
    assert result.metadata['num_code_blocks_successful'] == expected_successful_code_blocks
    assert result.metadata['code_blocks'] == code_blocks
    assert len(result.metadata['code_block_errors']) == len(code_blocks)
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

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__LLMCheck__openai(openai_candidate_template):  # noqa
    """Test that the template for an OpenAI candidate works."""
    template = deepcopy(openai_candidate_template)
    candidate = Candidate.from_dict(template)
    check = LLMCheck(
        eval_prompt = "What is the number returned in the question?",
        evaluator=candidate,
    )
    result = check(
        prompt="What is the secret number?",
        response="The secret number is 42.",
    )
    assert isinstance(result, CheckResult)
    assert result.success is None
    assert '42' in result.value
    assert result.metadata['check_type'] == CheckType.LLM
    assert result.metadata['check_metadata']['cost'] > 0

    candidate = Candidate.from_dict(template)
    check = LLMCheck(
        eval_prompt = "What is the number returned in the question?",
        evaluator=candidate,
        success=lambda x: '42' in x,
    )
    result = check(
        prompt="What is the secret number?",
        response="The secret number is 42.",
    )
    assert isinstance(result, CheckResult)
    assert result.success is True
    assert '42' in result.value
    assert result.metadata['check_type'] == CheckType.LLM
    assert result.metadata['check_metadata']['cost'] > 0

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OPENAI_API_KEY is not set")
def test__ToxicityCheck__openai(openai_candidate_template):  # noqa
    """Test that the template for an OpenAI candidate works."""
    template = deepcopy(openai_candidate_template)
    check = ToxicityCheck(evaluator=Candidate.from_dict(template))
    result = check(response="This is bullshit.")
    assert isinstance(result, CheckResult)
    assert result.success is False
    assert 'true' in result.value.lower()
    assert result.metadata['check_type'] == CheckType.TOXICITY
    assert result.metadata['check_metadata']['cost'] > 0

    check = ToxicityCheck(evaluator=Candidate.from_dict(template))
    result = check(response="This is great.")
    assert isinstance(result, CheckResult)
    assert result.success is True
    assert 'false' in result.value.lower()
    assert result.metadata['check_type'] == CheckType.TOXICITY
    assert result.metadata['check_metadata']['cost'] > 0
