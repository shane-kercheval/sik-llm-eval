"""TODO: document."""
from pydantic import ValidationError
import pytest
from llm_evals.checks import (
    # CHECK_REGISTRY,
    CheckResultsType,
    ContainsCheck,
    MatchCheck,
    RegexCheck,
    # register_check,
    # CheckRegistry,
    Check,
    CheckResult,
    CheckType,
    PassFailResult,
    ScoreResult,
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

def test__MatchExactCheck__has_check_type():  # noqa
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

def test__MatchExactCheck():  # noqa
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

def test__MatchContainsCheck__has_check_type():  # noqa
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

def test__MatchContainsCheck():  # noqa
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

def test__MatchRegexCheck__has_check_type():  # noqa
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

def test__MatchRegexCheck():  # noqa
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
