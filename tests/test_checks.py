"""TODO: document."""
import pytest
from llm_evals.checks import (
    CHECK_REGISTRY,
    register_check,
    CheckRegistery,
    Check,
    CheckResult,
    CheckType,
    PassFailResult,
    ScoreResult,
)


class FakeCheck(Check):
    """Mock test for testing."""

    def __init__(self, metadata: dict | None = None):
        super().__init__(metadata=metadata)

    def __call__(self, response: str) -> CheckResult:  # noqa: D102
        return PassFailResult(
            check_type=CheckType.PASS_FAIL,
            passed=response is not None,
            metadata=self.metadata,
        )

class FakeParamCheck(Check):
    """Mock test for testing."""

    def __init__(self, required_field: str, metadata: dict | None = None):
        super().__init__(metadata=metadata)
        self.required_field = required_field

    def __call__(self, response: str) -> CheckResult:  # noqa: D102
        return PassFailResult(
            check_type=CheckType.PASS_FAIL,
            passed=response is not None,
            metadata=self.metadata,
        )


def test__register_check__success__str__ensure_creation():  # noqa
    """Test successful registration of a check."""
    registry = CheckRegistery()
    registry.register('FakeCheck', FakeCheck)
    assert 'FakeCheck' in registry

    instance = registry.create_instance('FakeCheck')
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}
    assert instance.type == 'FAKECHECK'

    instance = registry.create_instance('FakeCheck', params=None)
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}
    assert instance.type == 'FAKECHECK'

    instance = registry.create_instance('FakeCheck', params={})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}
    assert instance.type == 'FAKECHECK'

    instance = registry.create_instance('FakeCheck', params={'metadata': {'foo': 'bar'}})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {'foo': 'bar'}
    assert instance.type == 'FAKECHECK'

def test__register_check__success__using_decorator():  # noqa
    """Test successful registration of a check using a decorator."""
    assert 'test' not in CHECK_REGISTRY

    @register_check('test')
    class TestCheck(Check):
        def __call__(self, response: str) -> CheckResult:
            return PassFailResult(value=True, metadata={'response': response})

    assert 'test' in CHECK_REGISTRY
    assert 'TEST' in CHECK_REGISTRY
    obj = CHECK_REGISTRY.create_instance('test')
    assert isinstance(obj, TestCheck)
    assert obj.type == 'TEST'
    assert obj.metadata == {}
    result = obj('foo')
    assert isinstance(result, PassFailResult)
    assert result.success
    assert result.metadata == {'response': 'foo'}
    # We should not be able to register a check with the same type
    with pytest.raises(AssertionError):
        @register_check('TEST')
        class TestCheck2(Check):
            def __call__(self, response: str) -> CheckResult:
                return PassFailResult(value=True, metadata={'response': response})
    # We should not be able to register a check that isn't a Check object
    with pytest.raises(AssertionError):
        @register_check('test2')
        class TestCheck3:
            def __call__(self, response: str) -> CheckResult:
                return PassFailResult(value=True, metadata={'response': response})

def test__register_check__success__CheckType__ensure_creation():  # noqa
    """Test successful registration of a check."""
    registry = CheckRegistery()
    registry.register(CheckType.MATCH_CONTAINS, FakeCheck)
    assert 'MATCH_CONTAINS' in registry
    assert CheckType.MATCH_CONTAINS in registry

    instance = registry.create_instance('MATCH_CONTAINS')
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance('MATCH_CONTAINS', params=None)
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance('MATCH_CONTAINS', params={})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance('MATCH_CONTAINS', params={'metadata': {'foo': 'bar'}})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {'foo': 'bar'}

    instance = registry.create_instance(CheckType.MATCH_CONTAINS)
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance(CheckType.MATCH_CONTAINS, params=None)
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance(CheckType.MATCH_CONTAINS, params={})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance(
        CheckType.MATCH_CONTAINS,
        params={'metadata': {'foo': 'bar'}},
    )
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {'foo': 'bar'}

def test__register_check__success__ensure_creation__with_required_params():  # noqa
    """Test successful registration of a check."""
    registry = CheckRegistery()
    registry.register('FakeParamCheck', FakeParamCheck)
    assert 'FakeParamCheck' in registry

    # if we don't pass the required param, we should get an error
    with pytest.raises(TypeError):
        _ = registry.create_instance('FakeParamCheck')
    with pytest.raises(TypeError):
        _ = registry.create_instance('FakeParamCheck', params=None)
    with pytest.raises(TypeError):
        _ = registry.create_instance('FakeParamCheck', params={})
    with pytest.raises(TypeError):
        _ = registry.create_instance('FakeParamCheck', params={'metadata': {'foo': 'bar'}})

    instance = registry.create_instance('FakeParamCheck', params={'required_field': 'foo'})
    assert isinstance(instance, FakeParamCheck)
    assert instance.metadata == {}
    assert instance.required_field == 'foo'

    instance = registry.create_instance(
        'FakeParamCheck',
        params={'required_field': 'foo', 'metadata': {'foo': 'bar'}},
    )
    assert isinstance(instance, FakeParamCheck)
    assert instance.metadata == {'foo': 'bar'}
    assert instance.required_field == 'foo'

def test__register_check_create_instance__case_insensitive():  # noqa
    """Test that the check name is case insensitive."""
    registry = CheckRegistery()
    registry.register('FakeCheck', FakeCheck)
    assert 'fakecheck' in registry
    assert 'FAKECHECK' in registry
    instance = registry.create_instance('fakecheck', params={'metadata': {'foo': 'bar'}})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {'foo': 'bar'}
    instance = registry.create_instance('FAKECHECK', params={'metadata': {'foo': 'bar'}})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {'foo': 'bar'}

def test__register_check__duplicate__str__():  # noqa
    """Test registering a check with a duplicate name raises an error."""
    registry = CheckRegistery()
    registry.register('FakeCheck', FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register('FakeCheck', FakeCheck)

def test__register_check__duplicate__CheckType_and_str():  # noqa
    """Test registering a check with a duplicate name raises an error."""
    registry = CheckRegistery()
    registry.register(CheckType.MATCH_EXACT, FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register(CheckType.MATCH_EXACT, FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register(CheckType.MATCH_EXACT.name, FakeCheck)

def test__register_check__duplicate__str_and_CheckType():  # noqa
    """Test registering a check with a duplicate name raises an error."""
    registry = CheckRegistery()
    registry.register(CheckType.MATCH_EXACT.name, FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register(CheckType.MATCH_EXACT, FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register(CheckType.MATCH_EXACT.name, FakeCheck)

def test__CheckType():  # noqa
    assert CheckType.MATCH_EXACT.name == 'MATCH_EXACT'
    assert CheckType.to_enum('MATCH_EXACT') == CheckType.MATCH_EXACT
    assert CheckType.to_enum('match_exact') == CheckType.MATCH_EXACT
    assert CheckType.MATCH_EXACT == 'MATCH_EXACT'
    assert CheckType.MATCH_EXACT == 'match_exact'
    with pytest.raises(ValueError):  # noqa: PT011
        CheckType.to_enum('foo')

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
