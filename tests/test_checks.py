"""TODO: document."""
import pytest
from llm_evals.checks import CheckRegistery, Check, CheckResult, CheckType, PassFailResult


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


@pytest.fixture()
def registry() -> CheckRegistery:
    """Fixture to provide a fresh instance of TestRegistry for each test."""
    return CheckRegistery()


def test__register_check__success__str__ensure_creation(registry):  # noqa
    """Test successful registration of a check."""
    registry.register('FakeCheck', FakeCheck)
    assert 'FakeCheck' in registry

    instance = registry.create_instance('FakeCheck')
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance('FakeCheck', params=None)
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance('FakeCheck', params={})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {}

    instance = registry.create_instance('FakeCheck', params={'metadata': {'foo': 'bar'}})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {'foo': 'bar'}

def test__register_check__success__CheckType__ensure_creation(registry):  # noqa
    """Test successful registration of a check."""
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

    instance = registry.create_instance(CheckType.MATCH_CONTAINS, params={'metadata': {'foo': 'bar'}})
    assert isinstance(instance, FakeCheck)
    assert instance.metadata == {'foo': 'bar'}

def test__register_check__success__ensure_creation__with_required_params(registry):  # noqa
    """Test successful registration of a check."""
    registry.register('FakeParamCheck', FakeParamCheck)
    assert 'FakeParamCheck' in registry.registered()

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

def test__register_check__duplicate__str__(registry):  # noqa
    """Test registering a check with a duplicate name raises an error."""
    registry.register('FakeCheck', FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register('FakeCheck', FakeCheck)

def test__register_check__duplicate__CheckType_and_str(registry):  # noqa
    """Test registering a check with a duplicate name raises an error."""
    registry.register(CheckType.MATCH_EXACT, FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register(CheckType.MATCH_EXACT, FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register(CheckType.MATCH_EXACT.name, FakeCheck)

def test__register_check__duplicate__str_and_CheckType(registry):  # noqa
    """Test registering a check with a duplicate name raises an error."""
    registry.register(CheckType.MATCH_EXACT.name, FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register(CheckType.MATCH_EXACT, FakeCheck)
    with pytest.raises(ValueError):  # noqa: PT011
        registry.register(CheckType.MATCH_EXACT.name, FakeCheck)
