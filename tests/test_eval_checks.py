"""TODO: document."""
import pytest
from llm_evals.checks import CheckRegistery, Check


class FakeTest(Check):
    """Mock test for testing."""

    def __init__(self, foo: str | None = None):
        self.foo = foo
        self._foo = foo
        self._responses = None

    def __call__(self, responses: list[str]) -> None:  # noqa
        self._responses = responses


@pytest.fixture()
def registry() -> CheckRegistery:
    """Fixture to provide a fresh instance of TestRegistry for each test."""
    return CheckRegistery()


def test_register_model_success(registry):  # noqa
    """Test successful registration of a model."""
    registry.register('FakeTest', FakeTest)
    assert 'FakeTest' in registry.registered()


def test_register_model_duplicate(registry):  # noqa
    """Test registering a model with a duplicate name raises an error."""
    registry.register('FakeTest', FakeTest)
    with pytest.raises(ValueError):  # noqa
        registry.register('FakeTest', FakeTest)


# def test_create_instance_success(registry):
#     """Test successful creation of a model instance."""
#     registry.register('FakeTest', FakeTest)
#     instance = registry.create_instance(
#         'FakeTest',
#         data_dimensions=DIMENSIONS,
#         in_channels=CHANNELS,
#         output_size=OUTPUT_SIZE,
#         model_parameters={},
#     )
#     assert isinstance(instance, FakeTest)


# # test with kwargs
# def test_create_instance_kwargs(registry):
#     """Test that kwargs are passed to model constructor."""
#     registry.register('FakeTest', FakeTest)
#     instance = registry.create_instance(
#         'FakeTest',
#         data_dimensions=DIMENSIONS,
#         in_channels=CHANNELS,
#         output_size=OUTPUT_SIZE,
#         model_parameters={'foo': 'bar'},
#     )
#     assert instance.foo == 'bar'


# def test_create_instance_unregistered(registry):
#     """Test error when creating an instance of an unregistered model."""
#     with pytest.raises(ValueError):
#         registry.create_instance(
#             'UnregisteredModel',
#             data_dimensions=DIMENSIONS,
#             in_channels=CHANNELS,
#             output_size=OUTPUT_SIZE,
#             model_parameters={},
#         )


def test_registered(registry):  # noqa
    """Test listing all registered models."""
    registry.register('Model1', FakeTest)
    registry.register('Model2', FakeTest)
    models = registry.registered()
    assert 'Model1' in models
    assert 'Model2' in models
    assert len(models) == 2
