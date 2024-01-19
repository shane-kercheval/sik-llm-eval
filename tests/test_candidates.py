

def test_register_candidate_success(registry):  # noqa
    """Test successful registration of a candidate."""
    registry.register('FakeTest', FakeTest)
    assert 'FakeTest' in registry.registered()