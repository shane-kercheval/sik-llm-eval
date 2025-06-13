"""Tests for the DelayedSemaphore class."""
import asyncio
import pytest
from time import perf_counter
from sik_llm_eval.delayed_semaphore import DelayedSemaphore


async def dummy_task(semaphore: DelayedSemaphore, task_time: float = 0.1) -> float:
    """Helper function that simulates a task using the semaphore."""
    start_time = perf_counter()
    async with semaphore:
        await asyncio.sleep(task_time)  # Simulate some work
    return perf_counter() - start_time


@pytest.mark.asyncio
async def test_delayed_semaphore_basic_functionality():
    """Test that the semaphore works as a basic semaphore without delays."""
    value = 2
    semaphore = DelayedSemaphore(value, batch_delay=0)

    # Test initial state
    assert semaphore.semaphore._value == value
    assert semaphore.batch_delay == 0

    # Test acquire/release
    await semaphore.acquire()
    assert semaphore.semaphore._value == value - 1

    semaphore.release()
    assert semaphore.semaphore._value == value


@pytest.mark.asyncio
async def test_delayed_semaphore_with_delay():
    """Test that the semaphore properly implements delays between batches."""
    semaphore = DelayedSemaphore(2, batch_delay=0.1)

    # Run multiple tasks that should be grouped into batches
    tasks = [dummy_task(semaphore, 0.05) for _ in range(4)]
    start_time = perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = perf_counter() - start_time

    # First batch of 2 tasks should take ~0.05s
    # Then delay of 0.1s
    # Second batch of 2 tasks should take ~0.05s
    # Total should be at least 0.2s
    assert total_time >= 0.2
    # Each task should individually take at least 0.05s
    assert all(duration >= 0.05 for duration in results)


@pytest.mark.asyncio
async def test_delayed_semaphore_zero_value():
    """Test that the semaphore raises ValueError for zero or negative values."""
    with pytest.raises(ValueError):  # noqa: PT011
        DelayedSemaphore(0)

    with pytest.raises(ValueError):  # noqa: PT011
        DelayedSemaphore(-1)


@pytest.mark.asyncio
async def test_delayed_semaphore_negative_delay():
    """Test that the semaphore raises ValueError for negative delays."""
    with pytest.raises(ValueError):  # noqa: PT011
        DelayedSemaphore(1, batch_delay=-1)


@pytest.mark.asyncio
async def test_delayed_semaphore_context_manager():
    """Test that the semaphore works properly as a context manager."""
    semaphore = DelayedSemaphore(1, batch_delay=0)

    # Test normal exit
    async with semaphore:
        assert semaphore.semaphore._value == 0
    assert semaphore.semaphore._value == 1

    # Test exit with exception
    with pytest.raises(RuntimeError):  # noqa: PT012
        async with semaphore:
            assert semaphore.semaphore._value == 0
            raise RuntimeError("Test exception")
    assert semaphore.semaphore._value == 1  # Should be released even after exception


@pytest.mark.asyncio
async def test_delayed_semaphore_multiple_batches():
    """Test that multiple batches of tasks are properly delayed."""
    batch_size = 3
    delay = 0.1
    semaphore = DelayedSemaphore(batch_size, batch_delay=delay)

    # Create 3 batches of tasks
    num_tasks = batch_size * 3
    task_duration = 0.05

    tasks = [dummy_task(semaphore, task_duration) for _ in range(num_tasks)]
    start_time = perf_counter()
    await asyncio.gather(*tasks)
    total_time = perf_counter() - start_time

    # Expected time:
    # Batch 1: task_duration
    # Delay: delay
    # Batch 2: task_duration
    # Delay: delay
    # Batch 3: task_duration
    min_expected_time = task_duration * 3 + delay * 2

    assert total_time >= min_expected_time


@pytest.mark.asyncio
async def test_delayed_semaphore_bounded():
    """Test that the semaphore properly bounds concurrent operations."""
    bound = 3
    semaphore = DelayedSemaphore(bound, batch_delay=0)

    # Track maximum concurrent operations
    max_concurrent = 0
    current_concurrent = 0

    async def tracked_task() -> None:
        nonlocal max_concurrent, current_concurrent
        async with semaphore:
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.1)  # Simulate work
            current_concurrent -= 1

    # Run more tasks than the bound
    tasks = [tracked_task() for _ in range(bound * 2)]
    await asyncio.gather(*tasks)

    assert max_concurrent == bound


@pytest.mark.asyncio
async def test_delayed_semaphore_fair_scheduling():
    """Test that tasks are processed in the order they are submitted."""
    semaphore = DelayedSemaphore(1, batch_delay=0)
    order = []

    async def ordered_task(task_id: int) -> None:
        async with semaphore:
            await asyncio.sleep(0.05)  # Small delay to ensure ordering matters
            order.append(task_id)

    # Create and gather tasks
    tasks = [ordered_task(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # Check that tasks were processed in order
    assert order == list(range(5))


@pytest.mark.asyncio
async def test_delayed_semaphore_stress():
    """Stress test the semaphore with many concurrent operations."""
    semaphore = DelayedSemaphore(5, batch_delay=0.01)
    num_tasks = 50

    async def stress_task() -> None:
        async with semaphore:
            await asyncio.sleep(0.01)  # Small work simulation

    # Run many concurrent tasks
    tasks = [stress_task() for _ in range(num_tasks)]
    await asyncio.gather(*tasks)

    # Verify final state
    assert semaphore.semaphore._value == 5


@pytest.mark.asyncio
async def test_delayed_semaphore_no_delay_no_counter():
    """Test that counter operations are skipped when batch_delay is 0."""
    semaphore = DelayedSemaphore(2, batch_delay=0)

    await semaphore.acquire()
    assert semaphore.counter == 0  # Counter should not increment when batch_delay is 0

    await semaphore.acquire()
    assert semaphore.counter == 0

    semaphore.release()
    semaphore.release()
