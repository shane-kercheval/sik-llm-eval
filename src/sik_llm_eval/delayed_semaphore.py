"""Provides the DelayedSemaphore class for rate-limited concurrency control."""

import asyncio

class DelayedSemaphore:
    """
    A semaphore that adds delay after a specified number of acquisitions.
    This is useful for rate limiting API calls by introducing delays between batches.
    """

    def __init__(self, value: int, batch_delay: float = 0.0):
        """
        Initialize the DelayedSemaphore.

        Args:
            value: The maximum number of concurrent acquisitions (semaphore value)
            batch_delay: The delay in seconds to add after each batch of acquisitions

        Raises:
            ValueError: If value <= 0 or batch_delay < 0
        """
        if value <= 0:
            raise ValueError("Semaphore value must be greater than 0")
        if batch_delay < 0:
            raise ValueError("Batch delay must be non-negative")

        self.semaphore = asyncio.Semaphore(value)
        self.batch_delay = batch_delay
        self.counter = 0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire the semaphore, adding delay if threshold reached.
        The delay is added after every batch_size acquisitions.
        """
        await self.semaphore.acquire()

        # Only add delay if we have a batch delay configured
        if self.batch_delay > 0:
            async with self.lock:
                self.counter += 1
                if self.counter >= self.semaphore._value:
                    # We've reached a full batch, add delay and reset counter
                    self.counter = 0
                    await asyncio.sleep(self.batch_delay)

    def release(self) -> None:
        """Release the semaphore."""
        self.semaphore.release()

    async def __aenter__(self):
        """Context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        """Context manager exit."""
        self.release()
