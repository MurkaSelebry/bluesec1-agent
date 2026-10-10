"""Monotonic wall-clock budgets shared by decisions and their caller."""

import math
import time
from collections.abc import Callable


class BudgetExceededError(TimeoutError):
    """No further work may start within the current task budget."""


class Budget:
    def __init__(self, max_seconds: float, *, clock: Callable[[], float] | None = None) -> None:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        self._clock = clock or time.monotonic
        self._deadline = self._clock() + max_seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self._deadline - self._clock())

    def with_reserve(self, seconds: float) -> "Budget":
        """Share this deadline/clock while preserving time for a later phase."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("reserve seconds must be finite and nonnegative")
        child = object.__new__(Budget)
        child._clock = self._clock
        child._deadline = self._deadline - seconds
        return child

    def check(self) -> None:
        if self.remaining_seconds() <= 0:
            raise BudgetExceededError("The task deadline was exceeded.")
