"""Monotonic wall-clock budgets shared by decisions and their caller."""

import math
import time


class BudgetExceededError(TimeoutError):
    """No further work may start within the current task budget."""


class Budget:
    def __init__(self, max_seconds: float) -> None:
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        self._deadline = time.monotonic() + max_seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def check(self) -> None:
        if self.remaining_seconds() <= 0:
            raise BudgetExceededError("The task deadline was exceeded.")
