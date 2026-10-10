import pytest


def test_budget_uses_monotonic_time_and_never_returns_negative(monkeypatch):
    from bluesec1_agent import budget as module

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    budget = module.Budget(3)
    assert budget.remaining_seconds() == 3
    now[0] = 102.5
    assert budget.remaining_seconds() == 0.5
    now[0] = 104
    assert budget.remaining_seconds() == 0
    with pytest.raises(module.BudgetExceededError):
        budget.check()


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_budget_rejects_unbounded_or_invalid_deadlines(seconds):
    from bluesec1_agent.budget import Budget

    with pytest.raises(ValueError):
        Budget(seconds)


def test_reserved_budget_uses_same_clock_and_leaves_parent_time():
    from bluesec1_agent.budget import Budget, BudgetExceededError

    now = [10.0]
    budget = Budget(300, clock=lambda: now[0])
    investigation = budget.with_reserve(90)
    now[0] += 210
    with pytest.raises(BudgetExceededError):
        investigation.check()
    assert budget.remaining_seconds() == 90
    final_decision = budget.with_reserve(15)
    now[0] += 75
    with pytest.raises(BudgetExceededError):
        final_decision.check()
    assert budget.remaining_seconds() == 15
