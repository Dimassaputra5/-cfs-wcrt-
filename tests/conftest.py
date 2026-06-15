"""Shared pytest fixtures for CFS-WCRT tests."""

from __future__ import annotations

import random

import pytest

from cfs_wcrt import CFSConfig, TaskParams


# ── Seed for reproducibility ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _seed_random() -> None:
    """Seed random for reproducible tests."""
    random.seed(42)


# ── Default config ───────────────────────────────────────────────────────────

@pytest.fixture
def config() -> CFSConfig:
    """Default CFS config matching kernel 5.15 defaults."""
    return CFSConfig()


# ── Single-task fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def single_task() -> TaskParams:
    """A single task with nice=0, period=100ms, wcet=10ms."""
    return TaskParams.from_nice(
        task_id=0,
        wcet=10.0,
        nice=0,
        period=100.0,
    )


@pytest.fixture
def short_task() -> TaskParams:
    """A task with nice=0, period=30ms, wcet=3ms."""
    return TaskParams.from_nice(
        task_id=0,
        wcet=3.0,
        nice=0,
        period=30.0,
    )


# ── Multi-task fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def two_tasks() -> list[TaskParams]:
    """Two tasks with different periods."""
    return [
        TaskParams.from_nice(task_id=0, wcet=10.0, nice=0, period=100.0),
        TaskParams.from_nice(task_id=1, wcet=20.0, nice=0, period=200.0),
    ]


@pytest.fixture
def three_tasks() -> list[TaskParams]:
    """Three tasks with varying periods (total util ~0.6)."""
    return [
        TaskParams.from_nice(task_id=0, wcet=10.0, nice=0, period=100.0),
        TaskParams.from_nice(task_id=1, wcet=20.0, nice=0, period=150.0),
        TaskParams.from_nice(task_id=2, wcet=30.0, nice=0, period=200.0),
    ]


@pytest.fixture
def schedulable_set() -> list[TaskParams]:
    """A set all schedulable under CFS (low utilisation)."""
    return [
        TaskParams.from_nice(task_id=0, wcet=5.0, nice=0, period=100.0),
        TaskParams.from_nice(task_id=1, wcet=10.0, nice=0, period=200.0),
        TaskParams.from_nice(task_id=2, wcet=15.0, nice=0, period=300.0),
    ]


@pytest.fixture
def nonschedulable_set() -> list[TaskParams]:
    """A set that is NOT schedulable (over-utilised)."""
    return [
        TaskParams.from_nice(task_id=0, wcet=80.0, nice=0, period=100.0),
        TaskParams.from_nice(task_id=1, wcet=90.0, nice=0, period=200.0),
    ]
