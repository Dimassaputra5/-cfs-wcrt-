"""Tests for core data models (CFSConfig, TaskParams, SchedulingEvent, etc.).

Tests cover:
    - CFSConfig validation (positive values, num_cores >= 1)
    - TaskParams validation (bounds, deadline <= period)
    - TaskParams.from_nice() factory
    - SchedPolicy enum values
    - SchedulingEvent dataclass
    - TaskResult and ExperimentResult dataclasses
"""

from __future__ import annotations

import pytest

from cfs_wcrt import (
    CFSConfig,
    ExperimentResult,
    SchedPolicy,
    SchedulingEvent,
    TaskParams,
    TaskResult,
)


# ── CFSConfig ────────────────────────────────────────────────────────────────

class TestCFSConfig:
    """Validation of CFS configuration parameters."""

    def test_default_config(self) -> None:
        cfg = CFSConfig()
        assert cfg.target_latency == 18.0
        assert cfg.min_granularity == 2.25
        assert cfg.jiffy == 1.0
        assert cfg.sched_nr_latency == 8
        assert cfg.num_cores == 1

    def test_custom_config(self) -> None:
        cfg = CFSConfig(target_latency=24.0, min_granularity=3.0, jiffy=4.0, num_cores=4)
        assert cfg.target_latency == 24.0
        assert cfg.num_cores == 4

    def test_invalid_target_latency_zero(self) -> None:
        with pytest.raises(ValueError, match="target_latency"):
            CFSConfig(target_latency=0)

    def test_invalid_target_latency_negative(self) -> None:
        with pytest.raises(ValueError, match="target_latency"):
            CFSConfig(target_latency=-1)

    def test_invalid_min_granularity_zero(self) -> None:
        with pytest.raises(ValueError, match="min_granularity"):
            CFSConfig(min_granularity=0)

    def test_invalid_jiffy_zero(self) -> None:
        with pytest.raises(ValueError, match="jiffy"):
            CFSConfig(jiffy=0)

    def test_invalid_sched_nr_latency_zero(self) -> None:
        with pytest.raises(ValueError, match="sched_nr_latency"):
            CFSConfig(sched_nr_latency=0)

    def test_invalid_num_cores_zero(self) -> None:
        with pytest.raises(ValueError, match="num_cores"):
            CFSConfig(num_cores=0)

    def test_frozen_dataclass(self) -> None:
        cfg = CFSConfig()
        with pytest.raises(AttributeError):
            cfg.target_latency = 99.0  # type: ignore[misc]

    @pytest.mark.parametrize("cores", [1, 2, 4, 8, 16])
    def test_valid_multi_core(self, cores: int) -> None:
        cfg = CFSConfig(num_cores=cores)
        assert cfg.num_cores == cores


# ── TaskParams ───────────────────────────────────────────────────────────────

class TestTaskParams:
    """Validation of task parameter model."""

    def test_valid_task(self) -> None:
        t = TaskParams(task_id=0, wcet=10.0, weight=1024, nice=0, deadline=100.0, period=100.0)
        assert t.task_id == 0
        assert t.wcet == 10.0
        assert t.weight == 1024

    def test_invalid_task_id_negative(self) -> None:
        with pytest.raises(ValueError, match="task_id"):
            TaskParams(task_id=-1, wcet=10.0, weight=1024, nice=0, deadline=100.0, period=100.0)

    def test_invalid_wcet_zero(self) -> None:
        with pytest.raises(ValueError, match="wcet"):
            TaskParams(task_id=0, wcet=0, weight=1024, nice=0, deadline=100.0, period=100.0)

    def test_invalid_weight_zero(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            TaskParams(task_id=0, wcet=10.0, weight=0, nice=0, deadline=100.0, period=100.0)

    def test_invalid_nice_too_low(self) -> None:
        with pytest.raises(ValueError, match="nice"):
            TaskParams(task_id=0, wcet=10.0, weight=1024, nice=-21, deadline=100.0, period=100.0)

    def test_invalid_nice_too_high(self) -> None:
        with pytest.raises(ValueError, match="nice"):
            TaskParams(task_id=0, wcet=10.0, weight=1024, nice=20, deadline=100.0, period=100.0)

    def test_invalid_deadline_zero(self) -> None:
        with pytest.raises(ValueError, match="deadline"):
            TaskParams(task_id=0, wcet=10.0, weight=1024, nice=0, deadline=0, period=100.0)

    def test_invalid_period_zero(self) -> None:
        with pytest.raises(ValueError, match="period"):
            TaskParams(task_id=0, wcet=10.0, weight=1024, nice=0, deadline=100.0, period=0)

    def test_deadline_exceeds_period(self) -> None:
        with pytest.raises(ValueError, match="deadline.*period"):
            TaskParams(task_id=0, wcet=10.0, weight=1024, nice=0, deadline=150.0, period=100.0)

    def test_deadline_equals_period(self) -> None:
        t = TaskParams(task_id=0, wcet=10.0, weight=1024, nice=0, deadline=100.0, period=100.0)
        assert t.deadline == t.period

    def test_frozen_dataclass(self) -> None:
        t = TaskParams(task_id=0, wcet=10.0, weight=1024, nice=0, deadline=100.0, period=100.0)
        with pytest.raises(AttributeError):
            t.wcet = 99.0  # type: ignore[misc]


class TestTaskParamsFromNice:
    """Factory method from_nice()."""

    def test_from_nice_default_deadline(self) -> None:
        t = TaskParams.from_nice(task_id=0, wcet=10.0, nice=0, period=100.0)
        assert t.deadline == 100.0
        assert t.weight == 1024

    def test_from_nice_custom_deadline(self) -> None:
        t = TaskParams.from_nice(task_id=0, wcet=10.0, nice=0, period=100.0, deadline=80.0)
        assert t.deadline == 80.0

    def test_from_nice_nonzero_weight(self) -> None:
        """Nice -5 should map to weight 3121."""
        t = TaskParams.from_nice(task_id=0, wcet=10.0, nice=-5, period=100.0)
        assert t.weight == 3121


# ── SchedPolicy ──────────────────────────────────────────────────────────────

class TestSchedPolicy:
    """Enum values."""

    def test_values(self) -> None:
        assert SchedPolicy.SCHED_OTHER.value == 1
        assert SchedPolicy.SCHED_FIFO.value == 2
        assert SchedPolicy.SCHED_RR.value == 3
        assert SchedPolicy.SCHED_DEADLINE.value == 4

    def test_distinct_values(self) -> None:
        vals = {m.value for m in SchedPolicy}
        assert len(vals) == 4  # all 4 must be unique


# ── SchedulingEvent ──────────────────────────────────────────────────────────

class TestSchedulingEvent:
    """Immutability and attributes."""

    def test_create(self) -> None:
        ev = SchedulingEvent(time=1.0, task_id=0, vruntime_at_start=100.0, timeslice=8.0)
        assert ev.time == 1.0
        assert ev.task_id == 0
        assert ev.vruntime_at_start == 100.0
        assert ev.timeslice == 8.0

    def test_frozen(self) -> None:
        ev = SchedulingEvent(time=1.0, task_id=0, vruntime_at_start=0.0, timeslice=4.0)
        with pytest.raises(AttributeError):
            ev.time = 99.0  # type: ignore[misc]


# ── TaskResult ───────────────────────────────────────────────────────────────

class TestTaskResult:
    """Mutable dataclass with defaults."""

    def test_defaults(self) -> None:
        r = TaskResult(task_id=0)
        assert r.task_id == 0
        assert r.measured_wcrt == 0.0
        assert r.estimated_wcrt == 0.0
        assert not r.schedulable_measured
        assert not r.schedulable_estimated

    def test_custom(self) -> None:
        r = TaskResult(task_id=1, measured_wcrt=5.0, estimated_wcrt=8.0, schedulable_measured=True)
        assert r.measured_wcrt == 5.0
        assert r.estimated_wcrt == 8.0
        assert r.schedulable_measured
        assert not r.schedulable_estimated


# ── ExperimentResult ─────────────────────────────────────────────────────────

class TestExperimentResult:
    """Full experiment result dataclass."""

    def test_defaults(self) -> None:
        r = ExperimentResult(task_params=[], num_tasks=0, utilization=0.0)
        assert r.tp == 0
        assert r.tn == 0
        assert r.fp == 0
        assert r.fn == 0
        assert r.task_results == []
        assert r.analysis_time_ms == 0.0

    def test_full(self, three_tasks: list[TaskParams]) -> None:
        r = ExperimentResult(
            task_params=three_tasks,
            num_tasks=3,
            utilization=0.55,
            tp=10,
            tn=5,
            fp=1,
            fn=0,
            analysis_time_ms=12.5,
        )
        assert r.tp == 10
        assert r.utilization == 0.55
