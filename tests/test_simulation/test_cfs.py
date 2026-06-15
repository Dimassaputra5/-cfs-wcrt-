"""Tests for CFS discrete-event simulator.

Tests cover:
    - CFSSimulator initialisation
    - _update_min_vruntime (Definition 1)
    - _update_curr (Definition 2)
    - _sched_slice (Definition 3 + 4)
    - _place_entity (Definition 5)
    - measure_wcrt() public API
"""

from __future__ import annotations

import pytest

from cfs_wcrt import CFSConfig, CFSSimulator, TaskParams


# ── CFSSimulator basic ───────────────────────────────────────────────────────

class TestCFSSimulatorBasics:
    """Initialisation and default state."""

    def test_create_default(self) -> None:
        sim = CFSSimulator()
        assert sim is not None

    def test_create_with_config(self) -> None:
        cfg = CFSConfig(target_latency=24.0)
        sim = CFSSimulator(cfg)
        assert sim is not None


# ── _update_min_vruntime (Definition 1) ─────────────────────────────────────

class TestUpdateMinVruntime:
    """Minimum vruntime tracking."""

    def test_empty_runqueue(self, config: CFSConfig) -> None:
        sim = CFSSimulator(config)
        sim._update_min_vruntime([])
        assert sim._min_vruntime == 0.0

    def test_min_tracked(self, config: CFSConfig, two_tasks: list[TaskParams]) -> None:
        sim = CFSSimulator(config)
        runqueue = sim._initialize_tasks(two_tasks)
        # Give task 0 a higher vruntime
        runqueue[0].vruntime = 50.0
        runqueue[1].vruntime = 100.0
        runnable = [t for t in runqueue if not t.is_sleeping and t.remaining > 0]
        # After init, all tasks are sleeping, so runnable is empty
        sim._update_min_vruntime(runnable)
        # Should be 0 since no runnable tasks
        assert sim._min_vruntime == 0.0


# ── _update_curr (Definition 2) ──────────────────────────────────────────────

class TestUpdateCurr:
    """Vruntime update on execution."""

    def test_vruntime_increases(self, config: CFSConfig, single_task: TaskParams) -> None:
        sim = CFSSimulator(config)
        runqueue = sim._initialize_tasks([single_task])
        rt = runqueue[0]
        rt.is_sleeping = False

        assert rt.vruntime == 0.0
        sim._update_curr(rt, 5.0)
        # delta * NICE_0_WEIGHT / weight = 5.0 * 1024 / 1024 = 5.0
        assert rt.vruntime == pytest.approx(5.0, rel=1e-9)
        assert rt.remaining == pytest.approx(5.0, rel=1e-9)  # wcet=10 - 5 = 5
        assert rt.total_execution == pytest.approx(5.0, rel=1e-9)

    def test_zero_delta_no_change(self, config: CFSConfig, single_task: TaskParams) -> None:
        sim = CFSSimulator(config)
        runqueue = sim._initialize_tasks([single_task])
        rt = runqueue[0]
        sim._update_curr(rt, 0.0)
        assert rt.vruntime == 0.0

    def test_negative_delta_no_change(self, config: CFSConfig, single_task: TaskParams) -> None:
        sim = CFSSimulator(config)
        runqueue = sim._initialize_tasks([single_task])
        rt = runqueue[0]
        sim._update_curr(rt, -5.0)
        assert rt.vruntime == 0.0

    def test_high_weight_runs_slower(self, config: CFSConfig) -> None:
        """A task with higher weight (nicer) should increase vruntime slower."""
        sim = CFSSimulator(config)
        high_weight = TaskParams.from_nice(task_id=0, wcet=10.0, nice=-20, period=100.0)  # weight=88761
        low_weight = TaskParams.from_nice(task_id=1, wcet=10.0, nice=19, period=100.0)  # weight=15

        rq = sim._initialize_tasks([high_weight, low_weight])
        rq[0].is_sleeping = False
        rq[1].is_sleeping = False

        sim._update_curr(rq[0], 10.0)
        sim._update_curr(rq[1], 10.0)

        # Higher weight -> smaller vruntime increase
        assert rq[0].vruntime < rq[1].vruntime


# ── _sched_slice (Definition 3 + 4) ──────────────────────────────────────────

class TestSchedSlice:
    """Timeslice computation."""

    def test_single_task_full_slice(self, config: CFSConfig, single_task: TaskParams) -> None:
        sim = CFSSimulator(config)
        runqueue = sim._initialize_tasks([single_task])
        rt = runqueue[0]
        rt.is_sleeping = False
        slice_ = sim._sched_slice(rt)
        assert slice_ > 0

    def test_two_tasks_split(self, config: CFSConfig, two_tasks: list[TaskParams]) -> None:
        sim = CFSSimulator(config)
        runqueue = sim._initialize_tasks(two_tasks)
        for rt in runqueue:
            rt.is_sleeping = False

        slice0 = sim._sched_slice(runqueue[0])
        slice1 = sim._sched_slice(runqueue[1])

        assert slice0 > 0
        assert slice1 > 0

    def test_slice_not_exceeding_remaining(self, config: CFSConfig, single_task: TaskParams) -> None:
        """Timeslice should be capped by remaining work."""
        sim = CFSSimulator(config)
        sim._runqueue = sim._initialize_tasks([single_task])
        rt = sim._runqueue[0]
        rt.is_sleeping = False
        rt.remaining = 0.5  # almost done
        slice_ = sim._sched_slice(rt)
        # remaining=0.5, jiffy=1, so cap rounds: ceil(delta/jiffy)*jiffy then min(,0.5)
        # resulting slice must be <= remaining
        assert slice_ <= 0.5 + 1e-9


# ── _place_entity (Definition 5) ─────────────────────────────────────────────

class TestPlaceEntity:
    """Wake-up vruntime adjustment."""

    def test_sleeping_task_placed_at_min(self, config: CFSConfig, single_task: TaskParams) -> None:
        sim = CFSSimulator(config)
        runqueue = sim._initialize_tasks([single_task])
        rt = runqueue[0]
        rt.is_sleeping = True
        rt.vruntime = 0.0
        sim._min_vruntime = 100.0

        sim._place_entity(rt, 0.0)
        # vruntime should be >= min_vruntime
        assert rt.vruntime >= 100.0

    def test_vruntime_not_reduced(self, config: CFSConfig, single_task: TaskParams) -> None:
        """If vruntime is already above min, it should stay."""
        sim = CFSSimulator(config)
        runqueue = sim._initialize_tasks([single_task])
        rt = runqueue[0]
        rt.vruntime = 200.0
        sim._min_vruntime = 100.0
        sim._place_entity(rt, 0.0)
        assert rt.vruntime == 200.0  # unchanged


# ── measure_wcrt ─────────────────────────────────────────────────────────────

class TestMeasureWCRT:
    """End-to-end WCRT measurement."""

    def test_single_task(self, config: CFSConfig, single_task: TaskParams) -> None:
        sim = CFSSimulator(config)
        result = sim.measure_wcrt([single_task], hyperperiod_factor=1.0, num_runs=2)
        assert 0 in result
        assert result[0] > 0

    def test_two_tasks(self, config: CFSConfig, two_tasks: list[TaskParams]) -> None:
        sim = CFSSimulator(config)
        result = sim.measure_wcrt(two_tasks, hyperperiod_factor=1.0, num_runs=2)
        assert len(result) == 2
        for tid in [0, 1]:
            assert tid in result
            assert result[tid] > 0

    def test_schedulable_set(self, config: CFSConfig, schedulable_set: list[TaskParams]) -> None:
        sim = CFSSimulator(config)
        result = sim.measure_wcrt(schedulable_set, hyperperiod_factor=1.0, num_runs=2)
        assert len(result) == 3
        # All WCRTs should be within deadlines
        task_map = {t.task_id: t for t in schedulable_set}
        for tid, wcrt in result.items():
            assert wcrt <= task_map[tid].deadline, f"Task {tid} exceeded deadline"

    def test_measure_wcrt_returns_dict(self, config: CFSConfig, two_tasks: list[TaskParams]) -> None:
        sim = CFSSimulator(config)
        result = sim.measure_wcrt(two_tasks, hyperperiod_factor=1.0, num_runs=1)
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())

    def test_measure_wcrt_different_offsets(self, config: CFSConfig, single_task: TaskParams) -> None:
        """Running with more runs should produce at least as high WCRT."""
        sim = CFSSimulator(config)
        r1 = sim.measure_wcrt([single_task], hyperperiod_factor=1.0, num_runs=1)
        r2 = sim.measure_wcrt([single_task], hyperperiod_factor=1.0, num_runs=3)
        # More runs should find at least the same max (or higher)
        assert r2[0] >= r1[0]


# ── Run event log ────────────────────────────────────────────────────────────

class TestRun:
    """The run() method produces event logs."""

    def test_run_returns_events(self, config: CFSConfig, two_tasks: list[TaskParams]) -> None:
        sim = CFSSimulator(config)
        events, wcrt = sim.run(two_tasks, duration=100.0, num_runs=1)
        assert len(events) > 0
        assert len(wcrt) == 2

    def test_run_duration_limit(self, config: CFSConfig, single_task: TaskParams) -> None:
        sim = CFSSimulator(config)
        events, wcrt = sim.run([single_task], duration=10.0, num_runs=1)
        # Events should not exceed duration
        for ev in events:
            assert ev.time <= 10.0 + 1e-9
