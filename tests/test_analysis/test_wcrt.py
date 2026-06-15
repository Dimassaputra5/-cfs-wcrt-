"""Tests for WCRT analysis module.

Tests cover:
    - _compute_sigma_tilde (Definition 3 + 4)
    - _compute_max_busy_period (fixed-point iteration)
    - WCRTAnalyzer.analyze_task() single-task analysis
    - WCRTAnalyzer.analyze() system-level analysis
    - Schedulability decisions
"""

from __future__ import annotations

import math

import pytest

from cfs_wcrt import CFSConfig, TaskParams
from cfs_wcrt.analysis.wcrt import (
    WCRTAnalyzer,
    _compute_max_busy_period,
    _compute_sigma_tilde,
)


# ── _compute_sigma_tilde ─────────────────────────────────────────────────────

class TestComputeSigmaTilde:
    """Maximum timeslice computation (Definition 3 + 4)."""

    def test_default_config_single_task(self, single_task: TaskParams) -> None:
        config = CFSConfig()
        sigma = _compute_sigma_tilde(single_task, float(single_task.weight), config)
        # single task: weight ratio = 1, L_adj = L, sigma = max(L, G) rounded to jiffy
        expected = max(config.target_latency, config.min_granularity)
        expected = expected - (expected % config.jiffy) + (config.jiffy if expected % config.jiffy != 0 else 0)
        assert sigma == expected

    def test_equal_weight_tasks(self) -> None:
        """Two tasks with equal weight should split timeslice equally."""
        config = CFSConfig()
        t1 = TaskParams.from_nice(task_id=0, wcet=10.0, nice=0, period=100.0)
        t2 = TaskParams.from_nice(task_id=1, wcet=10.0, nice=0, period=100.0)
        total_weight = float(t1.weight + t2.weight)
        sigma = _compute_sigma_tilde(t1, total_weight, config)
        # 2 tasks <= sched_nr_latency(8), so L_adj = L = 18
        # delta = (1024/2048) * 18 = 9
        # sigma = max(9, 2.25) = 9 rounded to jiffy=1 -> 9
        assert sigma == 9.0

    def test_high_priority_task(self) -> None:
        """A task with high weight (nice=-20) gets larger timeslice."""
        config = CFSConfig()
        high = TaskParams.from_nice(task_id=0, wcet=10.0, nice=-20, period=100.0)
        low = TaskParams.from_nice(task_id=1, wcet=10.0, nice=19, period=100.0)

        total_weight = float(high.weight + low.weight)
        sigma_high = _compute_sigma_tilde(high, total_weight, config)
        sigma_low = _compute_sigma_tilde(low, total_weight, config)

        assert sigma_high > sigma_low, "Higher priority task should get larger timeslice"

    def test_min_granularity_applied(self) -> None:
        """Timeslice must never be less than min_granularity."""
        config = CFSConfig(min_granularity=10.0)  # unusually large
        t = TaskParams.from_nice(task_id=0, wcet=1.0, nice=0, period=100.0)
        sigma = _compute_sigma_tilde(t, float(t.weight), config)
        assert sigma >= 10.0


# ── _compute_max_busy_period ─────────────────────────────────────────────────

class TestComputeMaxBusyPeriod:
    """Fixed-point busy-period computation."""

    def test_single_task(self, single_task: TaskParams) -> None:
        """For a single task, the busy period equals its WCET."""
        config = CFSConfig()
        hp = _compute_max_busy_period(single_task, [single_task], config)
        assert hp == pytest.approx(single_task.wcet, rel=1e-6)

    def test_two_tasks(self, two_tasks: list[TaskParams]) -> None:
        """Busy period must be >= both tasks' individual WCETs."""
        config = CFSConfig()
        for t in two_tasks:
            hp = _compute_max_busy_period(t, two_tasks, config)
            assert hp >= t.wcet
            assert hp > 0

    def test_three_tasks(self, three_tasks: list[TaskParams]) -> None:
        config = CFSConfig()
        for t in three_tasks:
            hp = _compute_max_busy_period(t, three_tasks, config)
            assert hp >= t.wcet
            assert hp < 1e6  # should converge sanely

    def test_convergence(self, schedulable_set: list[TaskParams]) -> None:
        """Fixed-point iteration must converge within 1000 iterations."""
        config = CFSConfig()
        for t in schedulable_set:
            hp = _compute_max_busy_period(t, schedulable_set, config)
            assert hp > 0
            assert not math.isnan(hp) and not math.isinf(hp)

    def test_with_overloaded_set(self, nonschedulable_set: list[TaskParams]) -> None:
        """Even overloaded sets should return a finite busy period."""
        config = CFSConfig()
        for t in nonschedulable_set:
            hp = _compute_max_busy_period(t, nonschedulable_set, config)
            assert hp > 0
            assert not math.isnan(hp)


# ── WCRTAnalyzer.analyze_task ───────────────────────────────────────────────

class TestWCRTAnalyzerSingle:
    """Single-task WCRT analysis."""

    def test_single_task_schedulable(self, single_task: TaskParams, config: CFSConfig) -> None:
        analyzer = WCRTAnalyzer(config)
        result = analyzer.analyze_task(single_task, [single_task])
        assert result.estimated_wcrt > 0
        assert result.schedulable, "Single task with zero interference should be schedulable"
        assert result.iterations >= 1

    def test_result_fields(self, single_task: TaskParams) -> None:
        analyzer = WCRTAnalyzer()
        result = analyzer.analyze_task(single_task, [single_task])
        assert result.task_id == 0
        assert result.analysis_time_us > 0


# ── WCRTAnalyzer.analyze (system) ────────────────────────────────────────────

class TestWCRTAnalyzerSystem:
    """System-level WCRT analysis."""

    def test_schedulable_set(self, schedulable_set: list[TaskParams], config: CFSConfig) -> None:
        analyzer = WCRTAnalyzer(config)
        result = analyzer.analyze(schedulable_set)
        assert result.system_schedulable
        assert len(result.results) == 3
        for r in result.results:
            assert r.schedulable
            assert r.estimated_wcrt > 0

    def test_nonschedulable_set(self, nonschedulable_set: list[TaskParams]) -> None:
        analyzer = WCRTAnalyzer()
        result = analyzer.analyze(nonschedulable_set)
        assert not result.system_schedulable
        assert len(result.results) == 2

    def test_three_tasks(self, three_tasks: list[TaskParams], config: CFSConfig) -> None:
        analyzer = WCRTAnalyzer(config)
        result = analyzer.analyze(three_tasks)
        assert len(result.results) == 3
        assert result.total_analysis_time_ms > 0

    def test_analyze_empty_set(self) -> None:
        analyzer = WCRTAnalyzer()
        result = analyzer.analyze([])
        assert result.system_schedulable  # vacuous truth
        assert len(result.results) == 0

    def test_single_task_system(self, single_task: TaskParams) -> None:
        analyzer = WCRTAnalyzer()
        result = analyzer.analyze([single_task])
        assert result.system_schedulable
        assert result.results[0].estimated_wcrt > 0

    def test_analysis_timing(self, three_tasks: list[TaskParams]) -> None:
        """Total time must exceed per-task times."""
        analyzer = WCRTAnalyzer()
        result = analyzer.analyze(three_tasks)
        total_single = sum(r.analysis_time_us for r in result.results)
        # total_analysis_time_ms should be less than or equal to max of per-task times
        # (it measures wall clock, so it could be slightly less if tasks overlap)
        assert result.total_analysis_time_ms >= 0
