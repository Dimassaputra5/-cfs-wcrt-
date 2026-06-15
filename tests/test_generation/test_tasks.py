"""Tests for synthetic task generation (UUniFast).

Tests cover:
    - _uunifast utilisation distribution
    - generate_task_set() with various parameters
    - generate_task_sets() bulk generation
    - compute_hyperperiod() edge cases
"""

from __future__ import annotations

import math
import random

import pytest

from cfs_wcrt import TaskGenConfig, generate_task_set, generate_task_sets
from cfs_wcrt.generation.tasks import _uunifast, compute_hyperperiod


# ── _uunifast (internal) ─────────────────────────────────────────────────────

class TestUUniFast:
    """Utilisation generation algorithm."""

    def test_sum_to_total(self) -> None:
        """Generated values must sum to the target utilisation."""
        for _ in range(20):
            utils = _uunifast(5, 0.8)
            assert abs(sum(utils) - 0.8) < 1e-9

    def test_all_positive(self) -> None:
        utils = _uunifast(5, 0.8)
        assert all(u > 0 for u in utils)

    def test_count_matches_n(self) -> None:
        assert len(_uunifast(3, 0.5)) == 3
        assert len(_uunifast(1, 0.9)) == 1

    def test_large_n(self) -> None:
        utils = _uunifast(50, 0.5)
        assert abs(sum(utils) - 0.5) < 1e-6

    def test_invalid_n_zero(self) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            _uunifast(0, 0.5)

    def test_invalid_n_negative(self) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            _uunifast(-1, 0.5)

    def test_invalid_util_zero(self) -> None:
        with pytest.raises(ValueError, match="total_util must be positive"):
            _uunifast(3, 0)

    def test_invalid_util_negative(self) -> None:
        with pytest.raises(ValueError, match="total_util must be positive"):
            _uunifast(3, -0.1)

    def test_reproducible_with_seed(self) -> None:
        random.seed(12345)
        a = _uunifast(4, 0.7)
        random.seed(12345)
        b = _uunifast(4, 0.7)
        assert a == b

    def test_single_task(self) -> None:
        """With n=1, the single value must equal total_util."""
        utils = _uunifast(1, 0.42)
        assert abs(utils[0] - 0.42) < 1e-12


# ── generate_task_set ───────────────────────────────────────────────────────

class TestGenerateTaskSet:
    """Full task set generation."""

    def test_basic_generation(self) -> None:
        tasks = generate_task_set(num_tasks=5, utilization=0.5)
        assert len(tasks) == 5
        for t in tasks:
            assert t.task_id >= 0
            assert t.wcet > 0
            assert t.period >= 30.0

    def test_utilization_input_output(self) -> None:
        """Total utilisation should approximately match input."""
        tasks = generate_task_set(num_tasks=8, utilization=0.6)
        actual = sum(t.wcet / t.period for t in tasks)
        assert abs(actual - 0.6) < 0.05  # within 5%

    def test_custom_config(self) -> None:
        config = TaskGenConfig(min_period=100.0, max_period=500.0, default_nice=-5)
        tasks = generate_task_set(num_tasks=3, utilization=0.5, config=config)
        assert all(t.nice == -5 for t in tasks)
        assert all(100.0 <= t.period <= 500.0 for t in tasks)

    def test_start_id(self) -> None:
        tasks = generate_task_set(num_tasks=3, utilization=0.3, start_id=10)
        assert [t.task_id for t in tasks] == [10, 11, 12]

    def test_invalid_num_tasks_zero(self) -> None:
        with pytest.raises(ValueError, match="num_tasks"):
            generate_task_set(num_tasks=0, utilization=0.5)

    def test_invalid_util_zero(self) -> None:
        with pytest.raises(ValueError, match="utilization"):
            generate_task_set(num_tasks=3, utilization=0)

    def test_invalid_util_above_1(self) -> None:
        with pytest.raises(ValueError, match="utilization"):
            generate_task_set(num_tasks=3, utilization=1.5)

    def test_wcet_at_least_1(self) -> None:
        tasks = generate_task_set(num_tasks=4, utilization=0.1)
        assert all(t.wcet >= 1.0 for t in tasks)


# ── generate_task_sets ───────────────────────────────────────────────────────

class TestGenerateTaskSets:
    """Bulk generation."""

    def test_correct_count(self) -> None:
        sets = generate_task_sets(num_sets=5, num_tasks=4, utilization=0.5)
        assert len(sets) == 5
        assert all(len(s) == 4 for s in sets)

    def test_unique_ids_across_sets(self) -> None:
        """Task IDs should be globally unique via start_id."""
        sets = generate_task_sets(num_sets=3, num_tasks=2, utilization=0.4)
        all_ids = [t.task_id for s in sets for t in s]
        assert len(all_ids) == len(set(all_ids))  # all unique

    def test_single_set(self) -> None:
        sets = generate_task_sets(num_sets=1, num_tasks=2, utilization=0.3)
        assert len(sets) == 1


# ── compute_hyperperiod ──────────────────────────────────────────────────────

class TestComputeHyperperiod:
    """LCM of task periods."""

    def test_single_period(self) -> None:
        assert compute_hyperperiod([100.0]) == 100

    def test_two_coprime(self) -> None:
        assert compute_hyperperiod([100.0, 37.0]) == 3700  # lcm(100, 37) = 3700

    def test_two_with_common_factor(self) -> None:
        assert compute_hyperperiod([100.0, 150.0]) == 300  # lcm(100, 150) = 300

    def test_three_periods(self) -> None:
        assert compute_hyperperiod([100.0, 150.0, 200.0]) == 600

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="periods list must not be empty"):
            compute_hyperperiod([])

    def test_rounded_periods(self) -> None:
        """Floating-point periods are rounded before LCM."""
        # round(100.4)=100, round(150.6)=151, lcm(100,151)=15100
        assert compute_hyperperiod([100.4, 150.6]) == 15100

    def test_with_task_set(self) -> None:
        tasks = generate_task_set(num_tasks=3, utilization=0.5)
        periods = [t.period for t in tasks]
        hp = compute_hyperperiod(periods)
        assert hp > 0
        for p in periods:
            assert hp % int(round(p)) == 0  # every period divides hyperperiod
