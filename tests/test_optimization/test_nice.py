"""Tests for nice-value assignment algorithms.

Tests cover:
    - DeadlineAwareHeuristic (Algorithm 2)
    - GeneticNiceAssignment (Algorithm 3)
    - baseline_assignment() reference
"""

from __future__ import annotations

import pytest

from cfs_wcrt import CFSConfig
from cfs_wcrt.analysis import WCRTAnalyzer
from cfs_wcrt.optimization.nice import (
    DeadlineAwareHeuristic,
    GeneticNiceAssignment,
    NiceAssignmentResult,
    baseline_assignment,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer() -> WCRTAnalyzer:
    """Default analyzer for optimisation tests."""
    return WCRTAnalyzer()


# ── Baseline ─────────────────────────────────────────────────────────────────

class TestBaselineAssignment:
    """Baseline: all tasks get nice=0."""

    def test_baseline_all_zero(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        result = baseline_assignment(schedulable_set, analyzer)
        assert all(v == 0 for v in result.nice_values.values())
        assert result.method == "baseline"
        assert result.analysis_time_ms > 0

    def test_baseline_result_type(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        result = baseline_assignment(schedulable_set, analyzer)
        assert isinstance(result, NiceAssignmentResult)

    def test_baseline_empty(self, analyzer: WCRTAnalyzer) -> None:
        result = baseline_assignment([], analyzer)
        assert result.nice_values == {}


# ── DeadlineAwareHeuristic (Algorithm 2) ─────────────────────────────────────

class TestDeadlineAwareHeuristic:
    """Lambda-driven nice assignment based on deadline ratios."""

    def test_schedulable_set(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        algo = DeadlineAwareHeuristic(analyzer)
        result = algo.assign(schedulable_set)
        assert len(result.nice_values) == 3
        assert result.method == "deadline_aware_heuristic"
        assert result.analysis_time_ms > 0

    def test_nice_values_in_range(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        algo = DeadlineAwareHeuristic(analyzer)
        result = algo.assign(schedulable_set)
        for v in result.nice_values.values():
            assert -20 <= v <= 19

    def test_empty_tasks(self, analyzer: WCRTAnalyzer) -> None:
        algo = DeadlineAwareHeuristic(analyzer)
        result = algo.assign([])
        assert result.nice_values == {}

    def test_configurable_lambda_range(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        """Narrow lambda range should still produce valid results."""
        algo = DeadlineAwareHeuristic(analyzer, lambda_min=0.0, lambda_max=5.0, lambda_gap=0.5)
        result = algo.assign(schedulable_set)
        assert len(result.nice_values) == 3

    def test_assign_internal(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        algo = DeadlineAwareHeuristic(analyzer)
        nice_map = algo._assign_nice_values(schedulable_set, lam=10.0)
        assert len(nice_map) == 3
        # Task with longest deadline should get nice=0
        d_max = max(t.deadline for t in schedulable_set)
        longest_deadline_task = [t for t in schedulable_set if t.deadline == d_max][0]
        assert nice_map[longest_deadline_task.task_id] == 0

    def test_create_tasks_with_nice(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        algo = DeadlineAwareHeuristic(analyzer)
        nice_map = {t.task_id: -10 for t in schedulable_set}
        new_tasks = algo._create_tasks_with_nice(schedulable_set, nice_map)
        assert all(t.nice == -10 for t in new_tasks)
        # Original tasks should be unchanged
        assert all(t.nice == 0 for t in schedulable_set)


# ── GeneticNiceAssignment (Algorithm 3) ──────────────────────────────────────

class TestGeneticNiceAssignment:
    """Evolutionary search for optimal nice values."""

    def test_schedulable_set(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        """GA should converge (or at least produce valid output)."""
        algo = GeneticNiceAssignment(
            analyzer,
            population_size=20,  # small for test speed
            max_generations=10,
            timeout_seconds=5.0,
        )
        result = algo.assign(schedulable_set)
        assert len(result.nice_values) == 3
        assert result.method == "genetic_algorithm"

    def test_nice_values_in_range(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        algo = GeneticNiceAssignment(analyzer, population_size=20, max_generations=10)
        result = algo.assign(schedulable_set)
        for v in result.nice_values.values():
            assert -20 <= v <= 19

    def test_random_chromosome(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        algo = GeneticNiceAssignment(analyzer)
        chrom = algo._random_chromosome(5)
        assert len(chrom.genes) == 5
        assert all(-20 <= g <= 19 for g in chrom.genes)
        assert chrom.fitness == 0

    def test_crossover(self, analyzer: WCRTAnalyzer) -> None:
        algo = GeneticNiceAssignment(analyzer)
        p1 = algo._random_chromosome(5)
        p2 = algo._random_chromosome(5)
        child = algo._crossover(p1, p2)
        assert len(child.genes) == 5

    def test_mutation(self, analyzer: WCRTAnalyzer) -> None:
        algo = GeneticNiceAssignment(analyzer, mutation_rate=1.0)  # always mutate
        parent = algo._random_chromosome(5)
        child = algo._mutate(parent)
        assert len(child.genes) == 5
        assert all(-20 <= g <= 19 for g in child.genes)

    def test_tournament_select(self, analyzer: WCRTAnalyzer) -> None:
        algo = GeneticNiceAssignment(analyzer)
        pop = [algo._random_chromosome(3) for _ in range(5)]
        for c in pop:
            c.fitness = c.genes[0]  # arbitrary fitness
        selected = algo._tournament_select(pop)
        assert selected in pop

    def test_small_generation(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        """Even 1 generation should not crash."""
        algo = GeneticNiceAssignment(analyzer, population_size=10, max_generations=1)
        result = algo.assign(schedulable_set)
        assert len(result.nice_values) == 3

    def test_crossover_single_gene(self, analyzer: WCRTAnalyzer) -> None:
        """Crossover with single gene should just copy."""
        algo = GeneticNiceAssignment(analyzer)
        from cfs_wcrt.optimization.nice import Chromosome
        p1 = Chromosome(genes=[5])
        p2 = Chromosome(genes=[10])
        child = algo._crossover(p1, p2)
        assert child.genes == [5]  # same as p1 since n <= 1


# ── Comparison sanity ────────────────────────────────────────────────────────

class TestComparison:
    """Baseline vs Heuristic vs GA should produce valid comparisons."""

    def test_all_three_methods(self, analyzer: WCRTAnalyzer, schedulable_set: list) -> None:
        base = baseline_assignment(schedulable_set, analyzer)
        heur = DeadlineAwareHeuristic(analyzer).assign(schedulable_set)
        ga = GeneticNiceAssignment(analyzer, population_size=20, max_generations=5).assign(schedulable_set)

        assert isinstance(base, NiceAssignmentResult)
        assert isinstance(heur, NiceAssignmentResult)
        assert isinstance(ga, NiceAssignmentResult)
