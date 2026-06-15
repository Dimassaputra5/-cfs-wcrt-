"""Nice value assignment algorithms for CFS schedulability.

Implements:
    - Algorithm 2: Deadline-aware heuristic for nice value assignment.
    - Algorithm 3: Genetic algorithm for optimal nice value search.

From: Yoon et al., "Worst case response time analysis for completely fair
    scheduling in Linux systems", Real-Time Systems, 2025.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import NICE_MAX, NICE_MIN, TaskParams

if TYPE_CHECKING:
    from ..analysis import WCRTAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NiceAssignmentResult:
    """Result of a nice value assignment algorithm."""

    nice_values: dict[int, int]
    schedulable: bool
    analysis_time_ms: float
    method: str


class DeadlineAwareHeuristic:
    """Heuristic for assigning nice values based on task deadlines.

    Algorithm 2: Assigns nice values proportional to the logarithm of
    the deadline ratio (D_max / D_i), scaled by parameter lambda.
    """

    def __init__(
        self,
        analyzer: WCRTAnalyzer,
        lambda_min: float = 0.0,
        lambda_max: float = 40.0,
        lambda_gap: float = 0.1,
    ) -> None:
        self._analyzer = analyzer
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max
        self._lambda_gap = lambda_gap

    def _assign_nice_values(
        self, tasks: list[TaskParams], lam: float
    ) -> dict[int, int]:
        """Assign nice values using the deadline-aware formula."""
        if not tasks:
            return {}

        d_max = max(t.deadline for t in tasks)
        nice_map: dict[int, int] = {}
        for t in tasks:
            if t.deadline >= d_max:
                nice_map[t.task_id] = 0
            else:
                raw = -lam * math.log2(d_max / t.deadline)
                nice_map[t.task_id] = max(NICE_MIN, min(NICE_MAX, round(raw)))
        return nice_map

    def _create_tasks_with_nice(
        self, tasks: list[TaskParams], nice_map: dict[int, int]
    ) -> list[TaskParams]:
        """Create new task params with assigned nice values."""
        return [
            TaskParams.from_nice(
                task_id=t.task_id,
                wcet=t.wcet,
                nice=nice_map.get(t.task_id, 0),
                period=t.period,
                deadline=t.deadline,
            )
            for t in tasks
        ]

    def assign(self, tasks: list[TaskParams]) -> NiceAssignmentResult:
        """Find the smallest lambda that makes the system schedulable."""
        start = time.perf_counter()
        best_nice: dict[int, int] = {t.task_id: 0 for t in tasks}
        schedulable = False

        lam = self._lambda_min
        while lam <= self._lambda_max:
            nice_map = self._assign_nice_values(tasks, lam)
            assigned = self._create_tasks_with_nice(tasks, nice_map)
            result = self._analyzer.analyze(assigned)

            if result.system_schedulable:
                best_nice = nice_map
                schedulable = True
                break

            best_nice = nice_map
            lam += self._lambda_gap

        elapsed = (time.perf_counter() - start) * 1000
        return NiceAssignmentResult(
            nice_values=best_nice,
            schedulable=schedulable,
            analysis_time_ms=elapsed,
            method="deadline_aware_heuristic",
        )


@dataclass
class Chromosome:
    """A candidate solution for nice value assignment."""

    genes: list[int]
    fitness: int = 0


class GeneticNiceAssignment:
    """Genetic algorithm for finding optimal nice value assignments.

    Algorithm 3: Uses evolutionary search over the space of 40^n possible
    nice value assignments (n = number of tasks).
    """

    def __init__(
        self,
        analyzer: WCRTAnalyzer,
        population_size: int = 100,
        mutation_rate: float = 0.05,
        max_generations: int = 200,
        timeout_seconds: float = 5.0,
        tournament_size: int = 3,
    ) -> None:
        self._analyzer = analyzer
        self._pop_size = population_size
        self._mutation_rate = mutation_rate
        self._max_gen = max_generations
        self._timeout = timeout_seconds
        self._tournament_size = tournament_size

    def _random_chromosome(self, n_tasks: int) -> Chromosome:
        genes = [random.randint(NICE_MIN, NICE_MAX) for _ in range(n_tasks)]
        return Chromosome(genes=genes)

    def _evaluate_fitness(self, chromosome: Chromosome, tasks: list[TaskParams]) -> int:
        assigned = [
            TaskParams.from_nice(
                task_id=t.task_id,
                wcet=t.wcet,
                nice=chromosome.genes[i],
                period=t.period,
                deadline=t.deadline,
            )
            for i, t in enumerate(tasks)
        ]
        result = self._analyzer.analyze(assigned)
        return sum(1 for r in result.results if r.schedulable)

    def _tournament_select(self, population: list[Chromosome]) -> Chromosome:
        candidates = random.sample(population, min(self._tournament_size, len(population)))
        return max(candidates, key=lambda c: c.fitness)

    def _crossover(self, p1: Chromosome, p2: Chromosome) -> Chromosome:
        n = len(p1.genes)
        if n <= 1:
            return Chromosome(genes=list(p1.genes))
        point = random.randint(1, n - 1)
        return Chromosome(genes=p1.genes[:point] + p2.genes[point:])

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        mutated = list(chromosome.genes)
        for i in range(len(mutated)):
            if random.random() < self._mutation_rate:
                offset = random.randint(-2, 2)
                mutated[i] = max(NICE_MIN, min(NICE_MAX, mutated[i] + offset))
        return Chromosome(genes=mutated)

    def assign(self, tasks: list[TaskParams]) -> NiceAssignmentResult:
        """Run genetic algorithm to find optimal nice value assignment."""
        start = time.perf_counter()
        n_tasks = len(tasks)

        population = [self._random_chromosome(n_tasks) for _ in range(self._pop_size)]
        for chrom in population:
            chrom.fitness = self._evaluate_fitness(chrom, tasks)

        best = max(population, key=lambda c: c.fitness)
        best_fitness = best.fitness

        for generation in range(self._max_gen):
            if time.perf_counter() - start > self._timeout:
                logger.info("GA timeout at generation %d", generation)
                break
            if best_fitness >= n_tasks:
                logger.info("GA converged at generation %d", generation)
                break

            new_population: list[Chromosome] = []
            new_population.append(Chromosome(genes=list(best.genes), fitness=best_fitness))

            while len(new_population) < self._pop_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                child = self._mutate(self._crossover(p1, p2))
                child.fitness = self._evaluate_fitness(child, tasks)
                new_population.append(child)

            population = new_population
            gen_best = max(population, key=lambda c: c.fitness)
            if gen_best.fitness > best_fitness:
                best = Chromosome(genes=list(gen_best.genes), fitness=gen_best.fitness)
                best_fitness = gen_best.fitness

        nice_map = {tasks[i].task_id: best.genes[i] for i in range(n_tasks)}
        assigned = [
            TaskParams.from_nice(
                task_id=t.task_id,
                wcet=t.wcet,
                nice=best.genes[i],
                period=t.period,
                deadline=t.deadline,
            )
            for i, t in enumerate(tasks)
        ]
        final = self._analyzer.analyze(assigned)
        elapsed = (time.perf_counter() - start) * 1000

        return NiceAssignmentResult(
            nice_values=nice_map,
            schedulable=final.system_schedulable,
            analysis_time_ms=elapsed,
            method="genetic_algorithm",
        )


def baseline_assignment(tasks: list[TaskParams], analyzer: WCRTAnalyzer) -> NiceAssignmentResult:
    """Baseline: assign nice 0 to all tasks."""
    start = time.perf_counter()
    nice_map = {t.task_id: 0 for t in tasks}
    result = analyzer.analyze(tasks)
    elapsed = (time.perf_counter() - start) * 1000
    return NiceAssignmentResult(
        nice_values=nice_map,
        schedulable=result.system_schedulable,
        analysis_time_ms=elapsed,
        method="baseline",
    )
