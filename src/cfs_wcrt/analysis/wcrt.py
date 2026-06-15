"""Worst-Case Response Time (WCRT) analysis for Linux CFS.

Implements Algorithm 1 and supporting lemmas from:
    Yoon et al., "Worst case response time analysis for completely fair
    scheduling in Linux systems", Real-Time Systems, 2025.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Final

from ..core import CFSConfig, TaskParams

logger = logging.getLogger(__name__)

NICE_0_WEIGHT: Final[int] = 1024


@dataclass(frozen=True)
class WCRTResult:
    """Result of WCRT analysis for a single task."""

    task_id: int
    estimated_wcrt: float
    schedulable: bool
    iterations: int
    analysis_time_us: float


@dataclass(frozen=True)
class SystemWCRTResult:
    """Result of WCRT analysis for an entire task set."""

    results: list[WCRTResult]
    system_schedulable: bool
    total_analysis_time_ms: float


def _compute_sigma_tilde(
    task: TaskParams,
    total_weight: float,
    config: CFSConfig,
) -> float:
    """Compute maximum timeslice for a task (Definition 3 + 4).

    sigma_tilde_i = max((w_i / total_weight) * L_adj, G) rounded to jiffy.
    """
    num_tasks = max(1, int(total_weight / NICE_0_WEIGHT))
    L = config.target_latency
    if num_tasks > config.sched_nr_latency:
        L_adj = float(num_tasks) * L / float(config.sched_nr_latency)
    else:
        L_adj = L

    delta = (float(task.weight) / total_weight) * L_adj
    sigma = max(delta, config.min_granularity)
    sigma = math.ceil(sigma / config.jiffy) * config.jiffy
    return sigma


def _compute_max_busy_period(
    task_i: TaskParams,
    all_tasks: list[TaskParams],
    config: CFSConfig,
) -> float:
    """Compute maximum busy period length L_i containing one job of task_i.

    Equation 33: L_i^{m+1} = C_i + sum_{tau_j != tau_i} ceil(L_i^m / T_j) * C_j
    """
    L_prev = task_i.wcet
    for _ in range(1000):
        total = task_i.wcet
        for tj in all_tasks:
            if tj.task_id == task_i.task_id:
                continue
            total += math.ceil(L_prev / tj.period) * tj.wcet

        L_new = total
        if abs(L_new - L_prev) < 1e-9:
            return L_new

        utilization = sum(t.wcet / t.period for t in all_tasks)
        if utilization > 1.0:
            return min(L_new, L_prev * 2)

        L_prev = L_new
    return L_prev


class WCRTAnalyzer:
    """Worst-Case Response Time analyzer for Linux CFS.

    Implements Algorithm 1: iterative fixed-point analysis computing
    a conservative upper bound on the response time of each task under CFS.
    """

    def __init__(self, config: CFSConfig | None = None) -> None:
        self._config = config or CFSConfig()

    def _compute_interference_bound(
        self,
        task_i: TaskParams,
        task_j: TaskParams,
        r_prev: float,
        total_weight: float,
        w_min_i: float,
        all_tasks: list[TaskParams],
    ) -> float:
        """Upper bound of interference from task_j to task_i (Corollary 2)."""
        config = self._config
        w_0 = float(NICE_0_WEIGHT)
        w_j = float(task_j.weight)

        # Lemma 2: alpha_i (max vruntime increase of task_i)
        sigma_i = _compute_sigma_tilde(task_i, total_weight, config)
        alpha_i = (w_0 / w_min_i) * sigma_i if w_min_i > 0 else sigma_i

        # Lemma 4: gamma_ij (upper bound of V_j[R_i])
        sigma_j = _compute_sigma_tilde(task_j, total_weight, config)
        gamma_ij = alpha_i + (w_0 / w_j) * sigma_j

        # Lemma 5: Interference from vruntime variance
        lemma5 = (gamma_ij + alpha_i) * w_j / w_0

        # Lemma 6: Total workload bound
        L_i = _compute_max_busy_period(task_i, all_tasks, config)
        n_jobs = self._compute_n_tilde(task_j, r_prev, L_i)
        workload_bound = float(n_jobs) * task_j.wcet

        return min(lemma5, workload_bound)

    def _compute_n_tilde(
        self, task_j: TaskParams, r_i: float, l_i: float
    ) -> int:
        """Upper bound on number of task_j jobs in [0, R_i] (Lemma 6)."""
        n1 = math.ceil(r_i / task_j.period) if task_j.period > 0 else 0
        n2 = math.ceil(l_i / task_j.period) if task_j.period > 0 else 0
        return min(n1, n2)

    def analyze_task(
        self,
        task_i: TaskParams,
        all_tasks: list[TaskParams],
        max_iterations: int = 1000,
    ) -> WCRTResult:
        """Analyze WCRT for a single task using fixed-point iteration."""
        start = time.perf_counter()

        total_weight = float(sum(t.weight for t in all_tasks))
        others = [t for t in all_tasks if t.task_id != task_i.task_id]
        w_min_i = float(min((t.weight for t in others), default=task_i.weight))

        r_prev = task_i.wcet
        iterations = 0

        for iteration in range(max_iterations):
            iterations = iteration + 1
            total_interference = 0.0
            for tj in others:
                total_interference += self._compute_interference_bound(
                    task_i, tj, r_prev, total_weight, w_min_i, all_tasks,
                )

            r_cur = task_i.wcet + total_interference
            if r_cur > task_i.deadline:
                elapsed = (time.perf_counter() - start) * 1e6
                return WCRTResult(
                    task_id=task_i.task_id,
                    estimated_wcrt=r_cur,
                    schedulable=False,
                    iterations=iterations,
                    analysis_time_us=elapsed,
                )

            if abs(r_cur - r_prev) < 1e-9:
                break
            r_prev = r_cur

        elapsed = (time.perf_counter() - start) * 1e6
        return WCRTResult(
            task_id=task_i.task_id,
            estimated_wcrt=r_prev,
            schedulable=r_prev <= task_i.deadline,
            iterations=iterations,
            analysis_time_us=elapsed,
        )

    def analyze(
        self,
        tasks: list[TaskParams],
        max_iterations: int = 1000,
    ) -> SystemWCRTResult:
        """Analyze WCRT for an entire task set."""
        start = time.perf_counter()
        results: list[WCRTResult] = []

        for task in tasks:
            result = self.analyze_task(task, tasks, max_iterations)
            results.append(result)
            logger.debug(
                "Task %d: WCRT=%.2f, Deadline=%.2f, Schedulable=%s (%d iter)",
                task.task_id,
                result.estimated_wcrt,
                task.deadline,
                result.schedulable,
                result.iterations,
            )

        system_schedulable = all(r.schedulable for r in results)
        elapsed = (time.perf_counter() - start) * 1000

        return SystemWCRTResult(
            results=results,
            system_schedulable=system_schedulable,
            total_analysis_time_ms=elapsed,
        )
