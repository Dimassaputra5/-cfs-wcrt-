"""Synthetic task set generation using UUniFast algorithm.

Implements the UUniFast algorithm for generating task sets with specified
total utilization, from:
    Bini and Buttazzo, "Measuring the performance of schedulability tests",
    Real-Time Systems, 2005.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..core import TaskParams


@dataclass(frozen=True)
class TaskGenConfig:
    """Configuration for synthetic task generation."""

    min_period: float = 30.0
    max_period: float = 3000.0
    min_wcet_ratio: float = 0.01
    max_wcet_ratio: float = 0.5
    default_nice: int = 0


def _uunifast(n: int, total_util: float) -> list[float]:
    """Generate n utilization values summing to total_util using UUniFast."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if total_util <= 0:
        raise ValueError(f"total_util must be positive, got {total_util}")

    utils: list[float] = []
    remaining = total_util
    for i in range(n - 1):
        next_sum = remaining * (random.random() ** (1.0 / (n - i)))
        utils.append(remaining - next_sum)
        remaining = next_sum
    utils.append(remaining)
    return utils


def _sample_period(config: TaskGenConfig) -> float:
    """Sample a task period from the configured range, rounded to nearest 10ms."""
    raw = random.uniform(config.min_period, config.max_period)
    return round(raw / 10.0) * 10.0


def generate_task_set(
    num_tasks: int,
    utilization: float,
    config: TaskGenConfig | None = None,
    start_id: int = 0,
) -> list[TaskParams]:
    """Generate a synthetic task set with specified utilization."""
    config = config or TaskGenConfig()

    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")
    if utilization <= 0 or utilization > 1.0:
        raise ValueError(f"utilization must be in (0, 1.0], got {utilization}")

    utilizations = _uunifast(num_tasks, utilization)
    tasks: list[TaskParams] = []

    for i in range(num_tasks):
        period = _sample_period(config)
        wcet = max(utilizations[i] * period, 1.0)
        wcet = round(wcet, 2)

        task = TaskParams.from_nice(
            task_id=start_id + i,
            wcet=wcet,
            nice=config.default_nice,
            period=period,
        )
        tasks.append(task)

    return tasks


def generate_task_sets(
    num_sets: int,
    num_tasks: int,
    utilization: float,
    config: TaskGenConfig | None = None,
) -> list[list[TaskParams]]:
    """Generate multiple synthetic task sets."""
    return [
        generate_task_set(num_tasks, utilization, config, start_id=i * num_tasks)
        for i in range(num_sets)
    ]


def compute_hyperperiod(periods: list[float]) -> int:
    """Compute the hyperperiod (LCM) of a list of periods."""
    if not periods:
        raise ValueError("periods list must not be empty")

    int_periods = [int(round(p)) for p in periods]
    result = int_periods[0]
    for p in int_periods[1:]:
        result = result * p // math.gcd(result, p)
    return result
