"""Data models for CFS scheduling and WCRT analysis.

References:
    Yoon et al., "Worst case response time analysis for completely fair
    scheduling in Linux systems", Real-Time Systems, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .constants import NICE_MAX, NICE_MIN, nice_to_weight


class SchedPolicy(Enum):
    """Linux scheduling policy."""

    SCHED_OTHER = auto()
    SCHED_FIFO = auto()
    SCHED_RR = auto()
    SCHED_DEADLINE = auto()


@dataclass(frozen=True)
class CFSConfig:
    """CFS configuration parameters matching Linux kernel 5.15 defaults.

    Attributes:
        target_latency: Desired time span (ms) for all tasks to run once (L).
        min_granularity: Minimum timeslice (ms) to avoid excessive context switches (G).
        jiffy: Timer tick interval (ms) based on CONFIG_HZ (J).
        sched_nr_latency: Max tasks for accurate weight-based timeslice (default 8).
        num_cores: Number of CPU cores (M). Single core = 1.
    """

    target_latency: float = 18.0
    min_granularity: float = 2.25
    jiffy: float = 1.0
    sched_nr_latency: int = 8
    num_cores: int = 1

    def __post_init__(self) -> None:
        if self.target_latency <= 0:
            raise ValueError("target_latency must be positive")
        if self.min_granularity <= 0:
            raise ValueError("min_granularity must be positive")
        if self.jiffy <= 0:
            raise ValueError("jiffy must be positive")
        if self.sched_nr_latency <= 0:
            raise ValueError("sched_nr_latency must be positive")
        if self.num_cores < 1:
            raise ValueError("num_cores must be >= 1")


@dataclass(frozen=True)
class TaskParams:
    """Task parameters for scheduling analysis.

    Attributes:
        task_id: Unique identifier.
        wcet: Worst-case execution time in ms (C_i).
        weight: CFS weight derived from nice value (w_i).
        nice: Nice value [-20, 19].
        deadline: Relative deadline in ms (D_i). Defaults to period.
        period: Period / minimum inter-arrival time in ms (T_i).
    """

    task_id: int
    wcet: float
    weight: int
    nice: int
    deadline: float
    period: float

    def __post_init__(self) -> None:
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if self.wcet <= 0:
            raise ValueError("wcet must be positive")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if not NICE_MIN <= self.nice <= NICE_MAX:
            raise ValueError(f"nice must be in [{NICE_MIN}, {NICE_MAX}]")
        if self.deadline <= 0:
            raise ValueError("deadline must be positive")
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.deadline > self.period:
            raise ValueError("deadline must be <= period (constrained system)")

    @classmethod
    def from_nice(
        cls,
        task_id: int,
        wcet: float,
        nice: int,
        period: float,
        deadline: float | None = None,
    ) -> TaskParams:
        """Create task from nice value instead of raw weight."""
        return cls(
            task_id=task_id,
            wcet=wcet,
            weight=nice_to_weight(nice),
            nice=nice,
            deadline=deadline if deadline is not None else period,
            period=period,
        )


@dataclass(frozen=True)
class SchedulingEvent:
    """A single scheduling event recorded during simulation.

    Attributes:
        time: Event timestamp in ms.
        task_id: ID of the task that starts running.
        vruntime_at_start: Task's vruntime when it starts executing.
        timeslice: Allocated timeslice in ms.
    """

    time: float
    task_id: int
    vruntime_at_start: float
    timeslice: float


@dataclass
class TaskResult:
    """Per-task result from simulation or analysis."""

    task_id: int
    measured_wcrt: float = 0.0
    estimated_wcrt: float = 0.0
    schedulable_measured: bool = False
    schedulable_estimated: bool = False


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    task_params: list[TaskParams]
    num_tasks: int
    utilization: float
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    task_results: list[TaskResult] = field(default_factory=list)
    analysis_time_ms: float = 0.0
