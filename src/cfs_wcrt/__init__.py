"""CFS WCRT Analysis Tool.

Worst-Case Response Time Analysis for Completely Fair Scheduling in Linux Systems.

Based on: Yoon et al., "Worst case response time analysis for completely fair
scheduling in Linux systems", Real-Time Systems, 2025.
"""

from __future__ import annotations

from .analysis import SystemWCRTResult, WCRTAnalyzer, WCRTResult
from .core import (
    NICE_0_WEIGHT,
    NICE_MAX,
    NICE_MIN,
    NICE_TO_WEIGHT,
    CFSConfig,
    ExperimentResult,
    SchedPolicy,
    SchedulingEvent,
    TaskParams,
    TaskResult,
    nice_to_weight,
)
from .generation import TaskGenConfig, compute_hyperperiod, generate_task_set, generate_task_sets
from .optimization import (
    DeadlineAwareHeuristic,
    GeneticNiceAssignment,
    NiceAssignmentResult,
    baseline_assignment,
)
from .simulation import CFSSimulator

__all__ = [
    "CFSConfig",
    "CFSSimulator",
    "DeadlineAwareHeuristic",
    "ExperimentResult",
    "GeneticNiceAssignment",
    "NiceAssignmentResult",
    "NICE_0_WEIGHT",
    "NICE_MAX",
    "NICE_MIN",
    "NICE_TO_WEIGHT",
    "SchedPolicy",
    "SchedulingEvent",
    "SystemWCRTResult",
    "TaskGenConfig",
    "TaskParams",
    "TaskResult",
    "WCRTAnalyzer",
    "WCRTResult",
    "baseline_assignment",
    "compute_hyperperiod",
    "generate_task_set",
    "generate_task_sets",
    "nice_to_weight",
]
