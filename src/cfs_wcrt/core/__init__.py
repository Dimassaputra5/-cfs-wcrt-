"""Core domain models and constants for CFS WCRT analysis."""
from .constants import (
    NICE_0_WEIGHT,
    NICE_MAX,
    NICE_MIN,
    NICE_TO_WEIGHT,
    nice_to_weight,
)
from .models import (
    CFSConfig,
    ExperimentResult,
    SchedPolicy,
    SchedulingEvent,
    TaskParams,
    TaskResult,
)

__all__ = [
    "CFSConfig",
    "ExperimentResult",
    "SchedPolicy",
    "SchedulingEvent",
    "TaskParams",
    "TaskResult",
    "NICE_0_WEIGHT",
    "NICE_MAX",
    "NICE_MIN",
    "NICE_TO_WEIGHT",
    "nice_to_weight",
]
