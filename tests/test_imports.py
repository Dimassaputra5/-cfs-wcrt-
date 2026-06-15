"""Verify all public API symbols are importable.

This is a basic sanity check that the package structure is correct.
"""

from __future__ import annotations

from cfs_wcrt import (
    CFSConfig,
    CFSSimulator,
    DeadlineAwareHeuristic,
    ExperimentResult,
    GeneticNiceAssignment,
    NICE_0_WEIGHT,
    NICE_MAX,
    NICE_MIN,
    NICE_TO_WEIGHT,
    NiceAssignmentResult,
    SchedPolicy,
    SchedulingEvent,
    SystemWCRTResult,
    TaskGenConfig,
    TaskParams,
    TaskResult,
    WCRTAnalyzer,
    WCRTResult,
    baseline_assignment,
    compute_hyperperiod,
    generate_task_set,
    generate_task_sets,
    nice_to_weight,
)


class TestPublicAPI:
    """All 23 public symbols must be importable and have correct types."""

    def test_cfs_config(self) -> None:
        obj = CFSConfig()
        assert isinstance(obj, CFSConfig)

    def test_cfs_simulator(self) -> None:
        obj = CFSSimulator()
        assert isinstance(obj, CFSSimulator)

    def test_deadline_aware_heuristic(self) -> None:
        obj = DeadlineAwareHeuristic(WCRTAnalyzer())
        assert isinstance(obj, DeadlineAwareHeuristic)

    def test_experiment_result(self) -> None:
        obj = ExperimentResult(task_params=[], num_tasks=0, utilization=0.0)
        assert isinstance(obj, ExperimentResult)

    def test_genetic_nice_assignment(self) -> None:
        obj = GeneticNiceAssignment(WCRTAnalyzer())
        assert isinstance(obj, GeneticNiceAssignment)

    def test_nice_assignment_result(self) -> None:
        from cfs_wcrt.optimization import NiceAssignmentResult
        obj = NiceAssignmentResult(nice_values={}, schedulable=False, analysis_time_ms=0.0, method="test")
        assert isinstance(obj, NiceAssignmentResult)

    def test_constants(self) -> None:
        assert isinstance(NICE_0_WEIGHT, int)
        assert isinstance(NICE_MIN, int)
        assert isinstance(NICE_MAX, int)
        assert isinstance(NICE_TO_WEIGHT, dict)

    def test_sched_policy(self) -> None:
        assert SchedPolicy.SCHED_OTHER in SchedPolicy

    def test_scheduling_event(self) -> None:
        obj = SchedulingEvent(time=0.0, task_id=0, vruntime_at_start=0.0, timeslice=0.0)
        assert isinstance(obj, SchedulingEvent)

    def test_system_wcrt_result(self) -> None:
        obj = SystemWCRTResult(results=[], system_schedulable=True, total_analysis_time_ms=0.0)
        assert isinstance(obj, SystemWCRTResult)

    def test_task_gen_config(self) -> None:
        from cfs_wcrt.generation import TaskGenConfig
        obj = TaskGenConfig()
        assert isinstance(obj, TaskGenConfig)

    def test_task_params(self) -> None:
        obj = TaskParams.from_nice(task_id=0, wcet=1.0, nice=0, period=100.0)
        assert isinstance(obj, TaskParams)

    def test_task_result(self) -> None:
        obj = TaskResult(task_id=0)
        assert isinstance(obj, TaskResult)

    def test_wcrt_analyzer(self) -> None:
        obj = WCRTAnalyzer()
        assert isinstance(obj, WCRTAnalyzer)

    def test_wcrt_result(self) -> None:
        from cfs_wcrt.analysis import WCRTResult
        obj = WCRTResult(task_id=0, estimated_wcrt=0.0, schedulable=True, iterations=1, analysis_time_us=0.0)
        assert isinstance(obj, WCRTResult)

    def test_functions(self) -> None:
        assert callable(baseline_assignment)
        assert callable(compute_hyperperiod)
        assert callable(generate_task_set)
        assert callable(generate_task_sets)
        assert callable(nice_to_weight)
