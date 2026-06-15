"""CFS (Completely Fair Scheduler) discrete-event simulator.

Implements the scheduling behavior described in Linux kernel 5.15 fair.c,
following the formal definitions from:
    Yoon et al., "Worst case response time analysis for completely fair
    scheduling in Linux systems", Real-Time Systems, 2025.
"""

from __future__ import annotations

import heapq
import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto

from ..core import CFSConfig, SchedulingEvent, TaskParams

logger = logging.getLogger(__name__)

NICE_0_WEIGHT: int = 1024


class EventType(Enum):
    """Simulation event types."""

    RELEASE = auto()
    TIMESLICE_EXPIRE = auto()
    COMPLETION = auto()
    JIFFY_TICK = auto()


@dataclass(order=True)
class SimEvent:
    """Priority-queue event for discrete-event simulation."""

    time: float
    priority: int = field(compare=True)
    event_type: EventType = field(compare=False)
    task_id: int = field(compare=False, default=-1)
    remaining_work: float = field(compare=False, default=0.0)

    def __init__(
        self,
        time: float,
        event_type: EventType,
        task_id: int = -1,
        remaining_work: float = 0.0,
    ) -> None:
        self.time = time
        self.priority = event_type.value
        self.event_type = event_type
        self.task_id = task_id
        self.remaining_work = remaining_work


@dataclass
class RunqueueTask:
    """Task state maintained in the CFS runqueue."""

    params: TaskParams
    vruntime: float = 0.0
    remaining: float = 0.0
    is_sleeping: bool = True
    sleep_vruntime: float = 0.0
    total_execution: float = 0.0
    release_count: int = 0


class CFSSimulator:
    """Discrete-event simulator for Linux CFS scheduling.

    Models vruntime tracking, minimum vruntime maintenance,
    dynamic timeslice allocation, wake-up adjustment, and jiffy-based preemption.
    """

    def __init__(self, config: CFSConfig | None = None) -> None:
        self._config = config or CFSConfig()
        self._runqueue: list[RunqueueTask] = []
        self._min_vruntime: float = 0.0
        self._event_log: list[SchedulingEvent] = []

    # --- Core CFS operations (Definitions 1-5) ---

    def _update_min_vruntime(self, runnable: list[RunqueueTask] | None = None) -> None:
        """Update minimum vruntime (Definition 1)."""
        source = runnable if runnable is not None else self._runqueue
        if not source:
            return
        min_vr = min(rt.vruntime for rt in source)
        self._min_vruntime = max(self._min_vruntime, min_vr)

    def _update_curr(self, rt: RunqueueTask, delta: float) -> None:
        """Update vruntime of a running task (Definition 2)."""
        if delta <= 0:
            return
        ratio = float(NICE_0_WEIGHT) / float(rt.params.weight)
        rt.vruntime += delta * ratio
        rt.remaining = max(0.0, rt.remaining - delta)
        rt.total_execution += delta
        runnable = [t for t in self._runqueue if not t.is_sleeping and t.remaining > 0]
        self._update_min_vruntime(runnable)

    def _sched_slice(self, rt: RunqueueTask) -> float:
        """Compute timeslice for a task (Definition 3 + 4)."""
        runnable = [t for t in self._runqueue if not t.is_sleeping and t.remaining > 0]
        total_weight = sum(t.params.weight for t in runnable)
        if total_weight == 0:
            return self._config.min_granularity

        L = self._config.target_latency
        num_tasks = len(runnable)
        L_adj = (
            float(num_tasks) * L / float(self._config.sched_nr_latency)
            if num_tasks > self._config.sched_nr_latency
            else L
        )

        delta = (float(rt.params.weight) / float(total_weight)) * L_adj
        sigma = max(delta, self._config.min_granularity)
        sigma = math.ceil(sigma / self._config.jiffy) * self._config.jiffy
        sigma = min(sigma, rt.remaining) if rt.remaining > 0 else sigma
        return sigma

    def _place_entity(self, rt: RunqueueTask, current_time: float) -> None:
        """Adjust vruntime on wake-up (Definition 5)."""
        threshold = self._min_vruntime - self._config.target_latency / 2.0
        if rt.vruntime < threshold:
            rt.vruntime = threshold
        rt.vruntime = max(rt.vruntime, self._min_vruntime)

    # --- Scheduling logic ---

    def _select_next(
        self, runnable: list[RunqueueTask]
    ) -> RunqueueTask | None:
        """Select task with lowest vruntime."""
        if not runnable:
            return None
        return min(runnable, key=lambda t: t.vruntime)

    # --- Simulation engine ---

    def _initialize_tasks(
        self, tasks: list[TaskParams], offsets: list[float] | None = None
    ) -> list[RunqueueTask]:
        """Create initial runqueue states."""
        runqueue: list[RunqueueTask] = []
        for _, tp in enumerate(tasks):
            rt = RunqueueTask(
                params=tp,
                vruntime=0.0,
                remaining=tp.wcet,
                is_sleeping=True,
                sleep_vruntime=0.0,
                total_execution=0.0,
                release_count=0,
            )
            runqueue.append(rt)
        return runqueue

    def _simulate_single(
        self,
        tasks: list[TaskParams],
        duration: float,
        offsets: list[float] | None = None,
    ) -> tuple[list[SchedulingEvent], dict[int, float]]:
        """Run a single simulation instance."""
        self._runqueue = self._initialize_tasks(tasks, offsets)
        self._min_vruntime = 0.0
        self._event_log = []

        event_queue: list[SimEvent] = []
        response_times: dict[int, float] = {}
        job_release_time: dict[tuple[int, int], float] = {}

        # Schedule initial releases
        for i, tp in enumerate(tasks):
            offset = offsets[i] if offsets else 0.0
            heapq.heappush(
                event_queue,
                SimEvent(offset, EventType.RELEASE, task_id=tp.task_id, remaining_work=tp.wcet),
            )

        # Schedule first jiffy tick
        heapq.heappush(event_queue, SimEvent(self._config.jiffy, EventType.JIFFY_TICK))

        current_running: RunqueueTask | None = None
        timeslice_remaining: float = 0.0
        current_time: float = 0.0
        next_jiffy_time: float = self._config.jiffy
        max_events = 50000
        event_count = 0

        while event_queue and event_count < max_events:
            event = heapq.heappop(event_queue)
            event_count += 1

            if event.time > duration:
                break

            # Advance time and update running task
            if current_running is not None and current_running.remaining > 0:
                exec_time = min(event.time - current_time, timeslice_remaining)
                if exec_time > 0:
                    self._update_curr(current_running, exec_time)
                    timeslice_remaining -= exec_time
                    if current_running.remaining <= 1e-9:
                        key = (current_running.params.task_id, current_running.release_count)
                        if key in job_release_time:
                            response_times[current_running.params.task_id] = max(
                                response_times.get(current_running.params.task_id, 0.0),
                                event.time - job_release_time[key],
                            )
                        current_running.is_sleeping = True
                        current_running = None

            current_time = event.time

            # Process event
            if event.event_type == EventType.RELEASE:
                self._handle_release(event, event_queue, job_release_time, current_time)
            elif event.event_type == EventType.JIFFY_TICK:
                next_jiffy_time += self._config.jiffy
                if next_jiffy_time <= duration:
                    heapq.heappush(event_queue, SimEvent(next_jiffy_time, EventType.JIFFY_TICK))

            # Scheduling decision
            current_running, timeslice_remaining = self._do_schedule(
                event_queue, current_time,
            )

            if current_running is not None:
                self._event_log.append(
                    SchedulingEvent(
                        time=current_time,
                        task_id=current_running.params.task_id,
                        vruntime_at_start=current_running.vruntime,
                        timeslice=timeslice_remaining,
                    )
                )

        return self._event_log, response_times

    def _handle_release(
        self,
        event: SimEvent,
        event_queue: list[SimEvent],
        job_release_time: dict[tuple[int, int], float],
        current_time: float,
    ) -> None:
        """Handle a task release."""
        rt = next(t for t in self._runqueue if t.params.task_id == event.task_id)
        if not rt.is_sleeping:
            return

        self._place_entity(rt, current_time)
        rt.is_sleeping = False
        rt.remaining = rt.params.wcet
        rt.release_count += 1

        key = (event.task_id, rt.release_count)
        job_release_time[key] = current_time

        next_release = current_time + rt.params.period
        heapq.heappush(
            event_queue,
            SimEvent(next_release, EventType.RELEASE, task_id=event.task_id),
        )

    def _do_schedule(
        self,
        event_queue: list[SimEvent],
        current_time: float,
    ) -> tuple[RunqueueTask | None, float]:
        """Perform scheduling decision at a scheduling point."""
        runnable = [
            rt for rt in self._runqueue
            if not rt.is_sleeping and rt.remaining > 0
        ]
        if not runnable:
            return None, 0.0

        self._update_min_vruntime(runnable)
        next_task = self._select_next(runnable)
        if next_task is None:
            return None, 0.0

        timeslice = self._sched_slice(next_task)
        completion_time = current_time + timeslice

        if next_task.remaining > 0 and next_task.remaining <= timeslice:
            completion_time = current_time + next_task.remaining
            heapq.heappush(
                event_queue,
                SimEvent(completion_time, EventType.COMPLETION, task_id=next_task.params.task_id),
            )
        else:
            heapq.heappush(
                event_queue,
                SimEvent(
                    completion_time,
                    EventType.TIMESLICE_EXPIRE,
                    task_id=next_task.params.task_id,
                ),
            )

        return next_task, timeslice

    # --- Public API ---

    def run(
        self,
        tasks: list[TaskParams],
        duration: float,
        num_runs: int = 10,
    ) -> tuple[list[SchedulingEvent], dict[int, float]]:
        """Run CFS simulation with multiple random offsets.

        Args:
            tasks: Task parameter set.
            duration: Simulation duration per run in ms.
            num_runs: Number of runs with different offsets.

        Returns:
            Tuple of (event_log from last run, measured_wcrt per task_id).
        """
        all_response_times: dict[int, list[float]] = {t.task_id: [] for t in tasks}

        for _ in range(num_runs):
            offsets = [random.uniform(0, t.period * 0.3) for t in tasks]
            _, response_times = self._simulate_single(tasks, duration, offsets)
            for task_id, rt in response_times.items():
                all_response_times[task_id].append(rt)

        measured_wcrt: dict[int, float] = {}
        for task_id, rts in all_response_times.items():
            measured_wcrt[task_id] = max(rts) if rts else 0.0

        return self._event_log, measured_wcrt

    def measure_wcrt(
        self,
        tasks: list[TaskParams],
        hyperperiod_factor: float = 2.0,
        num_runs: int = 5,
    ) -> dict[int, float]:
        """Measure actual WCRT by running tasks on the simulator.

        Args:
            tasks: Task parameter set.
            hyperperiod_factor: Multiplier on hyperperiod for simulation duration.
            num_runs: Number of runs to explore different scenarios.

        Returns:
            Measured WCRT per task_id.
        """
        periods = [int(t.period) for t in tasks]
        hyperperiod = periods[0]
        for p in periods[1:]:
            hyperperiod = hyperperiod * p // math.gcd(hyperperiod, p)

        duration = float(hyperperiod) * hyperperiod_factor
        _, measured_wcrt = self.run(tasks, duration, num_runs)
        return measured_wcrt
