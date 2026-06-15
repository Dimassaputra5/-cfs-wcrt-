"""Command-line interface for CFS WCRT experiments."""

from __future__ import annotations

import argparse
import logging
import random
import time
from typing import Final

from ..analysis import WCRTAnalyzer
from ..core import CFSConfig, ExperimentResult
from ..generation import TaskGenConfig, generate_task_set
from ..optimization import (
    DeadlineAwareHeuristic,
    GeneticNiceAssignment,
    baseline_assignment,
)
from ..simulation import CFSSimulator
from .charts import plot_wcrt_comparison
from .tables import (
    export_results_csv,
    format_nice_assignment_table,
    format_schedulability_table,
    format_task_detail_table,
)

logger = logging.getLogger(__name__)

NUM_TASK_SETS: Final[int] = 100
NUM_TASKS_LIST: Final[list[int]] = [4, 8, 12, 16, 20]
UTILIZATIONS: Final[list[float]] = [0.2, 0.4, 0.6, 0.8]


def run_experiment_1() -> None:
    """Evaluate WCRT analysis accuracy against simulation."""
    print("=" * 70)
    print("Experiment 1: WCRT Analysis Accuracy")
    print("=" * 70)
    print()

    config = CFSConfig()
    analyzer = WCRTAnalyzer(config)
    simulator = CFSSimulator(config)
    gen_config = TaskGenConfig()
    all_results: list[ExperimentResult] = []

    for num_tasks in NUM_TASKS_LIST:
        for util in UTILIZATIONS:
            tp = tn = fp = fn = 0
            total_time = 0.0
            for _ in range(NUM_TASK_SETS):
                tasks = generate_task_set(num_tasks, util, gen_config)
                measured = simulator.measure_wcrt(tasks, hyperperiod_factor=2.0, num_runs=5)
                a_start = time.perf_counter()
                analysis_result = analyzer.analyze(tasks)
                total_time += (time.perf_counter() - a_start) * 1000

                for t in tasks:
                    ms = measured.get(t.task_id, 0.0) <= t.deadline
                    es = any(
                        r.schedulable
                        for r in analysis_result.results
                        if r.task_id == t.task_id
                    )
                    if ms and es:
                        tp += 1
                    elif not ms and not es:
                        tn += 1
                    elif not ms and es:
                        fp += 1
                    else:
                        fn += 1

            result = ExperimentResult(
                task_params=[], num_tasks=num_tasks, utilization=util,
                tp=tp, tn=tn, fp=fp, fn=fn, analysis_time_ms=total_time / NUM_TASK_SETS,
            )
            all_results.append(result)
            total = tp + tn + fp + fn
            acc = (tp + tn) / total if total > 0 else 0.0
            print(
                f"  Tasks={num_tasks:>2}, Util={util:.1f}: "
                f"TP={tp:>4}, TN={tn:>4}, FP={fp:>2}, "
                f"FN={fn:>2}, Acc={acc:.4f}"
            )

    print()
    print("Schedulability Comparison Table:")
    print(format_schedulability_table(all_results))
    print()
    export_results_csv(all_results, "experiment_1_results.csv")
    print("Results saved to experiment_1_results.csv")


def run_experiment_2() -> None:
    """Compare nice value assignment strategies."""
    print("=" * 70)
    print("Experiment 2: Nice Value Assignment Comparison")
    print("=" * 70)
    print()

    config = CFSConfig()
    analyzer = WCRTAnalyzer(config)
    gen_config = TaskGenConfig()
    ga_task_list = [4, 8, 12]
    ga_sets = 20

    baseline_res: list[ExperimentResult] = []
    heuristic_res: list[ExperimentResult] = []
    ga_res: list[ExperimentResult] = []

    for num_tasks in NUM_TASKS_LIST:
        for util in UTILIZATIONS:
            b_tp = b_tn = h_tp = h_tn = g_tp = g_tn = 0
            actual = ga_sets if num_tasks in ga_task_list else NUM_TASK_SETS

            for _ in range(actual):
                tasks = generate_task_set(num_tasks, util, gen_config)
                bl = baseline_assignment(tasks, analyzer)
                if bl.schedulable:
                    b_tp += 1
                else:
                    b_tn += 1

                h = DeadlineAwareHeuristic(analyzer)
                hr = h.assign(tasks)
                if hr.schedulable:
                    h_tp += 1
                else:
                    h_tn += 1

                if num_tasks in ga_task_list:
                    ga = GeneticNiceAssignment(
                        analyzer, population_size=50,
                        max_generations=100, timeout_seconds=3.0,
                    )
                    gr = ga.assign(tasks)
                    if gr.schedulable:
                        g_tp += 1
                    else:
                        g_tn += 1

            baseline_res.append(ExperimentResult(
                task_params=[], num_tasks=num_tasks,
                utilization=util, tp=b_tp, tn=b_tn,
            ))
            heuristic_res.append(ExperimentResult(
                task_params=[], num_tasks=num_tasks,
                utilization=util, tp=h_tp, tn=h_tn,
            ))
            ga_res.append(ExperimentResult(
                task_params=[], num_tasks=num_tasks,
                utilization=util, tp=g_tp, tn=g_tn,
            ))
            print(
                f"  Tasks={num_tasks:>2}, Util={util:.1f}: "
                f"Baseline={b_tp:>3}/{actual}, "
                f"Heuristic={h_tp:>3}/{actual}, GA={g_tp:>3}/{actual}"
            )

    print()
    print("Nice Value Assignment Comparison:")
    print(format_nice_assignment_table(baseline_res, heuristic_res, ga_res))
    print()


def run_experiment_3() -> None:
    """Show detailed WCRT comparison for a single task set."""
    print("=" * 70)
    print("Experiment 3: Detailed WCRT Comparison (Single Task Set)")
    print("=" * 70)
    print()

    config = CFSConfig()
    analyzer = WCRTAnalyzer(config)
    simulator = CFSSimulator(config)
    gen_config = TaskGenConfig()
    random.seed(42)
    tasks = generate_task_set(8, 0.6, gen_config)

    print("Task Set:")
    print(f"  {'ID':>4} {'WCET':>8} {'Period':>8} {'Deadline':>10} {'Weight':>8} {'Nice':>5}")
    for t in tasks:
        print(
            f"  {t.task_id:>4} {t.wcet:>8.2f} {t.period:>8.1f} "
            f"{t.deadline:>10.1f} {t.weight:>8} {t.nice:>5}"
        )
    print()
    total_util = sum(t.wcet / t.period for t in tasks)
    print(f"Total Utilization: {total_util:.4f}")
    print()

    a_start = time.perf_counter()
    analysis_result = analyzer.analyze(tasks)
    a_time = (time.perf_counter() - a_start) * 1000

    s_start = time.perf_counter()
    measured_wcrt = simulator.measure_wcrt(tasks, hyperperiod_factor=2.0, num_runs=10)
    s_time = (time.perf_counter() - s_start) * 1000

    task_ids = [t.task_id for t in tasks]
    measured = [measured_wcrt.get(t.task_id, 0.0) for t in tasks]
    estimated = [r.estimated_wcrt for r in sorted(analysis_result.results, key=lambda r: r.task_id)]
    deadlines = [t.deadline for t in tasks]

    print("WCRT Comparison:")
    print(format_task_detail_table(task_ids, measured, estimated, deadlines))
    print()
    print(f"Analysis Time: {a_time:.2f}ms")
    print(f"Simulation Time: {s_time:.2f}ms")
    print(f"System Schedulable (Analysis): {analysis_result.system_schedulable}")
    print()

    print("Per-Task Analysis Details:")
    for r in sorted(analysis_result.results, key=lambda r: r.task_id):
        print(
            f"  Task {r.task_id}: WCRT={r.estimated_wcrt:.2f}ms, "
            f"Iterations={r.iterations}, Time={r.analysis_time_us:.1f}us"
        )
    print()

    try:
        plot_wcrt_comparison(task_ids, measured, estimated, deadlines,
                             title="Experiment 3: WCRT Measured vs Estimated",
                             save_path="experiment_3_wcrt_comparison.png")
        print("Plot saved to experiment_3_wcrt_comparison.png")
    except Exception as e:
        logger.warning("Could not generate plot: %s", e)


def main() -> None:
    """Parse arguments and run experiments."""
    parser = argparse.ArgumentParser(
        description="CFS WCRT Analysis - Reproducing experiments from Yoon et al. (2025)",
    )
    parser.add_argument("--experiment", choices=["exp1", "exp2", "exp3", "all"], default="all",
                        help="Which experiment to run (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    random.seed(args.seed if args.seed is not None else 42)
    print(f"Random seed: {args.seed if args.seed is not None else 42}")

    experiments = {"exp1": run_experiment_1, "exp2": run_experiment_2, "exp3": run_experiment_3}
    if args.experiment == "all":
        for func in experiments.values():
            func()
            print()
    else:
        experiments[args.experiment]()


if __name__ == "__main__":
    main()
