"""ASCII table formatting for experiment results."""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import ExperimentResult

logger = logging.getLogger(__name__)


def format_schedulability_table(results: list[ExperimentResult]) -> str:
    """Format schedulability comparison as an ASCII table."""
    lines: list[str] = []
    header = f"{'#Tasks':>6} {'Util':>6} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} {'Acc':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    total_tp = total_tn = total_fp = total_fn = 0
    for r in results:
        total = r.tp + r.tn + r.fp + r.fn
        acc = (r.tp + r.tn) / total if total > 0 else 0.0
        lines.append(
            f"{r.num_tasks:>6} {r.utilization:>6.1f} "
            f"{r.tp:>5} {r.tn:>5} {r.fp:>5} {r.fn:>5} {acc:>8.4f}"
        )
        total_tp += r.tp
        total_tn += r.tn
        total_fp += r.fp
        total_fn += r.fn

    t = total_tp + total_tn + total_fp + total_fn
    overall = (total_tp + total_tn) / t if t > 0 else 0.0
    lines.append("-" * len(header))
    lines.append(
        f"{'Total':>6} {'':>6} {total_tp:>5} {total_tn:>5} "
        f"{total_fp:>5} {total_fn:>5} {overall:>8.4f}"
    )
    return "\n".join(lines)


def format_nice_assignment_table(
    baseline: list[ExperimentResult],
    heuristic: list[ExperimentResult],
    ga: list[ExperimentResult],
) -> str:
    """Format nice value assignment comparison."""
    lines: list[str] = []
    header = f"{'#Tasks':>6} {'Util':>6} | {'Baseline':>10} {'Heuristic':>10} {'GA':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for b, h, g in zip(baseline, heuristic, ga):
        lines.append(
            f"{b.num_tasks:>6} {b.utilization:>6.1f} | "
            f"{b.tp + b.tn:>10} {h.tp + h.tn:>10} {g.tp + g.tn:>10}"
        )
    return "\n".join(lines)


def format_task_detail_table(
    task_ids: list[int],
    measured: list[float],
    estimated: list[float],
    deadlines: list[float],
) -> str:
    """Format per-task WCRT comparison."""
    lines: list[str] = []
    header = (
        f"{'Task':>6} {'Measured':>10} {'Estimated':>10} "
        f"{'Deadline':>10} {'Overest':>10} {'Meets':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for tid, m, e, d in zip(task_ids, measured, estimated, deadlines):
        over = (e - m) / m if m > 0 else 0.0
        meets = "YES" if e <= d else "NO"
        lines.append(f"{tid:>6} {m:>10.2f} {e:>10.2f} {d:>10.2f} {over:>10.2%} {meets:>8}")

    return "\n".join(lines)


def export_results_csv(results: list[ExperimentResult], file_path: str) -> None:
    """Export experiment results to CSV."""
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "num_tasks", "utilization", "tp", "tn",
            "fp", "fn", "accuracy", "analysis_time_ms",
        ])
        for r in results:
            total = r.tp + r.tn + r.fp + r.fn
            acc = (r.tp + r.tn) / total if total > 0 else 0.0
            writer.writerow([
                r.num_tasks, r.utilization, r.tp, r.tn,
                r.fp, r.fn, f"{acc:.4f}", f"{r.analysis_time_ms:.2f}",
            ])
    logger.info("Results exported to %s", file_path)


def export_wcrt_comparison_csv(
    task_ids: list[int],
    measured: list[float],
    estimated: list[float],
    deadlines: list[float],
    file_path: str,
) -> None:
    """Export per-task WCRT comparison to CSV."""
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "measured_wcrt", "estimated_wcrt",
            "deadline", "overestimation",
        ])
        for tid, m, e, d in zip(task_ids, measured, estimated, deadlines):
            over = (e - m) / m if m > 0 else 0.0
            writer.writerow([tid, f"{m:.2f}", f"{e:.2f}", f"{d:.2f}", f"{over:.4f}"])
    logger.info("WCRT comparison exported to %s", file_path)
