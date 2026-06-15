"""Chart generation for experiment results."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def plot_wcrt_comparison(
    task_ids: list[int],
    measured: list[float],
    estimated: list[float],
    deadlines: list[float],
    title: str = "WCRT: Measured vs Estimated",
    save_path: str | None = None,
) -> None:
    """Plot bar chart comparing measured vs estimated WCRT."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available")
        return

    x = np.arange(len(task_ids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, measured, w, label="Measured (Simulator)", color="#2196F3")
    ax.bar(x + w / 2, estimated, w, label="Estimated (Analysis)", color="#FF5722")

    for i, d in enumerate(deadlines):
        ax.plot([i - 0.5, i + 0.5], [d, d], "k--", alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Task ID")
    ax.set_ylabel("Response Time (ms)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(tid) for tid in task_ids])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", save_path)
    else:
        plt.show()


def plot_schedulability_heatmap(
    num_tasks_list: list[int],
    utilizations: list[float],
    schedulable_counts: list[list[int]],
    method: str = "WCRT Analysis",
    save_path: str | None = None,
) -> None:
    """Plot heatmap of schedulable task sets."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available")
        return

    data = np.array(schedulable_counts)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(utilizations)))
    ax.set_xticklabels([f"{u:.1f}" for u in utilizations])
    ax.set_yticks(range(len(num_tasks_list)))
    ax.set_yticklabels([str(n) for n in num_tasks_list])
    ax.set_xlabel("System Utilization")
    ax.set_ylabel("Number of Tasks")
    ax.set_title(f"Schedulable Task Sets ({method})")

    for i in range(len(num_tasks_list)):
        for j in range(len(utilizations)):
            ax.text(j, i, str(data[i, j]), ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax, label="Schedulable (out of 100)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Heatmap saved to %s", save_path)
    else:
        plt.show()
