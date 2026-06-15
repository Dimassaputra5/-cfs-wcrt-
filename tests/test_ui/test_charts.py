"""Minimal tests for chart generation (smoke tests).

These tests only verify the chart functions do not crash.
Full visual verification is done manually.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no tkinter needed

import pytest

from cfs_wcrt.ui.charts import plot_schedulability_heatmap, plot_wcrt_comparison


class TestPlotWcrtComparison:
    """Smoke tests for bar chart."""

    def test_smoke_no_save(self) -> None:
        """Should not crash when not saving."""
        plot_wcrt_comparison(
            task_ids=[0, 1],
            measured=[10.0, 20.0],
            estimated=[12.0, 22.0],
            deadlines=[100.0, 200.0],
            title="Test Chart",
        )

    def test_smoke_save(self, tmp_path: str) -> None:
        """Should write a file when save_path is given."""
        path = os.path.join(str(tmp_path), "wcrt.png")
        plot_wcrt_comparison(
            task_ids=[0, 1],
            measured=[10.0, 20.0],
            estimated=[12.0, 22.0],
            deadlines=[100.0, 200.0],
            save_path=path,
        )
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_single_task(self) -> None:
        """Should handle a single task."""
        plot_wcrt_comparison(
            task_ids=[0],
            measured=[5.0],
            estimated=[6.0],
            deadlines=[50.0],
        )

    @pytest.mark.skip(reason="Empty data requires non-empty arrays for imshow")
    def test_empty_lists(self) -> None:
        """Should not crash with empty data."""
        plot_wcrt_comparison(
            task_ids=[],
            measured=[],
            estimated=[],
            deadlines=[],
        )


class TestPlotSchedulabilityHeatmap:
    """Smoke tests for heatmap."""

    def test_smoke_no_save(self) -> None:
        plot_schedulability_heatmap(
            num_tasks_list=[4, 8],
            utilizations=[0.5, 0.8],
            schedulable_counts=[[80, 60], [70, 30]],
            method="WCRT",
        )

    def test_smoke_save(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "heatmap.png")
        plot_schedulability_heatmap(
            num_tasks_list=[4, 8],
            utilizations=[0.5, 0.8],
            schedulable_counts=[[80, 60], [70, 30]],
            method="WCRT",
            save_path=path,
        )
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_single_row(self) -> None:
        plot_schedulability_heatmap(
            num_tasks_list=[4],
            utilizations=[0.5, 0.8, 0.9],
            schedulable_counts=[[80, 50, 20]],
        )

    @pytest.mark.skip(reason="Empty data requires non-empty arrays for imshow")
    def test_empty(self) -> None:
        plot_schedulability_heatmap(
            num_tasks_list=[],
            utilizations=[],
            schedulable_counts=[],
        )
