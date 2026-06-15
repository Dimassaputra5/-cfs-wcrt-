"""Tests for ASCII table formatters and CSV export.

Tests cover:
    - format_schedulability_table()
    - format_nice_assignment_table()
    - format_task_detail_table()
    - export_results_csv()
    - export_wcrt_comparison_csv()
"""

from __future__ import annotations

import csv
import os

import pytest

from cfs_wcrt import ExperimentResult
from cfs_wcrt.ui.tables import (
    export_results_csv,
    export_wcrt_comparison_csv,
    format_nice_assignment_table,
    format_schedulability_table,
    format_task_detail_table,
)


# ── Sample data ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_results() -> list[ExperimentResult]:
    return [
        ExperimentResult(
            task_params=[], num_tasks=4, utilization=0.5,
            tp=8, tn=2, fp=1, fn=1,
        ),
        ExperimentResult(
            task_params=[], num_tasks=8, utilization=0.8,
            tp=5, tn=4, fp=2, fn=1,
        ),
    ]


@pytest.fixture
def sample_detail() -> tuple[list[int], list[float], list[float], list[float]]:
    return (
        [0, 1, 2],           # task_ids
        [10.0, 20.0, 30.0],  # measured
        [12.0, 22.0, 35.0],  # estimated
        [100.0, 200.0, 300.0],  # deadlines
    )


# ── format_schedulability_table ──────────────────────────────────────────────

class TestFormatSchedulabilityTable:
    """ASCII table for experiment results."""

    def test_header_present(self, sample_results: list[ExperimentResult]) -> None:
        table = format_schedulability_table(sample_results)
        assert "#Tasks" in table
        assert "Util" in table
        assert "TP" in table
        assert "TN" in table
        assert "FP" in table
        assert "FN" in table
        assert "Acc" in table

    def test_contains_data(self, sample_results: list[ExperimentResult]) -> None:
        table = format_schedulability_table(sample_results)
        assert "4" in table
        assert "8" in table
        assert "0.5" in table
        assert "0.8" in table

    def test_empty_results(self) -> None:
        table = format_schedulability_table([])
        assert "Total" in table  # still shows total row
        assert "#Tasks" in table

    def test_single_result(self) -> None:
        r = ExperimentResult(
            task_params=[], num_tasks=4, utilization=0.5,
            tp=10, tn=0, fp=0, fn=0,
        )
        table = format_schedulability_table([r])
        assert "1.0000" in table  # accuracy = (10+0)/10 = 1.0

    def test_total_row(self, sample_results: list[ExperimentResult]) -> None:
        table = format_schedulability_table(sample_results)
        assert "Total" in table

    def test_multiline(self, sample_results: list[ExperimentResult]) -> None:
        table = format_schedulability_table(sample_results)
        lines = table.strip().split("\n")
        assert len(lines) >= 4  # header + separator + 2 data + separator + total


# ── format_nice_assignment_table ─────────────────────────────────────────────

class TestFormatNiceAssignmentTable:
    """Comparison table for nice-value methods."""

    def test_header_present(self, sample_results: list[ExperimentResult]) -> None:
        table = format_nice_assignment_table(
            sample_results, sample_results, sample_results,
        )
        assert "#Tasks" in table
        assert "Util" in table
        assert "Baseline" in table
        assert "Heuristic" in table
        assert "GA" in table

    def test_three_columns(self, sample_results: list[ExperimentResult]) -> None:
        table = format_nice_assignment_table(
            sample_results, sample_results, sample_results,
        )
        assert "Baseline" in table
        assert "Heuristic" in table
        assert "GA" in table


# ── format_task_detail_table ─────────────────────────────────────────────────

class TestFormatTaskDetailTable:
    """Per-task WCRT comparison table."""

    def test_header_present(self, sample_detail: tuple) -> None:
        task_ids, measured, estimated, deadlines = sample_detail
        table = format_task_detail_table(task_ids, measured, estimated, deadlines)
        assert "Task" in table
        assert "Measured" in table
        assert "Estimated" in table
        assert "Deadline" in table
        assert "Overest" in table
        assert "Meets" in table

    def test_data_rows(self, sample_detail: tuple) -> None:
        task_ids, measured, estimated, deadlines = sample_detail
        table = format_task_detail_table(task_ids, measured, estimated, deadlines)
        for tid in task_ids:
            assert str(tid) in table

    def test_yes_no_flag(self, sample_detail: tuple) -> None:
        task_ids, measured, estimated, deadlines = sample_detail
        table = format_task_detail_table(task_ids, measured, estimated, deadlines)
        assert "YES" in table  # all estimated <= deadline in sample


# ── CSV export ───────────────────────────────────────────────────────────────

class TestCSVExport:
    """File export functionality."""

    def test_export_results_csv(self, tmp_path: str, sample_results: list[ExperimentResult]) -> None:
        path = os.path.join(str(tmp_path), "results.csv")
        export_results_csv(sample_results, path)
        assert os.path.exists(path)

        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows
        assert rows[0] == ["num_tasks", "utilization", "tp", "tn", "fp", "fn", "accuracy", "analysis_time_ms"]

    def test_export_wcrt_csv(self, tmp_path: str, sample_detail: tuple) -> None:
        task_ids, measured, estimated, deadlines = sample_detail
        path = os.path.join(str(tmp_path), "wcrt.csv")
        export_wcrt_comparison_csv(task_ids, measured, estimated, deadlines, path)
        assert os.path.exists(path)

        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 4  # header + 3 data rows
        assert rows[0][0] == "task_id"
