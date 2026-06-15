"""Tkinter GUI for CFS WCRT Analysis."""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Final

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    tk = None  # type: ignore

from typing import TYPE_CHECKING

from ..analysis import WCRTAnalyzer
from ..core import CFSConfig, TaskParams
from ..generation import TaskGenConfig, generate_task_set
from ..optimization import DeadlineAwareHeuristic, GeneticNiceAssignment
from ..simulation import CFSSimulator
from .charts import plot_wcrt_comparison
from .tables import export_wcrt_comparison_csv, format_task_detail_table

if TYPE_CHECKING:
    from ..analysis import SystemWCRTResult

logger = logging.getLogger(__name__)

MIN_TASKS: Final[int] = 2
MAX_TASKS: Final[int] = 50
DEF_TASKS: Final[int] = 8
MIN_UTIL: Final[float] = 0.05
MAX_UTIL: Final[float] = 0.95
DEF_UTIL: Final[float] = 0.5


class _StatusBar(ttk.Frame):
    """Status bar widget."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, relief=tk.SUNKEN, padding=2)
        self.label = ttk.Label(self, text="Ready", anchor=tk.W)
        self.label.pack(fill=tk.X, side=tk.LEFT)
        self.pack(fill=tk.X, side=tk.BOTTOM)

    def set(self, text: str) -> None:
        self.label.config(text=text)
        self.update_idletasks()


class _TaskSetFrame(ttk.LabelFrame):
    """Task set generation controls."""

    def __init__(self, parent: ttk.Frame, status: _StatusBar) -> None:
        super().__init__(parent, text="Task Set Generation", padding=10)
        self.status = status
        self._setup_widgets()

    def _setup_widgets(self) -> None:
        f = ttk.Frame(self)
        f.pack(fill=tk.X)

        ttk.Label(f, text="Number of Tasks:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2,
        )
        self.num_spin = ttk.Spinbox(f, from_=MIN_TASKS, to=MAX_TASKS, width=8)
        self.num_spin.set(str(DEF_TASKS))
        self.num_spin.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(f, text="Utilization:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2,
        )
        self.util_spin = ttk.Spinbox(
            f, from_=MIN_UTIL, to=MAX_UTIL, increment=0.05, width=8,
        )
        self.util_spin.set(str(DEF_UTIL))
        self.util_spin.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(f, text="Seed:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2,
        )
        self.seed_spin = ttk.Spinbox(f, from_=0, to=999999, width=8)
        self.seed_spin.set("42")
        self.seed_spin.grid(row=2, column=1, padx=5, pady=2)

        self.gen_btn = ttk.Button(f, text="Generate Tasks")
        self.gen_btn.grid(row=3, column=0, columnspan=2, pady=5)

    def get_config(self) -> tuple[int, float, int]:
        num = int(self.num_spin.get())
        util = float(self.util_spin.get())
        seed = int(self.seed_spin.get())
        return num, util, seed


class _TaskTable(ttk.Frame):
    """Scrollable task table."""

    def __init__(self, parent: ttk.Frame) -> None:
        super().__init__(parent)
        self.tree: ttk.Treeview | None = None
        self._setup_widgets()

    def _setup_widgets(self) -> None:
        cols = ("ID", "WCET", "Period", "Deadline", "Weight", "Nice")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=8)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=70, anchor=tk.CENTER)
        self.tree.column("ID", width=40)
        self.tree.column("Deadline", width=80)

        vsb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def set_tasks(self, tasks: list[TaskParams]) -> None:
        if not self.tree:
            return
        for item in self.tree.get_children():
            self.tree.delete(item)
        for t in tasks:
            self.tree.insert("", tk.END, values=(
                t.task_id, f"{t.wcet:.2f}", f"{t.period:.1f}",
                f"{t.deadline:.1f}", t.weight, t.nice,
            ))


class _ResultsFrame(ttk.LabelFrame):
    """Results display area."""

    def __init__(self, parent: ttk.Frame) -> None:
        super().__init__(parent, text="Results", padding=10)
        self.text: tk.Text | None = None
        self._setup_widgets()

    def _setup_widgets(self) -> None:
        f = ttk.Frame(self)
        f.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(f, wrap=tk.WORD, height=12, width=80)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(f, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def set_text(self, text: str) -> None:
        if not self.text:
            return
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, text)

    def append(self, text: str) -> None:
        if not self.text:
            return
        self.text.insert(tk.END, text)
        self.text.see(tk.END)


class HFApp:
    """Main application window for CFS WCRT Analysis."""

    def __init__(self) -> None:
        if tk is None:
            raise ImportError("tkinter is not available on this system")

        self.root = tk.Tk()
        self.root.title("CFS WCRT Analysis Tool")
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)

        self.config = CFSConfig()
        self.analyzer = WCRTAnalyzer(self.config)
        self.simulator = CFSSimulator(self.config)
        self.gen_config = TaskGenConfig()

        self._tasks: list[TaskParams] = []
        self._measured: dict[int, float] = {}
        self._analysis_result: SystemWCRTResult | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self._status = _StatusBar(self.root)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main_frame, width=350)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right.pack_propagate(False)

        self._gen_frame = _TaskSetFrame(left, self._status)
        self._gen_frame.pack(fill=tk.X, pady=(0, 5))
        self._gen_frame.gen_btn.config(command=self._on_generate)

        self._table = _TaskTable(left)
        self._table.pack(fill=tk.BOTH, expand=True, pady=5)

        self._results = _ResultsFrame(right)
        self._results.pack(fill=tk.BOTH, expand=True)

        self._build_action_buttons(right)

    def _build_action_buttons(self, parent: ttk.Frame) -> None:
        self._action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        self._action_frame.pack(fill=tk.X, pady=(5, 0))

        self._analyze_btn = ttk.Button(
            self._action_frame, text="1. Run Analysis",
            command=self._on_analyze,
        )
        self._analyze_btn.pack(fill=tk.X, pady=2)
        self._analyze_btn.config(state=tk.DISABLED)

        self._simulate_btn = ttk.Button(
            self._action_frame, text="2. Run Simulation",
            command=self._on_simulate,
        )
        self._simulate_btn.pack(fill=tk.X, pady=2)
        self._simulate_btn.config(state=tk.DISABLED)

        self._compare_btn = ttk.Button(
            self._action_frame, text="3. Compare Results",
            command=self._on_compare,
        )
        self._compare_btn.pack(fill=tk.X, pady=2)
        self._compare_btn.config(state=tk.DISABLED)

        ttk.Separator(self._action_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=5,
        )

        self._optimize_btn = ttk.Button(
            self._action_frame, text="Optimize Nice Values",
            command=self._on_optimize,
        )
        self._optimize_btn.pack(fill=tk.X, pady=2)
        self._optimize_btn.config(state=tk.DISABLED)

        ttk.Separator(self._action_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=5,
        )

        self._export_btn = ttk.Button(
            self._action_frame, text="Export CSV...",
            command=self._on_export,
        )
        self._export_btn.pack(fill=tk.X, pady=2)
        self._export_btn.config(state=tk.DISABLED)

        self._plot_btn = ttk.Button(
            self._action_frame, text="Show Chart",
            command=self._on_plot,
        )
        self._plot_btn.pack(fill=tk.X, pady=2)
        self._plot_btn.config(state=tk.DISABLED)

    def _on_generate(self) -> None:
        num, util, seed = self._gen_frame.get_config()
        random.seed(seed)
        self._status.set(f"Generating {num} tasks at utilization {util:.2f}...")
        self.root.update_idletasks()

        try:
            self._tasks = generate_task_set(num, util, self.gen_config)
            self._table.set_tasks(self._tasks)
            self._results.set_text(
                f"Generated {len(self._tasks)} tasks\n"
                f"Utilization: {util:.2f}\n"
                f"Seed: {seed}\n"
                + "-" * 40 + "\n"
            )
            self._analyze_btn.config(state=tk.NORMAL)
            self._simulate_btn.config(state=tk.DISABLED)
            self._compare_btn.config(state=tk.DISABLED)
            self._optimize_btn.config(state=tk.DISABLED)
            self._export_btn.config(state=tk.DISABLED)
            self._plot_btn.config(state=tk.DISABLED)
            self._status.set(f"Generated {len(self._tasks)} tasks. Ready to analyze.")
        except Exception as e:
            messagebox.showerror("Generation Error", str(e))
            self._status.set("Generation failed.")

    def _on_analyze(self) -> None:
        if not self._tasks:
            return
        self._run_async("Running WCRT Analysis...", self._do_analyze)

    def _do_analyze(self) -> str:
        start = time.perf_counter()
        result = self.analyzer.analyze(self._tasks)
        self._analysis_result = result
        elapsed = (time.perf_counter() - start) * 1000

        lines = ["WCRT Analysis Results", "=" * 40, ""]
        for r in sorted(result.results, key=lambda r: r.task_id):
            ok = "OK" if r.schedulable else "MISS"
            lines.append(f"Task {r.task_id}: WCRT={r.estimated_wcrt:.2f}ms ({ok})")
        lines.append("")
        lines.append(f"System Schedulable: {result.system_schedulable}")
        lines.append(f"Analysis Time: {elapsed:.2f}ms")
        return "\n".join(lines)

    def _on_simulate(self) -> None:
        if not self._tasks:
            return
        self._run_async("Running Simulation...", self._do_simulate)

    def _do_simulate(self) -> str:
        start = time.perf_counter()
        self._measured = self.simulator.measure_wcrt(
            self._tasks, hyperperiod_factor=2.0, num_runs=5,
        )
        elapsed = (time.perf_counter() - start) * 1000

        lines = ["Simulation Results", "=" * 40, ""]
        for t in self._tasks:
            wcrt = self._measured.get(t.task_id, 0.0)
            status = "OK" if wcrt <= t.deadline else "MISS"
            lines.append(f"Task {t.task_id}: Measured WCRT={wcrt:.2f}ms ({status})")
        lines.append("")
        lines.append(f"Simulation Time: {elapsed:.2f}ms")
        return "\n".join(lines)

    def _on_compare(self) -> None:
        if not self._tasks or self._analysis_result is None or not self._measured:
            return

        task_ids = [t.task_id for t in self._tasks]
        measured = [self._measured.get(t.task_id, 0.0) for t in self._tasks]
        estimated = [
            r.estimated_wcrt
            for r in sorted(self._analysis_result.results, key=lambda r: r.task_id)
        ]
        deadlines = [t.deadline for t in self._tasks]

        text = format_task_detail_table(task_ids, measured, estimated, deadlines)
        self._results.set_text(text)
        self._export_btn.config(state=tk.NORMAL)
        self._plot_btn.config(state=tk.NORMAL)
        self._status.set("Comparison complete. Export or plot results.")

    def _on_optimize(self) -> None:
        if not self._tasks:
            return
        self._run_async("Optimizing Nice Values...", self._do_optimize)

    def _do_optimize(self) -> str:
        lines = ["Nice Value Optimization", "=" * 40, ""]
        h = DeadlineAwareHeuristic(self.analyzer)
        h_result = h.assign(self._tasks)
        lines.append(f"Heuristic: {str(h_result.schedulable)}")
        lines.append("")

        lines.append("Running Genetic Algorithm (50 gen, 10s timeout)...")
        ga = GeneticNiceAssignment(
            self.analyzer, population_size=50,
            max_generations=50, timeout_seconds=10.0,
        )
        ga_result = ga.assign(self._tasks)
        lines.append(f"Genetic: {str(ga_result.schedulable)}")
        lines.append("")
        lines.append("Nice Assignments:")
        for task_id, nice_val in sorted(ga_result.nice_values.items()):
            lines.append(f"  Task {task_id}: nice={nice_val}")
        return "\n".join(lines)

    def _on_export(self) -> None:
        if not self._tasks or not self._measured or self._analysis_result is None:
            return
        fname = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if fname:
            task_ids = [t.task_id for t in self._tasks]
            measured = [self._measured.get(t.task_id, 0.0) for t in self._tasks]
            estimated = [
                r.estimated_wcrt
                for r in sorted(self._analysis_result.results, key=lambda r: r.task_id)
            ]
            deadlines = [t.deadline for t in self._tasks]
            export_wcrt_comparison_csv(task_ids, measured, estimated, deadlines, fname)
            self._status.set(f"Exported to {Path(fname).name}")

    def _on_plot(self) -> None:
        if not self._tasks or not self._measured or self._analysis_result is None:
            return
        task_ids = [t.task_id for t in self._tasks]
        measured = [self._measured.get(t.task_id, 0.0) for t in self._tasks]
        estimated = [
            r.estimated_wcrt
            for r in sorted(self._analysis_result.results, key=lambda r: r.task_id)
        ]
        deadlines = [t.deadline for t in self._tasks]

        try:
            plot_wcrt_comparison(task_ids, measured, estimated, deadlines)
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def _run_async(self, status_text: str, work):  # type: ignore[no-untyped-def]
        assert tk is not None
        self._status.set(status_text)
        for child in self._action_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.config(state=tk.DISABLED)

        result_text: str = ""

        def _done() -> None:
            self._results.set_text(result_text)
            self._status.set("Done.")
            for child in self._action_frame.winfo_children():
                if isinstance(child, ttk.Button):
                    child.config(state=tk.NORMAL)
            self._simulate_btn.config(
                state=tk.NORMAL if self._tasks else tk.DISABLED,
            )
            self._compare_btn.config(
                state=tk.NORMAL if self._tasks and self._measured else tk.DISABLED,
            )
            self._optimize_btn.config(
                state=tk.NORMAL if self._tasks else tk.DISABLED,
            )

        def _run() -> None:
            nonlocal result_text
            try:
                result_text = work()
            except Exception as e:
                result_text = f"Error: {e}"
                logger.exception("Async work failed")
            self.root.after(0, _done)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _on_close(self) -> None:
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    """Launch the GUI application."""
    if tk is None:
        print("tkinter is required for the GUI. Install python-tk or use --cli mode.")
        return
    app = HFApp()
    app.run()


if __name__ == "__main__":
    main()
