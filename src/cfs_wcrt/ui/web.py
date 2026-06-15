"""Streamlit web UI for CFS WCRT Analysis."""

from __future__ import annotations

import io
import logging
import random
from typing import Any

import streamlit as st

from cfs_wcrt.analysis import WCRTAnalyzer
from cfs_wcrt.core import CFSConfig
from cfs_wcrt.generation import TaskGenConfig, generate_task_set
from cfs_wcrt.optimization import DeadlineAwareHeuristic, GeneticNiceAssignment
from cfs_wcrt.simulation import CFSSimulator
from cfs_wcrt.ui.tables import format_task_detail_table

logger = logging.getLogger(__name__)

MIN_TASKS = 2
MAX_TASKS = 50
DEF_TASKS = 8
MIN_UTIL = 0.05
MAX_UTIL = 0.95
DEF_UTIL = 0.5

st.set_page_config(
    page_title="CFS WCRT Analysis",
    page_icon="⏱️",
    layout="wide",
)


def _init_session() -> None:
    """Initialize session state."""
    if "config" not in st.session_state:
        st.session_state.config = CFSConfig()
        st.session_state.analyzer = WCRTAnalyzer(st.session_state.config)
        st.session_state.simulator = CFSSimulator(st.session_state.config)
        st.session_state.gen_config = TaskGenConfig()

    defaults: list[tuple[str, Any]] = [
        ("tasks", []),
        ("task_df", None),
        ("measured", {}),
        ("analysis_result", None),
        ("analysis_done", False),
        ("simulation_done", False),
    ]
    for key, value in defaults:
        if key not in st.session_state:
            st.session_state[key] = value


def _generate_tasks(num: int, util: float, seed: int) -> None:
    """Generate task set and store in session state."""
    random.seed(seed)
    with st.spinner(f"Generating {num} tasks at utilization {util:.2f}..."):
        try:
            tasks = generate_task_set(num, util, st.session_state.gen_config)
            st.session_state.tasks = tasks
            st.session_state.analysis_done = False
            st.session_state.simulation_done = False
            st.session_state.measured = {}
            st.session_state.analysis_result = None
            st.session_state.task_df = _tasks_to_df(tasks)
            st.success(f"Generated {len(tasks)} tasks")
        except Exception as e:
            st.error(f"Generation failed: {e}")


def _tasks_to_df(tasks: list[Any]) -> Any:
    """Convert task list to display-friendly records."""
    import pandas as pd  # type: ignore[import-untyped]  # noqa: TCH002
    rows: list[dict[str, Any]] = []
    for t in tasks:
        rows.append({
            "ID": t.task_id,
            "WCET": t.wcet,
            "Period": t.period,
            "Deadline": t.deadline,
            "Weight": t.weight,
            "Nice": t.nice,
        })
    return pd.DataFrame(rows)


def _run_analysis() -> None:
    """Run WCRT analysis."""
    tasks = st.session_state.tasks
    if not tasks:
        st.warning("Generate tasks first")
        return
    with st.spinner("Running WCRT analysis..."):
        import time
        start = time.perf_counter()
        result = st.session_state.analyzer.analyze(tasks)
        elapsed = (time.perf_counter() - start) * 1000
        st.session_state.analysis_result = result
        st.session_state.analysis_done = True
        sched = result.system_schedulable
        st.success(f"Analysis done in {elapsed:.2f}ms — System schedulable: {sched}")


def _run_simulation() -> None:
    """Run CFS simulation."""
    tasks = st.session_state.tasks
    if not tasks:
        st.warning("Generate tasks first")
        return
    with st.spinner("Running simulation (this may take a while)..."):
        import time
        start = time.perf_counter()
        measured = st.session_state.simulator.measure_wcrt(
            tasks, hyperperiod_factor=2.0, num_runs=5,
        )
        elapsed = (time.perf_counter() - start) * 1000
        st.session_state.measured = measured
        st.session_state.simulation_done = True
        st.success(f"Simulation done in {elapsed:.2f}ms")


def _show_analysis_results() -> None:
    """Display analysis results."""
    result = st.session_state.analysis_result
    if not result:
        return

    col1, col2 = st.columns(2)
    col1.metric("System Schedulable", "✅ Yes" if result.system_schedulable else "❌ No")
    num_sched = sum(1 for r in result.results if r.schedulable)
    col2.metric("Tasks Schedulable", f"{num_sched}/{len(result.results)}")

    data = []
    for r in sorted(result.results, key=lambda x: x.task_id):
        data.append({
            "Task": r.task_id,
            "WCRT (ms)": f"{r.estimated_wcrt:.2f}",
            "Schedulable": "✅" if r.schedulable else "❌",
            "Iterations": r.iterations,
            "Time (us)": f"{r.analysis_time_us:.1f}",
        })
    st.dataframe(data, use_container_width=True, hide_index=True)


def _show_simulation_results() -> None:
    """Display simulation results."""
    tasks = st.session_state.tasks
    measured = st.session_state.measured
    if not tasks or not measured:
        return

    data = []
    for t in tasks:
        wcrt = measured.get(t.task_id, 0.0)
        ok = wcrt <= t.deadline
        data.append({
            "Task": t.task_id,
            "Measured WCRT (ms)": f"{wcrt:.2f}",
            "Deadline (ms)": f"{t.deadline:.1f}",
            "Status": "✅ OK" if ok else "❌ MISS",
        })
    st.dataframe(data, use_container_width=True, hide_index=True)


def _show_comparison() -> None:
    """Show WCRT comparison table + chart."""
    tasks = st.session_state.tasks
    measured = st.session_state.measured
    result = st.session_state.analysis_result
    if not tasks or not measured or not result:
        st.warning("Run analysis and simulation first")
        return

    task_ids = [t.task_id for t in tasks]
    m_vals = [measured.get(t.task_id, 0.0) for t in tasks]
    e_vals = [r.estimated_wcrt for r in sorted(result.results, key=lambda x: x.task_id)]
    deadlines = [t.deadline for t in tasks]

    text = format_task_detail_table(task_ids, m_vals, e_vals, deadlines)
    st.text(text)

    _plot_comparison_chart(task_ids, m_vals, e_vals, deadlines)


def _plot_comparison_chart(
    task_ids: list[int],
    measured: list[float],
    estimated: list[float],
    deadlines: list[float],
) -> None:
    """Render matplotlib comparison chart in Streamlit."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(len(task_ids))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - w / 2, measured, w, label="Measured (Simulator)", color="#2196F3")
        ax.bar(x + w / 2, estimated, w, label="Estimated (Analysis)", color="#FF5722")
        for i, d in enumerate(deadlines):
            ax.plot([i - 0.5, i + 0.5], [d, d], "k--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Task ID")
        ax.set_ylabel("Response Time (ms)")
        ax.set_title("WCRT: Measured vs Estimated")
        ax.set_xticks(x)
        ax.set_xticklabels([str(tid) for tid in task_ids])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.warning("matplotlib not available for chart")


def _run_optimize() -> None:
    """Run nice value optimization."""
    tasks = st.session_state.tasks
    if not tasks:
        st.warning("Generate tasks first")
        return

    with st.spinner("Running Deadline-Aware Heuristic..."):
        h = DeadlineAwareHeuristic(st.session_state.analyzer)
        h_result = h.assign(tasks)
        st.info(f"Heuristic — Schedulable: {h_result.schedulable}")

    with st.spinner("Running Genetic Algorithm (50 gen, 10s timeout)..."):
        ga = GeneticNiceAssignment(
            st.session_state.analyzer,
            population_size=50,
            max_generations=50,
            timeout_seconds=10.0,
        )
        ga_result = ga.assign(tasks)
        st.info(f"Genetic Algorithm — Schedulable: {ga_result.schedulable}")

        data = []
        for task_id, nice_val in sorted(ga_result.nice_values.items()):
            data.append({"Task": task_id, "Nice Value": nice_val})
        if data:
            import pandas as pd
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def _csv_download() -> bytes | None:
    """Generate CSV bytes for download."""
    tasks = st.session_state.tasks
    measured = st.session_state.measured
    result = st.session_state.analysis_result
    if not tasks or not measured or not result:
        return None

    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["task_id", "wcet", "period", "deadline",
                 "measured_wcrt", "estimated_wcrt", "schedulable"])
    for t in tasks:
        m = measured.get(t.task_id, 0.0)
        e = next(
            (r.estimated_wcrt for r in result.results if r.task_id == t.task_id),
            0.0,
        )
        sched = next(
            (r.schedulable for r in result.results if r.task_id == t.task_id),
            False,
        )
        w.writerow([t.task_id, t.wcet, t.period, t.deadline,
                     f"{m:.2f}", f"{e:.2f}", sched])
    return buf.getvalue().encode("utf-8")


def _hyperperiod(periods: list[float]) -> int:
    """Compute LCM of periods."""
    import math
    int_periods = [int(round(p)) for p in periods]
    result = int_periods[0]
    for p in int_periods[1:]:
        result = result * p // math.gcd(result, p)
    return result


# ── Main UI ──────────────────────────────────────────────────────────────

_init_session()

st.title("⏱️ CFS WCRT Analysis Tool")
st.markdown(
    "Worst-Case Response Time Analysis for Completely Fair Scheduling "
    "in Linux Systems — based on Yoon et al. (2025)"
)

# ── Sidebar: Task Generation ─────────────────────────────────────────────

with st.sidebar:
    st.header("Task Set Generation")

    num_tasks = st.number_input("Number of Tasks", MIN_TASKS, MAX_TASKS, DEF_TASKS)
    utilization = st.slider("Utilization", MIN_UTIL, MAX_UTIL, DEF_UTIL, 0.05)
    seed = st.number_input("Random Seed", 0, 999999, 42)

    if st.button("🎲 Generate Task Set", use_container_width=True, type="primary"):
        _generate_tasks(int(num_tasks), float(utilization), int(seed))

    st.divider()
    st.header("Actions")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔍 Analyze", use_container_width=True):
            _run_analysis()
    with col_b:
        if st.button("⚙️ Simulate", use_container_width=True):
            _run_simulation()

    if st.button("📊 Compare Results", use_container_width=True):
        _show_comparison()

    if st.button("🧬 Optimize Nice Values", use_container_width=True):
        _run_optimize()

    csv_bytes = _csv_download()
    if csv_bytes:
        st.download_button(
            "⬇️ Export CSV",
            data=csv_bytes,
            file_name="cfs_wcrt_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Panduan ──────────────────────────────────────────────────────────────

def _show_panduan() -> None:
    """Tampilkan panduan penggunaan dalam Bahasa Indonesia."""
    st.markdown("""
    ## 📖 Panduan Penggunaan CFS WCRT Analysis Tool

    ---

    ### 🎯 Apa Itu CFS WCRT Analysis?

    Tools ini menganalisis **Worst-Case Response Time (WCRT)** untuk penjadwalan
    **Completely Fair Scheduler (CFS)** di kernel Linux. Berdasarkan penelitian:

    > Yoon et al., *"Worst case response time analysis for completely fair
    > scheduling in Linux systems"*, Real-Time Systems, 2025.

    CFS adalah penjadwal default Linux yang membagi waktu CPU secara proporsional
    berdasarkan *nice value* setiap tugas. Tools ini menyediakan:

    - **Analisis WCRT** — Estimasi konservatif response time terburuk
    - **Simulasi CFS** — Pengukuran aktual via discrete-event simulator
    - **Optimasi Nice Value** — Algoritma penentuan prioritas tugas

    ---

    ### 🚀 Cara Penggunaan

    #### 1. Generate Task Set

    | Parameter | Deskripsi |
    |-----------|-----------|
    | **Number of Tasks** | Jumlah tugas dalam set (2–50) |
    | **Utilization** | Total utilisasi sistem (0.05–0.95) |
    | **Random Seed** | Nilai awal untuk reproduksibilitas |

    Klik **🎲 Generate Task Set** untuk membuat tugas sintetis menggunakan
    algoritma **UUniFast**. Tugas akan muncul di tab *📋 Task Set*.

    #### 2. Analisis WCRT

    Klik **🔍 Analyze** untuk menjalankan **Algoritma 1** (fixed-point iteration)
    yang menghitung estimasi WCRT konservatif untuk setiap tugas.

    Hasil ditampilkan di tab *📈 Analysis*:
    - **System Schedulable** — Apakah semua tugas memenuhi deadline?
    - **WCRT (ms)** — Estimated worst-case response time
    - **Iterations** — Jumlah iterasi hingga konvergen

    #### 3. Simulasi CFS

    Klik **⚙️ Simulate** untuk menjalankan discrete-event simulator CFS.
    Simulator ini meniru perilaku penjadwal CFS Linux:
    - Pelacakan *vruntime* (Definisi 1 & 2)
    - Alokasi *timeslice* dinamis (Definisi 3 & 4)
    - Penyesuaian *wake-up* (Definisi 5)
    - Tick jiffy periodik

    Hasil ditampilkan di tab *🎯 Simulation*.

    #### 4. Perbandingan

    Klik **📊 Compare Results** untuk melihat perbandingan antara:
    - **Measured WCRT** — Hasil simulasi aktual
    - **Estimated WCRT** — Hasil analisis
    - **Deadline** — Batas waktu setiap tugas
    - **Overestimation** — Seberapa konservatif estimasi

    Jika estimasi ≤ deadline → tugas **schedulable ✅**

    #### 5. Optimasi Nice Value

    Klik **🧬 Optimize Nice Values** untuk menjalankan:
    - **Algorithm 2** — Deadline-Aware Heuristic
    - **Algorithm 3** — Genetic Algorithm

    Kedua algoritma mencari assignment *nice value* optimal agar semua tugas
    schedulable.

    #### 6. Export CSV

    Klik **⬇️ Export CSV** untuk mendownload hasil analisis dan simulasi
    dalam format CSV.

    ---

    ### 📋 Penjelasan Parameter

    | Parameter | Default | Rentang | Penjelasan |
    |-----------|---------|---------|------------|
    | `target_latency` | 18 ms | > 0 | Interval waktu ideal untuk semua tugas berjalan sekali (L) |
    | `min_granularity` | 2.25 ms | > 0 | Timeslice minimum untuk cegah context switch berlebihan (G) |
    | `jiffy` | 1 ms | > 0 | Interval tick timer Linux (J) |
    | `sched_nr_latency` | 8 | > 0 | Maks tugas untuk pembagian timeslice akurat |
    | `num_cores` | 1 | ≥ 1 | Jumlah core CPU (M=1 untuk single-core) |
    | `min_period` | 30 ms | — | Periode minimum tugas yang di-generate |
    | `max_period` | 3000 ms | — | Periode maksimum tugas yang di-generate |
    | `default_nice` | 0 | -20–19 | Nice value default untuk semua tugas |

    ---

    ### 💻 Menjalankan dari CLI

    Tools ini juga bisa dijalankan dari command line:

    ```bash
    # Eksperimen 1: Analisis schedulability
    python -m cfs_wcrt --experiment exp1

    # Eksperimen 2: Perbandingan nice value assignment
    python -m cfs_wcrt --experiment exp2

    # Eksperimen 3: WCRT measured vs estimated (dengan chart)
    python -m cfs_wcrt --experiment exp3
    ```

    ---

    ### 📚 Referensi

    1. Yoon, P., Kim, J., & Lee, C. (2025). Worst case response time analysis
       for completely fair scheduling in Linux systems. *Real-Time Systems*.
    2. Bini, E. & Buttazzo, G. (2005). Measuring the performance of
       schedulability tests. *Real-Time Systems*.
    3. Linux kernel 5.15 — `kernel/sched/fair.c` (CFS implementation)

    ---

    ### 🏗️ Arsitektur Tools

    ```
    src/cfs_wcrt/
    ├── core/          → Model data & konstanta Linux
    ├── analysis/      → Algoritma 1: Analisis WCRT
    ├── simulation/    → Simulator discrete-event CFS
    ├── optimization/  → Algoritma 2 & 3: Optimasi nice value
    ├── generation/    → Pembangkitan tugas sintetis (UUniFast)
    └── ui/            → CLI, Streamlit, tabel, chart
    ```
    """)


# ── Main Panel ───────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Task Set", "📈 Analysis", "🎯 Simulation", "📊 Comparison", "📖 Panduan",
])

with tab1:
    if st.session_state.task_df is not None:
        st.dataframe(
            st.session_state.task_df,
            use_container_width=True,
            hide_index=True,
        )
        total_util = sum(t.wcet / t.period for t in st.session_state.tasks)
        st.caption(f"Total Utilization: {total_util:.4f}  |  "
                   f"Hyperperiod: {_hyperperiod([t.period for t in st.session_state.tasks])}ms")
    else:
        st.info("Generate a task set from the sidebar to begin.")

with tab2:
    if st.session_state.analysis_done:
        _show_analysis_results()
    else:
        st.info("Run analysis from the sidebar.")

with tab3:
    if st.session_state.simulation_done:
        _show_simulation_results()
    else:
        st.info("Run simulation from the sidebar.")

with tab4:
    if st.session_state.analysis_done and st.session_state.simulation_done:
        _show_comparison()
    else:
        st.info("Run both analysis and simulation first, then compare.")

with tab5:
    _show_panduan()
