"""Streamlit web UI for CFS WCRT Analysis — Bahasa Indonesia."""

from __future__ import annotations

import io
import logging
import math
import random
from typing import Any

import streamlit as st

from cfs_wcrt.analysis import WCRTAnalyzer
from cfs_wcrt.core import CFSConfig
from cfs_wcrt.generation import TaskGenConfig, generate_task_set
from cfs_wcrt.optimization import DeadlineAwareHeuristic, GeneticNiceAssignment
from cfs_wcrt.simulation import CFSSimulator

logger = logging.getLogger(__name__)

MIN_TASKS = 2
MAX_TASKS = 50
DEF_TASKS = 8
MIN_UTIL = 0.05
MAX_UTIL = 0.95
DEF_UTIL = 0.5

st.set_page_config(
    page_title="Analisis CFS WCRT",
    page_icon=None,
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
    with st.spinner(f"Membuat {num} tugas dengan utilisasi {util:.0%}..."):
        try:
            tasks = generate_task_set(num, util, st.session_state.gen_config)
            st.session_state.tasks = tasks
            st.session_state.analysis_done = False
            st.session_state.simulation_done = False
            st.session_state.measured = {}
            st.session_state.analysis_result = None
            st.session_state.task_df = _tasks_to_df(tasks)
            st.success(f"Berhasil membuat {len(tasks)} tugas")
        except Exception as e:
            st.error(f"Gagal membuat tugas: {e}")


def _tasks_to_df(tasks: list[Any]) -> Any:
    """Convert task list to display-friendly records."""
    import pandas as pd  # type: ignore[import-untyped]  # noqa: TCH002
    rows: list[dict[str, Any]] = []
    for t in tasks:
        rows.append({
            "ID": t.task_id,
            "WCET": t.wcet,
            "Periode": t.period,
            "Deadline": t.deadline,
            "Weight": t.weight,
            "Nice": t.nice,
        })
    return pd.DataFrame(rows)


def _run_analysis() -> None:
    """Run WCRT analysis."""
    tasks = st.session_state.tasks
    if not tasks:
        st.warning("Buat tugas terlebih dahulu")
        return
    with st.spinner("Menjalankan analisis WCRT..."):
        import time
        start = time.perf_counter()
        result = st.session_state.analyzer.analyze(tasks)
        elapsed = (time.perf_counter() - start) * 1000
        st.session_state.analysis_result = result
        st.session_state.analysis_done = True
        sched = result.system_schedulable
        label = ":green[Ya]" if sched else ":red[Tidak]"
        st.success(f"Analisis selesai dalam {elapsed:.2f}ms — Sistem schedulable: {label}")


def _run_simulation() -> None:
    """Run CFS simulation with progress per run."""
    tasks = st.session_state.tasks
    if not tasks:
        st.warning("Buat tugas terlebih dahulu")
        return

    # Compute hyperperiod and duration (same logic as measure_wcrt)
    periods = [int(t.period) for t in tasks]
    hyp = periods[0]
    for p in periods[1:]:
        hyp = hyp * p // math.gcd(hyp, p)
    duration = float(hyp) * 2.0  # hyperperiod_factor=2.0
    num_runs = 5

    status_text = st.status("Menjalankan simulasi CFS...", expanded=False)
    progress_bar = st.progress(0.0)
    all_response_times: dict[int, list[float]] = {t.task_id: [] for t in tasks}

    import time
    start = time.perf_counter()
    try:
        for run_idx in range(num_runs):
            label_progress = f"Simulasi: proses {run_idx + 1} dari {num_runs}..."
            status_text.update(label=label_progress)
            progress_bar.progress((run_idx + 1) / num_runs)
            random.seed(random.randint(0, 999999))
            _, response_times = st.session_state.simulator.run(
                tasks, duration, num_runs=1,
            )
            for task_id, rt in response_times.items():
                all_response_times[task_id].append(rt)

        measured_wcrt: dict[int, float] = {}
        for task_id, rts in all_response_times.items():
            measured_wcrt[task_id] = max(rts) if rts else 0.0

        elapsed = (time.perf_counter() - start) * 1000
        st.session_state.measured = measured_wcrt
        st.session_state.simulation_done = True
        progress_bar.empty()
        status_text.update(
            label=f"Simulasi selesai dalam {elapsed:.2f}ms",
            state="complete",
        )
    except Exception as e:
        progress_bar.empty()
        status_text.update(label=f"Simulasi gagal: {e}", state="error")
        st.error(f"Simulasi gagal: {e}")


def _show_analysis_results() -> None:
    """Display analysis results."""
    result = st.session_state.analysis_result
    if not result:
        return

    col1, col2 = st.columns(2)
    if result.system_schedulable:
        col1.markdown("**Sistem Schedulable**  \n:green[Ya]")
    else:
        col1.markdown("**Sistem Schedulable**  \n:red[Tidak]")
    num_sched = sum(1 for r in result.results if r.schedulable)
    col2.metric("Tugas Schedulable", f"{num_sched}/{len(result.results)}")

    data = []
    for r in sorted(result.results, key=lambda x: x.task_id):
        sched = "Ya" if r.schedulable else "Tidak"
        data.append({
            "Tugas": r.task_id,
            "WCRT (ms)": f"{r.estimated_wcrt:.2f}",
            "Schedulable": sched,
            "Iterasi": r.iterations,
            "Waktu (us)": f"{r.analysis_time_us:.1f}",
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
        status = "OK" if ok else "MISS"
        data.append({
            "Tugas": t.task_id,
            "WCRT Terukur (ms)": f"{wcrt:.2f}",
            "Deadline (ms)": f"{t.deadline:.1f}",
            "Status": status,
        })
    st.dataframe(data, use_container_width=True, hide_index=True)


def _show_comparison() -> None:
    """Show WCRT comparison as DataFrame + chart."""
    tasks = st.session_state.tasks
    measured = st.session_state.measured
    result = st.session_state.analysis_result
    if not tasks or not measured or not result:
        st.warning("Jalankan analisis dan simulasi terlebih dahulu")
        return

    task_ids = [t.task_id for t in tasks]
    m_vals = [measured.get(t.task_id, 0.0) for t in tasks]
    e_vals = [r.estimated_wcrt for r in sorted(result.results, key=lambda x: x.task_id)]
    deadlines = [t.deadline for t in tasks]

    # Rich DataFrame
    import pandas as pd  # type: ignore[import-untyped]
    rows = []
    for tid, m, e, d in zip(task_ids, m_vals, e_vals, deadlines):
        over = (e - m) / m if m > 0 else 0.0
        meets = "Ya" if e <= d else "Tidak"
        rows.append({
            "Tugas": tid,
            "Terukur (ms)": f"{m:.2f}",
            "Estimasi (ms)": f"{e:.2f}",
            "Deadline (ms)": f"{d:.2f}",
            "Overestimasi": f"{over:.2%}",
            "Schedulable": meets,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
        ax.bar(x - w / 2, measured, w, label="Terukur (Simulasi)", color="#2196F3")
        ax.bar(x + w / 2, estimated, w, label="Estimasi (Analisis)", color="#FF5722")
        for i, d in enumerate(deadlines):
            ax.plot([i - 0.5, i + 0.5], [d, d], "k--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("ID Tugas")
        ax.set_ylabel("Waktu Respon (ms)")
        ax.set_title("WCRT: Terukur vs Estimasi")
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
        st.warning("Buat tugas terlebih dahulu")
        return

    with st.spinner("Menjalankan Deadline-Aware Heuristic..."):
        h = DeadlineAwareHeuristic(st.session_state.analyzer)
        h_result = h.assign(tasks)
        sched_h = ":green[Ya]" if h_result.schedulable else ":red[Tidak]"
        st.info(f"Heuristic — Schedulable: {sched_h}")

    with st.spinner("Menjalankan Genetic Algorithm (50 gen, 10s batas)..."):
        ga = GeneticNiceAssignment(
            st.session_state.analyzer,
            population_size=50,
            max_generations=50,
            timeout_seconds=10.0,
        )
        ga_result = ga.assign(tasks)
        sched_ga = ":green[Ya]" if ga_result.schedulable else ":red[Tidak]"
        st.info(f"Genetic Algorithm — Schedulable: {sched_ga}")

        data = []
        for task_id, nice_val in sorted(ga_result.nice_values.items()):
            data.append({"Tugas": task_id, "Nice Value": nice_val})
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
    int_periods = [int(round(p)) for p in periods]
    result = int_periods[0]
    for p in int_periods[1:]:
        result = result * p // math.gcd(result, p)
    return result


# ── Workflow helpers ─────────────────────────────────────────────────────

def _workflow_status() -> None:
    """Show workflow step indicator in sidebar."""
    has_tasks = len(st.session_state.tasks) > 0
    has_analysis = st.session_state.analysis_done
    has_sim = st.session_state.simulation_done
    has_both = has_analysis and has_sim

    steps: list[tuple[str, str, bool]] = [
        ("1/5", "Generate Tugas", has_tasks),
        ("2/5", "Analisis WCRT", has_analysis),
        ("3/5", "Simulasi CFS", has_sim),
        ("4/5", "Bandingkan", has_both),
        ("5/5", "Optimasi", has_tasks),
    ]
    for num, label, done in steps:
        if done:
            st.markdown(f":green[**{num}**] {label}")
        else:
            st.markdown(f":gray[{num}] {label}")


# ── Main UI ──────────────────────────────────────────────────────────────

_init_session()

st.title("Alat Analisis CFS WCRT")
st.markdown(
    "Analisis Worst-Case Response Time untuk penjadwalan "
    "Completely Fair Scheduler di Linux — berdasarkan Yoon et al. (2025)"
)

# ── Sidebar: Workflow & Task Generation ──────────────────────────────────

with st.sidebar:
    st.header("Alur Kerja")
    _workflow_status()

    st.divider()
    st.header("Pembuatan Tugas")

    num_tasks = st.number_input(
        "Jumlah Tugas", MIN_TASKS, MAX_TASKS, DEF_TASKS,
        help="Jumlah tugas dalam set (2\u201350). Semakin banyak tugas, semakin kompleks analisis.",
    )
    utilization = st.slider(
        "Utilisasi CPU",
        MIN_UTIL, MAX_UTIL, DEF_UTIL, 0.05,
        format="%.0f%%",
        help="Total utilisasi CPU dari semua tugas. Contoh: 50% berarti CPU digunakan setengah kapasitasnya.",
    )
    seed = st.number_input(
        "Acak Seed", 0, 999999, 42,
        help="Nilai awal untuk generator angka acak. Gunakan seed yang sama untuk hasil yang reproducible.",
    )

    if st.button("Buat Tugas", use_container_width=True, type="primary"):
        _generate_tasks(int(num_tasks), float(utilization), int(seed))

    st.divider()
    st.header("Tindakan")

    has_tasks = len(st.session_state.tasks) > 0
    has_analysis = st.session_state.analysis_done
    has_sim = st.session_state.simulation_done

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Analisis", use_container_width=True, disabled=not has_tasks):
            _run_analysis()
    with col_b:
        if st.button("Simulasi", use_container_width=True, disabled=not has_tasks):
            _run_simulation()

    can_compare = has_analysis and has_sim
    if st.button("Bandingkan", use_container_width=True, disabled=not can_compare):
        _show_comparison()

    if st.button("Optimasi", use_container_width=True, disabled=not has_tasks):
        _run_optimize()

    st.divider()
    csv_bytes = _csv_download()
    st.download_button(
        "Ekspor CSV",
        data=csv_bytes if csv_bytes is not None else b"",
        disabled=csv_bytes is None,
        use_container_width=True,
        file_name="cfs_wcrt_results.csv",
        mime="text/csv",
    )

# ── Panduan ──────────────────────────────────────────────────────────────

def _show_panduan() -> None:
    """Tampilkan panduan penggunaan dalam Bahasa Indonesia."""
    st.markdown("## Panduan Penggunaan Alat Analisis CFS WCRT")
    st.markdown("---")

    with st.expander("**Apa Itu CFS WCRT Analysis?**", expanded=True):
        st.markdown("""
        Tools ini menganalisis **Worst-Case Response Time (WCRT)** untuk penjadwalan
        **Completely Fair Scheduler (CFS)** di kernel Linux. Berdasarkan penelitian:

        > Yoon et al., *"Worst case response time analysis for completely fair
        > scheduling in Linux systems"*, Real-Time Systems, 2025.

        CFS adalah penjadwal default Linux yang membagi waktu CPU secara proporsional
        berdasarkan *nice value* setiap tugas. Tools ini menyediakan:

        - **Analisis WCRT** — Estimasi konservatif response time terburuk
        - **Simulasi CFS** — Pengukuran aktual via discrete-event simulator
        - **Optimasi Nice Value** — Algoritma penentuan prioritas tugas
        """)

    with st.expander("**1. Generate Task Set**"):
        st.markdown("""
        | Parameter | Deskripsi |
        |-----------|-----------|
        | **Jumlah Tugas** | Jumlah tugas dalam set (2\u201350) |
        | **Utilisasi CPU** | Total utilisasi CPU (5%\u201395%) |
        | **Acak Seed** | Nilai awal untuk reproduksibilitas |

        Klik **Buat Tugas** untuk membuat tugas sintetis menggunakan algoritma
        **UUniFast**. Tugas akan muncul di tab *Tugas*.
        """)

    with st.expander("**2. Analisis WCRT**"):
        st.markdown("""
        Klik **Analisis** untuk menjalankan **Algoritma 1** (fixed-point iteration)
        yang menghitung estimasi WCRT konservatif untuk setiap tugas.

        **Parameter Analisis:**
        - `max_iterations` (1000) — Batas maksimum iterasi fixed-point
        - `tolerance` (1e-9) — Konvergensi iterasi

        **Output:**
        - **Sistem Schedulable** — Ya/Tidak: semua tugas memenuhi deadline?
        - **Tugas Schedulable** — Jumlah tugas schedulable dari total
        - **WCRT (ms)** — Estimated worst-case response time per tugas
        - **Iterasi** — Jumlah iterasi hingga konvergen
        - **Waktu (us)** — Waktu komputasi per tugas

        Jika WCRT estimasi ≤ deadline → tugas **schedulable**.
        Jika > deadline → perlu optimasi nice value.
        """)

    with st.expander("**3. Simulasi CFS**"):
        st.markdown("""
        Klik **Simulasi** untuk menjalankan discrete-event simulator CFS.
        Simulator meniru perilaku penjadwal CFS Linux secara *cycle-accurate*:

        **Mekanisme:**
        - Definisi 1: Update Min vruntime — V_min = min(V_i)
        - Definisi 2: Update Curr — V_i += delta * w0 / wi
        - Definisi 3 & 4: Timeslice — sigma = max((wi/W)*L_adj, G)
        - Definisi 5: Place Entity — V_i = max(V_i, V_min)

        **Parameter:**
        - `hyperperiod_factor` (2.0) — Durasi = hyperperiod x factor
        - `num_runs` (5) — Jumlah run dengan offset acak
        - `max_events` (50000) — Batas maks event

        **Output:**
        - WCRT Terukur (ms) — Response time terbesar dari semua run
        - Deadline (ms) — Batas waktu tugas
        - Status — OK jika ≤ deadline, MISS jika terlambat
        """)

    with st.expander("**4. Perbandingan**"):
        st.markdown("""
        Klik **Bandingkan** untuk melihat perbandingan analisis vs simulasi.

        **Metrik:**
        - **Terukur (ms)** — Response time dari simulator
        - **Estimasi (ms)** — Response time dari analisis
        - **Deadline (ms)** — Batas waktu absolut
        - **Overestimasi** — (E-M)/M: rasio konservatisme
        - **Schedulable** — Ya jika estimasi ≤ deadline

        **Interpretasi:**
        - Estimated ≈ Measured → Analisis akurat
        - Estimated > Measured → Analisis konservatif (safe)
        - Estimated > Deadline → Tidak schedulable, perlu optimasi
        """)

    with st.expander("**5. Optimasi Nice Value**"):
        st.markdown("""
        Klik **Optimasi** untuk mencari *nice value* optimal.

        **Algoritma 2 — Deadline-Aware Heuristic:**
        - Mencari lambda (0\u201340) step 0.1
        - Formula: nice = round(-lambda * log2(Dmax / Di))
        - Tugas dengan deadline pendek mendapat prioritas lebih tinggi

        **Algoritma 3 — Genetic Algorithm (GA):**
        - Population: 50, Generations: 50, Timeout: 10s
        - Seleksi turnamen, crossover, mutasi

        **Output:**
        - Schedulable? Ya/Tidak
        - Tabel assignment nice value per tugas
        """)

    with st.expander("**Penjelasan Parameter**"):
        st.markdown("""
        | Parameter | Default | Rentang | Penjelasan |
        |-----------|---------|---------|------------|
        | `target_latency` | 18 ms | > 0 | Interval ideal semua tugas berjalan sekali |
        | `min_granularity` | 2.25 ms | > 0 | Timeslice minimum |
        | `jiffy` | 1 ms | > 0 | Interval tick timer Linux |
        | `sched_nr_latency` | 8 | > 0 | Maks tugas utk pembagian timeslice akurat |
        | `num_cores` | 1 | ≥ 1 | Jumlah core CPU |
        | `min_period` | 30 ms | — | Periode minimum tugas |
        | `max_period` | 3000 ms | — | Periode maksimum tugas |
        | `default_nice` | 0 | -20\u201319 | Nice value default |
        """)

    with st.expander("**CLI & Docker**"):
        st.markdown("""
        ```bash
        # Streamlit web UI
        streamlit run src/cfs_wcrt/ui/web.py

        # Docker
        docker compose up -d --build
        # Buka http://localhost:8501

        # CLI experiments
        python -m cfs_wcrt --experiment exp1
        python -m cfs_wcrt --experiment exp2
        python -m cfs_wcrt --experiment exp3

        # Tests
        python -m pytest -v
        python -m pytest --cov --cov-report=term-missing
        ```
        """)

    with st.expander("**Referensi & Arsitektur**"):
        st.markdown("""
        **Referensi:**
        1. Yoon et al. (2025). Worst case response time analysis for completely fair
           scheduling in Linux systems. *Real-Time Systems*.
        2. Bini & Buttazzo (2005). Measuring the performance of schedulability tests.
        3. Linux kernel 5.15 — `kernel/sched/fair.c`

        **Arsitektur:**
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
    "Tugas", "Analisis", "Simulasi", "Perbandingan", "Panduan",
])

with tab1:
    if st.session_state.task_df is not None:
        st.dataframe(
            st.session_state.task_df,
            use_container_width=True,
            hide_index=True,
        )
        total_util = sum(t.wcet / t.period for t in st.session_state.tasks)
        st.caption(f"Total Utilisasi: {total_util:.4f}  |  "
                   f"Hyperperiod: {_hyperperiod([t.period for t in st.session_state.tasks])}ms")
    else:
        st.info("Buat tugas dari sidebar untuk memulai.")

with tab2:
    if st.session_state.analysis_done:
        _show_analysis_results()
    else:
        st.info("Jalankan analisis dari sidebar.")

with tab3:
    if st.session_state.simulation_done:
        _show_simulation_results()
    else:
        st.info("Jalankan simulasi dari sidebar.")

with tab4:
    if st.session_state.analysis_done and st.session_state.simulation_done:
        _show_comparison()
    else:
        st.info("Jalankan analisis dan simulasi terlebih dahulu.")

with tab5:
    _show_panduan()
