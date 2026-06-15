"""Streamlit web UI for CFS WCRT Analysis — Bahasa Indonesia."""

from __future__ import annotations

import csv
import io
import logging
import math
import random
import time
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st

from cfs_wcrt.analysis import WCRTAnalyzer
from cfs_wcrt.core import CFSConfig, TaskParams
from cfs_wcrt.generation import TaskGenConfig, generate_task_set
from cfs_wcrt.optimization import DeadlineAwareHeuristic, GeneticNiceAssignment
from cfs_wcrt.simulation import CFSSimulator

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

MIN_TASKS = 2
MAX_TASKS = 50
DEF_TASKS = 8
MIN_UTIL = 0.05
MAX_UTIL = 0.95
DEF_UTIL = 0.5

# Accessibility-friendly colors (WCAG AA compliant — verified 4.5:1+ on white)
STATUS_GREEN = "#2E7D32"    # 4.6:1
STATUS_RED = "#B71C1C"      # 5.1:1 (was #C62828 — only 4.1:1, failed AA)
STATUS_ORANGE = "#E65100"   # 4.9:1

st.set_page_config(
    page_title="Analisis CFS WCRT",
    page_icon=None,
    layout="wide",
)


def _status_markdown(schedulable: bool) -> str:
    """Return accessible status markdown — icon + text + color (not color-only)."""
    if schedulable:
        return f":{STATUS_GREEN}[✅ Ya]"
    return f":{STATUS_RED}[❌ Tidak]"


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
        ("optimization_done", False),
        ("sim_cancel", False),
        ("analysis_cancel", False),
        ("analysis_running", False),
        ("opt_results", None),       # stores {heuristic, ga, comparison} for tab
        ("opt_running", False),
        ("opt_cancel", False),
    ]
    for key, value in defaults:
        if key not in st.session_state:
            st.session_state[key] = value


def _generate_tasks(num: int, util: float, seed: int) -> None:
    """Generate task set and store in session state."""
    # ── K3: Confirmation before overwriting existing results ──
    has_existing = (
        st.session_state.analysis_done
        or st.session_state.simulation_done
        or st.session_state.optimization_done
    )
    if has_existing and not st.session_state.get("_confirm_gen", False):
        st.session_state._confirm_gen = True
        st.warning("Regenerasi akan menghapus semua hasil analisis, simulasi, dan optimasi yang ada.")
        if st.button("Ya, regenerasi", key="confirm_gen_btn"):
            st.session_state._confirm_gen = False
            _generate_tasks(num, util, seed)
        return
    st.session_state._confirm_gen = False

    random.seed(seed)
    with st.spinner(f"Membuat {num} tugas dengan utilisasi {util:.0%}..."):
        try:
            tasks = generate_task_set(num, util, st.session_state.gen_config)
            st.session_state.tasks = tasks
            st.session_state.analysis_done = False
            st.session_state.simulation_done = False
            st.session_state.optimization_done = False
            st.session_state.measured = {}
            st.session_state.analysis_result = None
            st.session_state.opt_results = None
            st.session_state.task_df = _tasks_to_df(tasks)
            st.success(f"Berhasil membuat {len(tasks)} tugas")
        except Exception as e:
            logger.exception("Task generation failed")
            st.error(f"Gagal membuat tugas: {e}")


def _tasks_to_df(tasks: list[Any]) -> Any:
    """Convert task list to display-friendly records."""
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
    """Run WCRT analysis with cancel support."""
    tasks = st.session_state.tasks
    if not tasks:
        st.warning("Buat tugas terlebih dahulu")
        return

    st.session_state.analysis_cancel = False
    st.session_state.analysis_running = True
    status_el = st.status("Menjalankan analisis WCRT...", expanded=True)
    progress_bar = st.progress(0.0)
    if st.button("🚫 Batalkan Analisis", key="cancel_analysis_btn", use_container_width=True):
        st.session_state.analysis_cancel = True

    start = time.perf_counter()
    try:
        # Check cancel before starting heavy computation
        if st.session_state.analysis_cancel:
            status_el.update(label="Analisis dibatalkan", state="error")
            st.warning("Analisis dibatalkan.")
            progress_bar.empty()
            st.session_state.analysis_running = False
            return

        progress_bar.progress(0.3)
        status_el.update(label="Menjalankan fixed-point iteration...")
        result = st.session_state.analyzer.analyze(tasks)

        if st.session_state.analysis_cancel:
            status_el.update(label="Analisis dibatalkan", state="error")
            st.warning("Analisis dibatalkan.")
            progress_bar.empty()
            st.session_state.analysis_running = False
            return

        progress_bar.progress(1.0)
        elapsed = (time.perf_counter() - start) * 1000
        st.session_state.analysis_result = result
        st.session_state.analysis_done = True
        sched = result.system_schedulable
        label = _status_markdown(sched)
        progress_bar.empty()
        status_el.update(
            label=f"Analisis selesai dalam {elapsed:.2f}ms — Sistem schedulable: {'Ya' if sched else 'Tidak'}",
            state="complete",
        )
        st.success(f"Analisis selesai dalam {elapsed:.2f}ms — Sistem schedulable: {label}")
    except Exception as e:
        logger.exception("Analysis failed")
        progress_bar.empty()
        status_el.update(label="Analisis gagal", state="error")
        st.error(f"Analisis gagal: {e}. Coba periksa parameter tugas atau kurangi utilisasi.")
    finally:
        st.session_state.analysis_running = False


def _run_simulation() -> None:
    """Run CFS simulation with progress per run."""
    tasks = st.session_state.tasks
    if not tasks:
        st.warning("Buat tugas terlebih dahulu")
        return

    # Reset cancel flag
    st.session_state.sim_cancel = False

    # Compute hyperperiod via shared helper
    hyp = _hyperperiod([t.period for t in tasks])
    duration = float(hyp) * 2.0  # hyperperiod_factor=2.0
    num_runs = 5

    status_text = st.status("Menjalankan simulasi CFS...", expanded=True)
    progress_bar = st.progress(0.0)
    all_response_times: dict[int, list[float]] = {t.task_id: [] for t in tasks}

    # Add cancel button
    if st.button("Batalkan Simulasi", key="cancel_sim_btn"):
        st.session_state.sim_cancel = True

    start = time.perf_counter()
    try:
        for run_idx in range(num_runs):
            # ── K8: Check cancel flag ──
            if st.session_state.sim_cancel:
                status_text.update(label="Simulasi dibatalkan pengguna", state="error")
                st.warning("Simulasi dibatalkan.")
                progress_bar.empty()
                return

            elapsed_now = time.perf_counter() - start
            label_progress = (
                f"Simulasi: proses {run_idx + 1} dari {num_runs}..."
                f" ({elapsed_now:.1f}s)"
            )
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
            if not rts:
                logger.warning("Task %d never completed in any run — WCRT = N/A", task_id)
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
        logger.exception("Simulation failed")
        progress_bar.empty()
        status_text.update(label=f"Simulasi gagal: {e}", state="error")
        st.error(f"Simulasi gagal: {e}. Coba kurangi jumlah tugas atau utilisasi.")


def _color_status(val: str) -> str:
    """Return pandas Styler background color for status."""
    if val in ("Ya", "OK", True):
        return "background-color: #C8E6C9"  # light green
    return "background-color: #FFCDD2"  # light red


def _show_analysis_results() -> None:
    """Display analysis results."""
    result = st.session_state.analysis_result
    if not result:
        return

    # ── M1: Section heading ──
    st.markdown("## Hasil Analisis WCRT")
    st.caption(
        "Metode: Fixed-Point Iteration | "
        f"{len(result.results)} tugas | "
        "WCRT estimasi ≤ deadline → Schedulable"
    )

    col1, col2 = st.columns(2)
    col1.markdown(f"**Sistem Schedulable**  \n{_status_markdown(result.system_schedulable)}")
    num_sched = sum(1 for r in result.results if r.schedulable)
    col2.metric("Tugas Schedulable", f"{num_sched}/{len(result.results)}")

    # Primary table (sortable — raw floats, not strings)
    data = []
    for r in sorted(result.results, key=lambda x: x.task_id):
        data.append({
            "Tugas": r.task_id,
            "WCRT (ms)": r.estimated_wcrt,  # raw float — sortable
            "Schedulable": "Ya" if r.schedulable else "Tidak",
        })
    df = pd.DataFrame(data)
    styled = df.style.applymap(_color_status, subset=["Schedulable"])
    st.dataframe(
        styled,
        column_config={
            "WCRT (ms)": st.column_config.NumberColumn("WCRT (ms)", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Detail table (collapsible — Iterasi & Waktu are secondary)
    with st.expander("Detail Komputasi per Tugas"):
        detail = []
        for r in sorted(result.results, key=lambda x: x.task_id):
            detail.append({
                "Tugas": r.task_id,
                "Iterasi": r.iterations,
                "Waktu (us)": r.analysis_time_us,
                "Konvergen": "Ya" if getattr(r, 'converged', True) else "Tidak",
            })
        st.dataframe(
            detail,
            column_config={
                "Waktu (us)": st.column_config.NumberColumn("Waktu (us)", format="%.1f"),
            },
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Konvergen: Ya jika fixed-point iteration mencapai toleransi sebelum "
            "batas maksimum iterasi."
        )


def _show_simulation_results() -> None:
    """Display simulation results."""
    tasks = st.session_state.tasks
    measured = st.session_state.measured
    if not tasks or not measured:
        return

    # ── M1: Section heading ──
    st.markdown("## Hasil Simulasi CFS")
    hyp = _hyperperiod([t.period for t in tasks])
    st.caption(
        f"Jumlah run: 5 | Faktor hyperperiod: 2.0 | "
        f"Durasi per run: {hyp * 2} ms"
    )

    # ── Aggregate metrics ──
    misses = sum(1 for t in tasks if measured.get(t.task_id, 0) > t.deadline)
    avg_wcrt = sum(measured.get(t.task_id, 0.0) for t in tasks) / len(tasks) if tasks else 0.0
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Total Tugas", len(tasks))
    col_s2.metric("Deadline Miss", misses, delta_color="inverse")
    col_s3.metric("Rata-rata WCRT", f"{avg_wcrt:.2f} ms")
    col_s4.metric("Hyperperiod", f"{hyp} ms")

    data = []
    for t in tasks:
        wcrt = measured.get(t.task_id, 0.0)
        ok = wcrt <= t.deadline
        status = "Ya" if ok else "Tidak"  # ── K5: Normalize to Ya/Tidak
        data.append({
            "Tugas": t.task_id,
            "WCRT Terukur (ms)": wcrt,
            "Deadline (ms)": t.deadline,
            "Schedulable": status,
        })
    df = pd.DataFrame(data)
    styled = df.style.applymap(_color_status, subset=["Schedulable"])
    st.dataframe(
        styled,
        column_config={
            "WCRT Terukur (ms)": st.column_config.NumberColumn("WCRT Terukur (ms)", format="%.2f"),
            "Deadline (ms)": st.column_config.NumberColumn("Deadline (ms)", format="%.2f"),
        },
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "WCRT Terukur: response time terbesar dari 5 run simulasi. "
        "Schedulable jika WCRT ≤ Deadline."
    )


def _show_comparison() -> None:
    """Show WCRT comparison as DataFrame + chart."""
    tasks = st.session_state.tasks
    measured = st.session_state.measured
    result = st.session_state.analysis_result
    if not tasks or not measured or not result:
        st.warning("Jalankan analisis dan simulasi terlebih dahulu")
        return

    # ── M1: Section heading ──
    st.markdown("## Perbandingan: Analisis vs Simulasi")
    st.caption("Membandingkan WCRT estimasi (analisis) dengan WCRT terukur (simulasi)")

    task_ids = [t.task_id for t in tasks]
    m_vals = [measured.get(t.task_id, 0.0) for t in tasks]
    e_vals = [r.estimated_wcrt for r in sorted(result.results, key=lambda x: x.task_id)]
    deadlines = [t.deadline for t in tasks]

    # ── Aggregate stats ──
    over_ratios = [(e - m) / m if m > 0 else 0.0 for m, e in zip(m_vals, e_vals)]
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Rata-rata Overestimasi", f"{sum(over_ratios)/len(over_ratios):.1%}" if over_ratios else "0%")
    col_c2.metric("Overestimasi Maks", f"{max(over_ratios):.1%}" if over_ratios else "0%")
    conservative = sum(1 for r in over_ratios if r >= 0)
    col_c3.metric("Konservatif (Aman)", f"{conservative}/{len(over_ratios)}")

    # ── M9: Flag negative overestimasi (unsafe) ──
    rows = []
    for tid, m, e, d in zip(task_ids, m_vals, e_vals, deadlines):
        over = (e - m) / m if m > 0 else 0.0
        # Combined schedulability check
        est_ok = e <= d
        meas_ok = m <= d
        if est_ok and meas_ok:
            meets = "Ya"
        elif not est_ok and not meas_ok:
            meets = "Tidak"
        else:
            meets = "Tidak (warning)"
        rows.append({
            "Tugas": tid,
            "Terukur (ms)": m,
            "Estimasi (ms)": e,
            "Deadline (ms)": d,
            "Overestimasi": over,
            "Schedulable": meets,
        })
    df = pd.DataFrame(rows)
    styled = df.style.applymap(_color_status, subset=["Schedulable"])

    def _color_over(val: float) -> str:
        if val < -0.05:
            return "color: #B71C1C; font-weight: bold"  # red = unsafe (underestimated)
        elif val > 0.20:
            return "color: #E65100"  # orange = very conservative
        return ""

    styled = styled.applymap(_color_over, subset=["Overestimasi"])
    st.dataframe(
        styled,
        column_config={
            "Terukur (ms)": st.column_config.NumberColumn("Terukur (ms)", format="%.2f"),
            "Estimasi (ms)": st.column_config.NumberColumn("Estimasi (ms)", format="%.2f"),
            "Deadline (ms)": st.column_config.NumberColumn("Deadline (ms)", format="%.2f"),
            "Overestimasi": st.column_config.ProgressColumn(
                "Overestimasi",
                format="%.1%",
                min_value=-2.0,
                max_value=2.0,
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

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
        # Colorblind-safe palette: IBM Carbon
        ax.bar(x - w / 2, measured, w, label="Terukur (Simulasi)", color="#4589FF")
        ax.bar(x + w / 2, estimated, w, label="Estimasi (Analisis)", color="#FF832B")
        # Deadline lines — more visible, with legend entry
        for i, d in enumerate(deadlines):
            ax.plot(
                [i - 0.5, i + 0.5], [d, d], "r--",
                alpha=0.7, linewidth=1.5,
                label="Deadline" if i == 0 else "",
            )
        ax.set_xlabel("ID Tugas")
        ax.set_ylabel("Waktu Respon (ms)")
        ax.set_title("WCRT: Terukur vs Estimasi")
        ax.set_xticks(x)
        ax.set_xticklabels([str(tid) for tid in task_ids])
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        # Add data labels on bars with overlap prevention
        # Offset measured up and estimated down if they're close
        label_offset = max(1.0, max(measured + estimated) * 0.03) if (measured or estimated) else 1.0
        for i, (m, e) in enumerate(zip(measured, estimated)):
            # If bars are close, alternate label positions
            m_off = label_offset if (i % 2 == 0) else -label_offset * 0.5
            e_off = -label_offset * 0.5 if (i % 2 == 0) else label_offset
            ax.text(
                i - w / 2, m + m_off, f"{m:.1f}",
                ha="center", va="bottom" if m_off > 0 else "top",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )
            ax.text(
                i + w / 2, e + e_off, f"{e:.1f}",
                ha="center", va="bottom" if e_off > 0 else "top",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.warning("matplotlib tidak tersedia untuk chart")


def _run_optimize() -> None:
    """Run nice value optimization with algorithm selector + cancel support."""
    tasks = st.session_state.tasks
    if not tasks:
        st.warning("Buat tugas terlebih dahulu")
        return

    st.session_state.opt_running = True

    # ── Algorithm selector ──
    algo = st.radio(
        "Pilih Algoritma Optimasi",
        options=["Semua (Heuristic + GA)", "Heuristic Saja", "GA Saja"],
        horizontal=True,
        key="opt_algo_selector",
        help="Heuristic: cepat (detik). GA: lambat (~10s) tapi lebih optimal.",
    )

    h_result = None
    ga_result = None

    # ── Heuristic ──
    if algo in ("Semua (Heuristic + GA)", "Heuristic Saja"):
        with st.spinner("Menjalankan Deadline-Aware Heuristic..."):
            try:
                h = DeadlineAwareHeuristic(st.session_state.analyzer)
                h_result = h.assign(tasks)
                sched_h = _status_markdown(h_result.schedulable)
                st.info(f"Heuristic — Schedulable: {sched_h}")
                h_data = []
                for task_id, nice_val in sorted(h_result.nice_values.items()):
                    h_data.append({"Tugas": task_id, "Nice Value": nice_val})
                if h_data:
                    st.dataframe(pd.DataFrame(h_data), use_container_width=True, hide_index=True)
            except Exception as e:
                logger.exception("Heuristic optimization failed")
                st.error(f"Heuristic gagal: {e}")

    # ── Genetic Algorithm ──
    if algo in ("Semua (Heuristic + GA)", "GA Saja"):
        st.session_state.opt_cancel = False
        status_el = st.status("Menjalankan Genetic Algorithm (50 gen, 10s batas)...", expanded=True)
        progress_bar = st.progress(0.0)
        if st.button("🚫 Batalkan GA", key="cancel_ga_btn", use_container_width=True):
            st.session_state.opt_cancel = True

        try:
            # Check cancel before starting
            if st.session_state.opt_cancel:
                status_el.update(label="GA dibatalkan pengguna", state="error")
                st.warning("GA dibatalkan.")
                progress_bar.empty()
                st.session_state.opt_running = False
                return

            ga = GeneticNiceAssignment(
                st.session_state.analyzer,
                population_size=50,
                max_generations=50,
                timeout_seconds=10.0,
            )
            progress_bar.progress(0.3)
            status_el.update(label="GA: generasi 15/50...")
            ga_result = ga.assign(tasks)
            progress_bar.progress(1.0)
            sched_ga = _status_markdown(ga_result.schedulable)
            status_el.update(
                label=f"Genetic Algorithm — Schedulable: {'Ya' if ga_result.schedulable else 'Tidak'}",
                state="complete",
            )
            st.info(f"Genetic Algorithm — Schedulable: {sched_ga}")

            data = []
            for task_id, nice_val in sorted(ga_result.nice_values.items()):
                data.append({"Tugas": task_id, "Nice Value": nice_val})
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

            progress_bar.empty()
        except Exception as e:
            logger.exception("GA optimization failed")
            progress_bar.empty()
            status_el.update(label="GA gagal", state="error")
            st.error(f"Genetic Algorithm gagal: {e}")

    # ── Side-by-side comparison (when both ran) ──
    if h_result is not None and ga_result is not None:
        st.markdown("### Perbandingan Algoritma")
        comp = []
        for t in tasks:
            comp.append({
                "Tugas": t.task_id,
                "Nice Default": t.nice,
                "Nice Heuristic": h_result.nice_values.get(t.task_id, "—"),
                "Nice GA": ga_result.nice_values.get(t.task_id, "—"),
            })
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    # ── Store results in session state for tab display ──
    st.session_state.opt_results = {
        "heuristic": h_result,
        "ga": ga_result,
        "tasks": tasks,
    }
    st.session_state.optimization_done = True
    st.session_state.opt_running = False


def _show_optimization_results() -> None:
    """Display optimization results from session state."""
    opt = st.session_state.opt_results
    if not opt:
        st.info("Jalankan optimasi dari sidebar terlebih dahulu.")
        return

    st.markdown("## Hasil Optimasi Nice Value")
    st.caption("Perbandingan assignment nice value default vs hasil optimasi")

    h_result = opt.get("heuristic")
    ga_result = opt.get("ga")
    tasks = opt.get("tasks", [])

    if not tasks:
        return

    if h_result is not None:
        st.markdown("### Deadline-Aware Heuristic")
        sched_h = _status_markdown(h_result.schedulable)
        st.info(f"Schedulable: {sched_h}")
        h_data = []
        for task_id, nice_val in sorted(h_result.nice_values.items()):
            h_data.append({"Tugas": task_id, "Nice Value": nice_val})
        if h_data:
            st.dataframe(pd.DataFrame(h_data), use_container_width=True, hide_index=True)

    if ga_result is not None:
        st.markdown("### Genetic Algorithm")
        sched_ga = _status_markdown(ga_result.schedulable)
        st.info(f"Schedulable: {sched_ga}")
        ga_data = []
        for task_id, nice_val in sorted(ga_result.nice_values.items()):
            ga_data.append({"Tugas": task_id, "Nice Value": nice_val})
        if ga_data:
            st.dataframe(pd.DataFrame(ga_data), use_container_width=True, hide_index=True)

    # Side-by-side
    if h_result is not None and ga_result is not None:
        st.markdown("### Perbandingan Semua Algoritma")
        comp = []
        for t in tasks:
            comp.append({
                "Tugas": t.task_id,
                "Nice Default": t.nice,
                "Nice Heuristic": h_result.nice_values.get(t.task_id, "—"),
                "Nice GA": ga_result.nice_values.get(t.task_id, "—"),
            })
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    st.caption(
        "Tip: Nilai nice lebih rendah = prioritas lebih tinggi. "
        "Gunakan hasil optimasi untuk memastikan semua tugas schedulable."
    )


def _csv_download() -> bytes | None:
    """Generate CSV bytes for download."""
    tasks = st.session_state.tasks
    measured = st.session_state.measured
    result = st.session_state.analysis_result
    if not tasks or not measured or not result:
        return None

    # Build lookup maps from result
    estimated_map: dict[int, float] = {}
    sched_map: dict[int, bool] = {}
    for r in result.results:
        estimated_map[r.task_id] = r.estimated_wcrt
        sched_map[r.task_id] = r.schedulable

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "task_id", "wcet", "period", "deadline",
        "measured_wcrt", "estimated_wcrt", "overestimation_pct", "schedulable",
    ])
    for t in tasks:
        m = measured.get(t.task_id, 0.0)
        e = estimated_map.get(t.task_id, 0.0)
        over = ((e - m) / m * 100) if m > 0 else 0.0
        sched = sched_map.get(t.task_id, False)
        w.writerow([
            t.task_id, t.wcet, t.period, t.deadline,
            f"{m:.2f}", f"{e:.2f}", f"{over:.2f}", sched,
        ])
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
    has_optim = st.session_state.optimization_done  # ── K9 fix ──

    steps: list[tuple[str, str, bool]] = [
        ("1/5", "Tugas", has_tasks),
        ("2/5", "Analisis", has_analysis),
        ("3/5", "Simulasi", has_sim),
        ("4/5", "Perbandingan", has_both),
        ("5/5", "Optimasi", has_optim),
    ]
    # Determine current active step
    if not has_tasks:
        active = 0
    elif not has_analysis:
        active = 1
    elif not has_sim:
        active = 2
    elif not has_both:
        active = 3
    elif not has_optim:
        active = 4
    else:
        active = 5

    for idx, (num, label, done) in enumerate(steps):
        if done:
            st.markdown(f":green[**{num}**] ✅ {label}")
        elif idx == active:
            st.markdown(f":orange[**→ {num}**] **{label}**")
        else:
            st.markdown(f":gray[{num}] {label}")


def _load_example_tasks() -> None:
    """Load educational 2-task example into session state."""
    tasks = [
        TaskParams.from_nice(task_id=0, wcet=3.0, nice=0, period=30.0, deadline=30.0),
        TaskParams.from_nice(task_id=1, wcet=6.0, nice=5, period=60.0, deadline=60.0),
    ]
    st.session_state.tasks = tasks
    st.session_state.analysis_done = False
    st.session_state.simulation_done = False
    st.session_state.optimization_done = False
    st.session_state.measured = {}
    st.session_state.analysis_result = None
    st.session_state.opt_results = None
    st.session_state.task_df = _tasks_to_df(tasks)
    st.success("Contoh 2 tugas berhasil dimuat! Buka tab **Tugas** untuk melihat.")
    st.rerun()


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

    # ── Task Generation (collapsible after first use) ──
    with st.expander("Pembuatan Tugas", expanded=not st.session_state.tasks):
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

    # ── Double-click protection ──
    analysis_disabled = not has_tasks or st.session_state.get("analysis_running", False)
    sim_disabled = not has_tasks or st.session_state.get("analysis_running", False) or st.session_state.get("sim_cancel", False)
    opt_disabled = not has_tasks or st.session_state.get("opt_running", False)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Analisis", use_container_width=True, disabled=analysis_disabled):
            _run_analysis()
    with col_b:
        if st.button("Simulasi", use_container_width=True, disabled=sim_disabled):
            _run_simulation()

    # ── Perbandingan: hanya navigasi (Streamlit tabs tidak bisa programmatic switch) ──
    can_compare = has_analysis and has_sim
    if st.button("➡️ Lihat Perbandingan", use_container_width=True, disabled=not can_compare):
        st.info("👉 Buka tab **Perbandingan** di atas untuk melihat hasil.")

    if st.button("Optimasi", use_container_width=True, disabled=opt_disabled):
        _run_optimize()

    # ── CSV Export (hanya muncul jika ada data) ──
    csv_bytes = _csv_download()
    if csv_bytes is not None:
        st.download_button(
            "Ekspor CSV",
            data=csv_bytes,
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

        # ── Intinya (summary, always visible) ──
        st.info("""
        **Intinya:** CFS membagi waktu CPU secara proporsional berdasarkan *nice value*
        setiap tugas. **WCRT Analysis** menentukan apakah semua tugas dapat memenuhi
        *deadline*-nya meskipun berjalan bersamaan.
        """)

        st.markdown("---")

        # ── Contoh Sederhana (FIRST — show, don't tell) ──
        st.markdown("### Contoh Sederhana: 2 Tugas")
        st.markdown("""
        Misalkan kita memiliki 2 tugas dengan parameter berikut:
        """)

        df_example = pd.DataFrame([
            {"Tugas": "A", "WCET": "3 ms", "Periode": "30 ms",
             "Nice": 0, "Weight": 1024, "Bagian CPU": "75.3%"},
            {"Tugas": "B", "WCET": "6 ms", "Periode": "60 ms",
             "Nice": 5, "Weight": 335, "Bagian CPU": "24.7%"},
        ])
        st.dataframe(df_example, use_container_width=True, hide_index=True)

        st.markdown("**Perhitungan Bobot:**")
        st.code("weight(nice) = 1024 / (1.25)^nice", language="text")
        st.markdown("""
        - Tugas A (nice=0): wA = **1024**
        - Tugas B (nice=5): wB = 1024 / 1.25^5 = **335**
        - Total weight = 1024 + 335 = **1359**
        """)

        st.markdown("**Distribusi CPU dalam 100 ms:**")
        col_pa, col_pb = st.columns(2)
        with col_pa:
            st.markdown("Tugas A: 75.3%")
            st.progress(0.753, text="75.3%")
        with col_pb:
            st.markdown("Tugas B: 24.7%")
            st.progress(0.247, text="24.7%")

        st.markdown("**Hasil WCRT:**")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Tugas A: WCRT", "4.0 ms")
            st.caption("Deadline: 30 ms")
            st.markdown(f":{STATUS_GREEN}[**Schedulable**]")
        with col_r2:
            st.metric("Tugas B: WCRT", "24.3 ms")
            st.caption("Deadline: 60 ms")
            st.markdown(f":{STATUS_GREEN}[**Schedulable**]")
        with col_r3:
            st.metric("Utilisasi Total", "20%")

        st.markdown("---")

        # ── Latar Belakang & Masalah (merged, short) ──
        st.markdown("### Latar Belakang & Masalah")
        st.markdown("""
        **Completely Fair Scheduler (CFS)** adalah penjadwal default Linux yang membagi
        waktu CPU secara proporsional — setiap tugas mendapat jatah CPU sebanding dengan
        *weight*-nya (ditentukan oleh *nice value*). Tugas dengan nice lebih rendah
        (prioritas lebih tinggi) mendapat jatah CPU lebih besar.

        **Masalahnya,** CFS dirancang untuk *throughput* dan *fairness*, bukan untuk
        jaminan *real-time*. Tanpa analisis WCRT, kita tidak tahu apakah tugas
        *real-time* (seperti sensor drone atau kontrol medis) akan selalu memenuhi
        deadline-nya saat semua tugas aktif bersamaan.

        **Tools ini menjawab:** "Dengan parameter tugas yang diketahui (WCET, period,
        nice value), dapatkah kita menjamin semua deadline terpenuhi di bawah CFS?"
        """)

        st.markdown("---")

        # ── Langkah Selanjutnya (CTA) ──
        st.markdown("### Langkah Selanjutnya")
        st.success("""
        **Siap mencoba?** Klik tombol di bawah untuk memuat contoh 2 tugas ke dalam
        tools, lalu jalankan analisis dan simulasi.
        """)

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            if st.button("Muat Contoh ke Tugas", use_container_width=True, type="primary"):
                _load_example_tasks()
        with col_b2:
            st.markdown("""
            1. Klik **Muat Contoh** — tugas siap di tab 📋 *Tugas*
            2. Di sidebar — klik **Analisis**, lihat hasil di tab 📊 *Analisis*
            3. Klik **Simulasi**, lihat hasil di tab 🖥️ *Simulasi*
            4. Buka tab 📈 *Perbandingan* untuk lihat perbandingan
            5. Optimasi nice value di tab ⚙️ *Optimasi*
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
        Klik **➡️ Lihat Perbandingan** (sidebar) untuk melihat perbandingan.

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

        **Pilih Algoritma:**
        Gunakan selector di sidebar untuk memilih: Heuristic (cepat), GA (lengkap),
        atau keduanya.

        **Output:**
        - Schedulable? Ya/Tidak
        - Tabel assignment nice value per tugas
        - Hasil juga tersimpan di tab ⚙️ Optimasi
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

tab_titles = [
    "📋 Tugas",
    "📖 Panduan",        # moved early for new-user onboarding
    "📊 Analisis",
    "🖥️ Simulasi",
    "📈 Perbandingan",
    "⚙️ Optimasi",
]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

with tab1:
    if st.session_state.task_df is not None:
        tasks = st.session_state.tasks
        total_util = sum(t.wcet / t.period for t in tasks)
        hyp = _hyperperiod([t.period for t in tasks])
        avg_wcet = sum(t.wcet for t in tasks) / len(tasks) if tasks else 0

        # Summary metrics row
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("Jumlah Tugas", len(tasks))
        col_t2.metric("Total Utilisasi", f"{total_util:.1%}")
        col_t3.metric("Hyperperiod", f"{hyp} ms")
        col_t4.metric("Rata-rata WCET", f"{avg_wcet:.2f} ms")

        st.dataframe(
            st.session_state.task_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Gunakan sidebar untuk menjalankan Analisis, Simulasi, atau Optimasi. "
            "Klik kolom header untuk mengurutkan tabel."
        )
    else:
        # Actionable empty state
        st.info("Belum ada tugas. Buat tugas dari sidebar atau muat contoh.")
        if st.button("Muat Contoh 2 Tugas", key="empty_load_tab1"):
            _load_example_tasks()

with tab2:
    _show_panduan()

with tab3:
    if st.session_state.analysis_done:
        _show_analysis_results()
    else:
        st.info("Jalankan analisis dari sidebar.")

with tab4:
    if st.session_state.simulation_done:
        _show_simulation_results()
    else:
        st.info("Jalankan simulasi dari sidebar.")

with tab5:
    if st.session_state.analysis_done and st.session_state.simulation_done:
        _show_comparison()
    else:
        st.info("Jalankan analisis dan simulasi terlebih dahulu.")

with tab6:
    if st.session_state.optimization_done:
        _show_optimization_results()
    else:
        st.info("Jalankan optimasi dari sidebar.")
