# 📖 Panduan Penggunaan CFS WCRT Analysis Tool

## 🎯 Apa Itu CFS WCRT Analysis?

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

## 🚀 Cara Penggunaan

### 1. Generate Task Set

| Parameter | Deskripsi |
|-----------|-----------|
| **Number of Tasks** | Jumlah tugas dalam set (2–50) |
| **Utilization** | Total utilisasi sistem (0.05–0.95) |
| **Random Seed** | Nilai awal untuk reproduksibilitas |

Klik **🎲 Generate Task Set** untuk membuat tugas sintetis menggunakan
algoritma **UUniFast**. Tugas akan muncul di tab *📋 Task Set*.

### 2. Analisis WCRT

Klik **🔍 Analyze** untuk menjalankan **Algoritma 1** (fixed-point iteration)
yang menghitung estimasi WCRT konservatif untuk setiap tugas.

Hasil ditampilkan di tab *📈 Analysis*:
- **System Schedulable** — Apakah semua tugas memenuhi deadline?
- **WCRT (ms)** — Estimated worst-case response time
- **Iterations** — Jumlah iterasi hingga konvergen

### 3. Simulasi CFS

Klik **⚙️ Simulate** untuk menjalankan discrete-event simulator CFS.
Simulator ini meniru perilaku penjadwal CFS Linux:
- Pelacakan *vruntime* (Definisi 1 & 2)
- Alokasi *timeslice* dinamis (Definisi 3 & 4)
- Penyesuaian *wake-up* (Definisi 5)
- Tick jiffy periodik

Hasil ditampilkan di tab *🎯 Simulation*.

### 4. Perbandingan

Klik **📊 Compare Results** untuk melihat perbandingan antara:
- **Measured WCRT** — Hasil simulasi aktual
- **Estimated WCRT** — Hasil analisis
- **Deadline** — Batas waktu setiap tugas
- **Overestimation** — Seberapa konservatif estimasi

Jika estimasi ≤ deadline → tugas **schedulable ✅**

### 5. Optimasi Nice Value

Klik **🧬 Optimize Nice Values** untuk menjalankan:
- **Algorithm 2** — Deadline-Aware Heuristic
- **Algorithm 3** — Genetic Algorithm

Kedua algoritma mencari assignment *nice value* optimal agar semua tugas
schedulable.

### 6. Export CSV

Klik **⬇️ Export CSV** untuk mendownload hasil analisis dan simulasi
dalam format CSV.

---

## 📋 Penjelasan Parameter

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

## 💻 Menjalankan dari CLI

```bash
# Masuk ke direktori project
cd cfs_wcrt

# Eksperimen 1: Analisis schedulability
python -m cfs_wcrt --experiment exp1

# Eksperimen 2: Perbandingan nice value assignment
python -m cfs_wcrt --experiment exp2

# Eksperimen 3: WCRT measured vs estimated (dengan chart)
python -m cfs_wcrt --experiment exp3

# Streamlit web UI (tanpa Docker)
streamlit run src/cfs_wcrt/ui/web.py

# Menjalankan semua test
python -m pytest -v

# Cek coverage
python -m pytest --cov --cov-report=term-missing
```

### Docker

```bash
cd cfs_wcrt
docker compose up -d --build
# Buka http://localhost:8501
```

---

## 📚 Referensi

1. Yoon, P., Kim, J., & Lee, C. (2025). Worst case response time analysis
   for completely fair scheduling in Linux systems. *Real-Time Systems*.
2. Bini, E. & Buttazzo, G. (2005). Measuring the performance of
   schedulability tests. *Real-Time Systems*.
3. Linux kernel 5.15 — `kernel/sched/fair.c` (CFS implementation)

---

## 🏗️ Arsitektur Tools

```
src/cfs_wcrt/
├── core/          → Model data & konstanta Linux
├── analysis/      → Algoritma 1: Analisis WCRT
├── simulation/    → Simulator discrete-event CFS
├── optimization/  → Algoritma 2 & 3: Optimasi nice value
├── generation/    → Pembangkitan tugas sintetis (UUniFast)
└── ui/            → CLI, Streamlit, tabel, chart
```
