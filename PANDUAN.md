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

**Parameter Analisis:**

| Parameter | Default | Penjelasan |
|-----------|---------|------------|
| `max_iterations` | 1000 | Batas maksimum iterasi fixed-point agar tidak infinite loop |
| `tolerance` | 1e-9 | Konvergensi: iterasi berhenti jika \|Rᵢⁿ⁺¹ - Rᵢⁿ\| < tolerance |
| **Algoritma** | — | Persamaan 33: Lᵢ = Cᵢ + Σ⌈Lᵢ/Tⱼ⌉·Cⱼ (busy period) |

**Output di tab *📈 Analysis*:**

| Metrik | Penjelasan |
|--------|------------|
| **System Schedulable** | ✅ Ya / ❌ Tidak — semua tugas memenuhi deadline? |
| **Tasks Schedulable** | Jumlah tugas schedulable / total tugas |
| **WCRT (ms)** | Estimated worst-case response time untuk setiap tugas |
| **Iterations** | Jumlah iterasi fixed-point hingga konvergen |
| **Time (us)** | Waktu komputasi analisis per tugas (mikrodetik) |

**Cara baca:** Jika WCRT estimasi ≤ deadline → tugas **schedulable ✅**.
Jika > deadline → perlu optimasi nice value.

---

### 3. Simulasi CFS

Klik **⚙️ Simulate** untuk menjalankan discrete-event simulator CFS.
Simulator ini meniru perilaku penjadwal CFS Linux secara *cycle-accurate*:

**Mekanisme Simulasi:**

| Definisi | Nama | Rumusan |
|----------|------|---------|
| Definisi 1 | **Update Min vruntime** | V_min = min(V_i dari semua task runnable) |
| Definisi 2 | **Update Curr** | V_i += Δ · w₀ / wᵢ |
| Definisi 3 & 4 | **Timeslice** | σ̃ = max((wᵢ / W)·L_adj, G) dibulatkan ke jiffy |
| Definisi 5 | **Place Entity** | V_i = max(V_i, V_min) saat wake-up |

**Parameter Simulasi:**

| Parameter | Default | Penjelasan |
|-----------|---------|------------|
| `hyperperiod_factor` | 2.0 | Durasi simulasi = hyperperiod × factor (ms) |
| `num_runs` | 5 | Jumlah run dengan random offset berbeda |
| `max_events` | 50000 | Batas maks event agar tidak infinite loop |
| **Offset** | random | Setiap run menggunakan offset fase acak 0–30% period |

**Output di tab *🎯 Simulation*:**

| Metrik | Penjelasan |
|--------|------------|
| **Measured WCRT (ms)** | Response time terbesar yang terukur dari semua run |
| **Deadline (ms)** | Batas waktu tugas |
| **Status** | ✅ OK jika ≤ deadline, ❌ MISS jika terlambat |

Simulasi menggunakan *random offset* setiap run untuk mendapatkan skenario
terburuk. Semakin banyak `num_runs`, semakin akurat pengukuran WCRT.

---

### 4. Perbandingan

Klik **📊 Compare Results** untuk melihat perbandingan antara hasil
analisis dan simulasi secara side-by-side.

**Metrik Perbandingan:**

| Metrik | Sumber | Penjelasan |
|--------|--------|------------|
| **Measured WCRT (ms)** | Simulator | Response time aktual dari discrete-event simulation |
| **Estimated WCRT (ms)** | Analisis | Response time estimasi dari Algoritma 1 |
| **Deadline (ms)** | Task set | Batas waktu absolut yang harus dipenuhi |
| **Overestimation** | (E-M)/M | Rasio konservatisme estimasi terhadap aktual |
| **Meets?** | E ≤ D | ✅ YES jika estimasi ≤ deadline, ❌ NO jika tidak |

**Interpretasi:**

| Skenario | Arti |
|----------|------|
| Estimated ≈ Measured | Analisis akurat, tidak berlebihan |
| Estimated > Measured | Analisis konservatif (safe — lebih baik) |
| Estimated > Deadline | Tugas tidak schedulable — perlu optimasi |
| Overestimation tinggi | Analisis terlalu konservatif (masih safe) |

Jika estimasi ≤ deadline → tugas **schedulable ✅**.
Tools juga menampilkan **bar chart** perbandingan visual.

---

### 5. Optimasi Nice Value

Klik **🧬 Optimize Nice Values** untuk mencari assignment *nice value*
optimal agar semua tugas schedulable.

**Algoritma 2 — Deadline-Aware Heuristic:**

| Parameter | Default | Penjelasan |
|-----------|---------|------------|
| `lambda_min` | 0.0 | Nilai awal lambda (λ) yang dicoba |
| `lambda_max` | 40.0 | Nilai maksimum lambda |
| `lambda_gap` | 0.1 | Step kenaikan lambda setiap iterasi |
| **Formula** | — | nice_i = round(-λ · log₂(D_max / D_i)) |

Prinsip: Tugas dengan deadline lebih pendek mendapat nice lebih rendah
(prioritas lebih tinggi). Lambda mengontrol seberapa agresif perbedaan
nice value antar tugas.

**Algoritma 3 — Genetic Algorithm (GA):**

| Parameter | Default | Penjelasan |
|-----------|---------|------------|
| `population_size` | 100 | Jumlah kromosom dalam satu generasi |
| `mutation_rate` | 0.05 (5%) | Probabilitas mutasi setiap gen |
| `max_generations` | 200 | Generasi maksimum sebelum berhenti |
| `timeout_seconds` | 5.0 | Waktu maksimum eksekusi (safety) |
| `tournament_size` | 3 | Jumlah kandidat seleksi tiap turnamen |
| **Fitness** | — | Jumlah tugas schedulable (max = n) |

**Output:**

| Metrik | Penjelasan |
|--------|------------|
| **Schedulable?** | ✅ Ya / ❌ Tidak — apakah semua tugas schedulable |
| **Nice Values** | Tabel assignment nice value per tugas |
| **Method** | Heuristic / Genetic Algorithm |

GA mencari kombinasi 40ⁿ kemungkinan nice value (n = jumlah tugas)
menggunakan evolusi: seleksi → crossover → mutasi.

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
