"""Benchmark real: mide ms/tick a distintos tamaños de mundo y densidad
poblacional, para poder extrapolar el costo del mundo SeedWorld real
(1024x512) sin adivinar.

No mide "cuánto tarda una corrida completa" — mide el costo marginal por
año simulado en función de:
    (a) tamaño del raster (H*W) — domina el costo de los campos globales
        (biomasa, K, forrajeo local) que se recalculan cada tick sin
        importar cuánta gente haya.
    (b) población total viva — domina el costo de reproducción/mortalidad/
        expresión fenotípica (todo vectorizado sobre N individuos).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stoneplace.simulator import Simulation, SimConfig


def bench_one(width, height, founder_plant, founder_fungi, founder_animal,
             warmup_years, measure_years, seed=1):
    cfg = SimConfig(
        world_bin=None, world_json=None,
        synthetic_size=(width, height),
        out_dir="outputs_bench",
        seed=seed,
        phase1_years=warmup_years,
        total_years=warmup_years + 1,   # se pisa: corremos manual abajo
        telemetry_every=10_000,          # no queremos overhead de I/O
        speciation_every=10_000,         # no queremos especiación en el bench
        founder_size_plant=founder_plant,
        founder_size_fungi=founder_fungi,
        founder_size_animal=founder_animal,
    )
    sim = Simulation(cfg)
    sim.seed_phase1()
    t0 = time.perf_counter()
    for _ in range(warmup_years):
        sim._step_one_year()
    t_phase1 = time.perf_counter() - t0

    sim.seed_phase2()
    pop_at_measure_start = sum(p.n for p in sim.pops)

    t0 = time.perf_counter()
    for _ in range(measure_years):
        sim._step_one_year()
    t_measure = time.perf_counter() - t0

    pop_end = sum(p.n for p in sim.pops)
    cells = width * height
    ms_per_tick = (t_measure / measure_years) * 1000
    print(f"world={width:>5}x{height:<5} cells={cells:>9,} "
          f"pop_start={pop_at_measure_start:>9,} pop_end={pop_end:>9,} "
          f"| {ms_per_tick:>8.1f} ms/año (con fauna) "
          f"| fase1 avg {t_phase1/warmup_years*1000:>7.1f} ms/año (solo flora)")
    return {
        "width": width, "height": height, "cells": cells,
        "pop_start": pop_at_measure_start, "pop_end": pop_end,
        "ms_per_tick": ms_per_tick,
        "ms_per_tick_phase1": t_phase1 / warmup_years * 1000,
    }


def main():
    print("=== Benchmark Stoneplace: ms/año simulado ===\n")
    results = []
    # Densidad de fundadores fija por celda para que la escala sea honesta:
    # usamos la misma densidad relativa que el run overnight (mundo 400x200,
    # founders 500/400/800).
    base_cells = 400 * 200
    configs = [
        (100, 50, 8),      # mundo chico (~equivalente smoke test grande)
        (200, 100, 10),
        (400, 200, 12),    # el tamaño del run overnight actual
    ]
    for w, h, years in configs:
        cells = w * h
        scale = cells / base_cells
        fp = max(int(500 * scale), 50)
        ff = max(int(400 * scale), 40)
        fa = max(int(800 * scale), 80)
        r = bench_one(w, h, fp, ff, fa, warmup_years=years, measure_years=8)
        results.append(r)
        print()

    print("=== Extrapolación a mundo real 1024x512 ===")
    # Regresión lineal simple ms/tick vs cells (con fauna activa,
    # población ya en régimen — ahí es donde vive el costo real).
    import numpy as np
    cells_arr = np.array([r["cells"] for r in results], dtype=np.float64)
    ms_arr = np.array([r["ms_per_tick"] for r in results], dtype=np.float64)
    # Ajuste lineal ms = a*cells + b
    A = np.vstack([cells_arr, np.ones_like(cells_arr)]).T
    a, b = np.linalg.lstsq(A, ms_arr, rcond=None)[0]
    real_cells = 1024 * 512
    est_ms = a * real_cells + b
    print(f"Ajuste: ms/año ≈ {a:.6f} * celdas + {b:.2f}")
    print(f"1024x512 = {real_cells:,} celdas → ~{est_ms:.0f} ms/año simulado")
    for total_years in (400, 1000, 5000, 10000):
        secs = est_ms * total_years / 1000
        hours = secs / 3600
        print(f"  {total_years:>6} años → ~{hours:.2f} horas de cómputo (single-core)")


if __name__ == "__main__":
    main()
