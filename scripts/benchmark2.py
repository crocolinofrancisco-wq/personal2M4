"""Benchmark v2: dos mediciones limpias (mismo protocolo, mismos años de
calentamiento) a dos tamaños de mundo con densidad de fundadores
proporcional al área, para poder extrapolar linealmente a 1024x512 sin el
sesgo del primer intento (que usaba distintos años de calentamiento por
config y comparaba poblaciones en fases de madurez distintas).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stoneplace.simulator import Simulation, SimConfig

BASE_W, BASE_H = 400, 200
BASE_FP, BASE_FF, BASE_FA = 500, 400, 800
PHASE1_YEARS = 60
MEASURE_YEARS = 40


def bench(width, height, seed=1):
    scale = (width * height) / (BASE_W * BASE_H)
    fp = max(int(BASE_FP * scale), 20)
    ff = max(int(BASE_FF * scale), 20)
    fa = max(int(BASE_FA * scale), 20)
    cfg = SimConfig(
        world_bin=None, world_json=None,
        synthetic_size=(width, height),
        out_dir="outputs_bench2",
        seed=seed,
        phase1_years=PHASE1_YEARS, total_years=PHASE1_YEARS + 1,
        telemetry_every=10_000, speciation_every=10_000,
        founder_size_plant=fp, founder_size_fungi=ff, founder_size_animal=fa,
    )
    sim = Simulation(cfg)
    sim.seed_phase1()
    for _ in range(PHASE1_YEARS):
        sim._step_one_year()
    sim.seed_phase2()
    for _ in range(20):   # dejar que la fauna llegue a régimen antes de medir
        sim._step_one_year()

    pop_start = sum(p.n for p in sim.pops)
    t0 = time.perf_counter()
    for _ in range(MEASURE_YEARS):
        sim._step_one_year()
    elapsed = time.perf_counter() - t0
    pop_end = sum(p.n for p in sim.pops)
    ms_per_year = elapsed / MEASURE_YEARS * 1000
    cells = width * height
    print(f"world={width}x{height} cells={cells:,} founders=({fp},{ff},{fa}) "
          f"pop {pop_start:,}->{pop_end:,} | {ms_per_year:.1f} ms/año "
          f"(promedio sobre {MEASURE_YEARS} años en régimen)")
    return cells, ms_per_year


def main():
    print("=== Benchmark v2 (motor optimizado hoy) ===\n")
    c1, t1 = bench(400, 200)
    c2, t2 = bench(700, 350)

    # Recta exacta por los dos puntos: ms = a*cells + b
    a = (t2 - t1) / (c2 - c1)
    b = t1 - a * c1
    print(f"\nRecta: ms/año ≈ {a:.6f} * celdas + {b:.2f}")

    real_cells = 1024 * 512
    est_ms = a * real_cells + b
    print(f"1024x512 = {real_cells:,} celdas → ~{est_ms:.0f} ms/año (motor optimizado)")

    UNOPT_MS_400x200 = 1800.0  # medido real anoche: 400 años / ~720s en GH Actions
    speedup = UNOPT_MS_400x200 / t1
    print(f"\nComparación con la corrida real de anoche (sin optimizar): "
          f"{UNOPT_MS_400x200:.0f} ms/año @ 400x200 → speedup medido hoy: {speedup:.2f}x")

    print("\n=== Estimación mundo real 1024x512 ===")
    for total_years in (400, 1000, 2000, 5000, 10000):
        hours = est_ms * total_years / 1000 / 3600
        print(f"  {total_years:>6} años → ~{hours:.2f} h (single-core, "
              f"población escalada proporcional al área)")


if __name__ == "__main__":
    main()
