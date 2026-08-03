"""Benchmark de rendimiento: mide ms/año simulado en 4 tamaños de grilla.

Ajusta un polinomio cuadrático (a*cells² + b*cells + c) porque el costo
NO es lineal exacto — el término _local_mean domina y tiene O(H·W)
después del fix con scipy.uniform_filter, pero los stores persistentes
y la expresión fenotípica añaden un término lineal-a-cuadrático más.
"""
from __future__ import annotations
import time
import numpy as np
from stoneplace.simulator import Simulation, SimConfig


SIZES = [(100, 50), (200, 100), (400, 200), (700, 350)]
YEARS = 30


def _bench(w, h):
    cfg = SimConfig(
        synthetic_size=(w, h),
        phase1_years=YEARS // 2, total_years=YEARS,
        telemetry_every=YEARS, speciation_every=YEARS + 1,
        seed=42,
        founder_size_plant=150, founder_size_fungi=100, founder_size_animal=200,
        out_dir=f"bench_{w}x{h}",
    )
    t0 = time.perf_counter()
    Simulation(cfg).run()
    dt = time.perf_counter() - t0
    ms_per_year = 1000.0 * dt / YEARS
    print(f"  {w}x{h} = {w*h:>7} celdas → {ms_per_year:>8.1f} ms/año  "
          f"(total {dt:.1f}s)")
    return w * h, ms_per_year


def main():
    print(f"[benchmark] Stoneplace v1.1 — {YEARS} años por tamaño")
    xs = []; ys = []
    for w, h in SIZES:
        c, ms = _bench(w, h)
        xs.append(c); ys.append(ms)
    xs = np.array(xs, dtype=np.float64); ys = np.array(ys, dtype=np.float64)
    a, b, c = np.polyfit(xs, ys, 2)
    print("\n[fit cuadrático]  ms/año ≈ "
          f"{a:.3e}·cells² + {b:.3e}·cells + {c:.1f}")
    # Extrapolación al mundo DDG completo (1024×512)
    target = 1024 * 512
    est_ms = a * target * target + b * target + c
    print(f"[extrapolación]  1024x512 = {target} celdas → "
          f"{est_ms:.0f} ms/año   (~{est_ms * 1000 / 3600e3:.2f} h / 1000 años)")


if __name__ == "__main__":
    main()
