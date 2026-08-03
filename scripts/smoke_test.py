"""Smoke test: corre una simulación corta con mundo sintético y verifica
que las 7 especies iniciales sigan vivas al final.

Regresión que motivó este script: tras la Fase 2 (siembra de fauna),
Yi qi y Triops se extinguían antes del año 25 porque sus baby_quantity
biológicos no llegaban al fenotipo y la mortalidad por inadaptación era
demasiado agresiva. Este test debe mantenerse verde.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stoneplace.simulator import Simulation, SimConfig


def main() -> int:
    cfg = SimConfig(
        world_bin=None, world_json=None,
        synthetic_size=(96, 48),
        out_dir="outputs_smoke",
        seed=7,
        phase1_years=20,
        total_years=60,
        telemetry_every=5,
        speciation_every=1000,   # desactivado para leer supervivencia limpia
        founder_size_plant=200,
        founder_size_fungi=150,
        founder_size_animal=400,
    )
    sim = Simulation(cfg)
    sim.run()

    print("\n=== VERIFICACIÓN ===")
    ok = True
    for pop in sim.pops:
        tpl = pop.template
        status = f"{tpl.scientific_name:40s} n={pop.n:>7}  born={tpl.born_at}"
        if pop.n == 0:
            status += "  ❌ EXTINTA"
            ok = False
        else:
            status += "  ✅ VIVA"
        print(status)
    if not ok:
        print("\n❌ Al menos una especie inicial se extinguió")
        return 1
    print("\n✅ Todas las especies iniciales sobrevivieron")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
