"""CLI: `python -m stoneplace.cli.run`."""
from __future__ import annotations
import argparse
import json
from ..simulator import (Simulation, SimConfig,
                         _DEFAULT_WORLD_BIN, _DEFAULT_WORLD_JSON)


def main():
    p = argparse.ArgumentParser("stoneplace")
    p.add_argument("--world-bin", default=_DEFAULT_WORLD_BIN,
                   help="Ruta al .bin DDG (por defecto: mapa real 768x384)")
    p.add_argument("--world-json", default=_DEFAULT_WORLD_JSON,
                   help="Ruta al .json DDG (por defecto: mapa real 768x384)")
    p.add_argument("--synthetic", action="store_true",
                   help="Fuerza mundo sintético en vez del mapa real.")
    p.add_argument("--synthetic-size", type=int, nargs=2, default=(128, 64),
                   metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--perf-log-every", type=int, default=25)
    p.add_argument("--out", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase1-years", type=int, default=200)
    p.add_argument("--total-years", type=int, default=800)
    p.add_argument("--telemetry-every", type=int, default=5)
    p.add_argument("--speciation-every", type=int, default=25)
    p.add_argument("--founder-plant", type=int, default=400)
    p.add_argument("--founder-fungi", type=int, default=300)
    p.add_argument("--founder-animal", type=int, default=600)
    p.add_argument("--mutation-rate", type=float, default=5e-4)
    p.add_argument("--mutation-sigma", type=float, default=2.5)
    p.add_argument("--env-sigma", type=float, default=8.0)
    p.add_argument("--warn-pop-per-species", type=int, default=500_000,
                   help="v1.5: warning si una especie supera este N (sin cap duro).")
    p.add_argument("--dump-every-snapshots", type=int, default=20)
    # v1.5
    p.add_argument("--climate-disabled", action="store_true",
                   help="Deshabilita ciclos Milankovitch/estacionales (mundo estático)")
    p.add_argument("--milankovitch-period", type=float, default=40_000.0)
    p.add_argument("--milankovitch-amp", type=float, default=3.0)
    p.add_argument("--seasonal-amp", type=float, default=4.0)
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="PNG del mundo cada N años (0 = deshabilitado)")
    p.add_argument("--perturbations-file", default=None,
                   help="JSON con lista de eventos catastróficos")
    args = p.parse_args()

    perturbation_events = []
    if args.perturbations_file:
        perturbation_events = json.loads(
            open(args.perturbations_file).read())

    world_bin = None if args.synthetic else args.world_bin
    world_json = None if args.synthetic else args.world_json
    cfg = SimConfig(
        world_bin=world_bin,
        world_json=world_json,
        synthetic_size=tuple(args.synthetic_size),
        out_dir=args.out,
        seed=args.seed,
        phase1_years=args.phase1_years,
        total_years=args.total_years,
        telemetry_every=args.telemetry_every,
        speciation_every=args.speciation_every,
        founder_size_plant=args.founder_plant,
        founder_size_fungi=args.founder_fungi,
        founder_size_animal=args.founder_animal,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        env_sigma=args.env_sigma,
        warn_pop_per_species=args.warn_pop_per_species,
        dump_every_snapshots=args.dump_every_snapshots,
        climate_enabled=not args.climate_disabled,
        milankovitch_period_years=args.milankovitch_period,
        milankovitch_amplitude_c=args.milankovitch_amp,
        seasonal_amplitude_c=args.seasonal_amp,
        snapshot_every=args.snapshot_every,
        perturbation_events=perturbation_events,
        perf_log_every=args.perf_log_every,
    )
    Simulation(cfg).run()


if __name__ == "__main__":
    main()
