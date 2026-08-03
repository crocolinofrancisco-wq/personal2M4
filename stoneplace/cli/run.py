"""CLI: `python -m stoneplace.cli.run`."""
from __future__ import annotations
import argparse
from ..simulator import Simulation, SimConfig


def main():
    p = argparse.ArgumentParser("stoneplace")
    p.add_argument("--world-bin", default=None,
                   help="Ruta al .bin DDG (opcional; sin él se genera un mundo sintético)")
    p.add_argument("--world-json", default=None,
                   help="Ruta al .json DDG (opcional; sin él se genera un mundo sintético)")
    p.add_argument("--synthetic-size", type=int, nargs=2, default=(128, 64),
                   metavar=("WIDTH", "HEIGHT"),
                   help="Tamaño del mundo sintético si no se pasa raster")
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
    p.add_argument("--max-pop-per-species", type=int, default=300_000,
                   help="Tope duro por especie (thinning aleatorio si se supera)")
    p.add_argument("--dump-every-snapshots", type=int, default=20,
                   help="Volcar telemetría a disco cada N snapshots")
    args = p.parse_args()

    cfg = SimConfig(
        world_bin=args.world_bin,
        world_json=args.world_json,
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
        max_pop_per_species=args.max_pop_per_species,
        dump_every_snapshots=args.dump_every_snapshots,
    )
    Simulation(cfg).run()


if __name__ == "__main__":
    main()
