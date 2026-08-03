"""Serialización JSON+NPZ del estado de Simulation (v1.5, N10).

El formato es intencionalmente simple: un JSON con la config +
metadata y un NPZ con los arrays pesados (genomas, coordenadas). Se
puede reanudar con `Simulation.load(path)` sin re-ejecutar Fase 1.

No pretende ser un formato de intercambio (para eso está `tskit`); es
un checkpoint operacional.
"""
from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np


def _cfg_to_dict(cfg) -> dict:
    d = asdict(cfg) if is_dataclass(cfg) else dict(cfg.__dict__)
    return d


def save_simulation(sim, path: str | Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": "1.5",
        "year": float(sim.year),
        "phase": int(sim.phase),
        "cfg": _cfg_to_dict(sim.cfg),
        "speciation_state": {
            "last_species_id": sim.speciation_state.last_species_id,
            "history_window": sim.speciation_state.history_window,
            "fst_sustain": sim.speciation_state.fst_sustain,
            "qst_sustain": sim.speciation_state.qst_sustain,
        },
        "pops": [],
    }
    arrays: dict[str, np.ndarray] = {
        "base_temperature": sim._base_temperature,
        "biome_map": sim.biome_map,
        "plant_biomass": sim.plant_store.biomass,
        "fungi_biomass": sim.fungi_store.biomass,
        "micro_bugs": sim.micro.bugs,
        "micro_microbes": sim.micro.microbes,
        "micro_plankton": sim.micro.plankton,
        "micro_detritus": sim.micro.detritus,
    }
    for i, pop in enumerate(sim.pops):
        tpl = pop.template
        pop_meta = {
            "idx": i,
            "species_id": tpl.species_id,
            "scientific_name": tpl.scientific_name,
            "kingdom": tpl.kingdom,
            "parent": tpl.parent,
            "born_at": tpl.born_at,
            "n_loci": tpl.n_loci,
            "n_chromosomes": tpl.n_chromosomes,
            "taxonomy": tpl.taxonomy,
            "hints": {k: v for k, v in tpl.hints.items()
                      if isinstance(v, (int, float, str, bool, list, dict))},
            "n": int(pop.n),
        }
        meta["pops"].append(pop_meta)
        if pop.n > 0:
            arrays[f"pop_{i}_alleles"] = pop.genome.alleles
            arrays[f"pop_{i}_chroms"] = pop.genome.chromosomes
            arrays[f"pop_{i}_y"] = pop.y
            arrays[f"pop_{i}_x"] = pop.x
            arrays[f"pop_{i}_age"] = pop.age_years
            arrays[f"pop_{i}_energy"] = pop.energy
            if pop.sex is not None:
                arrays[f"pop_{i}_sex"] = pop.sex
    (path / "state.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(path / "state.npz", **arrays)


def load_simulation(path: str | Path):
    """Carga básica — reconstruye un Simulation en el punto guardado.

    El GP map de cada especie se re-genera desde su seed conservada en
    hints, así que las mutaciones cis-regulatorias no persisten
    exactamente (para eso está `stoneplace.io.tskit_export`).
    """
    from ..simulator import Simulation, SimConfig       # lazy import
    from ..species.base import SpeciesPopulation, SpeciesTemplate
    from ..genetics.genome import Genome
    from ..genetics.gp_map import chordata_gpmap, flora_gpmap

    path = Path(path)
    meta = json.loads((path / "state.json").read_text(encoding="utf-8"))
    arrays = np.load(path / "state.npz")
    cfg = SimConfig(**{k: v for k, v in meta["cfg"].items()
                       if k in SimConfig.__dataclass_fields__})
    sim = Simulation(cfg)
    sim.year = float(meta["year"])
    sim.phase = int(meta["phase"])
    sim._base_temperature = arrays["base_temperature"]
    sim.biome_map = arrays["biome_map"]
    sim.plant_store.biomass = arrays["plant_biomass"]
    sim.fungi_store.biomass = arrays["fungi_biomass"]
    sim.micro.bugs = arrays["micro_bugs"]
    sim.micro.microbes = arrays["micro_microbes"]
    sim.micro.plankton = arrays["micro_plankton"]
    sim.micro.detritus = arrays["micro_detritus"]
    sim.speciation_state.last_species_id = meta["speciation_state"]["last_species_id"]
    # Reconstruimos las poblaciones
    sim.pops = []
    for pm in meta["pops"]:
        i = pm["idx"]
        if pm["n"] == 0:
            continue
        is_fungi = pm["kingdom"] == "fungi"
        if pm["kingdom"] == "chordata":
            gp = chordata_gpmap(pm["n_loci"], sim.map_rng, hints=pm["hints"],
                                 map_rng=sim.map_rng)
        else:
            gp = flora_gpmap(pm["n_loci"], sim.map_rng, is_fungi=is_fungi,
                              hints=pm["hints"], map_rng=sim.map_rng)
        tpl = SpeciesTemplate(
            species_id=pm["species_id"],
            scientific_name=pm["scientific_name"],
            taxonomy=list(pm["taxonomy"]),
            kingdom=pm["kingdom"],
            parent=pm["parent"], born_at=pm["born_at"],
            n_loci=pm["n_loci"], n_chromosomes=pm["n_chromosomes"],
            gp_map=gp, hints=dict(pm["hints"]),
        )
        genome = Genome(alleles=arrays[f"pop_{i}_alleles"],
                         chromosomes=arrays[f"pop_{i}_chroms"])
        pop = SpeciesPopulation(
            template=tpl, genome=genome,
            y=arrays[f"pop_{i}_y"], x=arrays[f"pop_{i}_x"],
            age_years=arrays[f"pop_{i}_age"],
            energy=arrays[f"pop_{i}_energy"],
            sex=arrays.get(f"pop_{i}_sex"),
        )
        pop.express(sim.rng, sigma_env=sim.cfg.env_sigma)
        sim.pops.append(pop)
    return sim
