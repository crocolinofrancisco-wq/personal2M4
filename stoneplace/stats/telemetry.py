"""Telemetría honesta v1.5: métricas por tick incluyendo pop-gen real.

v1.5 (roadmap v1.2 punto 6):
  * Cada snapshot añade π (nucleotide diversity aproximada) y H_obs
    (heterocigosidad) por especie.
  * `dump_species_cards` incluye Qst medio del linaje (si hay historia
    de divergencia guardada en `SpeciationState`).
"""
from __future__ import annotations
import csv
import json
import numpy as np
from pathlib import Path

from ..genetics.popgen import nucleotide_diversity_pi, heterozygosity


class Telemetry:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def snapshot(self, year: float, pops, biome_map,
                 speciation_state=None, narrative=None) -> dict:
        row = {"year": year, "total_pop": 0, "species_alive": 0}
        # v1.5.6 — imprimimos poblaciones bajas y detectamos extinciones
        # explícitamente en el log (antes sólo veíamos "N especies vivas").
        _watch = []
        for pop in pops:
            tpl = pop.template
            if pop.n == 0 and tpl.status == "alive":
                tpl.status = "extinct"
                tpl.extinct_at = year
                msg = (f"    ✘ año {year:>5.0f}: EXTINCIÓN de "
                       f"{tpl.scientific_name} (species_id={tpl.species_id})")
                print(msg, flush=True)
                if narrative is not None:
                    narrative.log(year, "extinction", tpl.species_id,
                                   f"Extinción de {tpl.scientific_name}.")
            elif 0 < pop.n < 200:
                _watch.append(f"{tpl.scientific_name}={pop.n}")
            if pop.n > 0:
                row["species_alive"] += 1
                row["total_pop"] += int(pop.n)
                tpl.historic_population = max(tpl.historic_population, int(pop.n))
                tpl.max_population = max(tpl.max_population, int(pop.n))
            sid = tpl.species_id
            safe = tpl.scientific_name.replace(" ", "_")
            row[f"n_{sid}_{safe}"] = int(pop.n)
            if pop.n > 0:
                # v1.5: π y H por especie (métricas honestas expuestas)
                additive = pop.genome.additive()
                row[f"pi_{sid}"] = round(nucleotide_diversity_pi(additive), 5)
                row[f"H_{sid}"] = round(heterozygosity(pop.genome.alleles), 5)
                # Qst en curso (última ventana registrada)
                if speciation_state is not None:
                    hist = speciation_state.divergence_history.get(sid, [])
                    if hist:
                        last_fst, last_qst = hist[-1]
                        row[f"Fst_{sid}"] = round(last_fst, 4)
                        row[f"Qst_{sid}"] = round(last_qst, 4)
                b = biome_map[pop.y, pop.x]
                counts = np.bincount(b, minlength=12)
                tpl.tile_distribution = counts.astype(np.int64)
                main = np.argsort(counts)[::-1][:3]
                from ..world.biomes import BIOME_NAMES
                tpl.main_biomes = [BIOME_NAMES.get(int(i), str(int(i)))
                                    for i in main if counts[int(i)] > 0]
        self.rows.append(row)
        if _watch:
            print(f"    ⚠ watch (n<200): {', '.join(_watch)}", flush=True)
        return row

    def dump(self):
        if not self.rows:
            return
        keys = sorted({k for r in self.rows for k in r.keys()})
        with (self.out_dir / "telemetry.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

    def dump_species_cards(self, pops):
        cards = []
        from ..world.biomes import BIOME_NAMES
        from .narrative import build_species_card
        for pop in pops:
            tpl = pop.template
            card = {
                "species_id": tpl.species_id,
                "scientific_name": tpl.scientific_name,
                "taxonomy": tpl.taxonomy,
                "kingdom": tpl.kingdom,
                "body_plan": tpl.hints.get("body_plan", tpl.kingdom),
                "parent": tpl.parent,
                "child": tpl.children,
                "born_at": tpl.born_at,
                "extinct_at": tpl.extinct_at,
                "status": tpl.status,
                "peak_population": tpl.historic_population,
                "max_population": tpl.max_population,
                "current_population": int(pop.n),
                "main_biomes": tpl.main_biomes,
                "tile_distribution": (
                    tpl.tile_distribution.tolist()
                    if hasattr(tpl, "tile_distribution")
                    and hasattr(tpl.tile_distribution, "tolist")
                    else list(getattr(tpl, "tile_distribution", []))
                ),
                "total_chromosomes": tpl.n_chromosomes,
                "total_loci": tpl.n_loci,
            }
            if pop.n > 0:
                for k, v in pop.phenotype.items():
                    card[f"pheno_mean_{k}"] = float(np.mean(v))
                add = pop.genome.additive().mean(axis=0)
                for i, val in enumerate(add):
                    card[f"Locus{i + 1:02d}"] = float(val)
                card["pi"] = nucleotide_diversity_pi(pop.genome.additive())
                card["heterozygosity"] = heterozygosity(pop.genome.alleles)
                # Ficha narrativa (rasgos desbloqueados, highlights)
                narrative = build_species_card(pop, phylo=None)
                card["narrative"] = narrative
            cards.append(card)
        (self.out_dir / "species.json").write_text(
            json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
