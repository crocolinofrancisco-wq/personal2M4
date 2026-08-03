"""Telemetría honesta: métricas por tick para exportar a CSV/JSON."""
from __future__ import annotations
import csv
import json
import numpy as np
from pathlib import Path


class Telemetry:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def snapshot(self, year: float, pops, biome_map) -> dict:
        row = {"year": year, "total_pop": 0, "species_alive": 0}
        for pop in pops:
            tpl = pop.template
            if pop.n == 0 and tpl.status == "alive":
                tpl.status = "extinct"
                tpl.extinct_at = year
            if pop.n > 0:
                row["species_alive"] += 1
                row["total_pop"] += int(pop.n)
                tpl.historic_population = max(tpl.historic_population,
                                              int(pop.n) + tpl.historic_population)
                tpl.max_population = max(tpl.max_population, int(pop.n))
            key = f"n_{tpl.species_id}_{tpl.scientific_name.replace(' ', '_')}"
            row[key] = int(pop.n)
            # Distribución por bioma
            if pop.n > 0:
                b = biome_map[pop.y, pop.x]
                counts = np.bincount(b, minlength=12)
                tpl.tile_distribution = counts.astype(np.int64)
                main = np.argsort(counts)[::-1][:3]
                from ..world.biomes import BIOME_NAMES
                tpl.main_biomes = [BIOME_NAMES.get(int(i), str(int(i)))
                                    for i in main if counts[int(i)] > 0]
        self.rows.append(row)
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
        for pop in pops:
            tpl = pop.template
            card = {
                "species_id": tpl.species_id,
                "scientific_name": tpl.scientific_name,
                "taxonomy": tpl.taxonomy,
                "kingdom": tpl.kingdom,
                "parent": tpl.parent,
                "child": tpl.children,
                "born_at": tpl.born_at,
                "extinct_at": tpl.extinct_at,
                "status": tpl.status,
                "historic_population": tpl.historic_population,
                "max_population": tpl.max_population,
                "current_population": int(pop.n),
                "main_biomes": tpl.main_biomes,
                "tile_distribution": tpl.tile_distribution.tolist(),
                "total_chromosomes": tpl.n_chromosomes,
                "total_loci": tpl.n_loci,
            }
            # Fenotipo medio actual (sirve como "carta de especie")
            if pop.n > 0:
                for k, v in pop.phenotype.items():
                    card[f"pheno_mean_{k}"] = float(np.mean(v))
                # Loci medios (0..100)
                add = pop.genome.additive().mean(axis=0)
                for i, val in enumerate(add):
                    card[f"Locus{i + 1:02d}"] = float(val)
            cards.append(card)
        (self.out_dir / "species.json").write_text(
            json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
