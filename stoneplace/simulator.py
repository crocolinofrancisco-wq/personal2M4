"""Simulador Stoneplace v1.5 — ciclo anual + clima dinámico + narrativa.

Cambios v1.5 respecto a v1.1 (respuestas2.txt):
  N4  — se elimina `max_pop_per_species` como cap duro; queda como
        umbral de warning (defensivo, no científico).
  N5  — clima dinámico: ciclo de Milankovitch (40 000 años) + ciclo
        estacional. Se ejecuta cada tick y actualiza world.layers["temperature"]
        respecto a un baseline capturado al inicio.
  N8  — se pasa el mapa de densidad por especie al `micro_dict` para
        que `habitat.suitability` aplique distribución libre ideal.
  N9  — la constante 50000 pasa a `CONSUMPTION_SCALE_KCAL` en el
        paquete raíz.
  N13/N14 — `SpeciationState` se posee en la instancia (adiós globals).
  Narrativa — cada evento (nacimiento, extinción, especiación,
        catástrofe) se anota en `NarrativeLog` (opcional; consumible por
        `stoneplace.narrative`).
  Perturbaciones — `PerturbationSchedule` puede aplicar impactos,
        glaciaciones, mega-erupciones parametrizables.
  Tree-sequence — cada especiación queda registrada en `PhyloTree` para
        exportar Newick con `stoneplace.io.newick_export`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from . import HOURS_PER_YEAR, CONSUMPTION_SCALE_KCAL
from .world.loader import World, load_world
from .world.biomes import classify, BIOME_NAMES
from .world.synthetic import make_synthetic_world
from .species.base import SpeciesPopulation, SpeciesTemplate
from .species.registry import all_prototype_species
from .genetics.genome import new_founder_population
from .ecology.microfauna import (MicrofaunaField, carrying_capacity,
                                  logistic_step)
from .ecology.producers import (biomass_field, meat_field, fish_field,
                                 PlantBiomassStore)
from .ecology.habitat import suitability, _local_mean
from .ecology.population import age_and_die, reproduce, _build_density
from .ecology.dispersal import animal_walk, disperse_seeds
from .ecology.speciation import try_speciate, SpeciationState
from .stats.telemetry import Telemetry
from .stats.phylo import PhyloTree
from .stats.narrative import NarrativeLog
from .ecology.climate import ClimateCycle
from .ecology.perturbation import PerturbationSchedule
from .utils import rng_from_seed
import time
from pathlib import Path as _Path
_DEFAULT_WORLD_BIN = str(_Path(__file__).parent.parent
                         / "data/samples/ddg_world_619267880_768x384_0Ma.bin")
_DEFAULT_WORLD_JSON = str(_Path(__file__).parent.parent
                          / "data/samples/ddg_world_619267880_768x384_0Ma.json")


@dataclass
class SimConfig:
    world_bin: str | None = None
    world_json: str | None = None
    synthetic_size: tuple[int, int] = (128, 64)
    out_dir: str = "outputs"
    seed: int = 42
    dt_years: float = 1.0
    mutation_rate: float = 5e-4
    mutation_sigma: float = 2.5
    env_sigma: float = 8.0
    speciation_every: int = 25
    telemetry_every: int = 5
    phase1_years: int = 200
    total_years: int = 800
    founder_size_plant: int = 400
    founder_size_fungi: int = 300
    founder_size_animal: int = 600
    # v1.5 (N4): ya NO se hace thinning duro. Sólo warning si se supera.
    warn_pop_per_species: int = 500_000
    dump_every_snapshots: int = 20
    reexpress_every: int = 4
    reclassify_biomes_every: int = 100
    # v1.5 (N5) — clima dinámico
    climate_enabled: bool = True
    milankovitch_period_years: float = 40_000.0
    milankovitch_amplitude_c: float = 3.0
    seasonal_amplitude_c: float = 4.0
    # Snapshots visuales (opcional; require matplotlib)
    snapshot_every: int = 0            # 0 = deshabilitado
    # Perturbaciones (impactos, glaciaciones)
    perturbation_events: list[dict] = field(default_factory=list)
    # v1.6 perf
    perf_log_every: int = 25
    # v1.5.4 — mundos DDG grandes: si la memoria por individuo escala mal,
    # el usuario puede topear founders en lugar de tocar el motor.


class Simulation:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = rng_from_seed(cfg.seed)
        # RNG dedicado a construir GP maps (N2). Determinístico por seed pero
        # ortogonal al RNG de la simulación, así reordenar traits[...] no
        # rompe la corrida.
        self.map_rng = rng_from_seed(cfg.seed ^ 0x9E3779B1)
        if cfg.world_bin and cfg.world_json:
            self.world: World = load_world(cfg.world_bin, cfg.world_json)
        else:
            w, h = cfg.synthetic_size
            self.world = make_synthetic_world(width=w, height=h, seed=cfg.seed)
        # Guardamos el clima "base" (año 0) para poder añadir Milankovitch/
        # estacionalidad sin que se acumulen los deltas tick a tick (N5).
        self._base_temperature = self.world.layers["temperature"].copy()
        self.climate = ClimateCycle(
            enabled=cfg.climate_enabled,
            milankovitch_period=cfg.milankovitch_period_years,
            milankovitch_amp=cfg.milankovitch_amplitude_c,
            seasonal_amp=cfg.seasonal_amplitude_c,
        )
        self.biome_map = classify(self.world)
        self.micro = MicrofaunaField.zeros(self.world.shape)
        self.plant_store = PlantBiomassStore.zeros(self.world.shape)
        self.fungi_store = PlantBiomassStore.zeros(self.world.shape)
        self.pops: list[SpeciesPopulation] = []
        self.templates = all_prototype_species(self.rng, map_rng=self.map_rng)
        self.year: float = 0.0
        self.phase: int = 1
        self.telemetry = Telemetry(cfg.out_dir)
        # v1.5 (N13/N14) — sin globals
        self.speciation_state = SpeciationState()
        self.phylo = PhyloTree()
        self.narrative = NarrativeLog(cfg.out_dir)
        self.perturbations = PerturbationSchedule(cfg.perturbation_events)
        # v1.6 telemetría de rendimiento (segundos por año, desgloses)
        self.perf_rows: list[dict] = []

    # ------------------------------------------------------------------
    # Siembra
    # ------------------------------------------------------------------
    def _sample_positions_for(self, tpl: SpeciesTemplate, n: int,
                              seed_point=None) -> tuple[np.ndarray, np.ndarray]:
        aquatic = tpl.hints.get("aquatic", False)
        elev = self.world.layers["elevation"]
        temp = self.world.layers["temperature"]
        salt = self.world.layers.get("water_salinity", np.zeros_like(elev))
        rains = self.world.layers.get("rains", np.zeros_like(elev))
        river = self.world.layers.get("riverFlow", np.zeros_like(elev))
        aw = self.world.layers.get("available_water", rains)
        if aquatic:
            sal_tol = float(tpl.hints.get("salt_tolerance", 3))
            mask = ((elev <= 0) & (salt <= sal_tol + 2)) | (river > 0.05) | \
                   ((elev > 0) & (elev < 400) & (aw > 250))
        else:
            mask = elev > 0
        tmin = tpl.hints.get("min_temp", -20)
        tmax = tpl.hints.get("max_temp", 45)
        mask &= (temp >= tmin - 2) & (temp <= tmax + 2)

        idxs = np.argwhere(mask)
        if idxs.size == 0:
            return np.array([], np.int32), np.array([], np.int32)

        if seed_point is not None:
            cy, cx = seed_point
            dy = idxs[:, 0] - cy
            dx = idxs[:, 1] - cx
            d2 = dy * dy + dx * dx
            weights = np.exp(-d2 / (2 * 80.0 ** 2))
            weights = weights / (weights.sum() + 1e-9)
            pick = self.rng.choice(idxs.shape[0], size=n, replace=True, p=weights)
        else:
            pick = self.rng.choice(idxs.shape[0], size=n, replace=True)
        chosen = idxs[pick]
        return chosen[:, 0].astype(np.int32), chosen[:, 1].astype(np.int32)

    def _seed_species(self, tpl: SpeciesTemplate, n: int, seed_point=None):
        y, x = self._sample_positions_for(tpl, n, seed_point)
        if y.size == 0:
            print(f"  [!] no hay hábitat inicial para {tpl.scientific_name}")
            return
        genome = new_founder_population(
            n=y.size, n_loci=tpl.n_loci, n_chromosomes=tpl.n_chromosomes,
            mean=50.0, sd=12.0, rng=self.rng)
        sex = None
        if tpl.kingdom == "chordata":
            sex = self.rng.integers(0, 2, size=y.size).astype(np.int8)
        pop = SpeciesPopulation(
            template=tpl, genome=genome, y=y, x=x,
            age_years=self.rng.uniform(0, 1, size=y.size).astype(np.float32),
            energy=np.full(y.size, 0.7, dtype=np.float32),
            sex=sex,
        )
        pop.express(self.rng, sigma_env=self.cfg.env_sigma)
        self.pops.append(pop)
        self.phylo.register(tpl, parent=None, born_at=self.year)
        self.narrative.log(self.year, "founding", tpl.species_id,
                            f"Se siembra {tpl.scientific_name}: "
                            f"{y.size} fundadores.")
        print(f"  · sembrada {tpl.scientific_name}: {y.size} individuos")

    def seed_phase1(self):
        print("[Fase 1] Siembra de productores primarios y descomponedores")
        for tpl in self.templates:
            if tpl.kingdom in ("plantae",):
                self._seed_species(tpl, self.cfg.founder_size_plant)
            elif tpl.kingdom == "fungi":
                self._seed_species(tpl, self.cfg.founder_size_fungi)

    def seed_phase2(self, seed_point=None):
        print("[Fase 2] Siembra de fauna")
        if seed_point is None:
            plant_b = self.plant_store.biomass
            score = np.where(self.world.layers["elevation"] > 0, plant_b, 0)
            idx = np.unravel_index(np.argmax(score), score.shape)
            seed_point = idx
        print(f"  Punto de siembra animal: y={seed_point[0]}, x={seed_point[1]}")
        for tpl in self.templates:
            if tpl.kingdom == "chordata":
                sp = seed_point
                if tpl.hints.get("aquatic", False):
                    sp = None
                self._seed_species(tpl, self.cfg.founder_size_animal,
                                    seed_point=sp)

    # ------------------------------------------------------------------
    # Dispersión σ por fenotipo (P4)
    # ------------------------------------------------------------------
    def _seed_sigma_for(self, pop: SpeciesPopulation) -> float:
        p = pop.phenotype
        if pop.template.kingdom in ("plantae", "fungi"):
            sr = float(np.mean(p.get("seed_resistance_level",
                                     np.full(pop.n, 25.0))))
            return 1.0 + sr / 20.0
        speed = float(np.mean(p.get("speed", np.full(pop.n, 45.0))))
        fly = float(np.mean(p.get("flying_level", np.full(pop.n, 5.0))))
        swim = float(np.mean(p.get("swiming_level", np.full(pop.n, 5.0))))
        return 0.5 + speed / 60.0 + max(fly, swim) / 40.0

    # ------------------------------------------------------------------
    # Ciclo anual
    # ------------------------------------------------------------------
    def _step_one_year(self):
        dt = self.cfg.dt_years
        _t_year_start = time.perf_counter()
        _perf = {"climate": 0.0, "biomass": 0.0, "micro": 0.0, "density": 0.0,
                 "walk": 0.0, "reproduce": 0.0, "die": 0.0, "consume": 0.0,
                 "speciation": 0.0, "telemetry": 0.0}
        def _tic():
            return time.perf_counter()

        _t = _tic()
        # v1.5 (N5): actualizamos temperatura respecto al baseline
        if self.climate.enabled:
            delta_t = self.climate.temperature_delta(self.year)
            self.world.layers["temperature"] = (self._base_temperature
                                                 + delta_t).astype(np.float32)

        # Perturbaciones (impactos, glaciaciones)
        for event in self.perturbations.due_for(self.year):
            event.apply(self.world, self.pops, self.year, self.narrative)

        # Reclasificar biomas (ahora sí cambia — la temperatura oscila)
        if (int(self.year) % self.cfg.reclassify_biomes_every == 0
                and self.year > 0):
            self.biome_map = classify(self.world)

        _perf["climate"] += time.perf_counter() - _t; _t = _tic()
        plant_target = biomass_field(self.pops, self.world, "plantae")
        fungi_target = biomass_field(self.pops, self.world, "fungi")
        self.plant_store.regenerate(plant_target, r=1.2, dt=dt)
        self.fungi_store.regenerate(fungi_target, r=1.0, dt=dt)
        plant_b = self.plant_store.biomass
        fungi_b = self.fungi_store.biomass

        _perf["biomass"] += time.perf_counter() - _t; _t = _tic()
        K = carrying_capacity(self.world, self.biome_map, plant_b, fungi_b)
        logistic_step(self.micro, K, r=2.5, dt_years=dt)

        micro_dict = self.micro.as_dict()
        micro_dict["plant_biomass"] = plant_b
        micro_dict["fungi_biomass"] = fungi_b
        micro_dict["meat_biomass"] = meat_field(self.pops, self.world)
        micro_dict["fish_biomass"] = fish_field(self.pops, self.world)
        micro_dict["bugs_local"] = _local_mean(micro_dict["bugs"], radius=2)
        micro_dict["meat_local"] = _local_mean(micro_dict["meat_biomass"], radius=2)
        micro_dict["plant_local"] = _local_mean(plant_b, radius=2)
        micro_dict["fish_local"] = _local_mean(micro_dict["fish_biomass"], radius=2)
        micro_dict["micro_local"] = _local_mean(
            micro_dict["microbes"] * 0.5 + micro_dict["plankton"] * 0.5, radius=2)

        _perf["micro"] += time.perf_counter() - _t; _t = _tic()
        # v1.5 (N8): densidad local por especie para IFD en habitat.suitability
        for pop in self.pops:
            if pop.n > 0:
                d_local = _build_density(pop, self.world.shape).astype(np.float32)
                micro_dict[f"density_{pop.template.species_id}"] = d_local
        _perf["density"] += time.perf_counter() - _t

        new_pops = []
        reexpress_now = (int(self.year) % self.cfg.reexpress_every == 0
                          and self.year > 0)

        for pop in self.pops:
            if pop.n == 0:
                continue
            # v1.5 (N4): SOLO warning; no thinning.
            if pop.n > self.cfg.warn_pop_per_species:
                print(f"  [warn] {pop.template.scientific_name} n={pop.n} "
                      f"> warn_pop_per_species={self.cfg.warn_pop_per_species}. "
                      f"La densodependencia debería frenarla; sin cap duro.")

            evolve = getattr(pop.template, "evolve", True)

            if reexpress_now and evolve:
                pop.express(self.rng, sigma_env=self.cfg.env_sigma)

            if pop.template.kingdom == "chordata":
                _tw = _tic()
                animal_walk(pop, self.world, self.biome_map, micro_dict, self.rng)
                _perf["walk"] += time.perf_counter() - _tw

            _tr = _tic()
            mu = self.cfg.mutation_rate if evolve else 0.0
            sig_mut = self.cfg.mutation_sigma if evolve else 0.0
            result = reproduce(pop, self.world, self.biome_map, micro_dict,
                               dt_years=dt, mu=mu,
                               sigma_mut=sig_mut, rng=self.rng)
            if result is not None:
                child_genome, mother_idx = result
                cy = pop.y[mother_idx].copy()
                cx = pop.x[mother_idx].copy()
                sigma = self._seed_sigma_for(pop)
                cy, cx = disperse_seeds(cy, cx, self.world, self.rng,
                                         sigma_cells=sigma)
                # Concat directo, sin construir un SpeciesPopulation efímero.
                pop.genome.alleles = np.concatenate([pop.genome.alleles,
                                                    child_genome.alleles])
                pop.y = np.concatenate([pop.y, cy.astype(np.int32)])
                pop.x = np.concatenate([pop.x, cx.astype(np.int32)])
                pop.age_years = np.concatenate(
                    [pop.age_years, np.zeros(cy.size, dtype=np.float32)])
                pop.energy = np.concatenate(
                    [pop.energy, np.full(cy.size, 0.6, dtype=np.float32)])
                if pop.sex is not None:
                    child_sex = self.rng.integers(
                        0, 2, size=cy.size).astype(np.int8)
                    pop.sex = np.concatenate([pop.sex, child_sex])
                if evolve:
                    # Reexpresar SOLO los hijos y concatenar.
                    tmp = SpeciesPopulation(
                        template=pop.template, genome=child_genome,
                        y=cy, x=cx,
                        age_years=np.zeros(cy.size, dtype=np.float32),
                        energy=np.full(cy.size, 0.6, dtype=np.float32),
                        sex=None,
                    )
                    tmp.express(self.rng, sigma_env=self.cfg.env_sigma)
                    for k in pop.phenotype:
                        pop.phenotype[k] = np.concatenate(
                            [pop.phenotype[k], tmp.phenotype[k]])
                else:
                    # Fenotipo del hijo = fenotipo de la madre (mismo genotipo).
                    for k in pop.phenotype:
                        pop.phenotype[k] = np.concatenate(
                            [pop.phenotype[k], pop.phenotype[k][mother_idx]])
            _perf["reproduce"] += time.perf_counter() - _tr

            _td = _tic()
            density = _build_density(pop, self.world.shape)
            age_and_die(pop, self.world, self.biome_map, micro_dict,
                        dt_years=dt, rng=self.rng, density_map=density)
            _perf["die"] += time.perf_counter() - _td

            if pop.template.kingdom == "chordata" and pop.n > 0:
                _tc = _tic()
                mass_factor = np.clip(
                    (pop.phenotype["mass_g"] / 100.0) ** 0.75, 0.05, 20.0)
                for cat, key in (("bug", "bugs"), ("micro", "microbes"),
                                  ("fish", "plankton")):
                    intake = (pop.phenotype[f"diet_{cat}"] / 100.0
                              * mass_factor / CONSUMPTION_SCALE_KCAL * dt)
                    consume = np.zeros(self.world.shape, dtype=np.float32)
                    np.add.at(consume, (pop.y, pop.x), intake)
                    layer = getattr(self.micro, key)
                    setattr(self.micro, key,
                            np.clip(layer - consume, 0.0, 1.0).astype(np.float32))
                intake_veg = (pop.phenotype["diet_vegetal"] / 100.0
                              * mass_factor / CONSUMPTION_SCALE_KCAL * dt)
                self.plant_store.consume(pop.y, pop.x, intake_veg)
                _perf["consume"] += time.perf_counter() - _tc

        # Especiación periódica — k=2..4 con silhouette (N7)
        _ts = _tic()
        if int(self.year) % self.cfg.speciation_every == 0 and self.year > 0:
            for pop in list(self.pops):
                if not getattr(pop.template, "evolve", True):
                    continue
                children = try_speciate(pop, self.world, self.year, self.rng,
                                         self.speciation_state)
                for child in children:
                    child.express(self.rng, sigma_env=self.cfg.env_sigma)
                    new_pops.append(child)
                    self.phylo.register(child.template,
                                         parent=pop.template.species_id,
                                         born_at=self.year)
                    self.narrative.log(
                        self.year, "speciation", child.template.species_id,
                        f"Especiación desde {pop.template.scientific_name} → "
                        f"{child.template.scientific_name} (n={child.n}).")
                    print(f"    ✦ año {self.year:>5.0f}: especiación → "
                          f"{child.template.scientific_name}")
        self.pops.extend(new_pops)
        _perf["speciation"] += time.perf_counter() - _ts

        # Telemetría
        _tt = _tic()
        if int(self.year) % self.cfg.telemetry_every == 0:
            row = self.telemetry.snapshot(self.year, self.pops, self.biome_map,
                                          speciation_state=self.speciation_state)
            print(f"  año {self.year:>5.0f} | especies vivas: "
                  f"{row['species_alive']:>3} | población total: "
                  f"{row['total_pop']:>7}", flush=True)
            if (len(self.telemetry.rows) % self.cfg.dump_every_snapshots) == 0:
                self.telemetry.dump()
                self.telemetry.dump_species_cards(self.pops)
                self.phylo.dump_newick(Path(self.cfg.out_dir) / "phylo.nwk")
                self.narrative.dump()

        _perf["telemetry"] += time.perf_counter() - _tt

        # Registro de rendimiento
        year_dt = time.perf_counter() - _t_year_start
        perf_row = {"year": int(self.year), "wall_s": year_dt,
                    "n_pops": sum(1 for p in self.pops if p.n > 0),
                    "total_pop": sum(int(p.n) for p in self.pops), **_perf}
        self.perf_rows.append(perf_row)
        if (self.cfg.perf_log_every > 0
                and int(self.year) % self.cfg.perf_log_every == 0
                and self.year > 0):
            top = sorted(((v, k) for k, v in _perf.items()), reverse=True)[:3]
            top_str = ", ".join(f"{k}={v*1000:.0f}ms" for v, k in top)
            print(f"  [perf] año {int(self.year):>5} | {year_dt*1000:.0f}ms | {top_str}",
                  flush=True)

        # Snapshot PNG (opcional; matplotlib no bloquea si no está instalado)
        if self.cfg.snapshot_every > 0 and int(self.year) % self.cfg.snapshot_every == 0:
            try:
                from .stats.snapshot import save_snapshot_png
                save_snapshot_png(self, out_dir=self.cfg.out_dir)
            except Exception as e:                          # pragma: no cover
                print(f"  [snapshot] omitido: {e}")

        self.year += dt

    def run(self):
        t0 = time.perf_counter()
        self.seed_phase1()
        while self.year < self.cfg.phase1_years:
            self._step_one_year()
        self.seed_phase2()
        while self.year < self.cfg.total_years:
            self._step_one_year()
        self.telemetry.dump()
        self.telemetry.dump_species_cards(self.pops)
        self.phylo.dump_newick(Path(self.cfg.out_dir) / "phylo.nwk")
        self.narrative.dump()
        elapsed = time.perf_counter() - t0
        yrs = max(self.year, 1e-6)
        print(f"\n✓ simulación completa: {self.year:.0f} años "
              f"({self.year * HOURS_PER_YEAR:.0f} horas del mundo)")
        print(f"  tiempo total: {elapsed:.1f}s "
              f"({elapsed / yrs * 1000:.1f} ms/año)")
        print(f"  outputs → {self.cfg.out_dir}/")
        # Volcado de rendimiento + gráficos finales
        try:
            from .stats.plots import render_all, dump_perf_csv
            dump_perf_csv(self.cfg.out_dir, self.perf_rows)
            render_all(self.cfg.out_dir, self.perf_rows)
        except Exception as e:                                # pragma: no cover
            print(f"  [plots] omitidos: {e}")

    # ------------------------------------------------------------------
    # Serialización (v1.5, N10)
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        from .io.snapshot import save_simulation
        save_simulation(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Simulation":
        from .io.snapshot import load_simulation
        return load_simulation(path)
