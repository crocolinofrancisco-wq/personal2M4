"""Simulador Stoneplace — ciclo anual con siembra en dos capas.

Fase 1: se siembran plantas y hongos por todo el planeta. Corren solos
        hasta que colonizan al menos N celdas viables.
Fase 2: se sueltan los animales en un punto del mapa y el ecosistema
        completo evoluciona a la par.

Reglas de tiempo del mundo (constantes en `stoneplace/__init__.py`):
    1 año = 400 días = 10 000 horas (día = 25 h).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from . import HOURS_PER_YEAR
from .world.loader import World, load_world
from .world.biomes import classify, BIOME_NAMES
from .world.synthetic import make_synthetic_world
from .species.base import SpeciesPopulation, SpeciesTemplate
from .species.registry import all_prototype_species
from .genetics.genome import new_founder_population
from .ecology.microfauna import (MicrofaunaField, carrying_capacity,
                                  logistic_step)
from .ecology.producers import biomass_field, meat_field, fish_field
from .ecology.habitat import suitability
from .ecology.population import age_and_die, reproduce, _build_density
from .ecology.dispersal import animal_walk, disperse_seeds
from .ecology.speciation import try_speciate
from .stats.telemetry import Telemetry
from .utils import rng_from_seed


@dataclass
class SimConfig:
    # Ruta al raster DDG real. Si ambos son None se genera un mundo
    # sintético (útil para tests sin data binaria).
    world_bin: str | None = None
    world_json: str | None = None
    synthetic_size: tuple[int, int] = (128, 64)   # (width, height)
    out_dir: str = "outputs"
    seed: int = 42
    dt_years: float = 1.0            # 1 año por tick = 10 000 h
    mutation_rate: float = 5e-4      # por locus y por meiosis
    mutation_sigma: float = 2.5      # DFE exponencial, sigma en unidades de locus
    env_sigma: float = 8.0           # ruido ambiental de la expresión G→P
    speciation_every: int = 25       # años entre chequeos de especiación
    telemetry_every: int = 5
    phase1_years: int = 200          # años sólo con productores
    total_years: int = 800
    founder_size_plant: int = 400
    founder_size_fungi: int = 300
    founder_size_animal: int = 600
    # Tope duro por especie: si una supera este número, se descarta
    # aleatoriamente el excedente. Evita que mundos grandes revienten
    # la memoria del runner cuando una especie explota demográficamente.
    max_pop_per_species: int = 300_000
    # Cada cuántos snapshots de telemetría volcar a disco (checkpoint).
    dump_every_snapshots: int = 20


class Simulation:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = rng_from_seed(cfg.seed)
        if cfg.world_bin and cfg.world_json:
            self.world: World = load_world(cfg.world_bin, cfg.world_json)
        else:
            w, h = cfg.synthetic_size
            self.world = make_synthetic_world(width=w, height=h, seed=cfg.seed)
        self.biome_map = classify(self.world)
        self.micro = MicrofaunaField.zeros(self.world.shape)
        self.pops: list[SpeciesPopulation] = []
        self.templates = all_prototype_species(self.rng)
        self.year: float = 0.0
        self.phase: int = 1
        self.telemetry = Telemetry(cfg.out_dir)

    # ------------------------------------------------------------------
    # Siembra
    # ------------------------------------------------------------------
    def _sample_positions_for(self, tpl: SpeciesTemplate, n: int,
                              seed_point=None) -> tuple[np.ndarray, np.ndarray]:
        """Elige n coordenadas viables para la especie según sus hints."""
        aquatic = tpl.hints.get("aquatic", False)
        elev = self.world.layers["elevation"]
        temp = self.world.layers["temperature"]
        salt = self.world.layers.get("water_salinity", np.zeros_like(elev))
        rains = self.world.layers.get("rains", np.zeros_like(elev))
        river = self.world.layers.get("riverFlow", np.zeros_like(elev))
        aw = self.world.layers.get("available_water", rains)
        if aquatic:
            # Agua dulce: mar poco salado + ríos + suelos anegados/costeros
            sal_tol = float(tpl.hints.get("salt_tolerance", 3))
            mask = ((elev <= 0) & (salt <= sal_tol + 2)) | (river > 0.05) | \
                   ((elev > 0) & (elev < 400) & (aw > 250))
        else:
            mask = elev > 0
        # Filtrar por temperatura razonable de la especie
        tmin = tpl.hints.get("min_temp", -20)
        tmax = tpl.hints.get("max_temp", 45)
        mask &= (temp >= tmin - 2) & (temp <= tmax + 2)

        idxs = np.argwhere(mask)
        if idxs.size == 0:
            return np.array([], np.int32), np.array([], np.int32)

        if seed_point is not None:
            # Fase 2: los animales caen cerca de un punto concreto pero
            # con dispersión amplia (sigma = 80 celdas) para probar suerte
            # en varias regiones del mismo continente en vez de amontonarse.
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
        print(f"  · sembrada {tpl.scientific_name}: {y.size} individuos")

    def seed_phase1(self):
        """Plantas + hongos por todo el planeta."""
        print("[Fase 1] Siembra de productores primarios y descomponedores")
        for tpl in self.templates:
            if tpl.kingdom in ("plantae",):
                self._seed_species(tpl, self.cfg.founder_size_plant)
            elif tpl.kingdom == "fungi":
                self._seed_species(tpl, self.cfg.founder_size_fungi)

    def seed_phase2(self, seed_point=None):
        """Animales alrededor de un punto (por defecto, un continente cálido)."""
        print("[Fase 2] Siembra de fauna")
        if seed_point is None:
            # Buscar un punto de tierra templada rica en plantas
            plant_b = biomass_field(self.pops, self.world, "plantae")
            score = np.where(self.world.layers["elevation"] > 0, plant_b, 0)
            idx = np.unravel_index(np.argmax(score), score.shape)
            seed_point = idx
        print(f"  Punto de siembra animal: y={seed_point[0]}, x={seed_point[1]}")
        for tpl in self.templates:
            if tpl.kingdom == "chordata":
                sp = seed_point
                if tpl.hints.get("aquatic", False):
                    # Triops: cerca de un lago/costa
                    sp = None
                self._seed_species(tpl, self.cfg.founder_size_animal,
                                    seed_point=sp)

    # ------------------------------------------------------------------
    # Ciclo anual
    # ------------------------------------------------------------------
    def _step_one_year(self):
        dt = self.cfg.dt_years

        # Campos derivados
        plant_b = biomass_field(self.pops, self.world, "plantae")
        fungi_b = biomass_field(self.pops, self.world, "fungi")
        K = carrying_capacity(self.world, self.biome_map, plant_b, fungi_b)
        logistic_step(self.micro, K, r=0.4, dt_years=dt)

        micro_dict = self.micro.as_dict()
        micro_dict["plant_biomass"] = plant_b
        micro_dict["fungi_biomass"] = fungi_b
        micro_dict["meat_biomass"] = meat_field(self.pops, self.world)
        micro_dict["fish_biomass"] = fish_field(self.pops, self.world)

        new_pops = []
        for pop in self.pops:
            if pop.n == 0:
                continue
            # Tope global por especie: si excede el máximo, thinning aleatorio.
            if pop.n > self.cfg.max_pop_per_species:
                keep_p = self.cfg.max_pop_per_species / pop.n
                keep = self.rng.random(pop.n) < keep_p
                pop.kill_mask(keep)
                if pop.n == 0:
                    continue

            # 1. Movimiento (animales) / dispersión pasiva (plantas)
            if pop.template.kingdom == "chordata":
                animal_walk(pop, self.world, self.biome_map, micro_dict, self.rng)

            # 2. Reproducción
            result = reproduce(pop, self.world, self.biome_map, micro_dict,
                               dt_years=dt, mu=self.cfg.mutation_rate,
                               sigma_mut=self.cfg.mutation_sigma, rng=self.rng)
            if result is not None:
                child_genome, mother_idx = result
                cy = pop.y[mother_idx].copy()
                cx = pop.x[mother_idx].copy()
                if pop.template.kingdom in ("plantae", "fungi"):
                    cy, cx = disperse_seeds(cy, cx, self.world, self.rng,
                                             sigma_cells=3.0)
                else:
                    cy, cx = disperse_seeds(cy, cx, self.world, self.rng,
                                             sigma_cells=1.0)
                child_pop = SpeciesPopulation(
                    template=pop.template,
                    genome=child_genome,
                    y=cy.astype(np.int32), x=cx.astype(np.int32),
                    age_years=np.zeros(cy.size, dtype=np.float32),
                    energy=np.full(cy.size, 0.6, dtype=np.float32),
                    sex=(self.rng.integers(0, 2, size=cy.size).astype(np.int8)
                         if pop.template.kingdom == "chordata" else None),
                )
                child_pop.express(self.rng, sigma_env=self.cfg.env_sigma)
                # Fusionar en la misma especie
                pop.genome.alleles = np.concatenate([pop.genome.alleles,
                                                    child_pop.genome.alleles])
                pop.y = np.concatenate([pop.y, child_pop.y])
                pop.x = np.concatenate([pop.x, child_pop.x])
                pop.age_years = np.concatenate([pop.age_years, child_pop.age_years])
                pop.energy = np.concatenate([pop.energy, child_pop.energy])
                if pop.sex is not None:
                    pop.sex = np.concatenate([pop.sex, child_pop.sex])
                for k in pop.phenotype:
                    pop.phenotype[k] = np.concatenate([pop.phenotype[k],
                                                       child_pop.phenotype[k]])

            # 3. Mortalidad
            density = _build_density(pop, self.world.shape)
            age_and_die(pop, self.world, self.biome_map, micro_dict,
                        dt_years=dt, rng=self.rng, density_map=density)

            # 4. Consumo de microfauna donde hay animales activos.
            # Intake per cápita proporcional a masa^0.75 (Kleiber) y muy
            # rebajado — con 50k animales todos en la misma celda antes se
            # arrasaba la biomasa local en un solo tick.
            if pop.template.kingdom == "chordata" and pop.n > 0:
                mass_factor = np.clip(
                    (pop.phenotype["mass_g"] / 100.0) ** 0.75, 0.05, 20.0)
                for cat, key in (("bug", "bugs"), ("micro", "microbes"),
                                  ("fish", "plankton")):
                    intake = (pop.phenotype[f"diet_{cat}"] / 100.0
                              * mass_factor / 50000.0 * dt)
                    consume = np.zeros(self.world.shape, dtype=np.float32)
                    np.add.at(consume, (pop.y, pop.x), intake)
                    layer = getattr(self.micro, key)
                    setattr(self.micro, key,
                            np.clip(layer - consume, 0.0, 1.0).astype(np.float32))

        # 5. Especiación periódica
        if int(self.year) % self.cfg.speciation_every == 0 and self.year > 0:
            for pop in list(self.pops):
                child = try_speciate(pop, self.world, self.year, self.rng)
                if child is not None:
                    child.express(self.rng, sigma_env=self.cfg.env_sigma)
                    new_pops.append(child)
                    print(f"    ✦ año {self.year:>5.0f}: especiación → "
                          f"{child.template.scientific_name}")
        self.pops.extend(new_pops)

        # 6. Telemetría
        if int(self.year) % self.cfg.telemetry_every == 0:
            row = self.telemetry.snapshot(self.year, self.pops, self.biome_map)
            print(f"  año {self.year:>5.0f} | especies vivas: "
                  f"{row['species_alive']:>3} | población total: "
                  f"{row['total_pop']:>7}", flush=True)
            # Checkpoint: volcar a disco cada N snapshots para no perder
            # progreso si el proceso muere (timeout, OOM…).
            if (len(self.telemetry.rows) % self.cfg.dump_every_snapshots) == 0:
                self.telemetry.dump()
                self.telemetry.dump_species_cards(self.pops)

        self.year += dt

    def run(self):
        self.seed_phase1()
        while self.year < self.cfg.phase1_years:
            self._step_one_year()
        self.seed_phase2()
        while self.year < self.cfg.total_years:
            self._step_one_year()
        self.telemetry.dump()
        self.telemetry.dump_species_cards(self.pops)
        print(f"\n✓ simulación completa: {self.year:.0f} años "
              f"({self.year * HOURS_PER_YEAR:.0f} horas del mundo)")
        print(f"  outputs → {self.cfg.out_dir}/")
