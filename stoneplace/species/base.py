"""Estructura de datos de una especie viva en Stoneplace.

Cumple exactamente los campos de metadata / phenotype / abilities / genetics
descritos en la plantilla del usuario, tanto para Chordata como para
Fungi/Plantae. Los individuos se almacenan vectorizados (SoA) por especie.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
from ..genetics.genome import Genome
from ..genetics.gp_map import GPMap


Kingdom = Literal["chordata", "plantae", "fungi"]


@dataclass
class SpeciesTemplate:
    """Plantilla estable de una especie (no cambia con el tick)."""
    species_id: int
    scientific_name: str
    taxonomy: list[str]
    kingdom: Kingdom
    parent: Optional[int]
    born_at: float
    n_loci: int
    n_chromosomes: int
    gp_map: GPMap
    # Óptimos de nicho iniciales (se usan como valores fenotípicos objetivo)
    hints: dict = field(default_factory=dict)
    # Estado histórico
    children: list[int] = field(default_factory=list)
    extinct_at: Optional[float] = None
    status: str = "alive"
    historic_population: int = 0
    max_population: int = 0
    main_biomes: list[str] = field(default_factory=list)


@dataclass
class SpeciesPopulation:
    """Individuos vivos de una especie. Struct-of-Arrays vectorizado."""
    template: SpeciesTemplate
    genome: Genome
    # Posiciones (y, x) en el raster mundial
    y: np.ndarray
    x: np.ndarray
    age_years: np.ndarray
    energy: np.ndarray            # reserva energética normalizada 0..1
    sex: Optional[np.ndarray] = None   # 0=hembra 1=macho (solo chordata sexuales)
    # Cache del fenotipo expresado — se recalcula tras nacer / mutar
    phenotype: dict[str, np.ndarray] = field(default_factory=dict)
    # Distribución por bioma (histograma) — se actualiza en stats
    tile_distribution: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int64))

    @property
    def n(self) -> int:
        return self.y.shape[0]

    def express(self, rng, sigma_env: float = 8.0):
        self.phenotype = self.template.gp_map.express(self.genome, rng, sigma_env)

    def kill_mask(self, alive: np.ndarray):
        """Filtra individuos según una máscara booleana `alive`."""
        self.genome = self.genome.slice(alive)
        self.y = self.y[alive]
        self.x = self.x[alive]
        self.age_years = self.age_years[alive]
        self.energy = self.energy[alive]
        if self.sex is not None:
            self.sex = self.sex[alive]
        for k in list(self.phenotype.keys()):
            self.phenotype[k] = self.phenotype[k][alive]
