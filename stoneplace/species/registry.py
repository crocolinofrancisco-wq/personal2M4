"""Registro de las especies del prototipo.

Cada función devuelve un `SpeciesTemplate` con los hints biológicos
(óptimos de temperatura, agua, dieta base…) que empujan la G→P hacia la
biología real del organismo. Los hints son solo el punto de partida — la
selección natural mueve los loci de ahí en adelante.
"""
from __future__ import annotations
from typing import Callable
import numpy as np

from .base import SpeciesTemplate
from ..genetics.gp_map import chordata_gpmap, flora_gpmap


_NEXT_ID = 1


def _next_id() -> int:
    global _NEXT_ID
    v = _NEXT_ID
    _NEXT_ID += 1
    return v


# ---------------------------------------------------------------------------
# CHORDATA
# ---------------------------------------------------------------------------

def yi_qi(rng, born_at: float = 0.0, n_loci: int = 64) -> SpeciesTemplate:
    """Yi qi — pequeño escansorioptérigido membraniplano del Jurásico.

    Hints biológicos reales (Xu et al. 2015): ~380 g, ~60 cm, planador
    arborícola insectívoro-carnívoro.
    """
    hints = {
        "mass_g_opt": 380, "size_mm_opt": 600,
        "temp_opt": 24.0, "temp_sigma": 8.0,
        "min_temp": 5, "max_temp": 38,
        "salt_tolerance": 2,
        "diet_bias": {"bug": 55, "meat": 25, "vegetal": 5, "fish": 10, "micro": 5},
        "abilities_bias": {"flying_level": 55, "climbing_level": 75,
                            "sight_level": 70, "smell_level": 55},
        "biomes_ok": {"bosque templado", "selva tropical",
                      "bosque boreal", "sabana"},
        "aquatic": False,
        # Yi qi jurásico: hembra pone ~6 huevos, superviven 2-4; le damos 5.
        "breeding_lapse": 1.0,
        "baby_quantity": 5,
        "life_expectancy": 12,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Yi qi",
        taxonomy=["Animalia", "Chordata", "Reptilia", "Dinosauria",
                  "Theropoda", "Scansoriopterygidae"],
        kingdom="chordata",
        parent=None,
        born_at=born_at,
        n_loci=n_loci,
        n_chromosomes=8,
        gp_map=chordata_gpmap(n_loci, rng, hints=hints),
        hints=hints,
    )


def triops_longicaudatus(rng, born_at: float = 0.0, n_loci: int = 48) -> SpeciesTemplate:
    """Triops longicaudatus — braquiópodo notostráceo de pozas efímeras.

    Es un artrópodo, NO un vertebrado, pero el usuario lo pidió como
    "vertebrado acuático principal": se implementa con la plantilla
    Chordata (misma estructura de rasgos) marcando modo acuático.
    """
    hints = {
        "mass_g_opt": 1.5, "size_mm_opt": 40,
        "temp_opt": 26.0, "temp_sigma": 6.0,
        "min_temp": 12, "max_temp": 40,
        "salt_tolerance": 4,
        "diet_bias": {"bug": 15, "meat": 5, "vegetal": 10, "fish": 5, "micro": 65},
        "abilities_bias": {"swiming_level": 80, "dig_level": 55,
                            "breath_water_level": 90, "sight_level": 40},
        "biomes_ok": {"agua dulce", "costa", "sabana", "desierto"},
        "aquatic": True,
        # r-estratega extremo: ciclo corto y puestas grandes que compensan
        # la mortalidad casi total anual de pozas efímeras.
        "breeding_lapse": 0.05,
        "baby_quantity": 200,
        "life_expectancy": 0.35,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Triops longicaudatus",
        taxonomy=["Animalia", "Arthropoda", "Branchiopoda",
                  "Notostraca", "Triopsidae"],
        kingdom="chordata",   # se rige por la plantilla Chordata del usuario
        parent=None,
        born_at=born_at,
        n_loci=n_loci,
        n_chromosomes=6,
        gp_map=chordata_gpmap(n_loci, rng, hints=hints),
        hints=hints,
    )


# ---------------------------------------------------------------------------
# PLANTAE
# ---------------------------------------------------------------------------

def helianthus_annuus(rng, born_at: float = 0.0, n_loci: int = 48) -> SpeciesTemplate:
    """Girasol — anual heliófilo, biomas abiertos, altos requerimientos hídricos."""
    hints = {
        "mass_g_opt": 2500, "size_mm_opt": 2000,
        "temp_opt": 22.0, "temp_sigma": 7.0,
        "min_temp": 6, "max_temp": 38,
        "min_ph": 6.0, "max_ph": 7.8,
        "salt_tolerance": 2,
        "biomes_ok": {"pradera templada", "sabana", "bosque templado"},
        "life_expectancy": 0.5, "breeding_lapse": 0.5, "baby_quantity": 800,
        "photosynthesis": 95, "aquatic": False,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Helianthus annuus",
        taxonomy=["Plantae", "Tracheophyta", "Magnoliopsida",
                  "Asterales", "Asteraceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=17,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints),
        hints=hints,
    )


def poa_annua(rng, born_at: float = 0.0, n_loci: int = 48) -> SpeciesTemplate:
    """Poa annua — césped común, colonizador universal templado."""
    hints = {
        "mass_g_opt": 5, "size_mm_opt": 150,
        "temp_opt": 15.0, "temp_sigma": 10.0,
        "min_temp": -5, "max_temp": 32,
        "min_ph": 5.0, "max_ph": 8.0,
        "salt_tolerance": 4,
        "biomes_ok": {"pradera templada", "tundra", "bosque templado",
                      "bosque boreal", "sabana", "costa"},
        "life_expectancy": 1.0, "breeding_lapse": 0.2, "baby_quantity": 2000,
        "photosynthesis": 90, "overcrowd_tolerance": 80, "aquatic": False,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Poa annua",
        taxonomy=["Plantae", "Tracheophyta", "Liliopsida",
                  "Poales", "Poaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=7,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints),
        hints=hints,
    )


def phyllostachys_edulis(rng, born_at: float = 0.0, n_loci: int = 56) -> SpeciesTemplate:
    """Bambú moso — gramínea gigante, expansión clonal."""
    hints = {
        "mass_g_opt": 40000, "size_mm_opt": 15000,
        "temp_opt": 20.0, "temp_sigma": 6.0,
        "min_temp": -5, "max_temp": 35,
        "min_ph": 5.0, "max_ph": 7.0,
        "salt_tolerance": 1,
        "biomes_ok": {"bosque templado", "selva tropical", "sabana"},
        "life_expectancy": 60, "breeding_lapse": 20.0, "baby_quantity": 400,
        "photosynthesis": 92, "allelopathy": 65, "aquatic": False,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Phyllostachys edulis",
        taxonomy=["Plantae", "Tracheophyta", "Liliopsida",
                  "Poales", "Poaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=12,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints),
        hints=hints,
    )


def orchis_stoneplace(rng, born_at: float = 0.0, n_loci: int = 48) -> SpeciesTemplate:
    """Flor a elección: orquídea genérica ombrófila."""
    hints = {
        "mass_g_opt": 40, "size_mm_opt": 400,
        "temp_opt": 21.0, "temp_sigma": 5.0,
        "min_temp": 8, "max_temp": 32,
        "min_ph": 5.5, "max_ph": 7.0,
        "salt_tolerance": 1,
        "biomes_ok": {"selva tropical", "bosque templado"},
        "life_expectancy": 8, "breeding_lapse": 1.0, "baby_quantity": 5000,
        "photosynthesis": 78, "attractiveness": 85, "aquatic": False,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Orchis stoneplacensis",
        taxonomy=["Plantae", "Tracheophyta", "Liliopsida",
                  "Asparagales", "Orchidaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=10,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints),
        hints=hints,
    )


# ---------------------------------------------------------------------------
# FUNGI
# ---------------------------------------------------------------------------

def pleurotus_ostreatus(rng, born_at: float = 0.0, n_loci: int = 48) -> SpeciesTemplate:
    """Hongo a elección: Pleurotus ostreatus (gírgola) — saprófito lignícola."""
    hints = {
        "mass_g_opt": 300, "size_mm_opt": 150,
        "temp_opt": 18.0, "temp_sigma": 8.0,
        "min_temp": 2, "max_temp": 32,
        "min_ph": 4.5, "max_ph": 7.5,
        "salt_tolerance": 0.5,
        "biomes_ok": {"bosque templado", "bosque boreal", "selva tropical"},
        "life_expectancy": 3, "breeding_lapse": 0.2, "baby_quantity": 100000,
        "saprotrophy": 90, "aquatic": False,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Pleurotus ostreatus",
        taxonomy=["Fungi", "Basidiomycota", "Agaricomycetes",
                  "Agaricales", "Pleurotaceae"],
        kingdom="fungi",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=11,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=True, hints=hints),
        hints=hints,
    )


def all_prototype_species(rng) -> list[Callable[..., SpeciesTemplate]]:
    """Devuelve las 7 especies del prototipo, en orden de siembra sugerido."""
    return [
        # Fase 1: productores primarios y descomponedores
        poa_annua(rng),
        helianthus_annuus(rng),
        phyllostachys_edulis(rng),
        orchis_stoneplace(rng),
        pleurotus_ostreatus(rng),
        # Fase 2: animales
        yi_qi(rng),
        triops_longicaudatus(rng),
    ]
