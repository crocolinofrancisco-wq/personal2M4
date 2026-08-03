"""Registro de las especies del prototipo.

Cada función devuelve un `SpeciesTemplate` con los hints biológicos
(óptimos de temperatura, agua, dieta base…) que empujan la G→P hacia la
biología real del organismo. Los hints son solo el punto de partida — la
selección natural mueve los loci de ahí en adelante.

P10 fix: `body_plan` separado de `kingdom` para no falsear a Triops
como cordado. La taxonomía queda limpia; el motor sigue usando la misma
plantilla mecánica.
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
def yi_qi(rng, born_at: float = 0.0, n_loci: int = 64, map_rng=None) -> SpeciesTemplate:
    hints = {
        "mass_g_opt": 380, "size_mm_opt": 600,
        "temp_opt": 24.0, "temp_sigma": 8.0,
        "min_temp": 5, "max_temp": 38,
        "salt_tolerance": 2,
        # v1.5.7 — rebalance realista basado en biología del clado
        # Scansoriopterygidae: pequeño teropodo arborícola con dentición
        # de insectívoro (Zhang et al. 2008, Nature). Se mueve por el
        # dosel cazando insectos y ocasionalmente comiendo frutos. Fish
        # bajo (arborícola, no acuático). Meat oportunista (5-10 %,
        # pequeños vertebrados si aparecen). Sin veg dominante.
        "diet_bias": {"bug": 70, "meat": 8, "vegetal": 12, "fish": 3, "micro": 7},
        # Yi qi: sensorial (vista, olfato), motor arborícola (climbing, flying).
        # NADA de bioluminiscencia / toxicidad / electric / fire — deben
        # desbloquearse por evolución si acaso, no venir de fábrica.
        "abilities_bias": {
            "flying_level": 55, "climbing_level": 75,
            "sight_level": 70, "smell_level": 55, "hearing_level": 50,
        },
        "biomes_ok": {"bosque templado", "selva tropical",
                      "bosque boreal", "sabana"},
        "aquatic": False,
        "body_plan": "reptilia",
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
        gp_map=chordata_gpmap(n_loci, rng, hints=hints, map_rng=map_rng),
        hints=hints,
    )


def triops_longicaudatus(rng, born_at: float = 0.0, n_loci: int = 48, map_rng=None) -> SpeciesTemplate:
    """Triops — artrópodo notostráceo. Usa la mecánica Chordata (mismos
    rasgos) pero el body_plan queda etiquetado como arthropoda."""
    hints = {
        "mass_g_opt": 1.5, "size_mm_opt": 40,
        "temp_opt": 26.0, "temp_sigma": 6.0,
        "min_temp": 12, "max_temp": 40,
        "salt_tolerance": 4,
        "diet_bias": {"bug": 15, "meat": 5, "vegetal": 10, "fish": 5, "micro": 65},
        "abilities_bias": {
            "swiming_level": 80, "dig_level": 55,
            "breath_water_level": 90, "breath_air_level": 15,
            "sight_level": 40,
        },
        "biomes_ok": {"agua dulce", "costa", "sabana", "desierto"},
        "aquatic": True,
        "body_plan": "arthropoda",
        "breeding_lapse": 0.05,
        "baby_quantity": 200,
        "life_expectancy": 0.35,
        # r-estratega extremo: aguanta muchísima densidad local
        "overcrowd_tolerance": 90,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Triops longicaudatus",
        taxonomy=["Animalia", "Arthropoda", "Branchiopoda",
                  "Notostraca", "Triopsidae"],
        kingdom="chordata",   # plantilla mecánica; body_plan lleva la taxonomía real
        parent=None,
        born_at=born_at,
        n_loci=n_loci,
        n_chromosomes=6,
        gp_map=chordata_gpmap(n_loci, rng, hints=hints, map_rng=map_rng),
        hints=hints,
    )


# ---------------------------------------------------------------------------
# PLANTAE
# ---------------------------------------------------------------------------
def helianthus_annuus(rng, born_at: float = 0.0, n_loci: int = 48, map_rng=None) -> SpeciesTemplate:
    hints = {
        "mass_g_opt": 2500, "size_mm_opt": 2000,
        "temp_opt": 22.0, "temp_sigma": 7.0,
        "min_temp": 6, "max_temp": 38,
        "min_ph": 6.0, "max_ph": 7.8,
        "salt_tolerance": 2,
        "biomes_ok": {"pradera templada", "sabana", "bosque templado"},
        "life_expectancy": 0.5, "breeding_lapse": 0.5, "baby_quantity": 800,
        "photosynthesis": 95, "aquatic": False, "body_plan": "angiosperm",
        "seed_resistance_level": 45,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Helianthus annuus",
        taxonomy=["Plantae", "Tracheophyta", "Magnoliopsida",
                  "Asterales", "Asteraceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=17,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints, map_rng=map_rng),
        hints=hints,
    )


def poa_annua(rng, born_at: float = 0.0, n_loci: int = 48, map_rng=None) -> SpeciesTemplate:
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
        "body_plan": "grass",
        "seed_resistance_level": 65,   # semillas viajan lejos
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Poa annua",
        taxonomy=["Plantae", "Tracheophyta", "Liliopsida",
                  "Poales", "Poaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=7,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints, map_rng=map_rng),
        hints=hints,
    )


def phyllostachys_edulis(rng, born_at: float = 0.0, n_loci: int = 56, map_rng=None) -> SpeciesTemplate:
    hints = {
        # Bambú: la masa por INDIVIDUO en el simulador la bajamos ~4× para
        # que un solo bosquecillo no ocupe la mitad del ecosistema (P: bug
        # de dominancia de bambú). Se compensa con más individuos.
        "mass_g_opt": 10000, "size_mm_opt": 15000,
        "temp_opt": 20.0, "temp_sigma": 6.0,
        "min_temp": -5, "max_temp": 35,
        "min_ph": 5.0, "max_ph": 7.0,
        "salt_tolerance": 1,
        "biomes_ok": {"bosque templado", "selva tropical", "sabana"},
        "life_expectancy": 60, "breeding_lapse": 20.0, "baby_quantity": 400,
        "photosynthesis": 92, "allelopathy": 65, "aquatic": False,
        "body_plan": "grass",
        "seed_resistance_level": 15,   # rizomas locales, casi no viajan
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Phyllostachys edulis",
        taxonomy=["Plantae", "Tracheophyta", "Liliopsida",
                  "Poales", "Poaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=12,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints, map_rng=map_rng),
        hints=hints,
    )


def orchis_stoneplace(rng, born_at: float = 0.0, n_loci: int = 48, map_rng=None) -> SpeciesTemplate:
    hints = {
        "mass_g_opt": 40, "size_mm_opt": 400,
        "temp_opt": 21.0, "temp_sigma": 5.0,
        "min_temp": 8, "max_temp": 32,
        "min_ph": 5.5, "max_ph": 7.0,
        "salt_tolerance": 1,
        "biomes_ok": {"selva tropical", "bosque templado"},
        "life_expectancy": 8, "breeding_lapse": 1.0, "baby_quantity": 5000,
        "photosynthesis": 78, "attractiveness": 85, "aquatic": False,
        "body_plan": "angiosperm",
        "seed_resistance_level": 85,   # esporas orquídea: km de dispersión
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Orchis stoneplacensis",
        taxonomy=["Plantae", "Tracheophyta", "Liliopsida",
                  "Asparagales", "Orchidaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=10,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints, map_rng=map_rng),
        hints=hints,
    )


# ---------------------------------------------------------------------------
# FUNGI
# ---------------------------------------------------------------------------
def pleurotus_ostreatus(rng, born_at: float = 0.0, n_loci: int = 48, map_rng=None) -> SpeciesTemplate:
    hints = {
        "mass_g_opt": 300, "size_mm_opt": 150,
        "temp_opt": 18.0, "temp_sigma": 8.0,
        "min_temp": 2, "max_temp": 32,
        "min_ph": 4.5, "max_ph": 7.5,
        "salt_tolerance": 0.5,
        "biomes_ok": {"bosque templado", "bosque boreal", "selva tropical"},
        "life_expectancy": 3, "breeding_lapse": 0.2, "baby_quantity": 100000,
        "saprotrophy": 90, "aquatic": False, "body_plan": "fungi",
        "seed_resistance_level": 90,   # esporas: dispersión enorme
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Pleurotus ostreatus",
        taxonomy=["Fungi", "Basidiomycota", "Agaricomycetes",
                  "Agaricales", "Pleurotaceae"],
        kingdom="fungi",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=11,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=True, hints=hints, map_rng=map_rng),
        hints=hints,
    )


def quercus_robur(rng, born_at: float = 0.0, n_loci: int = 56, map_rng=None) -> SpeciesTemplate:
    """v1.5.7 — Roble común. Árbol de biomasa enorme en bosques templados
    y hospedador de muchísima biomasa de insectos (>500 especies asociadas
    en biología real). Reemplaza al nicho de bosque de biomasa alta que
    dejó vacante el bambú retirado en v1.5.1.
    """
    hints = {
        "mass_g_opt": 60000, "size_mm_opt": 25000,
        "temp_opt": 12.0, "temp_sigma": 8.0,
        "min_temp": -10, "max_temp": 32,
        "min_ph": 4.5, "max_ph": 7.5,
        "salt_tolerance": 1,
        "biomes_ok": {"bosque templado", "bosque boreal", "pradera templada"},
        "life_expectancy": 400, "breeding_lapse": 5.0, "baby_quantity": 3000,
        "photosynthesis": 88, "aquatic": False, "body_plan": "angiosperm_tree",
        "seed_resistance_level": 30,
        "attractiveness": 30,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Quercus robur",
        taxonomy=["Plantae", "Tracheophyta", "Magnoliopsida",
                  "Fagales", "Fagaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=12,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints, map_rng=map_rng),
        hints=hints,
    )


def sphagnum_sp(rng, born_at: float = 0.0, n_loci: int = 40, map_rng=None) -> SpeciesTemplate:
    """v1.5.7 — Musgo Sphagnum. Cubre inmensas extensiones de biomas fríos
    y húmedos (tundra, turberas, bosque boreal, costa). Provee biomasa
    vegetal donde nada más crece → sostiene insectos en climas fríos.
    """
    hints = {
        "mass_g_opt": 8, "size_mm_opt": 60,
        "temp_opt": 5.0, "temp_sigma": 8.0,
        "min_temp": -20, "max_temp": 22,
        "min_ph": 3.5, "max_ph": 6.0,
        "salt_tolerance": 1,
        "biomes_ok": {"tundra", "bosque boreal", "costa", "pradera templada"},
        "life_expectancy": 25, "breeding_lapse": 0.5, "baby_quantity": 8000,
        "photosynthesis": 75, "aquatic": False, "body_plan": "bryophyte",
        "seed_resistance_level": 92,          # esporas extremadamente ligeras
        "water_storage_level": 85,
        "overcrowd_tolerance": 90,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Sphagnum sp.",
        taxonomy=["Plantae", "Bryophyta", "Sphagnopsida",
                  "Sphagnales", "Sphagnaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=19,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints, map_rng=map_rng),
        hints=hints,
    )


def opuntia_ficus_indica(rng, born_at: float = 0.0, n_loci: int = 44, map_rng=None) -> SpeciesTemplate:
    """v1.5.7 — Nopal. Cactus columnar que ocupa el nicho desértico —
    biomas donde plantas normales no sobreviven. Su CAM y almacenamiento
    hídrico permiten sostener bugs incluso en desierto.
    """
    hints = {
        "mass_g_opt": 3000, "size_mm_opt": 2500,
        "temp_opt": 28.0, "temp_sigma": 10.0,
        "min_temp": 2, "max_temp": 48,
        "min_ph": 6.0, "max_ph": 8.5,
        "salt_tolerance": 6,
        "biomes_ok": {"desierto", "sabana", "pradera templada"},
        "life_expectancy": 80, "breeding_lapse": 2.0, "baby_quantity": 500,
        "photosynthesis": 80, "aquatic": False, "body_plan": "cactus",
        "seed_resistance_level": 60,
        "water_storage_level": 95,            # CAM + tejido suculento
        "fire_resistance_level": 40,
    }
    return SpeciesTemplate(
        species_id=_next_id(),
        scientific_name="Opuntia ficus-indica",
        taxonomy=["Plantae", "Tracheophyta", "Magnoliopsida",
                  "Caryophyllales", "Cactaceae"],
        kingdom="plantae",
        parent=None, born_at=born_at,
        n_loci=n_loci, n_chromosomes=11,
        gp_map=flora_gpmap(n_loci, rng, is_fungi=False, hints=hints, map_rng=map_rng),
        hints=hints,
    )


def all_prototype_species(rng, map_rng=None) -> list[SpeciesTemplate]:
    """v1.5: si se pasa `map_rng`, cada especie recibe un RNG dedicado
    para construir su GPMap. Esto congela la disposición locus↔rasgo
    independientemente del orden de creación (N2).
    """
    # v1.5.1 — Phyllostachys edulis (bambú) eliminado del set por defecto.
    # v1.5.7 — añadidos Quercus (bosque bulk), Sphagnum (frío/húmedo) y
    # Opuntia (desierto) para enriquecer la cobertura vegetal y sostener
    # más biomasa de insectos en biomas donde Yi qi puede llegar.
    factories = [poa_annua, helianthus_annuus, orchis_stoneplace,
                 quercus_robur, sphagnum_sp, opuntia_ficus_indica,
                 pleurotus_ostreatus,
                 yi_qi, triops_longicaudatus]
    out = []
    for i, fn in enumerate(factories):
        if map_rng is not None:
            # RNG por-especie determinista a partir del map_rng maestro
            seed = int(map_rng.integers(0, 2**31 - 1))
            per_map = __import__("numpy").random.default_rng(seed)
            tpl = fn(rng, map_rng=per_map)
        else:
            tpl = fn(rng)
        out.append(tpl)
    return out
