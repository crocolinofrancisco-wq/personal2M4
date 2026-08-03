"""Matriz G→P: poligenia + pleiotropía + heredabilidad h².

Cada rasgo del fenotipo se define como una combinación lineal de varios
loci; cada locus puede afectar varios rasgos (pleiotropía). La expresión
final añade ruido ambiental con h² controlada:

    P = h * G_normalizado + (1-h) * ruido_ambiental * scale + baseline

Los rasgos siguen exactamente las plantillas del usuario (Chordata y
Fungi/Plantae).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .genome import Genome


@dataclass
class TraitSpec:
    """Definición de un rasgo del fenotipo.

    Attributes:
        loci: índices de los loci que contribuyen
        weights: peso de cada locus (mismo largo que loci); se normaliza
        baseline: valor esperado cuando los loci están en su media (50)
        scale: cuánto varía el rasgo por unidad de desvío genético
        h2: heredabilidad (0..1). Alta = poco ruido ambiental
        clip: (min, max) opcional para forzar rango biológico
        unit: string informativo (mm, g, kcal/d, ...)
    """
    loci: np.ndarray
    weights: np.ndarray
    baseline: float
    scale: float
    h2: float = 0.6
    clip: tuple[float, float] | None = None
    unit: str = ""

    @classmethod
    def make(cls, loci, weights, baseline, scale, h2=0.6, clip=None, unit=""):
        loci = np.asarray(loci, dtype=np.int32)
        w = np.asarray(weights, dtype=np.float32)
        w = w / (np.abs(w).sum() + 1e-9)
        return cls(loci, w, baseline, scale, h2, clip, unit)


@dataclass
class GPMap:
    """Matriz G→P completa para una especie."""
    traits: dict[str, TraitSpec]

    def express(self, genome: Genome, rng, sigma_env: float = 8.0) -> dict[str, np.ndarray]:
        additive = genome.additive()                    # (N, L), 0..100
        # Centrar en 50 y normalizar → contribución en [-1, 1] típica
        centered = (additive - 50.0) / 50.0
        out = {}
        for name, spec in self.traits.items():
            g_component = centered[:, spec.loci] @ spec.weights   # (N,)
            env = rng.normal(0.0, sigma_env / 50.0, size=g_component.shape)
            expression = spec.h2 * g_component + (1.0 - spec.h2) * env
            value = spec.baseline + expression * spec.scale
            if spec.clip is not None:
                value = np.clip(value, spec.clip[0], spec.clip[1])
            out[name] = value.astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Constructores por reino: mapean la plantilla del usuario a una GPMap concreta
# ---------------------------------------------------------------------------

def _bl(hints, key, default):
    """Devuelve `hints[key]` (float) si existe; si no, el default.

    Es la vía para que la biología declarada en `species/registry.py`
    (baby_quantity=120 para Triops, mass=380 g para Yi qi, etc.) llegue
    realmente al fenotipo esperado — antes los hints se ignoraban y todo
    chordata compartía baseline 3 crías, lo que condenaba a los r-estrategas.
    """
    if not hints or key not in hints:
        return float(default)
    return float(hints[key])


def chordata_gpmap(n_loci: int, rng, hints: dict | None = None) -> GPMap:
    """Genera una matriz G→P coherente con la plantilla Chordata.

    Cada rasgo toma 4-10 loci al azar con pesos aleatorios; hay solape
    para forzar pleiotropía. Baselines respetan `hints` cuando existen.
    """
    def pick(k, positive_bias=0.5):
        loci = rng.choice(n_loci, size=k, replace=False)
        w = rng.normal(positive_bias, 1.0, size=k)
        return loci, w

    diet_bias = (hints or {}).get("diet_bias", {}) or {}
    abil_bias = (hints or {}).get("abilities_bias", {}) or {}

    # Escalas de variación (scale) proporcionales a la baseline biológica,
    # para que la selección tenga margen de maniobra en cada especie.
    mass_bl = _bl(hints, "mass_g_opt", 500)
    size_bl = _bl(hints, "size_mm_opt", 350)
    life_bl = _bl(hints, "life_expectancy", 12)
    breed_bl = _bl(hints, "breeding_lapse", 1.0)
    baby_bl = _bl(hints, "baby_quantity", 3)

    traits = {}
    traits["size_length_mm"]   = TraitSpec.make(*pick(8), size_bl, size_bl * 0.45, h2=0.55, clip=(1, 1e6), unit="mm")
    traits["mass_g"]           = TraitSpec.make(*pick(8), mass_bl, mass_bl * 0.45, h2=0.55, clip=(0.01, 1e8), unit="g")
    traits["life_expectancy"]  = TraitSpec.make(*pick(6), life_bl, max(life_bl * 0.4, 0.05), h2=0.45, clip=(0.05, 200), unit="años")
    # Metabolismo escala aproximadamente ~M^0.75 (ley de Kleiber);
    # aquí lo aproximamos linealmente para no explotar los valores.
    metab_bl = max(6.0, 25.0 * (mass_bl ** 0.75) / 100.0)
    traits["basal_metabolism"] = TraitSpec.make(*pick(6), metab_bl, metab_bl * 0.5, h2=0.5, clip=(1, 1e6), unit="kcal/día")
    traits["water_turnover"]   = TraitSpec.make(*pick(5), max(200.0, mass_bl * 50), max(150.0, mass_bl * 30), h2=0.4, clip=(1, 1e9), unit="mm³/día")
    traits["max_temp"]         = TraitSpec.make(*pick(5), _bl(hints, "max_temp", 38), 6, h2=0.6, clip=(-5, 65), unit="°C")
    traits["min_temp"]         = TraitSpec.make(*pick(5), _bl(hints, "min_temp", 0),  6, h2=0.6, clip=(-60, 30), unit="°C")
    traits["salt_tolerance"]   = TraitSpec.make(*pick(4), _bl(hints, "salt_tolerance", 5), 4, h2=0.5, clip=(0, 60), unit="PSU")
    traits["intelligence"]     = TraitSpec.make(*pick(6), 40, 25, h2=0.4, clip=(0, 100))
    traits["speed"]            = TraitSpec.make(*pick(6), 45, 30, h2=0.55, clip=(0, 100))
    traits["resistance"]       = TraitSpec.make(*pick(6), 45, 25, h2=0.5, clip=(0, 100))
    traits["ferocity"]         = TraitSpec.make(*pick(4), 30, 25, h2=0.45, clip=(0, 100))
    traits["stamina"]          = TraitSpec.make(*pick(5), 45, 25, h2=0.5, clip=(0, 100))
    traits["attractiveness"]   = TraitSpec.make(*pick(5), 45, 25, h2=0.35, clip=(0, 100))
    # Reproducción — respetan hints biológicos
    traits["breeding_lapse"]   = TraitSpec.make(*pick(4), breed_bl, max(breed_bl * 0.35, 0.05), h2=0.5, clip=(0.02, 20), unit="años")
    traits["baby_quantity"]    = TraitSpec.make(*pick(4), baby_bl, max(baby_bl * 0.35, 1.0), h2=0.5, clip=(1, 5000))
    traits["hatch_time"]       = TraitSpec.make(*pick(3), 0.15, 0.1, h2=0.5, clip=(0.01, 3), unit="años")
    # growth_time debe ser < life_expectancy y ≤ breeding_lapse * 2
    growth_bl = min(life_bl * 0.5, max(breed_bl * 1.5, 0.05))
    traits["growth_time"]      = TraitSpec.make(*pick(4), growth_bl, max(growth_bl * 0.4, 0.02), h2=0.5, clip=(0.02, 30), unit="años")
    # Comportamiento
    traits["sociability"]      = TraitSpec.make(*pick(4), 50, 25, h2=0.3, clip=(0, 100))
    traits["fleeing"]          = TraitSpec.make(*pick(4), 50, 25, h2=0.35, clip=(0, 100))
    traits["aggressivity"]     = TraitSpec.make(*pick(4), 40, 25, h2=0.35, clip=(0, 100))
    traits["curiosity"]        = TraitSpec.make(*pick(4), 45, 20, h2=0.3, clip=(0, 100))
    traits["circadian"]        = TraitSpec.make(*pick(3), 50, 30, h2=0.4, clip=(0, 100))
    # Dieta (0..100 cada uno, luego se normalizan a 1 al usarlos)
    for diet in ("bug", "meat", "vegetal", "fish", "micro"):
        traits[f"diet_{diet}"] = TraitSpec.make(*pick(3), float(diet_bias.get(diet, 20)), 25, h2=0.5, clip=(0, 100))
    # Habilidades
    ability_names = [
        "sight_level", "infrared_sight_level", "electroreception_level",
        "bioluminiscence_level", "camouflage_level", "smell_level",
        "hearing_level", "swiming_level", "flying_level", "climbing_level",
        "dig_level", "toxicity_level", "echolocation_level",
        "breath_air_level", "breath_water_level", "fire_breath_level",
    ]
    for a in ability_names:
        traits[a] = TraitSpec.make(*pick(3), float(abil_bias.get(a, 30)), 22, h2=0.4, clip=(0, 100))
    return GPMap(traits=traits)


def flora_gpmap(n_loci: int, rng, is_fungi: bool = False,
                hints: dict | None = None) -> GPMap:
    def pick(k):
        loci = rng.choice(n_loci, size=k, replace=False)
        w = rng.normal(0.4, 1.0, size=k)
        return loci, w

    mass_bl = _bl(hints, "mass_g_opt", 200)
    size_bl = _bl(hints, "size_mm_opt", 300)
    life_bl = _bl(hints, "life_expectancy", 5)
    breed_bl = _bl(hints, "breeding_lapse", 1.0)
    baby_bl = _bl(hints, "baby_quantity", 200)

    traits = {}
    traits["size_length_mm"]   = TraitSpec.make(*pick(6), size_bl, max(size_bl * 0.45, 10), h2=0.6, clip=(1, 1e5), unit="mm")
    traits["mass_g"]           = TraitSpec.make(*pick(6), mass_bl, max(mass_bl * 0.45, 0.1), h2=0.6, clip=(0.001, 1e7), unit="g")
    traits["life_expectancy"]  = TraitSpec.make(*pick(4), life_bl, max(life_bl * 0.4, 0.05), h2=0.5, clip=(0.05, 500), unit="años")
    traits["basal_metabolism"] = TraitSpec.make(*pick(4), 5, 4, h2=0.5, clip=(0.01, 500), unit="kcal/día")
    traits["water_turnover"]   = TraitSpec.make(*pick(4), 5000, 4000, h2=0.5, clip=(1, 1e7), unit="mm³/día")
    traits["max_temp"]         = TraitSpec.make(*pick(4), _bl(hints, "max_temp", 38), 6, h2=0.65, clip=(-5, 65))
    traits["min_temp"]         = TraitSpec.make(*pick(4), _bl(hints, "min_temp", 2),  6, h2=0.65, clip=(-60, 30))
    traits["salt_tolerance"]   = TraitSpec.make(*pick(3), _bl(hints, "salt_tolerance", 3), 4, h2=0.6, clip=(0, 60))
    traits["max_ph"]           = TraitSpec.make(*pick(3), _bl(hints, "max_ph", 7.5), 1.0, h2=0.55, clip=(2, 12))
    traits["min_ph"]           = TraitSpec.make(*pick(3), _bl(hints, "min_ph", 5.5), 1.0, h2=0.55, clip=(2, 12))
    traits["intelligence"]     = TraitSpec.make(*pick(2), 2,  4,  h2=0.3, clip=(0, 100))
    traits["speed"]            = TraitSpec.make(*pick(2), 1,  3,  h2=0.3, clip=(0, 100))
    traits["resistance"]       = TraitSpec.make(*pick(4), 45, 25, h2=0.55, clip=(0, 100))
    traits["attractiveness"]   = TraitSpec.make(*pick(3), _bl(hints, "attractiveness", 40), 22, h2=0.35, clip=(0, 100))
    # Reproducción — respetan hints biológicos
    traits["breeding_lapse"]   = TraitSpec.make(*pick(3), breed_bl, max(breed_bl * 0.35, 0.05), h2=0.5, clip=(0.02, 30), unit="años")
    traits["baby_quantity"]    = TraitSpec.make(*pick(3), baby_bl, max(baby_bl * 0.35, 5), h2=0.55, clip=(1, 5e5))
    traits["germination_time"] = TraitSpec.make(*pick(3), 0.05, 0.05, h2=0.5, clip=(0.01, 3), unit="años")
    growth_bl = min(life_bl * 0.4, max(breed_bl, 0.05))
    traits["growth_time"]      = TraitSpec.make(*pick(3), growth_bl, max(growth_bl * 0.4, 0.05), h2=0.5, clip=(0.02, 100), unit="años")
    # Crecimiento y defensa
    for a in ("overcrowd_tolerance", "allelopathy", "regeneration", "dormancy_capability"):
        traits[a] = TraitSpec.make(*pick(3), _bl(hints, a, 40), 22, h2=0.45, clip=(0, 100))
    # Nutrición
    if is_fungi:
        traits["photosynthesis"] = TraitSpec.make(*pick(2), _bl(hints, "photosynthesis", 0), 5, h2=0.5, clip=(0, 100))
        traits["saprotrophy"]    = TraitSpec.make(*pick(3), _bl(hints, "saprotrophy", 70), 20, h2=0.55, clip=(0, 100))
        traits["parasitism"]     = TraitSpec.make(*pick(3), _bl(hints, "parasitism", 15), 18, h2=0.55, clip=(0, 100))
        traits["carnivory"]      = TraitSpec.make(*pick(3), _bl(hints, "carnivory", 5), 12, h2=0.5, clip=(0, 100))
    else:
        traits["photosynthesis"] = TraitSpec.make(*pick(3), _bl(hints, "photosynthesis", 90), 12, h2=0.6, clip=(0, 100))
        traits["saprotrophy"]    = TraitSpec.make(*pick(2), _bl(hints, "saprotrophy", 5),  8, h2=0.5, clip=(0, 100))
        traits["parasitism"]     = TraitSpec.make(*pick(2), _bl(hints, "parasitism", 3),   6, h2=0.5, clip=(0, 100))
        traits["carnivory"]      = TraitSpec.make(*pick(2), _bl(hints, "carnivory", 2),    6, h2=0.5, clip=(0, 100))
    # Habilidades
    for a in ("fire_resistance_level", "carnivorous_trap_level", "toxicity_level",
              "water_storage_level", "bioluminiscence_level", "seed_resistance_level"):
        traits[a] = TraitSpec.make(*pick(3), _bl(hints, a, 25), 22, h2=0.4, clip=(0, 100))
    return GPMap(traits=traits)
