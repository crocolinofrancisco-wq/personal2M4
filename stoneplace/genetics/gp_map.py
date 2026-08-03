"""Matriz G→P: poligenia + pleiotropía + heredabilidad h² + gating blando.

v1.5:
  * Los rasgos GATED usan compuertas BLANDAS (`soft_gate`, N3):
    partial floor + rampa lineal + sigmoide. La selección puede empujar
    el rasgo hacia arriba porque hay pequeñas recompensas antes del
    umbral duro (Nilsson & Pelger 1994 sobre el ojo).
  * Se puede pasar un `rng` dedicado al construir cada GPMap (N2). Así
    la disposición de loci NO depende del orden de creación de rasgos
    dentro del diccionario, y dos linajes de la misma especie con la
    misma seed no comparten identidad de mapa a menos que quieras.
  * `mutate_gp_map()` implementa mutación cis-regulatoria rara: al
    especiar, con probabilidad `p_cis` se reasigna 1 locus del mapa de
    la hija (evo-devo, N2).

Cada rasgo del fenotipo se define como una combinación lineal de varios
loci; cada locus puede afectar varios rasgos (pleiotropía). La expresión
final añade ruido ambiental con h² controlada:

    P = h * G_normalizado + (1-h) * ruido_ambiental * scale + baseline
"""
from __future__ import annotations
from dataclasses import dataclass, field
import copy
import numpy as np
from .genome import Genome
from ..utils import soft_gate as _soft_gate, sigmoid as _sigmoid_util


# ---------------------------------------------------------------------------
# TraitSpec: rasgo lineal poligénico con h² y clip opcional
# ---------------------------------------------------------------------------
@dataclass
class TraitSpec:
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


# ---------------------------------------------------------------------------
# GatedTraitSpec: habilidad especial con prerequisitos duros
# ---------------------------------------------------------------------------
@dataclass
class GatedTraitSpec:
    """Rasgo que requiere que otros rasgos/loci superen umbrales.

    El rasgo se expresa así:

        raw   = poligenia normal(loci, weights)   # 0..100 potencial
        gate  = ∏ sigmoid( (req_value - threshold) / softness )
        final = raw * gate    (queda en 0..100, casi 0 si algún req falla)

    Ejemplo: bioluminiscencia_level en Yi qi requiere
        - intelligence      >= 55   (control neuronal del órgano)
        - energy_surplus    ~ Kleiber-margen positivo (metabolismo capaz)
        - photic_seed_locus >= 60   (locus dedicado al órgano fotogénico)
    Hasta que la selección natural NO empuje esos tres, el rasgo queda ~0.
    """
    base: TraitSpec                              # rasgo poligénico "potencial"
    gates: list[tuple[str, float, float]]         # (nombre_req, umbral, softness)
    trait_of_locus: str | None = None             # "locus_direct" opcional
    seed_loci: np.ndarray | None = None           # loci dedicados al órgano
    seed_threshold: float = 55.0                  # media loci dedicados
    seed_softness: float = 8.0
    # v1.5: pleiotropía blanda. Un mini-beneficio "latente" mantiene el
    # rasgo bajo selección débil incluso cuando la puerta está casi cerrada
    # → evita el valle-fitness que hacía a las gated dead-code (N3).
    latent_floor: float = 0.05


def _sigmoid(x):
    return _sigmoid_util(x)


# ---------------------------------------------------------------------------
# GPMap
# ---------------------------------------------------------------------------
@dataclass
class GPMap:
    """Matriz G→P completa para una especie."""
    traits: dict[str, TraitSpec]
    gated: dict[str, GatedTraitSpec] = field(default_factory=dict)

    def express(self, genome: Genome, rng, sigma_env: float = 8.0) -> dict[str, np.ndarray]:
        additive = genome.additive()                   # (N, L), 0..100
        centered = (additive - 50.0) / 50.0
        out = {}

        # 1) rasgos lineales (poligénicos normales)
        for name, spec in self.traits.items():
            g_component = centered[:, spec.loci] @ spec.weights
            env = rng.normal(0.0, sigma_env / 50.0, size=g_component.shape)
            expression = spec.h2 * g_component + (1.0 - spec.h2) * env
            value = spec.baseline + expression * spec.scale
            if spec.clip is not None:
                value = np.clip(value, spec.clip[0], spec.clip[1])
            out[name] = value.astype(np.float32)

        # 2) rasgos gated (habilidades desbloqueables)
        for name, gspec in self.gated.items():
            spec = gspec.base
            g_component = centered[:, spec.loci] @ spec.weights
            env = rng.normal(0.0, sigma_env / 50.0, size=g_component.shape)
            expression = spec.h2 * g_component + (1.0 - spec.h2) * env
            raw = spec.baseline + expression * spec.scale
            if spec.clip is not None:
                raw = np.clip(raw, spec.clip[0], spec.clip[1])

            # Compuertas de prerequisito con BLANDURA (N3): recompensa
            # parcial ya con val = umbral - 3·softness. La evolución puede
            # subir la habilidad de forma gradual.
            gate = np.ones_like(raw, dtype=np.float32)
            for req_name, thresh, soft in gspec.gates:
                if req_name in out:
                    req_val = out[req_name]
                elif req_name == "mass_log":
                    req_val = np.log10(np.maximum(out.get("mass_g", raw), 1e-3))
                else:
                    continue
                gate = gate * _soft_gate(req_val, thresh, soft,
                                          floor=gspec.latent_floor)

            # Compuerta adicional: locus "semilla" dedicado al órgano
            if gspec.seed_loci is not None and gspec.seed_loci.size > 0:
                seed_mean = additive[:, gspec.seed_loci].mean(axis=1)
                gate = gate * _soft_gate(seed_mean, gspec.seed_threshold,
                                          gspec.seed_softness,
                                          floor=gspec.latent_floor)

            final = raw * gate
            out[name] = np.clip(final, 0.0, 100.0).astype(np.float32)

        # ------------------------------------------------------------------
        # v1.5: coste pleiotrópico de attractiveness (N6, Kirkpatrick 1982).
        # Antes attractiveness era freebie → runaway sexual selection sin
        # freno. Ahora paga con basal_metabolism y speed reducida.
        # ------------------------------------------------------------------
        if "attractiveness" in out and "basal_metabolism" in out:
            attr = out["attractiveness"]
            # +0..15 % basal metabolic cost al máximo del ornamento
            cost = 1.0 + 0.0015 * np.maximum(attr - 40.0, 0.0)
            out["basal_metabolism"] = (out["basal_metabolism"] * cost).astype(np.float32)
            if "speed" in out:
                out["speed"] = np.clip(
                    out["speed"] - 0.10 * np.maximum(attr - 40.0, 0.0),
                    0.0, 100.0).astype(np.float32)

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bl(hints, key, default):
    if not hints or key not in hints:
        return float(default)
    return float(hints[key])


def _pick_seed_loci(rng, n_loci, k=3):
    """Loci dedicados a un órgano concreto (semilla del gating)."""
    return rng.choice(n_loci, size=k, replace=False).astype(np.int32)


# ---------------------------------------------------------------------------
# v1.5 — Mutación cis-regulatoria del mapa G→P (evo-devo, N2)
# ---------------------------------------------------------------------------
def mutate_gp_map(gp_map: "GPMap", n_loci: int, rng,
                  p_cis: float = 0.05, p_seed: float = 0.05) -> "GPMap":
    """Devuelve un GPMap NUEVO con mutaciones raras del cableado gen→rasgo.

    Con probabilidad `p_cis` reasigna 1 locus de un rasgo aleatorio
    (mutación regulatoria). Con probabilidad `p_seed` desplaza un seed_locus
    de una habilidad gated. Sirve para simular divergencia arquitectural
    lenta entre linajes (biología evo-devo real: cis-regulatory shifts).
    """
    new = copy.deepcopy(gp_map)
    if rng.random() < p_cis and new.traits:
        name = rng.choice(list(new.traits.keys()))
        spec = new.traits[name]
        idx = int(rng.integers(0, spec.loci.size))
        # elegir un locus nuevo que no esté ya usado por este rasgo
        used = set(int(v) for v in spec.loci)
        avail = [i for i in range(n_loci) if i not in used]
        if avail:
            new_locus = int(rng.choice(avail))
            spec.loci = spec.loci.copy()
            spec.loci[idx] = new_locus
    if rng.random() < p_seed and new.gated:
        gname = rng.choice(list(new.gated.keys()))
        gspec = new.gated[gname]
        if gspec.seed_loci is not None and gspec.seed_loci.size > 0:
            j = int(rng.integers(0, gspec.seed_loci.size))
            gspec.seed_loci = gspec.seed_loci.copy()
            gspec.seed_loci[j] = int(rng.integers(0, n_loci))
    return new


# ---------------------------------------------------------------------------
# CHORDATA — GPMap
# ---------------------------------------------------------------------------
def chordata_gpmap(n_loci: int, rng, hints: dict | None = None,
                   *, map_rng=None) -> GPMap:
    """Genera una matriz G→P coherente con la plantilla Chordata.

    v1.5 (N2): si se pasa `map_rng`, el sorteo de loci usa ese RNG
    dedicado y NO el RNG global de simulación. Esto evita que un cambio
    de orden en el diccionario `traits[...]` altere toda la simulación.
    Recomendado: `map_rng = default_rng(species_id * 0x9E3779B1 ^ seed)`.
    """
    _rng = map_rng if map_rng is not None else rng
    def pick(k, positive_bias=0.0):
        loci = _rng.choice(n_loci, size=k, replace=False)
        w = _rng.normal(positive_bias, 1.0, size=k)
        return loci, w

    diet_bias = (hints or {}).get("diet_bias", {}) or {}
    abil_bias = (hints or {}).get("abilities_bias", {}) or {}

    mass_bl = _bl(hints, "mass_g_opt", 500)
    size_bl = _bl(hints, "size_mm_opt", 350)
    life_bl = _bl(hints, "life_expectancy", 12)
    breed_bl = _bl(hints, "breeding_lapse", 1.0)
    baby_bl = _bl(hints, "baby_quantity", 3)

    traits = {}
    traits["size_length_mm"]   = TraitSpec.make(*pick(8), size_bl, size_bl * 0.45, h2=0.55, clip=(1, 1e6), unit="mm")
    traits["mass_g"]           = TraitSpec.make(*pick(8), mass_bl, mass_bl * 0.45, h2=0.55, clip=(0.01, 1e8), unit="g")
    traits["life_expectancy"]  = TraitSpec.make(*pick(6), life_bl, max(life_bl * 0.4, 0.05), h2=0.45, clip=(0.05, 200), unit="años")
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
    traits["breeding_lapse"]   = TraitSpec.make(*pick(4), breed_bl, max(breed_bl * 0.35, 0.05), h2=0.5, clip=(0.02, 20), unit="años")
    traits["baby_quantity"]    = TraitSpec.make(*pick(4), baby_bl, max(baby_bl * 0.35, 1.0), h2=0.5, clip=(1, 5000))
    traits["hatch_time"]       = TraitSpec.make(*pick(3), 0.15, 0.1, h2=0.5, clip=(0.01, 3), unit="años")
    growth_bl = min(life_bl * 0.5, max(breed_bl * 1.5, 0.05))
    traits["growth_time"]      = TraitSpec.make(*pick(4), growth_bl, max(growth_bl * 0.4, 0.02), h2=0.5, clip=(0.02, 30), unit="años")
    traits["sociability"]      = TraitSpec.make(*pick(4), 50, 25, h2=0.3, clip=(0, 100))
    traits["fleeing"]          = TraitSpec.make(*pick(4), 50, 25, h2=0.35, clip=(0, 100))
    traits["aggressivity"]     = TraitSpec.make(*pick(4), 40, 25, h2=0.35, clip=(0, 100))
    traits["curiosity"]        = TraitSpec.make(*pick(4), 45, 20, h2=0.3, clip=(0, 100))
    traits["circadian"]        = TraitSpec.make(*pick(3), 50, 30, h2=0.4, clip=(0, 100))
    for diet in ("bug", "meat", "vegetal", "fish", "micro"):
        traits[f"diet_{diet}"] = TraitSpec.make(*pick(3), float(diet_bias.get(diet, 20)), 25, h2=0.5, clip=(0, 100))

    # -----------------------------------------------------------------------
    # HABILIDADES BASE (siempre expresables, poca traba biológica):
    # sight, smell, hearing, climbing, dig, swiming, breath_air/water.
    # -----------------------------------------------------------------------
    base_abilities = [
        ("sight_level", 30, 22, 0.4),
        ("smell_level", 30, 22, 0.4),
        ("hearing_level", 30, 22, 0.4),
        ("climbing_level", 25, 20, 0.4),
        ("dig_level", 20, 18, 0.4),
        ("swiming_level", 20, 20, 0.4),
        ("breath_air_level", 65, 25, 0.4),
        ("breath_water_level", 20, 25, 0.4),
    ]
    for a, bl, sc, h in base_abilities:
        traits[a] = TraitSpec.make(*pick(3), float(abil_bias.get(a, bl)), sc, h2=h, clip=(0, 100))

    # Vuelo: gating suave (requiere masa baja, alta stamina, morfología alada)
    traits["flying_level"] = TraitSpec.make(*pick(3), float(abil_bias.get("flying_level", 5)), 22, h2=0.4, clip=(0, 100))

    # -----------------------------------------------------------------------
    # HABILIDADES ESPECIALES CON GATING — arrancan ~0 en el fenotipo real
    # aunque el rasgo poligénico exista. Solo se desbloquean si la selección
    # acumula los prerequisitos.
    # -----------------------------------------------------------------------
    gated: dict[str, GatedTraitSpec] = {}

    # BIOLUMINISCENCIA — requiere metabolismo excedente, órgano fotogénico
    # dedicado (seed loci) y control neural. Yi qi NO la desbloqueará
    # fácilmente porque su intelligence baseline es 40.
    gated["bioluminiscence_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 60.0, 25.0, h2=0.45, clip=(0, 100)),
        gates=[
            ("intelligence", 55.0, 6.0),          # control neural del órgano
            ("basal_metabolism", metab_bl * 0.9, metab_bl * 0.2),
        ],
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=62.0, seed_softness=6.0,
    )

    # TOXICIDAD — requiere defensa química (metabolismo excedente) y baja
    # velocidad relativa (los tóxicos suelen ser lentos: pastilla evolutiva).
    gated["toxicity_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 55.0, 25.0, h2=0.5, clip=(0, 100)),
        gates=[
            ("basal_metabolism", metab_bl * 0.85, metab_bl * 0.25),
        ],
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=60.0, seed_softness=8.0,
    )

    # ECOLOCALIZACIÓN — requiere hearing_level MUY alto e intelligence.
    gated["echolocation_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 50.0, 25.0, h2=0.5, clip=(0, 100)),
        gates=[
            ("hearing_level", 65.0, 6.0),
            ("intelligence", 50.0, 6.0),
        ],
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=60.0, seed_softness=8.0,
    )

    # ELECTRORRECEPCIÓN — casi imposible fuera del agua.
    is_aquatic = bool((hints or {}).get("aquatic", False))
    electro_gates = [("intelligence", 45.0, 8.0)]
    if not is_aquatic:
        # gate durísima para no-acuáticos: el rasgo queda anclado a ~0.
        electro_gates.append(("swiming_level", 55.0, 6.0))
    gated["electroreception_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 45.0, 25.0, h2=0.5, clip=(0, 100)),
        gates=electro_gates,
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=60.0, seed_softness=8.0,
    )

    # VISIÓN INFRARROJA — requiere sight_level alto + circadian nocturno.
    gated["infrared_sight_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 50.0, 25.0, h2=0.5, clip=(0, 100)),
        gates=[
            ("sight_level", 60.0, 6.0),
        ],
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=62.0, seed_softness=8.0,
    )

    # CAMUFLAJE — requiere baja aggressivity O alta intelligence.
    gated["camouflage_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 55.0, 25.0, h2=0.5, clip=(0, 100)),
        gates=[
            ("intelligence", 40.0, 10.0),
        ],
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=58.0, seed_softness=8.0,
    )

    # ALIENTO DE FUEGO — rasgo especulativo, endurecemos MUCHO los gates.
    gated["fire_breath_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 40.0, 25.0, h2=0.5, clip=(0, 100)),
        gates=[
            ("basal_metabolism", metab_bl * 1.5, metab_bl * 0.3),
            ("aggressivity", 65.0, 6.0),
            ("intelligence", 55.0, 6.0),
        ],
        seed_loci=_pick_seed_loci(rng, n_loci, k=4),
        seed_threshold=72.0, seed_softness=5.0,
    )

    return GPMap(traits=traits, gated=gated)


# ---------------------------------------------------------------------------
# FLORA (plantae / fungi) — GPMap
# ---------------------------------------------------------------------------
def flora_gpmap(n_loci: int, rng, is_fungi: bool = False,
                hints: dict | None = None, *, map_rng=None) -> GPMap:
    """P11 fix aplicado también aquí + RNG dedicado (N2)."""
    _rng = map_rng if map_rng is not None else rng
    def pick(k):
        loci = _rng.choice(n_loci, size=k, replace=False)
        w = _rng.normal(0.0, 1.0, size=k)
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
    traits["breeding_lapse"]   = TraitSpec.make(*pick(3), breed_bl, max(breed_bl * 0.35, 0.05), h2=0.5, clip=(0.02, 30), unit="años")
    traits["baby_quantity"]    = TraitSpec.make(*pick(3), baby_bl, max(baby_bl * 0.35, 5), h2=0.55, clip=(1, 5e5))
    traits["germination_time"] = TraitSpec.make(*pick(3), 0.05, 0.05, h2=0.5, clip=(0.01, 3), unit="años")
    growth_bl = min(life_bl * 0.4, max(breed_bl, 0.05))
    traits["growth_time"]      = TraitSpec.make(*pick(3), growth_bl, max(growth_bl * 0.4, 0.05), h2=0.5, clip=(0.02, 100), unit="años")

    for a in ("overcrowd_tolerance", "allelopathy", "regeneration", "dormancy_capability"):
        traits[a] = TraitSpec.make(*pick(3), _bl(hints, a, 40), 22, h2=0.45, clip=(0, 100))

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

    # Habilidades base (resistencia, dispersión, almacenamiento agua)
    traits["fire_resistance_level"]  = TraitSpec.make(*pick(3), _bl(hints, "fire_resistance_level", 25), 22, h2=0.4, clip=(0, 100))
    traits["water_storage_level"]    = TraitSpec.make(*pick(3), _bl(hints, "water_storage_level", 25), 22, h2=0.4, clip=(0, 100))
    traits["seed_resistance_level"]  = TraitSpec.make(*pick(3), _bl(hints, "seed_resistance_level", 25), 22, h2=0.4, clip=(0, 100))

    # GATED (flora): trampa carnívora, toxicidad, bioluminiscencia (setas)
    gated: dict[str, GatedTraitSpec] = {}

    # TRAMPA CARNÍVORA — requiere carnivory alto + baja fotosíntesis.
    gated["carnivorous_trap_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 55.0, 25.0, h2=0.45, clip=(0, 100)),
        gates=[
            ("carnivory", 40.0, 8.0),
        ],
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=60.0, seed_softness=8.0,
    )

    # TOXICIDAD DEFENSIVA — requiere metabolismo secundario (parasitism o allelopathy)
    gated["toxicity_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 55.0, 25.0, h2=0.5, clip=(0, 100)),
        gates=[
            ("allelopathy", 45.0, 8.0),
        ],
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=58.0, seed_softness=8.0,
    )

    # BIOLUMINISCENCIA (hongos: setas fosforescentes son reales)
    gated["bioluminiscence_level"] = GatedTraitSpec(
        base=TraitSpec.make(*pick(3), 55.0, 25.0, h2=0.5, clip=(0, 100)),
        gates=(
            [("saprotrophy", 50.0, 8.0)] if is_fungi
            else [("attractiveness", 60.0, 6.0)]
        ),
        seed_loci=_pick_seed_loci(rng, n_loci, k=3),
        seed_threshold=62.0, seed_softness=6.0,
    )

    return GPMap(traits=traits, gated=gated)
