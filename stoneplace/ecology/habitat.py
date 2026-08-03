"""Cálculo de idoneidad de hábitat por individuo.

Combina las capas del mundo con el fenotipo expresado del individuo para
producir una probabilidad de supervivencia/reproducción por tick. Sigue
la ley del mínimo de Liebig — el factor peor manda.
"""
from __future__ import annotations
import numpy as np
from ..utils import clip01, gaussian_tolerance, liebig_min
from ..world import biomes as B

try:
    from scipy.ndimage import uniform_filter as _scipy_uniform_filter
    _HAS_SCIPY = True
except Exception:                                       # pragma: no cover
    _HAS_SCIPY = False


def _local_mean(field: np.ndarray, radius: int = 2) -> np.ndarray:
    """Media local en un kernel (2*radius+1)² con wraparound toroidal.

    Optimización P12/perf: cuando scipy está disponible se usa
    `uniform_filter` (una pasada en C, ~10× más rápido que 25 `np.roll`).
    Fallback a np.roll cuando scipy no está.
    """
    if radius <= 0:
        return field
    if _HAS_SCIPY:
        size = 2 * radius + 1
        # mode='wrap' respeta el mundo toroidal que ya usa disperse_seeds.
        return _scipy_uniform_filter(
            field.astype(np.float32, copy=False), size=size, mode="wrap"
        ).astype(np.float32)
    acc = np.zeros_like(field, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            acc += np.roll(np.roll(field, dy, axis=0), dx, axis=1)
            count += 1
    return acc / max(count, 1)


def _ecological_release(pop, current_year: float | None) -> float:
    """Bonus a `f_food` para fundadores recientes (< 80 años).

    Justificación biológica: cuando una especie coloniza un nicho, hay una
    fase inicial de "ecological release" antes de que la competencia
    interespecífica se equilibre. Aquí lo modelamos como un multiplicador
    de 1.5× que decae linealmente a 1.0× a los 80 años desde `born_at`.

    Evita que Yi qi (y otros fundadores animales) se extingan en el
    período crítico post-seed, cuando la selección disruptiva y la
    variabilidad mutacional pueden dar dinámicas caóticas.
    """
    if current_year is None:
        return 1.0
    age = current_year - pop.template.born_at
    if age >= 80.0:
        return 1.0
    return float(1.0 + 0.5 * (1.0 - age / 80.0))


def suitability(pop, world, biome_map, micro_dict) -> np.ndarray:
    """Devuelve un array (N,) con la idoneidad 0..1 de cada individuo."""
    if pop.n == 0:
        return np.zeros(0, dtype=np.float32)
    y = pop.y; x = pop.x
    T = world.layers["temperature"][y, x]
    hum = world.layers["humidity"][y, x]
    elev_here = world.layers["elevation"][y, x]
    water_here = elev_here <= 0
    salt_layer = world.layers.get("water_salinity",
                                  np.zeros_like(world.layers["elevation"]))
    salt_here = salt_layer[y, x]
    river_here = world.layers.get("riverFlow",
                                  np.zeros_like(world.layers["elevation"]))[y, x]
    ph = world.layers.get("soil_ph",
                          np.full_like(world.layers["elevation"], 6.5))[y, x]
    fert = world.layers.get("fertility",
                            np.full_like(world.layers["elevation"], 0.4))[y, x]
    aw_raw = world.layers.get("available_water", world.layers["rains"])[y, x]
    aw = aw_raw / 1200.0

    p = pop.phenotype
    aquatic = pop.template.hints.get("aquatic", False)

    # Temperatura
    t_opt = (p["max_temp"] + p["min_temp"]) * 0.5
    t_sigma = np.maximum((p["max_temp"] - p["min_temp"]) * 0.35, 1.0)
    f_temp = gaussian_tolerance(T, t_opt, t_sigma)

    # Agua / medio
    if aquatic:
        in_pool = (river_here > 0.05) | (aw_raw > 400)
        habitat_water = water_here | in_pool
        f_medium = np.where(habitat_water, 1.0, 0.05).astype(np.float32)
        eff_salt = np.where(in_pool & ~water_here, 0.0, salt_here)
        f_salt = clip01(1.0 - np.maximum(eff_salt - p["salt_tolerance"], 0.0) / 30.0)
    else:
        f_medium = np.where(water_here, 0.02, 1.0).astype(np.float32)
        f_medium = f_medium * clip01(aw + 0.15)
        f_salt = clip01(1.0 - np.maximum(salt_here - p["salt_tolerance"], 0.0) / 30.0)

    factors = [f_temp, f_salt, f_medium]

    if pop.template.kingdom in ("plantae", "fungi"):
        ph_opt = (p["max_ph"] + p["min_ph"]) * 0.5
        ph_sigma = np.maximum((p["max_ph"] - p["min_ph"]) * 0.4, 0.3)
        f_ph = gaussian_tolerance(ph, ph_opt, ph_sigma)
        factors.append(f_ph)
        weight = 0.9 if pop.template.kingdom == "plantae" else 0.5
        f_nut = clip01(fert * weight + (1.0 - weight))
        factors.append(f_nut)

    if pop.template.kingdom == "chordata":
        diet = np.stack([
            p["diet_bug"], p["diet_meat"], p["diet_vegetal"],
            p["diet_fish"], p["diet_micro"],
        ], axis=1)
        diet = diet / np.maximum(diet.sum(axis=1, keepdims=True), 1e-6)

        # v1.5.1 — SELECCIÓN DISRUPTIVA sobre dieta (estilo Serina).
        # Dos bonos que empujan la población a las esquinas del simplex:
        #   1. Especialista: Simpson's index (∑ p²) alto = dieta concentrada
        #      en un único recurso → mejor conversión trófica.
        #   2. Divergente: la individua cuyo perfil dietético se aleja del
        #      perfil poblacional MEDIO evita competencia intraespecífica
        #      (frequency-dependent selection, Rosenzweig 1978).
        # Combinados, los generalistas medios pierden frente a los
        # especialistas de nicho — el driver clásico de radiación adaptativa.
        specialization = (diet ** 2).sum(axis=1)                  # 0.2 (uniforme) .. 1.0 (all-in)
        pop_mean_diet = diet.mean(axis=0, keepdims=True)
        divergence = 0.5 * np.abs(diet - pop_mean_diet).sum(axis=1)  # 0 .. 1
        # Bonos NON-NEGATIVOS: el generalista no está penalizado; sólo el
        # especialista y el divergente reciben ventaja. Evitamos la
        # extinción de fundadores en mundos apretados.
        specialist_bonus = 1.0 + 0.25 * (specialization - 0.2) / 0.8  # 1.00 (uniforme) .. 1.25 (all-in)
        divergent_bonus = 1.0 + 0.20 * divergence                     # 1.00 .. 1.20

        if "bugs_local" in micro_dict:
            bugs_l = micro_dict["bugs_local"]
            meat_l = micro_dict["meat_local"]
            plant_l = micro_dict["plant_local"]
            fish_l = micro_dict["fish_local"]
            micro_l = micro_dict["micro_local"]
        else:
            bugs_l = _local_mean(micro_dict["bugs"], radius=2)
            meat_l = _local_mean(micro_dict.get("meat_biomass", micro_dict["bugs"]), radius=2)
            plant_l = _local_mean(micro_dict.get("plant_biomass", micro_dict["detritus"]), radius=2)
            fish_l = _local_mean(micro_dict.get("fish_biomass", micro_dict["plankton"]), radius=2)
            micro_l = _local_mean(micro_dict["microbes"] * 0.5 + micro_dict["plankton"] * 0.5, radius=2)

        food_sources = np.stack([
            bugs_l[y, x], meat_l[y, x], plant_l[y, x],
            fish_l[y, x], micro_l[y, x],
        ], axis=1)
        raw_food = (diet * food_sources).sum(axis=1) * 1.5 + 0.1
        # v1.5.1 — aplicar los dos bonos disruptivos
        raw_food = raw_food * specialist_bonus * divergent_bonus
        # v1.5.3 — ecological release para especies jóvenes
        current_year = micro_dict.get("_current_year")
        release_mult = _ecological_release(pop, current_year)
        raw_food = raw_food * release_mult
        # v1.5 (N8): distribución libre ideal — la comida por individuo
        # se diluye con la densidad LOCAL de la propia especie. Antes,
        # animal_walk atraía a todos al mismo pico → aglomeración.
        density_local = micro_dict.get(f"density_{pop.template.species_id}")
        if density_local is not None:
            d_here = density_local[y, x].astype(np.float32)
            # IFD blanda: la comida por individuo se reduce en tiles con
            # densidad muy alta. K generosa + umbral inferior "gratis"
            # (una manada nace amontonada, no la castigamos hasta que
            # supera cierta densidad). La mortalidad density-dependent de
            # `age_and_die` sigue siendo el freno principal.
            crowd_tol = np.maximum(p.get("overcrowd_tolerance",
                                          np.full(pop.n, 40.0)), 10.0)
            free_density = 30.0 + crowd_tol            # margen gratis
            K_local = 800.0 + 15.0 * crowd_tol
            excess = np.maximum(d_here - free_density, 0.0)
            competition = np.clip(excess / K_local, 0.0, 0.5)
            raw_food = raw_food * (1.0 - competition)
        f_food = clip01(raw_food)
        factors.append(f_food)

    return liebig_min(*factors).astype(np.float32)
