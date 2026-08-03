"""Cálculo de idoneidad de hábitat por individuo.

Combina las capas del mundo con el fenotipo expresado del individuo para
producir una probabilidad de supervivencia/reproducción por tick. Sigue
la ley del mínimo de Liebig — el factor peor manda.
"""
from __future__ import annotations
import numpy as np
from ..utils import clip01, gaussian_tolerance, liebig_min
from ..world import biomes as B


def _local_mean(field: np.ndarray, radius: int = 2) -> np.ndarray:
    """Media 8-conexa aproximada en un kernel (2*radius+1)². Sirve para
    modelar el forrajeo local del animal (no come solo de la celda exacta).
    Implementado con `np.roll` para mantenerse en NumPy puro y wraparound.
    """
    if radius <= 0:
        return field
    acc = np.zeros_like(field, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            acc += np.roll(np.roll(field, dy, axis=0), dx, axis=1)
            count += 1
    return acc / max(count, 1)


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

    # Temperatura: exige estar entre min_temp y max_temp con curva suave
    t_opt = (p["max_temp"] + p["min_temp"]) * 0.5
    t_sigma = np.maximum((p["max_temp"] - p["min_temp"]) * 0.35, 1.0)
    f_temp = gaussian_tolerance(T, t_opt, t_sigma)

    # Agua / medio
    if aquatic:
        # Los acuáticos toleran también ríos activos y suelos anegados
        # (pozas efímeras / humedales) — no sólo el mar/lago propiamente dicho.
        in_pool = (river_here > 0.05) | (aw_raw > 400)
        habitat_water = water_here | in_pool
        f_medium = np.where(habitat_water, 1.0, 0.05).astype(np.float32)
        # Si están en río o poza dulce, la salinidad efectiva es ~0 (agua dulce).
        eff_salt = np.where(in_pool & ~water_here, 0.0, salt_here)
        f_salt = clip01(1.0 - np.maximum(eff_salt - p["salt_tolerance"], 0.0) / 30.0)
    else:
        f_medium = np.where(water_here, 0.02, 1.0).astype(np.float32)
        # Sequía: la disponibilidad de agua limita a los terrestres
        f_medium = f_medium * clip01(aw + 0.15)
        f_salt = clip01(1.0 - np.maximum(salt_here - p["salt_tolerance"], 0.0) / 30.0)

    factors = [f_temp, f_salt, f_medium]

    if pop.template.kingdom in ("plantae", "fungi"):
        # pH: entre min_ph y max_ph
        ph_opt = (p["max_ph"] + p["min_ph"]) * 0.5
        ph_sigma = np.maximum((p["max_ph"] - p["min_ph"]) * 0.4, 0.3)
        f_ph = gaussian_tolerance(ph, ph_opt, ph_sigma)
        factors.append(f_ph)
        # Nutrientes: fertilidad importa mucho a plantas, algo a hongos
        weight = 0.9 if pop.template.kingdom == "plantae" else 0.5
        f_nut = clip01(fert * weight + (1.0 - weight))
        factors.append(f_nut)

    if pop.template.kingdom == "chordata":
        # Comida disponible según dieta (biomasa local de las categorías).
        # Se lee de un promedio 5x5 alrededor del individuo — modelo simple
        # de forrajeo: el animal recorre metros a la redonda para comer,
        # así que agotar la celda exacta no le mata al instante.
        diet = np.stack([
            p["diet_bug"], p["diet_meat"], p["diet_vegetal"],
            p["diet_fish"], p["diet_micro"],
        ], axis=1)
        diet = diet / np.maximum(diet.sum(axis=1, keepdims=True), 1e-6)

        # Preferir los campos ya promediados por el simulador (una vez por
        # tick); si no están (p.ej. tests que llaman suitability directo),
        # calcularlos al vuelo como fallback.
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
        # Piso 0.1 para no colapsar suit a cero por comida escasa (los
        # animales tienen reservas). Aún así, sin apenas comida f_food<<1.
        f_food = clip01((diet * food_sources).sum(axis=1) * 1.5 + 0.1)
        factors.append(f_food)

    return liebig_min(*factors).astype(np.float32)
