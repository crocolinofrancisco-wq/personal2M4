"""Clasificación emergente de biomas tipo Whittaker (T, precipitación).

No se lee de una tabla fija: se calcula por celda a partir de las capas del
raster y se re-evalúa cada vez que cambia el clima. Es el mapa de habitats
donde caen semillas y sobre el que se calcula la biomasa de microfauna.
"""
from __future__ import annotations
import numpy as np

# IDs de bioma (código estable para exports y stats)
OCEAN = 0
COAST = 1
DESERT = 2
TUNDRA = 3
BOREAL = 4
TEMPERATE_GRASS = 5
TEMPERATE_FOREST = 6
SAVANNA = 7
TROPICAL_FOREST = 8
ALPINE = 9
GLACIER = 10
FRESHWATER = 11

BIOME_NAMES = {
    OCEAN: "océano", COAST: "costa", DESERT: "desierto", TUNDRA: "tundra",
    BOREAL: "bosque boreal", TEMPERATE_GRASS: "pradera templada",
    TEMPERATE_FOREST: "bosque templado", SAVANNA: "sabana",
    TROPICAL_FOREST: "selva tropical", ALPINE: "alpino",
    GLACIER: "glaciar", FRESHWATER: "agua dulce",
}


def classify(world) -> np.ndarray:
    """Devuelve un array (H,W) uint8 con el bioma de cada celda."""
    elev = world.layers["elevation"]
    temp = world.layers["temperature"]
    rain = world.layers["rains"]
    salt = world.layers.get("water_salinity", np.zeros_like(elev))

    out = np.full(elev.shape, OCEAN, dtype=np.uint8)
    water = elev <= 0
    land = ~water

    # Agua
    fresh = water & (salt < 0.5)
    out[water] = OCEAN
    out[fresh] = FRESHWATER

    # Tierra: primero extremos térmicos
    glacier = land & (temp < -15)
    tundra = land & (temp >= -15) & (temp < 0)
    alpine = land & (elev > 2500) & (temp < 5)
    out[glacier] = GLACIER
    out[tundra] = TUNDRA
    out[alpine] = ALPINE

    # Resto por Whittaker simplificado
    warm = land & ~glacier & ~tundra & ~alpine
    hot = warm & (temp >= 20)
    temperate = warm & (temp >= 5) & (temp < 20)
    cold = warm & (temp >= 0) & (temp < 5)

    out[cold] = BOREAL
    out[temperate & (rain < 500)] = TEMPERATE_GRASS
    out[temperate & (rain >= 500)] = TEMPERATE_FOREST
    out[hot & (rain < 300)] = DESERT
    out[hot & (rain >= 300) & (rain < 1200)] = SAVANNA
    out[hot & (rain >= 1200)] = TROPICAL_FOREST

    # Costa (sobrescribe una franja de tierra baja pegada al mar)
    coast = world.coastal_mask(1) & (elev < 50)
    out[coast] = COAST
    return out
