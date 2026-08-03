"""Biomasa de microfauna / decomponedores como campo raster (no individuos).

En vez de simular cada insecto, plancton o microbio, se llevan campos de
biomasa por celda que crecen en función de:
  - biomasa vegetal local (fotosíntesis → herbívoros → insectos)
  - humedad y temperatura
  - materia orgánica del suelo

Y decrecen cuando los animales las comen (consumo desde la ecología de
chordata). Se regeneran con tasa logística.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from ..utils import clip01, gaussian_tolerance
from ..world.biomes import (OCEAN, COAST, FRESHWATER, DESERT, GLACIER,
                             TROPICAL_FOREST, SAVANNA, TEMPERATE_FOREST,
                             TEMPERATE_GRASS, BOREAL, TUNDRA)

MICRO_LAYERS = ("bugs", "microbes", "plankton", "detritus")


@dataclass
class MicrofaunaField:
    """Campos de biomasa 0..1 por celda para cada categoría de microfauna."""
    bugs: np.ndarray       # insectos / artrópodos terrestres
    microbes: np.ndarray   # microbios del suelo (descomponedores)
    plankton: np.ndarray   # productores/consumidores acuáticos microscópicos
    detritus: np.ndarray   # materia orgánica muerta acumulada

    @classmethod
    def zeros(cls, shape) -> "MicrofaunaField":
        return cls(
            bugs=np.zeros(shape, np.float32),
            microbes=np.zeros(shape, np.float32),
            plankton=np.zeros(shape, np.float32),
            detritus=np.zeros(shape, np.float32),
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"bugs": self.bugs, "microbes": self.microbes,
                "plankton": self.plankton, "detritus": self.detritus}


def carrying_capacity(world, biomes: np.ndarray, plant_biomass: np.ndarray,
                       fungi_biomass: np.ndarray) -> dict[str, np.ndarray]:
    """K por categoría en cada celda (rango 0..1).

    - bugs      ~ f(plantas, T, humedad)
    - microbes  ~ f(materia orgánica del suelo + detritus)
    - plankton  ~ f(agua, luz-latitud, nutrientes de escorrentía)
    - detritus  ~ f(plantas + hongos: cae hojarasca)
    """
    temp = world.layers["temperature"]
    hum = world.layers["humidity"]
    om = world.layers["organic_matter"] + world.layers.get("om_potential",
                                                          np.zeros_like(temp)) * 0.3
    water = world.layers["elevation"] <= 0
    salt = world.layers.get("water_salinity", np.zeros_like(temp))
    runoff = world.layers.get("runoff", np.zeros_like(temp)) / 800.0

    # Bugs: sólo tierra emergida con vegetación y clima cálido-húmedo.
    # v1.5.7 — pesos elevados: los insectos son la mayor biomasa animal
    # real del planeta. sigma de temperatura amplia (16 vs 12) — hay bugs
    # activos también en climas fríos (springtails, dipteros). Bosques
    # nativos rebalanceados con Quercus como hospedador principal.
    temp_bugs = gaussian_tolerance(temp, 22.0, 16.0)
    K_bugs = clip01(plant_biomass * 1.4 + fungi_biomass * 0.35) * temp_bugs * clip01(hum + 0.25)
    K_bugs = np.where(water, 0.0, K_bugs)
    boost = np.isin(biomes, [TROPICAL_FOREST, SAVANNA, TEMPERATE_FOREST])
    K_bugs = np.where(boost, K_bugs * 1.3, K_bugs)
    K_bugs = np.where(np.isin(biomes, [DESERT, GLACIER]), K_bugs * 0.20, K_bugs)
    K_bugs = np.where(np.isin(biomes, [TUNDRA]), K_bugs * 0.35, K_bugs)  # ya no tan hostil

    # Microbios: dependen del suelo (om + detritus latente)
    K_microbes = clip01(om * 0.9 + gaussian_tolerance(temp, 20.0, 15.0) * 0.4)
    K_microbes = np.where(water, K_microbes * 0.3, K_microbes)

    # Plankton: sólo agua; mejor en zonas templadas y con aportes de escorrentía
    K_plankton = np.where(water,
                          clip01(gaussian_tolerance(temp, 18.0, 12.0)
                                 + clip01(runoff) * 0.4
                                 - clip01(salt / 60.0) * 0.2),
                          0.0)

    # Detritus: aportes de plantas + hongos, más en climas fríos (descomposición lenta)
    slow_decomp = gaussian_tolerance(temp, 5.0, 10.0)
    K_detritus = clip01(plant_biomass * 0.5 + fungi_biomass * 0.4
                        + slow_decomp * 0.3)
    return {"bugs": K_bugs, "microbes": K_microbes,
            "plankton": K_plankton, "detritus": K_detritus}


def logistic_step(field: MicrofaunaField, K: dict[str, np.ndarray],
                  r: float = 2.5, dt_years: float = 1.0):
    """Crecimiento logístico dB/dt = r * B * (1 - B/K).

    r sube a 2.5/año — los insectos y microbios se reponen mucho más
    rápido de lo que se agota una celda por presión de forrajeo, así que
    aunque un enjambre limpie un tile, se rellena en semanas del mundo.
    """
    for name in MICRO_LAYERS:
        B = getattr(field, name)
        Kv = np.maximum(K[name], 1e-6)
        dB = r * B * (1.0 - B / Kv) * dt_years
        # Semilla mínima donde K > 0 pero B = 0 (colonización microbiana).
        seed = (B < 1e-3) & (Kv > 0.02)
        dB = np.where(seed, 0.05 * dt_years + Kv * 0.15, dB)
        setattr(field, name, np.clip(B + dB, 0.0, 1.0).astype(np.float32))
