"""Generador de mundo sintético para tests/smoke sin data raster real.

Produce un `World` totalmente válido (mismas capas que `loader.load_world`
lee del `.bin`) hecho a mano con NumPy: continentes gaussianos, gradiente
climático latitudinal, ríos y variables edáficas. Sirve para probar la
simulación en CI o en cualquier equipo sin bajar los mapas DDG completos.
"""
from __future__ import annotations
import numpy as np
from .loader import World


def make_synthetic_world(width: int = 128, height: int = 64,
                        seed: int = 0) -> World:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    # Dos continentes gaussianos + islas: elevation en metros (-4000..4000)
    def bump(cy, cx, sy, sx, amp):
        return amp * np.exp(-((yy - cy) ** 2) / (2 * sy ** 2)
                            - ((xx - cx) ** 2) / (2 * sx ** 2))

    elev = (bump(height * 0.35, width * 0.25, 12, 18, 1800)
            + bump(height * 0.6, width * 0.7, 14, 22, 2200)
            + bump(height * 0.5, width * 0.5, 6, 8, 400)
            - 800  # mar dominante
            + rng.normal(0, 120, size=(height, width)))
    elev = elev.astype(np.float32)

    # Temperatura: baja hacia los polos, un pelo más frío en altura
    lat_norm = (yy / max(height - 1, 1)) * 2 - 1        # -1..1
    temp = 28.0 - 45.0 * lat_norm ** 2 - np.clip(elev, 0, None) * 0.005
    temp += rng.normal(0, 1.2, size=(height, width))
    temp = temp.astype(np.float32)

    # Lluvia: bastante en trópicos (ecuador) y costas, seco en el interior
    rains = 1600 * np.exp(-lat_norm ** 2 * 3.0) + rng.normal(0, 100, (height, width))
    # Sombra de lluvia hacia el interior de continentes altos
    interior_pen = np.clip(elev / 800.0, 0, 3) ** 1.4
    rains = np.clip(rains - interior_pen * 250, 20, 3500).astype(np.float32)

    humidity = np.clip(rains / 2500.0 + rng.normal(0, 0.05, (height, width)),
                       0.05, 1.0).astype(np.float32)

    # Salinidad del agua: mar ~35 PSU, bahías interiores más bajas
    salt = np.where(elev <= 0, 32 + rng.normal(0, 2, (height, width)),
                    rng.uniform(0.0, 0.4, (height, width))).astype(np.float32)
    # Franjas costeras con agua algo menos salada
    coast_band = (elev > -50) & (elev <= 20)
    salt = np.where(coast_band, np.clip(salt - 6, 0, 40), salt).astype(np.float32)

    # Ríos: cauce que baja de los continentes al mar (flujo pequeño distribuido)
    river = np.clip(
        rng.gamma(0.4, 0.3, size=(height, width))
        * np.where((elev > 0) & (elev < 900), 1.0, 0.05),
        0, 1.0,
    ).astype(np.float32)

    # Agua disponible en el suelo: en función de lluvia, cerca de ríos y agua
    available_water = np.clip(rains * 0.6 + river * 300, 0, 2500).astype(np.float32)

    # Fertilidad: alta en llanuras templadas, baja en desiertos y hielo
    fertility = np.clip(
        0.4 + 0.35 * np.exp(-((temp - 18) ** 2) / (2 * 12 ** 2))
        + 0.20 * (rains / 2000.0)
        - 0.15 * (np.abs(temp - 18) > 25),
        0.05, 0.95,
    ).astype(np.float32)

    # pH edáfico (5..8, algo neutro)
    soil_ph = np.clip(6.2 + rng.normal(0, 0.4, (height, width))
                      - 0.2 * (rains > 1800),
                      3.5, 9.0).astype(np.float32)

    # Materia orgánica: alta en zonas frías-húmedas
    om = np.clip(0.4 * humidity + 0.4 * np.exp(-((temp - 10) ** 2) / 200)
                 + rng.normal(0, 0.05, (height, width)),
                 0.0, 1.0).astype(np.float32)
    om_potential = (om * 1.2).astype(np.float32)
    runoff = (rains * 0.5 * (elev > 0)).astype(np.float32)

    layers = {
        "elevation": elev,
        "temperature": temp,
        "rains": rains,
        "humidity": humidity,
        "water_salinity": salt,
        "riverFlow": river,
        "available_water": available_water,
        "fertility": fertility,
        "soil_ph": soil_ph,
        "organic_matter": om,
        "om_potential": om_potential,
        "runoff": runoff,
    }
    meta = {
        "width": width, "height": height, "seaLevelNorm": 0.5,
        "synthetic": True, "seed": seed,
    }
    return World(meta=meta, layers=layers)
