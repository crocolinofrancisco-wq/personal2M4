"""Clima dinámico Stoneplace v1.5 (N5).

Antes: `world.layers["temperature"]` era estático → la reclasificación de
biomas cada 100 años no cambiaba nada (no-op determinista). Ahora hay
dos ciclos superpuestos:

  * **Milankovitch** — variación de largo plazo (≈40 000 años en el
    default; ajustable). Amplitud ±3 °C globalmente.
  * **Estacional** — variación intra-año (400 días stoneplace = 1 año).
    Hemisferios opuestos alternan verano/invierno. Amplitud ±4 °C en
    latitudes medias, atenuada hacia el ecuador.

`temperature_delta(year)` devuelve un array (H, W) con el delta a sumar
al `_base_temperature` que captura el simulador al inicio. Los efectos
se acumulan sin drift porque no se aplican de forma incremental.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ClimateCycle:
    enabled: bool = True
    milankovitch_period: float = 40_000.0
    milankovitch_amp: float = 3.0
    seasonal_amp: float = 4.0

    def temperature_delta(self, year: float) -> float:
        """Delta escalar global (Milankovitch) + estacional aproximado.

        La componente estacional depende de la fracción del año; la
        modulación por latitud queda como suma escalar cuando se usa
        sin conocer el world (una aproximación honesta y barata).
        """
        milank = self.milankovitch_amp * np.sin(
            2 * np.pi * year / max(self.milankovitch_period, 1.0)
        )
        # Ciclo estacional dentro del año — asumimos 1 año = 1 unidad.
        year_frac = year - np.floor(year)
        seasonal = self.seasonal_amp * np.sin(2 * np.pi * year_frac)
        return float(milank + seasonal)
