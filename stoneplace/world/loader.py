"""Cargador del raster mundial DDG (formato ddg-world-layers/1).

Lee los 24 canales del `.bin` con memmap y expone cada capa como un
`np.ndarray` 2D re-escalado a sus unidades reales. Sin copias hasta que se
solicita explícitamente.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class World:
    meta: dict
    layers: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    @property
    def height(self) -> int:
        return int(self.meta["height"])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def sea_level(self) -> float:
        return float(self.meta.get("seaLevelNorm", 0.5))

    def land_mask(self) -> np.ndarray:
        """True donde hay tierra emergida."""
        return self.layers["elevation"] > 0

    def water_mask(self) -> np.ndarray:
        return self.layers["elevation"] <= 0

    def coastal_mask(self, kernel: int = 1) -> np.ndarray:
        """Píxeles de tierra con al menos un vecino de agua (útil para triops)."""
        land = self.land_mask()
        water = ~land
        h, w = land.shape
        coast = np.zeros_like(land)
        for dy in range(-kernel, kernel + 1):
            for dx in range(-kernel, kernel + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(water, dy, axis=0), dx, axis=1)
                coast |= land & shifted
        return coast


def _ensure_derived_layers(layers: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Completa las capas que la ecología necesita pero un `.bin` real de
    SeedWorld puede no traer (formato "prebiótico": sólo sustrato +
    fertilidad, sin agua/suelo biológico todavía).

    Sólo rellena claves AUSENTES — un mundo sintético que ya trae todo
    (`rains`, `water_salinity`, `soil_ph`, `available_water`, `runoff`,
    `organic_matter`) pasa sin cambios. Cada derivación usa la capa real
    más cercana disponible en vez de inventar un número sin base:
      - rains          ← `rainfall` (0..1 relativo) escalado a mm/año.
      - water_salinity ← `material` (ids 0-3 = fondo oceánico real en el
                          esquema ddg-world-layers/1) o elevación si no
                          hay `material`.
      - soil_ph        ← `weathering` (régimen de meteorización: la
                          lixiviación química intensa acidifica, la árida
                          alcaliniza) + penalización por lluvia alta.
      - organic_matter ← ~0 (mundo prebiótico: el detritus se construye
                          con el tiempo a partir de la biomasa sembrada),
                          con un pequeño baseline por humedad/fertilidad.
      - available_water← `rains` + `riverFlow` + `soilDepth` (retención).
      - runoff         ← `rains` * pendiente normalizada (`slope`), si
                          está disponible; más pendiente = más escorrentía.
    """
    elev = layers["elevation"]

    if "rains" not in layers:
        if "rainfall" in layers:
            layers["rains"] = (layers["rainfall"] * 2500.0).astype(np.float32)
        else:
            layers["rains"] = np.full_like(elev, 600.0, dtype=np.float32)
    rains = layers["rains"]

    if "water_salinity" not in layers:
        water = elev <= 0
        if "material" in layers:
            # 0 océano abisal, 1 océano, 2 plataforma continental, 3 fosa
            ocean_material = np.isin(layers["material"].astype(np.int32),
                                     [0, 1, 2, 3])
            salt = np.where(ocean_material, 34.0, 0.0)
        else:
            salt = np.where(water, 34.0, 0.0)
        layers["water_salinity"] = salt.astype(np.float32)

    if "soil_ph" not in layers:
        base_ph = np.full_like(elev, 6.8, dtype=np.float32)
        if "weathering" in layers:
            wreg = layers["weathering"].astype(np.int32)
            # 0 glaciar, 1 periglaciar, 2 mecánica árida, 3 química
            # moderada, 4 química intensa (lixiviación → ácido), 5 débil
            ph_by_regime = {0: 6.5, 1: 6.3, 2: 7.8, 3: 6.5, 4: 5.3, 5: 7.2}
            conds = [wreg == k for k in ph_by_regime]
            choices = [np.full_like(elev, v, dtype=np.float32)
                      for v in ph_by_regime.values()]
            base_ph = np.select(conds, choices, default=base_ph)
        acidify = np.clip(rains / 3000.0, 0.0, 1.0)
        layers["soil_ph"] = np.clip(base_ph - acidify, 4.0, 9.0).astype(np.float32)

    if "organic_matter" not in layers:
        hum = layers.get("humidity", np.full_like(elev, 0.4))
        fert = layers.get("fertility", np.full_like(elev, 0.4))
        layers["organic_matter"] = np.clip(
            hum * 0.15 + fert * 0.05, 0.0, 1.0).astype(np.float32)
    if "om_potential" not in layers:
        layers["om_potential"] = layers["organic_matter"].copy()

    if "available_water" not in layers:
        river = layers.get("riverFlow", np.zeros_like(elev))
        soil_depth = layers.get("soilDepth", np.full_like(elev, 1.0))
        layers["available_water"] = np.clip(
            rains * 0.6 + river * 400.0 + soil_depth * 30.0,
            0.0, 3000.0).astype(np.float32)

    if "runoff" not in layers:
        if "slope" in layers:
            slope_norm = np.clip(layers["slope"] / 200.0, 0.0, 1.0)
        else:
            slope_norm = np.full_like(elev, 0.3, dtype=np.float32)
        layers["runoff"] = (rains * (0.2 + 0.6 * slope_norm)).astype(np.float32)

    return layers


def load_world(bin_path: str | Path, json_path: str | Path) -> World:
    bin_path = Path(bin_path)
    json_path = Path(json_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    w, h = int(meta["width"]), int(meta["height"])
    raw = np.memmap(bin_path, dtype=np.uint8, mode="r")

    dtype_map = {"int16": np.int16, "uint16": np.uint16, "uint8": np.uint8}
    layers: dict[str, np.ndarray] = {}
    for spec in meta["layers"]:
        name = spec["name"]
        dtype = dtype_map[spec["dtype"]]
        offset = int(spec["offset"])
        nbytes = int(spec["bytes"])
        buf = raw[offset:offset + nbytes]
        arr = np.frombuffer(buf, dtype=dtype).reshape(h, w).astype(np.float32)
        scale = float(spec.get("scale", 1.0))
        if scale != 1.0:
            arr = arr * scale
        layers[name] = arr
    layers = _ensure_derived_layers(layers)
    return World(meta=meta, layers=layers)
