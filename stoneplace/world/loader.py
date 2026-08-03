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
    return World(meta=meta, layers=layers)
