"""Campos de biomasa vegetal y fúngica.

P7 fix: la biomasa vegetal ahora es PERSISTENTE (un `PlantBiomassStore`
mantenido por el simulador). Cada tick se regenera parcialmente en
función de las plantas vivas, pero el consumo animal la mantiene
consumida. Sin esto los herbívoros nunca competían con las plantas.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PlantBiomassStore:
    """Capa persistente de biomasa vegetal consumible por herbívoros."""
    biomass: np.ndarray                       # (H, W) 0..1

    @classmethod
    def zeros(cls, shape) -> "PlantBiomassStore":
        return cls(biomass=np.zeros(shape, dtype=np.float32))

    def regenerate(self, target: np.ndarray, r: float, dt: float):
        """Logística hacia `target` (biomasa potencial dada por las plantas vivas).

        dB/dt = r * B * (1 - B/target) + seed_if_empty
        """
        B = self.biomass
        T = np.maximum(target, 1e-6)
        dB = r * B * (1.0 - B / T) * dt
        seed = (B < 1e-3) & (T > 0.02)
        dB = np.where(seed, T * 0.15, dB)
        self.biomass = np.clip(B + dB, 0.0, 1.0).astype(np.float32)

    def consume(self, y, x, amount: np.ndarray):
        """Resta consumo animal celda a celda (in-place)."""
        consume_map = np.zeros_like(self.biomass)
        np.add.at(consume_map, (y, x), amount)
        self.biomass = np.clip(
            self.biomass - consume_map, 0.0, 1.0
        ).astype(np.float32)


def biomass_field(pops, world, kingdom: str) -> np.ndarray:
    """Biomasa potencial (target) según las poblaciones vivas del reino.

    Se sigue usando como "target" al que aspira el store persistente.
    """
    h, w = world.shape
    field = np.zeros((h, w), dtype=np.float32)
    for pop in pops:
        if pop.template.kingdom != kingdom or pop.n == 0:
            continue
        mass = pop.phenotype["mass_g"] / 5000.0
        mass = np.clip(mass, 0.0, 4.0)
        np.add.at(field, (pop.y, pop.x), mass)
    return np.clip(field / 20.0, 0.0, 1.0)


def meat_field(pops, world) -> np.ndarray:
    h, w = world.shape
    field = np.zeros((h, w), dtype=np.float32)
    for pop in pops:
        if pop.template.kingdom != "chordata" or pop.n == 0:
            continue
        if pop.template.hints.get("aquatic", False):
            continue
        mass = pop.phenotype["mass_g"] / 2000.0
        np.add.at(field, (pop.y, pop.x), mass)
    return np.clip(field / 40.0, 0.0, 1.0)


def fish_field(pops, world) -> np.ndarray:
    h, w = world.shape
    field = np.zeros((h, w), dtype=np.float32)
    for pop in pops:
        if pop.template.kingdom != "chordata" or pop.n == 0:
            continue
        if not pop.template.hints.get("aquatic", False):
            continue
        mass = pop.phenotype["mass_g"] / 100.0
        np.add.at(field, (pop.y, pop.x), mass)
    return np.clip(field / 40.0, 0.0, 1.0)
