"""Campos agregados de biomasa vegetal y fúngica para alimentar la microfauna
y la dieta de los animales.

Cada tick se recomputan a partir de las especies vivas (contando individuos
por celda ponderados por su masa y edad relativa).
"""
from __future__ import annotations
import numpy as np


def biomass_field(pops, world, kingdom: str) -> np.ndarray:
    """Suma la biomasa (masa_g escalada) de todas las poblaciones del reino."""
    h, w = world.shape
    field = np.zeros((h, w), dtype=np.float32)
    for pop in pops:
        if pop.template.kingdom != kingdom or pop.n == 0:
            continue
        mass = pop.phenotype["mass_g"] / 5000.0    # escala a ~0..1 para 5 kg
        mass = np.clip(mass, 0.0, 4.0)
        np.add.at(field, (pop.y, pop.x), mass)
    return np.clip(field / 20.0, 0.0, 1.0)


def meat_field(pops, world) -> np.ndarray:
    """Biomasa animal disponible como presa (para dieta 'meat')."""
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
    """Biomasa animal acuática (dieta 'fish')."""
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
