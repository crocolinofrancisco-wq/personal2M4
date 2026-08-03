"""Dispersión de individuos y semillas por el raster.

- Plantas y hongos: dispersión por difusión aleatoria (semillas y esporas).
- Animales: caminata aleatoria sesgada por idoneidad de las 8 celdas vecinas.
"""
from __future__ import annotations
import numpy as np


def wrap(coord, size, seamless=True):
    if seamless:
        return coord % size
    return np.clip(coord, 0, size - 1)


def disperse_seeds(y, x, world, rng, sigma_cells: float = 2.0):
    """Añade ruido gaussiano de dispersión al punto de nacimiento."""
    h, w = world.shape
    dy = rng.normal(0.0, sigma_cells, size=y.shape).round().astype(np.int32)
    dx = rng.normal(0.0, sigma_cells, size=x.shape).round().astype(np.int32)
    ny = wrap(y.astype(np.int32) + dy, h)
    nx = wrap(x.astype(np.int32) + dx, w)
    return ny, nx


def animal_walk(pop, world, biome_map, micro_dict, rng, step: int = 1):
    """Camina cada animal a la mejor de sus 9 celdas locales (incl. quedarse)."""
    from .habitat import suitability
    if pop.n == 0:
        return
    h, w = world.shape
    best_y = pop.y.copy(); best_x = pop.x.copy()
    best_score = suitability(pop, world, biome_map, micro_dict)
    # Guardar coords originales para restaurarlas al probar cada offset
    orig_y = pop.y.copy(); orig_x = pop.x.copy()
    for dy in (-step, 0, step):
        for dx in (-step, 0, step):
            if dy == 0 and dx == 0:
                continue
            ny = (orig_y + dy) % h
            nx = (orig_x + dx) % w
            pop.y = ny; pop.x = nx
            score = suitability(pop, world, biome_map, micro_dict)
            better = score > best_score + 1e-4
            best_y = np.where(better, ny, best_y)
            best_x = np.where(better, nx, best_x)
            best_score = np.where(better, score, best_score)
    # Un poco de ruido para evitar que se estanquen todos en un óptimo local
    jitter_y = rng.integers(-1, 2, size=best_y.shape)
    jitter_x = rng.integers(-1, 2, size=best_x.shape)
    pop.y = ((best_y + jitter_y) % h).astype(np.int32)
    pop.x = ((best_x + jitter_x) % w).astype(np.int32)
