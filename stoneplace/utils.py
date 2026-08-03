"""Utilidades numéricas compartidas."""
from __future__ import annotations
import numpy as np


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def clip100(x):
    return np.clip(x, 0.0, 100.0)


def gaussian_tolerance(value, optimum, sigma):
    """Curva de tolerancia gaussiana normalizada 0..1.

    Se usa por todos los módulos ecológicos: mide qué tan cerca está una
    variable ambiental (temperatura, pH, salinidad, …) del óptimo del taxón.
    `sigma` puede ser escalar o array (uno por individuo).
    """
    sigma = np.maximum(np.asarray(sigma, dtype=np.float32), 1e-6)
    return np.exp(-0.5 * ((value - optimum) / sigma) ** 2)


def rng_from_seed(seed: int | None):
    return np.random.default_rng(seed)


def liebig_min(*factors):
    """Ley del mínimo de Liebig: el factor limitante manda."""
    stacked = np.stack(factors, axis=0)
    return stacked.min(axis=0)
