"""Utilidades numéricas compartidas."""
from __future__ import annotations
import numpy as np


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def clip100(x):
    return np.clip(x, 0.0, 100.0)


def gaussian_tolerance(value, optimum, sigma):
    """Curva de tolerancia gaussiana normalizada 0..1."""
    sigma = np.maximum(np.asarray(sigma, dtype=np.float32), 1e-6)
    return np.exp(-0.5 * ((value - optimum) / sigma) ** 2)


def rng_from_seed(seed: int | None):
    return np.random.default_rng(seed)


def liebig_min(*factors):
    """Ley del mínimo de Liebig: el factor limitante manda."""
    stacked = np.stack(factors, axis=0)
    return stacked.min(axis=0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def soft_gate(value, threshold, softness,
              floor: float = 0.05, ramp_start: float | None = None):
    """Compuerta blanda (N3): recompensa parcial ANTES del umbral duro.

    Antes: sigmoid((v - t)/s) daba salto brusco → landscape con valle
    fitness intransitable → los gated traits eran dead-code evolutivo.

    Ahora: la puerta arranca en `floor` (5% por defecto) desde `ramp_start`
    y sube linealmente hasta el sigmoide clásico en el umbral. Imita la
    evolución gradual del ojo (Nilsson & Pelger 1994).
    """
    softness = max(float(softness), 1e-3)
    if ramp_start is None:
        ramp_start = threshold - 3.0 * softness
    ramp_start = float(ramp_start)
    sig = sigmoid((np.asarray(value, dtype=np.float32) - threshold) / softness)
    below = np.asarray(value, dtype=np.float32) < threshold
    ramp = clip01((np.asarray(value, dtype=np.float32) - ramp_start)
                  / max(threshold - ramp_start, 1e-3))
    partial = floor + (1.0 - floor) * ramp * 0.5   # hasta 0.55 al llegar al umbral
    return np.where(below, np.maximum(partial, sig), sig).astype(np.float32)
