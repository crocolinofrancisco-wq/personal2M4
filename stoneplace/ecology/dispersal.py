"""Dispersión de individuos y semillas por el raster.

- Plantas y hongos: difusión gaussiana modulada por seed_resistance_level.
- Animales: caminata aleatoria SESGADA por idoneidad (P2 fix), con coste
  metabólico y probabilidad de moverse dependiente del fenotipo.
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


def _biased_choice(scores: np.ndarray, rng) -> np.ndarray:
    """Muestreo categórico vectorizado — scores (N, K) → índices (N,)."""
    s = np.maximum(scores, 1e-6)
    # Endurece un poco el sesgo hacia lo bueno sin volverlo determinista.
    s = s ** 2.0
    s = s / s.sum(axis=1, keepdims=True)
    cum = np.cumsum(s, axis=1)
    u = rng.random(scores.shape[0])[:, None]
    return (u < cum).argmax(axis=1).astype(np.int32)


def animal_walk(pop, world, biome_map, micro_dict, rng, step: int = 1):
    """Caminata aleatoria sesgada por idoneidad (biased random walk).

    P2 fix: en vez de hill-climbing determinista, cada animal elige una
    de las 9 celdas locales con probabilidad ∝ suitability². Esto:
      - evita que TODOS converjan a un mismo pozo (dispersión emergente),
      - paga coste metabólico proporcional a la distancia efectiva,
      - respeta rasgos: los sedentarios se mueven menos (p_move bajo).
    Sigue siendo vectorizado — 9 llamadas a suitability y un choice.
    """
    from .habitat import suitability
    if pop.n == 0:
        return
    h, w = world.shape
    orig_y = pop.y.copy(); orig_x = pop.x.copy()

    # Probabilidad de intentar movimiento este tick según el fenotipo.
    p = pop.phenotype
    if "speed" in p:
        speed = p["speed"] / 100.0
    else:
        speed = np.full(pop.n, 0.5, dtype=np.float32)
    fly = p.get("flying_level", np.zeros(pop.n, dtype=np.float32)) / 100.0
    swim = p.get("swiming_level", np.zeros(pop.n, dtype=np.float32)) / 100.0
    mobility = np.clip(0.25 + 0.5 * speed + 0.25 * np.maximum(fly, swim), 0.1, 1.0)
    moves_mask = rng.random(pop.n) < mobility

    # Puntajes de las 9 celdas (incluyendo la propia)
    scores = np.empty((pop.n, 9), dtype=np.float32)
    ys = np.empty((pop.n, 9), dtype=np.int32)
    xs = np.empty((pop.n, 9), dtype=np.int32)
    k = 0
    for dy in (-step, 0, step):
        for dx in (-step, 0, step):
            ny = (orig_y + dy) % h
            nx = (orig_x + dx) % w
            pop.y = ny; pop.x = nx
            scores[:, k] = suitability(pop, world, biome_map, micro_dict)
            ys[:, k] = ny; xs[:, k] = nx
            k += 1

    choice = _biased_choice(scores, rng)
    idx = np.arange(pop.n)
    new_y = ys[idx, choice]
    new_x = xs[idx, choice]

    # Los sedentarios se quedan donde estaban
    new_y = np.where(moves_mask, new_y, orig_y)
    new_x = np.where(moves_mask, new_x, orig_x)

    pop.y = new_y.astype(np.int32)
    pop.x = new_x.astype(np.int32)

    # Coste energético del movimiento (Kleiber-lite): sólo los que se movieron
    moved = (new_y != orig_y) | (new_x != orig_x)
    if moved.any() and hasattr(pop, "energy"):
        cost = np.where(moved, 0.015 + 0.020 * speed, 0.0).astype(np.float32)
        pop.energy = np.clip(pop.energy - cost, 0.0, 1.0)
