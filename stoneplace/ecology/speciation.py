"""Detección de especiación por divergencia genética + aislamiento espacial.

Cada N años se toma la población de una especie y se agrupa por k-means
sobre los loci aditivos + coordenadas. Si dos clusters superan un umbral
de distancia genética (Fst proxy) Y llevan tiempo separados
espacialmente, se declara una nueva especie hija.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from ..species.base import SpeciesPopulation, SpeciesTemplate
from ..genetics.genome import Genome


def _kmeans2(X: np.ndarray, rng, n_iter: int = 8):
    """K-means barato para k=2 (evita sklearn por dependencia)."""
    n = X.shape[0]
    if n < 20:
        return np.zeros(n, dtype=np.int32), None
    idx = rng.choice(n, size=2, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        d0 = ((X - centers[0]) ** 2).sum(axis=1)
        d1 = ((X - centers[1]) ** 2).sum(axis=1)
        labels = (d1 < d0).astype(np.int32)
        for k in (0, 1):
            m = labels == k
            if m.any():
                centers[k] = X[m].mean(axis=0)
    return labels, centers


def try_speciate(pop: SpeciesPopulation, world, current_year: float,
                 rng, fst_threshold: float = 0.18) -> SpeciesPopulation | None:
    """Devuelve una nueva SpeciesPopulation hija o None si no hay divergencia."""
    if pop.n < 60:
        return None
    additive = pop.genome.additive()             # (N, L)
    coords = np.stack([pop.y / world.height, pop.x / world.width], axis=1) * 40.0
    X = np.concatenate([additive, coords], axis=1)
    labels, _ = _kmeans2(X, rng)
    if labels is None:
        return None
    m0 = labels == 0; m1 = labels == 1
    if m0.sum() < 30 or m1.sum() < 30:
        return None
    # Divergencia genética: distancia entre medias de loci / rango
    mean0 = additive[m0].mean(axis=0)
    mean1 = additive[m1].mean(axis=0)
    fst_proxy = np.mean(np.abs(mean0 - mean1)) / 100.0
    if fst_proxy < fst_threshold:
        return None
    # Aislamiento espacial: baja densidad de individuos entre clusters
    dy = pop.y[m0].mean() - pop.y[m1].mean()
    dx = pop.x[m0].mean() - pop.x[m1].mean()
    dist = float(np.hypot(dy, dx))
    if dist < 15.0:
        return None

    # Fabricar la hija con el cluster minoritario
    child_mask = m1 if m1.sum() < m0.sum() else m0
    parent_template = pop.template
    child_template = SpeciesTemplate(
        species_id=_new_species_id(parent_template),
        scientific_name=f"{parent_template.scientific_name} {chr(65 + len(parent_template.children))}",
        taxonomy=list(parent_template.taxonomy),
        kingdom=parent_template.kingdom,
        parent=parent_template.species_id,
        born_at=current_year,
        n_loci=parent_template.n_loci,
        n_chromosomes=parent_template.n_chromosomes,
        gp_map=parent_template.gp_map,        # comparte matriz G→P
        hints=dict(parent_template.hints),
    )
    parent_template.children.append(child_template.species_id)

    child_pop = SpeciesPopulation(
        template=child_template,
        genome=Genome(
            alleles=pop.genome.alleles[child_mask].copy(),
            chromosomes=pop.genome.chromosomes.copy()),
        y=pop.y[child_mask].copy(),
        x=pop.x[child_mask].copy(),
        age_years=pop.age_years[child_mask].copy(),
        energy=pop.energy[child_mask].copy(),
        sex=(pop.sex[child_mask].copy() if pop.sex is not None else None),
    )
    # Quitar los individuos de la especie madre
    pop.kill_mask(~child_mask)
    return child_pop


_LAST_ID = [10_000]


def _new_species_id(parent_template: SpeciesTemplate) -> int:
    _LAST_ID[0] += 1
    return _LAST_ID[0]
