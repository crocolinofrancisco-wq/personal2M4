"""Genoma por loci (0..100) — el corazón de la plantilla del usuario.

Cada especie tiene N loci diploides. Cada valor fenotípico se obtiene de
una **combinación poligénica** de varios loci (matriz G→P). Ningún rasgo
depende de un único locus.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ..utils import clip100


@dataclass
class Genome:
    """Genoma diploide vectorizado.

    Attributes:
        alleles: shape (N_individuos, N_loci, 2), valores 0..100
        chromosomes: mapeo locus -> id de cromosoma (para recombinación)
    """
    alleles: np.ndarray               # (N, L, 2) float32
    chromosomes: np.ndarray            # (L,) uint8

    @property
    def n(self) -> int:
        return self.alleles.shape[0]

    @property
    def n_loci(self) -> int:
        return self.alleles.shape[1]

    def additive(self) -> np.ndarray:
        """Valor aditivo por locus = media de los dos alelos → (N, L)."""
        return self.alleles.mean(axis=2)

    def slice(self, idx) -> "Genome":
        return Genome(alleles=self.alleles[idx].copy(),
                      chromosomes=self.chromosomes.copy())


def new_founder_population(n: int, n_loci: int, n_chromosomes: int,
                           mean: float = 50.0, sd: float = 12.0,
                           rng=None) -> Genome:
    """Crea una población fundadora con loci ~N(mean, sd) clip [0,100]."""
    rng = rng or np.random.default_rng()
    alleles = clip100(rng.normal(mean, sd, size=(n, n_loci, 2))).astype(np.float32)
    chroms = (np.arange(n_loci) * n_chromosomes // max(n_loci, 1)).astype(np.uint8)
    return Genome(alleles=alleles, chromosomes=chroms)


def mate(parents_a: Genome, parents_b: Genome, mu: float, sigma_mut: float,
         rng) -> Genome:
    """Recombinación por cromosoma + mutación gaussiana con DFE exponencial.

    Cada cromosoma se hereda entero de un padre (aproximación honesta y
    barata; evita el 'shuffling amateur' criticado en v0.2).
    """
    n, L, _ = parents_a.alleles.shape
    chroms = parents_a.chromosomes
    n_ch = int(chroms.max()) + 1

    # Máscara (N, n_ch) que decide de qué padre viene cada cromosoma.
    pick_a = rng.random(size=(n, n_ch)) < 0.5
    pick_a_locus = pick_a[:, chroms]                     # (N, L)
    # Y de qué copia del padre (gamete)
    gamete_a = rng.integers(0, 2, size=(n, L))
    gamete_b = rng.integers(0, 2, size=(n, L))

    ii = np.arange(n)[:, None]
    ll = np.arange(L)[None, :]
    allele_from_a = parents_a.alleles[ii, ll, gamete_a]
    allele_from_b = parents_b.alleles[ii, ll, gamete_b]
    child_a = np.where(pick_a_locus, allele_from_a, allele_from_b)

    # Segundo alelo: gameta del otro padre (el que NO fue elegido antes)
    pick_a2 = ~pick_a
    pick_a2_locus = pick_a2[:, chroms]
    allele2_a = parents_a.alleles[ii, ll, 1 - gamete_a]
    allele2_b = parents_b.alleles[ii, ll, 1 - gamete_b]
    child_b = np.where(pick_a2_locus, allele2_a, allele2_b)

    child = np.stack([child_a, child_b], axis=2).astype(np.float32)

    # Mutación: probabilidad mu por locus; magnitud exponencial con signo
    mut_mask = rng.random(size=child.shape) < mu
    signs = rng.choice([-1.0, 1.0], size=child.shape)
    magnitudes = rng.exponential(sigma_mut, size=child.shape).astype(np.float32)
    child = child + mut_mask * signs * magnitudes
    child = clip100(child).astype(np.float32)
    return Genome(alleles=child, chromosomes=chroms.copy())


def clone_asexual(parent: Genome, mu: float, sigma_mut: float, rng,
                  n_children: int) -> Genome:
    """Reproducción asexual (para fungi/plantas asexuales)."""
    idx = rng.integers(0, parent.n, size=n_children)
    base = parent.alleles[idx].copy()
    mut_mask = rng.random(size=base.shape) < mu
    signs = rng.choice([-1.0, 1.0], size=base.shape)
    magnitudes = rng.exponential(sigma_mut, size=base.shape).astype(np.float32)
    base = clip100(base + mut_mask * signs * magnitudes).astype(np.float32)
    return Genome(alleles=base, chromosomes=parent.chromosomes.copy())
