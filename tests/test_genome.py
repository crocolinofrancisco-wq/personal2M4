"""Tests de invariantes de genoma / mate() / mutación."""
import numpy as np
import pytest

from stoneplace.genetics.genome import (Genome, new_founder_population,
                                          mate, clone_asexual)


def test_founder_population_in_range():
    rng = np.random.default_rng(0)
    g = new_founder_population(n=100, n_loci=32, n_chromosomes=4,
                                mean=50.0, sd=12.0, rng=rng)
    assert g.alleles.min() >= 0.0
    assert g.alleles.max() <= 100.0
    assert g.alleles.shape == (100, 32, 2)


def test_mate_preserves_bounds():
    rng = np.random.default_rng(1)
    a = new_founder_population(50, 24, 4, rng=rng)
    b = new_founder_population(50, 24, 4, rng=rng)
    child = mate(a, b, mu=0.1, sigma_mut=5.0, rng=rng)
    assert (child.alleles >= 0.0).all()
    assert (child.alleles <= 100.0).all()
    assert child.alleles.shape == a.alleles.shape


def test_mutation_exponential_distribution_variance():
    """El DFE es exponencial con signo; verificamos su media |x|=σ."""
    rng = np.random.default_rng(2)
    n_samples = 20_000
    magnitudes = rng.exponential(2.5, size=n_samples)
    # Media teórica = σ = 2.5, tolerancia 5%
    assert abs(magnitudes.mean() - 2.5) < 0.15


def test_clone_asexual_preserves_chromosomes():
    rng = np.random.default_rng(3)
    a = new_founder_population(30, 16, 4, rng=rng)
    child = clone_asexual(a, mu=0.05, sigma_mut=3.0, rng=rng, n_children=10)
    assert child.chromosomes.tolist() == a.chromosomes.tolist()
    assert child.n == 10


def test_reproducibility_with_seed():
    """Dos corridas con la misma seed deben dar bit-a-bit el mismo genoma."""
    r1 = np.random.default_rng(42)
    g1 = new_founder_population(20, 16, 4, rng=r1)
    r2 = np.random.default_rng(42)
    g2 = new_founder_population(20, 16, 4, rng=r2)
    np.testing.assert_array_equal(g1.alleles, g2.alleles)
    np.testing.assert_array_equal(g1.chromosomes, g2.chromosomes)
