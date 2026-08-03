"""Tests de las métricas de pop-gen (Fst / Qst / π)."""
import numpy as np

from stoneplace.genetics.popgen import (fst_wright_locus, qst_quantitative,
                                          nucleotide_diversity_pi,
                                          heterozygosity)


def test_fst_zero_when_no_divergence():
    rng = np.random.default_rng(0)
    additive = rng.normal(50, 10, size=(200, 16))
    # Etiquetas aleatorias → no debe haber divergencia detectable
    labels = rng.integers(0, 2, size=200).astype(bool)
    fst = fst_wright_locus(additive.astype(np.float32), labels, ~labels)
    assert fst < 0.05, f"Fst random-labels debería ser ~0, fue {fst}"


def test_fst_high_on_clear_divergence():
    rng = np.random.default_rng(1)
    additive = np.concatenate([
        rng.normal(30, 5, size=(100, 16)),
        rng.normal(70, 5, size=(100, 16)),
    ]).astype(np.float32)
    labels = np.zeros(200, bool); labels[:100] = True
    fst = fst_wright_locus(additive, labels, ~labels)
    assert fst > 0.6, f"Fst con clara separación bimodal debería ser alto, fue {fst}"


def test_qst_matches_intuition():
    rng = np.random.default_rng(2)
    additive = np.concatenate([
        rng.normal(20, 3, size=(80, 8)),
        rng.normal(80, 3, size=(80, 8)),
    ]).astype(np.float32)
    m_a = np.zeros(160, bool); m_a[:80] = True
    qst = qst_quantitative(additive, m_a, ~m_a)
    assert 0.5 < qst < 1.5, f"Qst con separación fuerte fuera de rango: {qst}"


def test_pi_and_h_bounded():
    rng = np.random.default_rng(3)
    additive = rng.uniform(0, 100, size=(50, 24)).astype(np.float32)
    assert 0.0 <= nucleotide_diversity_pi(additive) <= 1.0
    alleles = rng.uniform(0, 100, size=(50, 24, 2)).astype(np.float32)
    assert 0.0 <= heterozygosity(alleles) <= 1.0
