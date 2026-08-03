"""Métricas de genética de poblaciones honestas para Stoneplace v1.5.

Antes (v1.1) sólo teníamos un proxy Qst (varianza aditiva entre-grupos
sobre varianza total) que se llamaba equivocadamente "Fst". Aquí:

* :func:`fst_wright_locus` — Fst clásico por locus, sobre frecuencias
  alélicas de la alta/baja mitad de cada locus (discretización mediana
  local).  Es lo que la literatura entiende por Fst en modelos QTL.
* :func:`qst_quantitative` — Qst honesto: varianza de rasgos aditivos
  entre subpoblaciones sobre varianza total.  Este es el proxy antiguo,
  renombrado sin engaño.
* :func:`nucleotide_diversity_pi` — π aproximado. Como los loci son
  cuantitativos, usamos la varianza de la media alélica por locus como
  proxy de heterocigosidad.
"""
from __future__ import annotations
import numpy as np


def _binarize(additive: np.ndarray) -> np.ndarray:
    """Discretiza un genoma (N,L) 0..100 → (N,L) uint8 con 1 si el locus está
    por encima de la mediana global de ese locus."""
    med = np.median(additive, axis=0, keepdims=True)
    return (additive >= med).astype(np.uint8)


def fst_wright_locus(additive: np.ndarray, mask_a: np.ndarray,
                     mask_b: np.ndarray) -> float:
    """Fst de Wright (H_T - H_S) / H_T por locus, promediado.

    Discretizamos cada locus en alelo "alto/bajo" respecto a la mediana
    global; luego H = 2 p q y F_ST = (H_T - H_S) / H_T.
    """
    if mask_a.sum() < 4 or mask_b.sum() < 4:
        return 0.0
    bin_all = _binarize(additive)
    p_a = bin_all[mask_a].mean(axis=0)
    p_b = bin_all[mask_b].mean(axis=0)
    n_a = float(mask_a.sum()); n_b = float(mask_b.sum())
    w_a = n_a / (n_a + n_b)
    w_b = 1.0 - w_a
    p_bar = w_a * p_a + w_b * p_b
    H_T = 2.0 * p_bar * (1.0 - p_bar)                 # heterocig. total
    H_S = w_a * 2.0 * p_a * (1.0 - p_a) + w_b * 2.0 * p_b * (1.0 - p_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        fst_locus = np.where(H_T > 1e-9, (H_T - H_S) / H_T, 0.0)
    return float(np.clip(np.nanmean(fst_locus), 0.0, 1.0))


def qst_quantitative(additive: np.ndarray, mask_a: np.ndarray,
                     mask_b: np.ndarray) -> float:
    """Qst de rasgos cuantitativos: var_entre / var_total sobre loci aditivos."""
    if mask_a.sum() < 4 or mask_b.sum() < 4:
        return 0.0
    mean_a = additive[mask_a].mean(axis=0)
    mean_b = additive[mask_b].mean(axis=0)
    n_a = float(mask_a.sum()); n_b = float(mask_b.sum())
    w_a = n_a / (n_a + n_b); w_b = 1.0 - w_a
    mean_g = w_a * mean_a + w_b * mean_b
    var_between = (w_a * (mean_a - mean_g) ** 2
                   + w_b * (mean_b - mean_g) ** 2).mean()
    var_total = additive.var(axis=0).mean()
    return float(var_between / (var_total + 1e-9))


def nucleotide_diversity_pi(additive: np.ndarray) -> float:
    """π aproximado: media de la varianza por locus normalizada."""
    if additive.shape[0] < 2:
        return 0.0
    # Escala 0..1: varianza(loci en 0..100) / (50^2 = 2500)
    return float(np.mean(additive.var(axis=0)) / 2500.0)


def heterozygosity(alleles: np.ndarray) -> float:
    """H observada media por locus (fracción individuos con alelos distintos)."""
    if alleles.shape[0] < 2:
        return 0.0
    het = np.abs(alleles[:, :, 0] - alleles[:, :, 1]) > 1e-3
    return float(het.mean())
