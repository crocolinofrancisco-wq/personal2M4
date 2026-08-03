"""Tests unitarios de utilidades numéricas (v1.5 punto 5 del roadmap)."""
import numpy as np
import pytest

from stoneplace.utils import liebig_min, clip100, clip01, soft_gate


def test_liebig_min_is_actual_minimum():
    # El factor peor debe mandar. NUNCA multiplicativo disfrazado.
    a = np.array([1.0, 0.5, 0.0])
    b = np.array([1.0, 0.5, 0.0])
    c = np.array([1.0, 0.5, 0.0])
    assert liebig_min(a, b, c).tolist() == [1.0, 0.5, 0.0]
    # Un solo cero mata la cosecha entera
    assert liebig_min(np.array([0.9]), np.array([0.9]),
                       np.array([0.0]))[0] == 0.0
    # y un valor alto no compensa al bajo
    assert liebig_min(np.array([1.0]), np.array([0.3]),
                       np.array([1.0]))[0] == 0.3


def test_clip100_bounds():
    assert clip100(-5).tolist() == 0.0
    assert clip100(150).tolist() == 100.0
    assert clip100(50).tolist() == 50.0


def test_clip01_bounds():
    assert clip01(-1).tolist() == 0.0
    assert clip01(2).tolist() == 1.0


def test_soft_gate_is_never_zero_before_threshold():
    # N3 fix: la puerta ANTES del umbral debe dar al menos `floor` de reward
    v = np.array([20.0, 30.0, 40.0, 50.0, 55.0, 60.0])
    g = soft_gate(v, threshold=55.0, softness=6.0, floor=0.05)
    assert (g >= 0.05).all(), f"soft_gate cayó bajo floor: {g}"
    # y monótonamente creciente (más rasgo → más recompensa)
    assert (np.diff(g) >= -1e-6).all(), f"no monótono: {g}"
