"""Corrida corta reproducible bit-a-bit dado el seed (v1.2 punto 5)."""
import numpy as np
import pytest

from stoneplace.simulator import Simulation, SimConfig


@pytest.fixture(scope="module")
def small_cfg():
    return SimConfig(
        synthetic_size=(32, 16),
        out_dir="/tmp/stoneplace_test_repro",
        seed=1234,
        phase1_years=3,
        total_years=6,
        telemetry_every=1,
        speciation_every=1000,
        founder_size_plant=40,
        founder_size_fungi=30,
        founder_size_animal=40,
        dump_every_snapshots=1000,
        climate_enabled=False,     # aislamos test del clima
    )


def _run(cfg):
    sim = Simulation(cfg)
    sim.run()
    return {
        "pops_n": [p.n for p in sim.pops],
        "total_pop": sum(p.n for p in sim.pops),
    }


def test_reproducible_same_seed(small_cfg):
    a = _run(small_cfg)
    b = _run(small_cfg)
    assert a == b, f"corridas con misma seed divergen: {a} vs {b}"
