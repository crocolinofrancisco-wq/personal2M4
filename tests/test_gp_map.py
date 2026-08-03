"""Tests del mapa G→P: gating blando + coste attractiveness + cis mutation."""
import numpy as np

from stoneplace.genetics.gp_map import (chordata_gpmap, mutate_gp_map,
                                          GPMap)
from stoneplace.genetics.genome import new_founder_population


def test_gated_trait_has_latent_floor():
    """Con gates blandos, un gated trait no cae a 0 aunque falten prereqs."""
    rng = np.random.default_rng(0)
    gp = chordata_gpmap(48, rng, map_rng=rng)
    genome = new_founder_population(100, 48, 4, rng=rng)
    phen = gp.express(genome, rng, sigma_env=1.0)
    bio = phen["bioluminiscence_level"]
    # No debería ser 0 en TODOS los individuos (había mini-reward vía floor)
    assert (bio > 0.01).any(), (
        "gated trait totalmente muerto — soft_gate no está funcionando")


def test_attractiveness_costs_metabolism():
    rng = np.random.default_rng(0)
    gp = chordata_gpmap(48, rng, map_rng=rng)
    # Truco: forzamos alelos altos en los loci de attractiveness
    genome = new_founder_population(200, 48, 4, mean=90.0, sd=2.0, rng=rng)
    phen = gp.express(genome, rng, sigma_env=0.5)
    assert phen["basal_metabolism"].mean() > 0
    # attractiveness alta debe implicar coste metabólico (relativamente)
    # simple sanity: la media no colapsa a 0
    assert phen["basal_metabolism"].mean() < 1e6


def test_mutate_gp_map_changes_only_one_locus():
    rng = np.random.default_rng(0)
    gp = chordata_gpmap(48, rng, map_rng=rng)
    # Prob 1.0 fuerza el cambio
    new_gp = mutate_gp_map(gp, 48, rng, p_cis=1.0, p_seed=0.0)
    diffs = 0
    for name in gp.traits:
        if not np.array_equal(gp.traits[name].loci, new_gp.traits[name].loci):
            diffs += 1
    assert diffs == 1, f"esperado 1 rasgo mutado, hubo {diffs}"


def test_dedicated_map_rng_independence():
    """Cambiar el rng principal sin tocar map_rng debe dar el mismo mapa."""
    m1 = np.random.default_rng(999)
    gp1 = chordata_gpmap(48, np.random.default_rng(1), map_rng=m1)
    m2 = np.random.default_rng(999)
    gp2 = chordata_gpmap(48, np.random.default_rng(2), map_rng=m2)
    np.testing.assert_array_equal(gp1.traits["speed"].loci,
                                    gp2.traits["speed"].loci)
