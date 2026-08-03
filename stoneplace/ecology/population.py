"""Dinámica poblacional: envejecimiento, mortalidad y reproducción.

Todo vectorizado. Los tiempos de reproducción se miden en años del mundo
Stoneplace (400 días de 25 h = 10 000 h por año).
"""
from __future__ import annotations
import numpy as np

from ..utils import clip01
from ..genetics.genome import mate, clone_asexual, Genome
from ..species.base import SpeciesPopulation
from .habitat import suitability


def age_and_die(pop: SpeciesPopulation, world, biome_map, micro_dict,
                dt_years: float, rng, density_map: np.ndarray | None = None):
    if pop.n == 0:
        return
    pop.age_years = pop.age_years + dt_years
    life = np.maximum(pop.phenotype["life_expectancy"], 0.05)
    p_old = 1.0 - np.exp(-dt_years / life)

    suit = suitability(pop, world, biome_map, micro_dict)
    p_unfit = clip01(0.25 * dt_years * (1.0 - suit) ** 2)

    metab_cost = clip01(pop.phenotype["basal_metabolism"] / 8000.0) * dt_years
    pop.energy = clip01(pop.energy - metab_cost + np.sqrt(np.maximum(suit, 0.0)) * dt_years * 0.5)
    p_hunger = clip01(np.maximum(0.10 - pop.energy, 0.0) * 1.5)

    p_density = np.zeros(pop.n, dtype=np.float32)
    if density_map is not None:
        d = density_map[pop.y, pop.x]
        crowd_tol = np.maximum(pop.phenotype.get(
            "overcrowd_tolerance", np.full(pop.n, 40.0)), 10.0)
        max_local = 40.0 + crowd_tol * 3.0
        p_density = clip01((d - max_local) / 250.0) * dt_years

    survive_p = (1.0 - p_old) * (1.0 - p_unfit) * (1.0 - p_hunger) * (1.0 - p_density)
    p_die = 1.0 - survive_p
    survive = rng.random(pop.n) >= p_die
    pop.kill_mask(survive)


def _build_density(pop: SpeciesPopulation, shape) -> np.ndarray:
    m = np.zeros(shape, dtype=np.int32)
    np.add.at(m, (pop.y, pop.x), 1)
    return m


# ---------------------------------------------------------------------------
# v1.5.1 — Apareamiento asortativo por ecomorfo (estilo Serina)
# ---------------------------------------------------------------------------
def _ecomorph_vector(pop: "SpeciesPopulation", idx) -> np.ndarray:
    """Vector fenotípico "ecomorfo" — la coordenada donde una hembra busca
    macho similar. Combina dieta normalizada + masa (log) + circadiano.

    Estas son las tres dimensiones donde la selección disruptiva puede
    partir una población: qué come, qué tan grande es, y cuándo está
    activo. Un carnívoro grande diurno no se aparea con un herbívoro
    pequeño nocturno aunque compartan celda.
    """
    p = pop.phenotype
    diet = np.stack([
        p["diet_bug"][idx], p["diet_meat"][idx], p["diet_vegetal"][idx],
        p["diet_fish"][idx], p["diet_micro"][idx],
    ], axis=1)
    diet_norm = diet / np.maximum(diet.sum(axis=1, keepdims=True), 1e-6)
    log_mass = np.log10(np.maximum(p["mass_g"][idx], 1e-3))[:, None] / 5.0
    circadian = p["circadian"][idx][:, None] / 100.0
    return np.concatenate([diet_norm, log_mass, circadian], axis=1).astype(np.float32)


def _assortative_choice(pop, mother_idx, males, rng,
                         alpha_assort: float = 2.5,
                         assort_p: float = 0.55,
                         min_pop_for_assort: int = 500) -> np.ndarray:
    """Elige un padre por cada madre con sesgo por similitud fenotípica.

    - Con probabilidad `assort_p` la hembra usa apareamiento asortativo:
      pondera cada macho por `exp(-α · d_L1(v_hembra, v_macho)) · attractiveness`.
    - Con probabilidad `1 - assort_p` mate is random attractiveness-biased
      (mantiene flujo génico residual que evita fixation immediata).

    Para no explotar O(N_f × N_m), asignamos cada macho a un "bin ecomorfo"
    (round del vector fenotípico) y cada hembra elige entre machos del bin
    más cercano con prob mayor.
    """
    if males.size == 0:
        return np.array([], dtype=np.int64)
    if males.size == 1 or mother_idx.size == 0:
        return np.full(mother_idx.size, males[0], dtype=np.int64)

    # Fundadoras: mientras la población sea pequeña, se aparean random-
    # attractiveness. El asortativo aparece cuando hay suficientes machos
    # para que el sesgo por ecomorfo tenga significado biológico.
    if pop.n < min_pop_for_assort:
        attr = pop.phenotype["attractiveness"][males] + 1e-3
        return rng.choice(males, size=mother_idx.size,
                           p=attr / attr.sum()).astype(np.int64)

    # Vectores fenotípicos
    v_f = _ecomorph_vector(pop, mother_idx)          # (F, D)
    v_m = _ecomorph_vector(pop, males)               # (M, D)

    # Attractiveness base de cada macho
    attr = pop.phenotype["attractiveness"][males] + 1e-3
    attr = attr / attr.sum()

    # Máscara: ¿esta hembra usa asortativo este turno?
    use_assort = rng.random(mother_idx.size) < assort_p

    # Para muestras aleatorias (no asortativas): choice por attractiveness
    random_partners = rng.choice(males, size=mother_idx.size, p=attr)

    # Para asortativas: aproximación en 2 pasos —
    #  1) sample K candidatos por attractiveness (K=8),
    #  2) elegir entre esos K el más similar.
    # O(F * K * D) — barato aunque F=100k.
    K = 8
    cand_idx_in_males = rng.choice(males.size, size=(mother_idx.size, K), p=attr)
    cand_males = males[cand_idx_in_males]                    # (F, K)
    cand_vecs = v_m[cand_idx_in_males]                        # (F, K, D)
    # Distancia L1 entre la hembra y cada uno de sus K candidatos
    d = np.abs(cand_vecs - v_f[:, None, :]).sum(axis=2)      # (F, K)
    weights = np.exp(-alpha_assort * d)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-9)
    # Muestreo categórico vectorizado por fila
    cum = np.cumsum(weights, axis=1)
    u = rng.random(mother_idx.size)[:, None]
    pick = (u < cum).argmax(axis=1)
    assort_partners = cand_males[np.arange(mother_idx.size), pick]

    return np.where(use_assort, assort_partners, random_partners).astype(np.int64)


def reproduce(pop: SpeciesPopulation, world, biome_map, micro_dict,
              dt_years: float, mu: float, sigma_mut: float, rng,
              max_children_cap: int = 60000) -> tuple | None:
    """Reproducción sexual o asexual.

    Fixes aplicados:
      P1: chordata — solo hembras cuentan como progenitoras, con broods
          recalculado sobre ellas (antes se sobreestimaba x2 la natalidad).
      P9: supervivencia de huevos density-dependent ANTES del cap global
          (evita que los r-estrategas como Triops choquen contra un tope
          arbitrario que sesga la selección).
    """
    if pop.n == 0:
        return None

    breed = np.maximum(pop.phenotype["breeding_lapse"], 0.02)
    suit = suitability(pop, world, biome_map, micro_dict)
    broods_per_tick = np.clip(dt_years / breed, 1.0, 25.0)
    p_breed = np.minimum(1.0, dt_years / breed) * np.maximum(suit, 0.05)

    if pop.template.kingdom == "chordata":
        mature_age = pop.phenotype.get("growth_time", np.full(pop.n, 1.0)) * 0.25
    else:
        mature_age = pop.phenotype.get("germination_time", np.full(pop.n, 0.1)) * 0.25
    p_breed = np.where(pop.age_years >= mature_age, p_breed, 0.0)

    kingdom = pop.template.kingdom

    # -----------------------------------------------------------------
    # P1: chordata sexuada — descartamos machos ANTES de calcular hijos.
    # -----------------------------------------------------------------
    if kingdom == "chordata" and pop.sex is not None:
        females_mask = (pop.sex == 0)
        p_breed = np.where(females_mask, p_breed, 0.0)

    breeders = np.where(rng.random(pop.n) < p_breed)[0]
    if breeders.size == 0:
        return None

    baby_q = pop.phenotype["baby_quantity"][breeders]
    n_children_per = np.maximum(
        (baby_q * broods_per_tick[breeders] * np.maximum(suit[breeders], 0.15)
         ).astype(np.int32),
        1)

    if kingdom == "plantae":
        n_children_per = np.maximum(n_children_per // 40, 1)
    elif kingdom == "fungi":
        n_children_per = np.maximum(n_children_per // 200, 1)

    # -----------------------------------------------------------------
    # P9: mortalidad temprana density-dependent (egg_survival)
    # -----------------------------------------------------------------
    density_map = _build_density(pop, world.shape)
    local_d = density_map[pop.y[breeders], pop.x[breeders]].astype(np.float32)
    # K_local depende del tamaño (microorganismos y r-estrategas soportan más)
    crowd_tol = pop.phenotype.get(
        "overcrowd_tolerance", np.full(pop.n, 40.0))[breeders]
    K_local = 50.0 + 3.0 * np.maximum(crowd_tol, 10.0)
    egg_survival = 1.0 / (1.0 + local_d / K_local)
    n_children_per = np.maximum(
        (n_children_per.astype(np.float32) * egg_survival).astype(np.int32), 0)

    total = int(n_children_per.sum())
    if total == 0:
        return None
    # Cap defensivo (memoria) — poco frecuente ya con egg_survival
    if total > max_children_cap:
        keep = max_children_cap / total
        n_children_per = np.maximum((n_children_per * keep).astype(np.int32), 0)
        total = int(n_children_per.sum())
        if total == 0:
            return None

    mother_idx = np.repeat(breeders, n_children_per)

    sexual = (kingdom == "chordata") or (kingdom == "plantae" and rng.random() > 0.2)
    father_idx = None
    if sexual and pop.n > 1:
        if pop.sex is not None and kingdom == "chordata":
            males = np.where(pop.sex == 1)[0]
            if males.size == 0:
                sexual = False
            else:
                father_idx = _assortative_choice(
                    pop, mother_idx, males, rng)
        else:
            father_idx = rng.choice(pop.n, size=mother_idx.size)

    if sexual and father_idx is not None:
        parents_a = Genome(
            alleles=pop.genome.alleles[mother_idx].copy(),
            chromosomes=pop.genome.chromosomes.copy())
        parents_b = Genome(
            alleles=pop.genome.alleles[father_idx].copy(),
            chromosomes=pop.genome.chromosomes.copy())
        return mate(parents_a, parents_b, mu, sigma_mut, rng), mother_idx
    return clone_asexual(pop.genome.slice(mother_idx),
                        mu, sigma_mut, rng, n_children=mother_idx.size), mother_idx
