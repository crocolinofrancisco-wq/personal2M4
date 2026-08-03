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
    """Envejece a los individuos y aplica mortalidad.

    Componentes de mortalidad por tick (probabilidades independientes,
    combinadas como 1-∏(1-pᵢ) para no saturar en 1.0 con dos malos números):
        - vejez: prob por año = 1 - exp(-dt / life_expectancy)
        - inadaptación: escala cuadrática en (1 - suitability); con suit≥0.6
          la mortalidad ambiental es casi nula.
        - hambre: solo cuando la energía cae por debajo de 0.1.
        - densodependiente: si hay demasiados individuos en la misma celda.
    """
    if pop.n == 0:
        return
    pop.age_years = pop.age_years + dt_years
    life = np.maximum(pop.phenotype["life_expectancy"], 0.05)
    # Aging exponencial: p_old = 1 - e^(-dt/life). Para life<<dt (Triops)
    # esto se acerca a 1 sin explotar, y da 8% para life=12, dt=1.
    p_old = 1.0 - np.exp(-dt_years / life)

    suit = suitability(pop, world, biome_map, micro_dict)
    # Cuadrático: suit=0.6→0.04, suit=0.3→0.12, suit=0.1→0.20 por año.
    p_unfit = clip01(0.25 * dt_years * (1.0 - suit) ** 2)

    # Energía: baja con metabolismo, sube con idoneidad (comida virtual).
    # La recuperación ahora es proporcional a suit^0.5 para que incluso con
    # comida escasa haya un poco de reposición.
    metab_cost = clip01(pop.phenotype["basal_metabolism"] / 8000.0) * dt_years
    pop.energy = clip01(pop.energy - metab_cost + np.sqrt(np.maximum(suit, 0.0)) * dt_years * 0.5)
    p_hunger = clip01(np.maximum(0.10 - pop.energy, 0.0) * 1.5)

    p_density = np.zeros(pop.n, dtype=np.float32)
    if density_map is not None:
        d = density_map[pop.y, pop.x]
        # Densidad tolerada depende del tamaño: microorganismos aguantan más.
        crowd_tol = np.maximum(pop.phenotype.get(
            "overcrowd_tolerance", np.full(pop.n, 40.0)), 10.0)
        max_local = 40.0 + crowd_tol * 3.0
        p_density = clip01((d - max_local) / 250.0) * dt_years

    # Combinación multiplicativa: cada causa quita una fracción del "vivo".
    survive_p = (1.0 - p_old) * (1.0 - p_unfit) * (1.0 - p_hunger) * (1.0 - p_density)
    p_die = 1.0 - survive_p
    survive = rng.random(pop.n) >= p_die
    pop.kill_mask(survive)


def _build_density(pop: SpeciesPopulation, shape) -> np.ndarray:
    m = np.zeros(shape, dtype=np.int32)
    np.add.at(m, (pop.y, pop.x), 1)
    return m


def reproduce(pop: SpeciesPopulation, world, biome_map, micro_dict,
              dt_years: float, mu: float, sigma_mut: float, rng,
              max_children_cap: int = 60000) -> Genome | None:
    """Reproducción sexual (chordata + plantas sexuales) o asexual.

    Devuelve el genoma de los hijos (o None si no hubo nacimientos).
    Los hijos se agregan a la población en `Simulation`.

    Las especies con `breeding_lapse` mucho menor que `dt_years` (Triops,
    hongos) obtienen múltiples camadas por tick — se traduce en un
    multiplicador `broods_per_tick` sobre el número de crías, lo que
    respeta la biología sin explotar el conteo de progenitores.
    """
    if pop.n == 0:
        return None

    breed = np.maximum(pop.phenotype["breeding_lapse"], 0.02)
    suit = suitability(pop, world, biome_map, micro_dict)
    # Camadas por tick: al menos 1 si maduraron; si breeding_lapse << dt
    # se acumulan varias, capadas a 25 para no volar la memoria.
    broods_per_tick = np.clip(dt_years / breed, 1.0, 25.0)
    # Prob. de reproducirse este tick: siempre 1 para especies rápidas,
    # y para las lentas es dt/breed (Poisson simplificado).
    p_breed = np.minimum(1.0, dt_years / breed) * np.maximum(suit, 0.05)

    if pop.template.kingdom == "chordata":
        mature_age = pop.phenotype.get("growth_time", np.full(pop.n, 1.0)) * 0.25
    else:
        mature_age = pop.phenotype.get("germination_time", np.full(pop.n, 0.1)) * 0.25
    p_breed = np.where(pop.age_years >= mature_age, p_breed, 0.0)

    breeders = np.where(rng.random(pop.n) < p_breed)[0]
    if breeders.size == 0:
        return None

    kingdom = pop.template.kingdom
    baby_q = pop.phenotype["baby_quantity"][breeders]
    n_children_per = np.maximum(
        (baby_q * broods_per_tick[breeders] * np.maximum(suit[breeders], 0.15)
         ).astype(np.int32),
        1)
    # Las plantas/hongos generan cantidades absurdas por hembra: reducimos
    # a "semillas viables que germinan" con un divisor moderado.
    if kingdom == "plantae":
        n_children_per = np.maximum(n_children_per // 40, 1)
    elif kingdom == "fungi":
        n_children_per = np.maximum(n_children_per // 200, 1)

    total = int(n_children_per.sum())
    if total == 0:
        return None
    if total > max_children_cap:
        keep = max_children_cap / total
        n_children_per = np.maximum((n_children_per * keep).astype(np.int32), 0)
        total = int(n_children_per.sum())
        if total == 0:
            return None

    # Índice del padre "madre" por hijo
    mother_idx = np.repeat(breeders, n_children_per)

    sexual = (kingdom == "chordata") or (kingdom == "plantae" and rng.random() > 0.2)
    father_idx = None
    if sexual and pop.n > 1:
        # Elegir "padre" al azar (los animales evalúan atractiveness)
        if pop.sex is not None and pop.template.kingdom == "chordata":
            males = np.where(pop.sex == 1)[0]
            females = np.where(pop.sex == 0)[0]
            if males.size == 0 or females.size == 0:
                sexual = False
            else:
                # Sólo hembras maduras se reproducen sexualmente
                fem_breeders = np.intersect1d(breeders, females)
                if fem_breeders.size == 0:
                    return None
                # Recalcular children a partir de las hembras
                sel = np.isin(breeders, fem_breeders)
                mother_idx = np.repeat(breeders[sel], n_children_per[sel])
                if mother_idx.size == 0:
                    return None
                # Padres: sesgo por attractiveness
                attr = pop.phenotype["attractiveness"][males] + 1e-3
                p = attr / attr.sum()
                father_idx = rng.choice(males, size=mother_idx.size, p=p)
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
    # Asexual (hongos y algunas plantas)
    return clone_asexual(pop.genome.slice(mother_idx),
                        mu, sigma_mut, rng, n_children=mother_idx.size), mother_idx
