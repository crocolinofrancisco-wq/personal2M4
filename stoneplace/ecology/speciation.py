"""Detección de especiación por Qst + Fst acumulados + aislamiento espacial.

v1.5 (N1, N7, N13, N14):
  * El proxy antiguo (varianza aditiva entre-grupos sobre varianza total)
    se renombra **Qst** (Fst de rasgos cuantitativos) — que es lo que era
    en realidad.  Se añade el **Fst clásico** por frecuencia alélica
    (binarización por mediana global).  Ambos se acumulan y se usan de
    forma complementaria: Qst dispara el hallazgo, Fst confirma.
  * K-means k=2..4 con selección por silhouette (radiaciones adaptativas
    simultáneas, no solo bifurcaciones binarias).
  * `_LAST_ID` y `_DIVERGENCE_HISTORY` **ya no son globales de módulo**.
    Viven en un objeto :class:`SpeciationState` inyectado por Simulation.
    Corridas paralelas en el mismo proceso son ahora seguras.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from ..species.base import SpeciesPopulation, SpeciesTemplate
from ..genetics.genome import Genome
from ..genetics.popgen import fst_wright_locus, qst_quantitative
from ..genetics.gp_map import mutate_gp_map


# ---------------------------------------------------------------------------
# Estado por-simulación (sin globals — N13, N14)
# ---------------------------------------------------------------------------
@dataclass
class SpeciationState:
    """Estado inyectable por :class:`stoneplace.simulator.Simulation`.

    Antes: `_LAST_ID = [10_000]` y `_DIVERGENCE_HISTORY = {}` a nivel de
    módulo → dos simulaciones concurrentes se pisaban los IDs y la
    memoria de divergencia. Ahora todo vive aquí y `Simulation` posee
    una instancia por corrida.
    """
    last_species_id: int = 10_000
    divergence_history: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    history_window: int = 4
    fst_sustain: float = 0.10        # umbral Fst honesto por chequeo
    qst_sustain: float = 0.14        # umbral Qst (antiguo Fst proxy)
    cis_mutation_prob: float = 0.05  # prob. de mutar mapa G→P al especiar

    def next_species_id(self) -> int:
        self.last_species_id += 1
        return self.last_species_id


# ---------------------------------------------------------------------------
# K-means k=2..4 con silhouette (N7)
# ---------------------------------------------------------------------------
def _kmeans_pp(X: np.ndarray, k: int, rng, n_iter: int = 15):
    n = X.shape[0]
    if n < 8 * k:
        return None, None
    # k-means++: primer centro aleatorio, resto ponderados por d²
    i0 = int(rng.integers(0, n))
    centers = [X[i0]]
    for _ in range(1, k):
        d2 = np.min(np.stack([((X - c) ** 2).sum(axis=1) for c in centers]), axis=0)
        probs = d2 / (d2.sum() + 1e-9)
        centers.append(X[int(rng.choice(n, p=probs))])
    centers = np.stack(centers).astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(n_iter):
        dists = np.stack([((X - c) ** 2).sum(axis=1) for c in centers])
        new_labels = np.argmin(dists, axis=0).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = X[m].mean(axis=0)
    return labels, centers


def _silhouette_score(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Silhouette medio aproximado — barato: usa distancia a centroide."""
    if k <= 1:
        return -1.0
    centers = np.stack([X[labels == j].mean(axis=0) for j in range(k)])
    n = X.shape[0]
    # a = distancia a su propio centroide
    a = np.linalg.norm(X - centers[labels], axis=1)
    # b = distancia mínima a cualquier otro centroide
    all_d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    all_d[np.arange(n), labels] = np.inf
    b = all_d.min(axis=1)
    s = (b - a) / (np.maximum(a, b) + 1e-9)
    return float(np.mean(s))


def _pick_best_split(X: np.ndarray, rng, k_max: int = 4):
    """Devuelve (labels, k) del mejor k∈{2..k_max} por silhouette."""
    best = (None, 1, -np.inf)
    for k in range(2, k_max + 1):
        labels, _ = _kmeans_pp(X, k, rng)
        if labels is None:
            continue
        # Todos los clústeres deben tener al menos 25 miembros para ser útil
        sizes = np.bincount(labels, minlength=k)
        if (sizes < 25).any():
            continue
        s = _silhouette_score(X, labels, k)
        if s > best[2]:
            best = (labels, k, s)
    return best[0], best[1]


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------
def try_speciate(pop: SpeciesPopulation, world, current_year: float,
                 rng, state: SpeciationState,
                 seed_master: int = 0) -> list[SpeciesPopulation]:
    """Detecta hasta k-1 splits en una especie madre.

    Devuelve una lista de hijas nuevas (potencialmente vacía). Si k=2 y hay
    divergencia sostenida, sale 1 hija (equivalente a v1.1). Si k=3/4 y la
    silhouette lo respalda, salen 2/3 hijas simultáneas — radiación
    adaptativa.
    """
    if pop.n < 60:
        return []
    additive = pop.genome.additive()
    coords = np.stack([pop.y / world.height, pop.x / world.width], axis=1) * 40.0
    X = np.concatenate([additive, coords], axis=1)

    labels, k = _pick_best_split(X, rng, k_max=4)
    if labels is None or k < 2:
        return []

    # Distancia máxima entre centroides como proxy de aislamiento
    centers_y = np.array([pop.y[labels == j].mean() for j in range(k)])
    centers_x = np.array([pop.x[labels == j].mean() for j in range(k)])
    dist_max = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            d = float(np.hypot(centers_y[a] - centers_y[b],
                                centers_x[a] - centers_x[b]))
            dist_max = max(dist_max, d)
    is_isolated = dist_max >= 10.0

    # Fst y Qst globales entre TODOS los grupos (media ponderada de pares)
    fst_total, qst_total, weight_total = 0.0, 0.0, 0.0
    for a in range(k):
        for b in range(a + 1, k):
            m_a = labels == a; m_b = labels == b
            w = float(min(m_a.sum(), m_b.sum()))
            fst_total += w * fst_wright_locus(additive, m_a, m_b)
            qst_total += w * qst_quantitative(additive, m_a, m_b)
            weight_total += w
    if weight_total <= 0:
        return []
    fst = fst_total / weight_total
    qst = qst_total / weight_total

    sid = pop.template.species_id
    hist = state.divergence_history.setdefault(sid, [])
    hist.append((fst if is_isolated else 0.0, qst if is_isolated else 0.0))
    if len(hist) > state.history_window:
        hist.pop(0)
    if len(hist) < state.history_window:
        return []
    if not all(f >= state.fst_sustain * 0.5 and q >= state.qst_sustain
               for (f, q) in hist):
        return []
    if fst < state.fst_sustain or qst < state.qst_sustain * 1.2:
        return []

    state.divergence_history[sid] = []

    # Extraemos cada clúster distinto: el más grande se queda como madre.
    sizes = [(int((labels == j).sum()), j) for j in range(k)]
    sizes.sort(reverse=True)
    mother_cluster = sizes[0][1]
    child_clusters = [j for _, j in sizes[1:]]

    parent_template = pop.template
    child_pops: list[SpeciesPopulation] = []
    for cj in child_clusters:
        m = labels == cj
        if m.sum() < 25:
            continue
        # Mutación cis-regulatoria rara del mapa G→P (N2)
        child_gpmap = parent_template.gp_map
        if rng.random() < state.cis_mutation_prob:
            child_gpmap = mutate_gp_map(child_gpmap, parent_template.n_loci, rng)
        child_template = SpeciesTemplate(
            species_id=state.next_species_id(),
            scientific_name=f"{parent_template.scientific_name} {chr(65 + len(parent_template.children))}",
            taxonomy=list(parent_template.taxonomy),
            kingdom=parent_template.kingdom,
            parent=parent_template.species_id,
            born_at=current_year,
            n_loci=parent_template.n_loci,
            n_chromosomes=parent_template.n_chromosomes,
            gp_map=child_gpmap,
            hints=dict(parent_template.hints),
        )
        parent_template.children.append(child_template.species_id)
        child_pops.append(SpeciesPopulation(
            template=child_template,
            genome=Genome(
                alleles=pop.genome.alleles[m].copy(),
                chromosomes=pop.genome.chromosomes.copy()),
            y=pop.y[m].copy(),
            x=pop.x[m].copy(),
            age_years=pop.age_years[m].copy(),
            energy=pop.energy[m].copy(),
            sex=(pop.sex[m].copy() if pop.sex is not None else None),
        ))

    # La madre se queda solo con el clúster más grande.
    keep = labels == mother_cluster
    pop.kill_mask(keep)
    return child_pops


# ---------------------------------------------------------------------------
# Compatibilidad hacia atrás (v1.1) — Deprecated pero se mantiene el nombre
# ---------------------------------------------------------------------------
def _fst_wright(*_args, **_kwargs):    # pragma: no cover
    """Deprecated. Usa `stoneplace.genetics.popgen.qst_quantitative`."""
    raise RuntimeError("v1.1 _fst_wright ha sido renombrado a qst_quantitative "
                       "(era un proxy de Qst, no un Fst honesto). Ver "
                       "stoneplace.genetics.popgen.")
