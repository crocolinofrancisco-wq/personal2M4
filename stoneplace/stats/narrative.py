"""Narrativa emergente (v1.5, v1.4 en el roadmap).

Traduce cambios cuantitativos en texto para el worldbuilder. Cada
evento se guarda como una entrada JSONL en `<out_dir>/narrative.jsonl`.
El objetivo declarado del proyecto es alimentar historias de SeedWorld —
esta capa hace que el usuario ya no tenga que releer telemetry.csv.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class NarrativeLog:
    out_dir: str
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    def log(self, year: float, kind: str, subject_id: int, message: str,
            extra: dict | None = None):
        entry = {
            "year": round(float(year), 3),
            "kind": kind,
            "subject_id": int(subject_id),
            "message": str(message),
        }
        if extra:
            entry["extra"] = extra
        self.events.append(entry)

    def dump(self):
        path = Path(self.out_dir) / "narrative.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Generación de tarjetas narrativas al estilo Speculative Biology
# ---------------------------------------------------------------------------
_LATIN_ROOTS = [
    "sol", "luna", "nox", "aqua", "silva", "mons", "flamma", "petra",
    "umbra", "aurum", "argenta", "spina", "corvus", "draco", "vulpes",
    "lumen", "vesper", "aurora", "borealis", "meridi", "stella", "arbor",
]
_LATIN_SUFFIX = ["us", "a", "ensis", "atus", "orum", "ianum", "ale"]


def coin_scientific_name(rng, genus: str) -> str:
    """Inventa un epíteto latino barato."""
    root = rng.choice(_LATIN_ROOTS)
    suff = rng.choice(_LATIN_SUFFIX)
    return f"{genus} {root}{suff}"


def build_species_card(pop, phylo) -> dict:
    """Ficha narrativa para volcar como species_cards_narrative.json."""
    tpl = pop.template
    card = {
        "species_id": tpl.species_id,
        "scientific_name": tpl.scientific_name,
        "kingdom": tpl.kingdom,
        "body_plan": tpl.hints.get("body_plan", tpl.kingdom),
        "born_at": tpl.born_at,
        "parent": tpl.parent,
        "children": list(tpl.children),
        "current_population": int(pop.n),
        "main_biomes": list(tpl.main_biomes),
        "unlocked_gated": [],
        "highlighted_traits": {},
    }
    if pop.n == 0:
        return card
    # Rasgos gated desbloqueados (> 20 medio)
    for k, v in pop.phenotype.items():
        m = float(v.mean()) if hasattr(v, "mean") else float(v)
        if any(k.endswith(s) for s in ("_level",)) and m > 20:
            card["unlocked_gated"].append({"trait": k, "mean": round(m, 2)})
    # Highlights: rasgos que se movieron mucho del baseline (baseline = 45)
    for k, v in pop.phenotype.items():
        m = float(v.mean()) if hasattr(v, "mean") else float(v)
        if abs(m - 45) > 20:
            card["highlighted_traits"][k] = round(m, 2)
    return card
