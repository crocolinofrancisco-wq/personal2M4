"""Registrador de tree-sequences ligero (v1.5, hook estilo tskit).

No implementa el formato completo de tskit — para eso Stoneplace tendría
que trackear identidad de gametas por locus, que aún no hace. Lo que sí
graba es un log de nodos (especies) y edges (padre→hija) con timestamps,
suficiente para reconstruir la genealogía y exportar a Newick.

Si en un futuro se quiere alimentar `msprime` / `pyslim`, esta clase es
el punto de intercepción a extender.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class TreeSequenceRecorder:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)

    def add_node(self, species_id: int, name: str, born_at: float,
                 population_size: int):
        self.nodes.append({
            "id": species_id, "name": name, "time": born_at,
            "n": int(population_size),
        })

    def add_edge(self, parent: int, child: int, at_year: float):
        self.edges.append({"parent": parent, "child": child, "time": at_year})

    def dump(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "nodes.json").write_text(
            json.dumps(self.nodes, indent=2), encoding="utf-8")
        (path / "edges.json").write_text(
            json.dumps(self.edges, indent=2), encoding="utf-8")
