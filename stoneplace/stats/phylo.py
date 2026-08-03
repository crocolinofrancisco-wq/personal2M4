"""Árbol filogenético de Stoneplace (v1.5).

Registra cada nodo (especie) con su padre y su año de aparición. Exporta
en formato Newick que abren FigTree, iTOL o dendropy.

Es un mini tree-sequence: cada nodo tiene id, parent y born_at, similar
al patrón `tskit` sin la complicación completa de coalescente por locus.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Node:
    species_id: int
    name: str
    parent: int | None
    born_at: float
    children: list = field(default_factory=list)
    extinct_at: float | None = None


class PhyloTree:
    def __init__(self):
        self.nodes: dict[int, _Node] = {}

    def register(self, template, parent: int | None, born_at: float):
        node = _Node(species_id=template.species_id,
                      name=template.scientific_name.replace(" ", "_"),
                      parent=parent, born_at=born_at)
        self.nodes[template.species_id] = node
        if parent is not None and parent in self.nodes:
            self.nodes[parent].children.append(template.species_id)

    def mark_extinct(self, species_id: int, at_year: float):
        if species_id in self.nodes:
            self.nodes[species_id].extinct_at = at_year

    def _newick_of(self, node: _Node, now: float) -> str:
        end = node.extinct_at if node.extinct_at is not None else now
        length = max(end - node.born_at, 0.0)
        if not node.children:
            return f"{node.name}:{length:.2f}"
        kids = ",".join(self._newick_of(self.nodes[c], now)
                        for c in node.children if c in self.nodes)
        return f"({kids}){node.name}:{length:.2f}"

    def dump_newick(self, path: str | Path, now: float = 0.0):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.nodes:
            path.write_text(";", encoding="utf-8")
            return
        # Raíces: nodos sin parent en el diccionario
        roots = [n for n in self.nodes.values() if n.parent is None]
        # Sincroniza "now" con el mayor born_at si no se pasa
        if now == 0.0:
            now = max((n.born_at for n in self.nodes.values()), default=0.0) + 1
        if len(roots) == 1:
            nw = self._newick_of(roots[0], now) + ";"
        else:
            nw = "(" + ",".join(self._newick_of(r, now) for r in roots) + ")root;"
        path.write_text(nw, encoding="utf-8")

    def summary(self) -> dict:
        return {
            "total_nodes": len(self.nodes),
            "extinct": sum(1 for n in self.nodes.values() if n.extinct_at is not None),
            "roots": [n.name for n in self.nodes.values() if n.parent is None],
        }
