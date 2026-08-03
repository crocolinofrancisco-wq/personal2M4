"""Snapshots visuales — PNG del mundo + poblaciones (v1.4/v1.5).

Requiere matplotlib. Se llama desde `Simulation` cuando
`cfg.snapshot_every > 0`. Cada PNG muestra biomas de fondo + puntos de
cada especie coloreados por species_id.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


_KINGDOM_MARKER = {"plantae": "s", "fungi": "^", "chordata": "o"}


def save_snapshot_png(sim, out_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"world_year_{int(sim.year):06d}.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    biome = sim.biome_map
    ax.imshow(biome, cmap="terrain", origin="upper", alpha=0.6)

    rng = np.random.default_rng(0xC0DE)
    for pop in sim.pops:
        if pop.n == 0:
            continue
        color = np.array(rng.random(3)) * 0.7 + 0.15
        marker = _KINGDOM_MARKER.get(pop.template.kingdom, ".")
        n_sample = min(pop.n, 250)
        idx = rng.choice(pop.n, n_sample, replace=False)
        ax.scatter(pop.x[idx], pop.y[idx], s=4, marker=marker, c=[color],
                    linewidths=0, alpha=0.7,
                    label=pop.template.scientific_name)
    ax.set_title(f"Stoneplace — año {int(sim.year)}")
    ax.set_xticks([]); ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=6,
                  loc="lower left", framealpha=0.7, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
