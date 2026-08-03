"""Gráficos post-simulación con matplotlib + seaborn.

Genera un puñado de PNG a partir de `telemetry.csv` y las filas de
rendimiento acumuladas por `Simulation.perf_rows`. Falla suave si
matplotlib/seaborn no están instalados.
"""
from __future__ import annotations
from pathlib import Path
import csv


def _load_telemetry(csv_path: Path):
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def render_all(out_dir: str | Path, perf_rows: list[dict] | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:                                # pragma: no cover
        print(f"  [plots] omitidos (falta matplotlib/seaborn): {e}")
        return []
    sns.set_theme(style="whitegrid", context="talk")
    saved: list[Path] = []

    tel_csv = out_dir / "telemetry.csv"
    if tel_csv.exists():
        rows = _load_telemetry(tel_csv)
        if rows:
            years = [float(r["year"]) for r in rows]
            total = [int(r.get("total_pop", 0) or 0) for r in rows]
            alive = [int(r.get("species_alive", 0) or 0) for r in rows]

            # 1) Población total + especies vivas
            fig, ax1 = plt.subplots(figsize=(10, 5))
            sns.lineplot(x=years, y=total, ax=ax1, color="#3467eb",
                         label="Población total")
            ax1.set_xlabel("Año"); ax1.set_ylabel("Población")
            ax2 = ax1.twinx()
            sns.lineplot(x=years, y=alive, ax=ax2, color="#eb7134",
                         label="Especies vivas")
            ax2.set_ylabel("Especies vivas")
            ax1.set_title("Población y biodiversidad")
            fig.tight_layout()
            p = plots_dir / "population.png"
            fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

            # 2) Población por especie (columnas n_*)
            sp_cols = [k for k in rows[0].keys() if k.startswith("n_")]
            if sp_cols:
                fig, ax = plt.subplots(figsize=(11, 6))
                for col in sp_cols:
                    ys = [int(r.get(col) or 0) for r in rows]
                    if max(ys) == 0:
                        continue
                    label = col.split("_", 2)[-1].replace("_", " ")
                    ax.plot(years, ys, label=label, alpha=0.85)
                ax.set_yscale("symlog")
                ax.set_xlabel("Año"); ax.set_ylabel("N individuos")
                ax.set_title("Población por especie")
                ax.legend(fontsize=7, loc="best", ncols=2, framealpha=0.7)
                fig.tight_layout()
                p = plots_dir / "species_populations.png"
                fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

            # 3) Diversidad genética π por especie
            pi_cols = [k for k in rows[0].keys() if k.startswith("pi_")]
            if pi_cols:
                fig, ax = plt.subplots(figsize=(11, 5))
                for col in pi_cols:
                    ys = [float(r[col]) if r.get(col) not in (None, "", "None")
                          else None for r in rows]
                    if all(v is None or v == 0 for v in ys):
                        continue
                    ax.plot(years, ys, label=col.replace("pi_", "sp "), alpha=0.85)
                ax.set_xlabel("Año"); ax.set_ylabel("π (nucleotide diversity)")
                ax.set_title("Diversidad genética por especie")
                ax.legend(fontsize=7, loc="best", ncols=2, framealpha=0.7)
                fig.tight_layout()
                p = plots_dir / "genetic_diversity.png"
                fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

    # 4) Rendimiento: segundos por año y desglose
    if perf_rows:
        yrs = [r["year"] for r in perf_rows]
        wall = [r["wall_s"] for r in perf_rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=yrs, y=wall, ax=ax, color="#c0392b")
        ax.set_xlabel("Año simulado"); ax.set_ylabel("Segundos de reloj / año")
        ax.set_title("Rendimiento del motor")
        fig.tight_layout()
        p = plots_dir / "perf_wall.png"
        fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

        keys = [k for k in perf_rows[0].keys()
                if k not in ("year", "wall_s", "n_pops", "total_pop")]
        import numpy as np
        stacks = np.array([[r.get(k, 0.0) for k in keys] for r in perf_rows]).T
        fig, ax = plt.subplots(figsize=(11, 6))
        palette = sns.color_palette("tab20", n_colors=len(keys))
        ax.stackplot(yrs, stacks, labels=keys, colors=palette, alpha=0.9)
        ax.set_xlabel("Año simulado"); ax.set_ylabel("Segundos por fase")
        ax.set_title("Desglose de tiempo por año")
        ax.legend(fontsize=8, loc="upper left", ncols=2)
        fig.tight_layout()
        p = plots_dir / "perf_breakdown.png"
        fig.savefig(p, dpi=110); plt.close(fig); saved.append(p)

    print(f"  [plots] {len(saved)} figuras en {plots_dir}/")
    return saved


def dump_perf_csv(out_dir: str | Path, perf_rows: list[dict]) -> Path | None:
    if not perf_rows:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in perf_rows for k in r.keys()})
    path = out_dir / "perf.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in perf_rows:
            w.writerow(r)
    return path
