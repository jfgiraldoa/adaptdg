"""Plot ./out/<preset>: error versus DoF^(1/3) (L2 and Vh, both targets, adaptive versus uniform,
reference slopes -p and -(p+1)) and the cell size versus distance to the corner at the last level."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"
COLORS = {1: "C0", 2: "C1", 3: "C2"}
TARGETS = (("", "exact solution"), ("_proj", "DG(p) L2 projection"))


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def runs(folder: Path) -> dict[float, list[tuple[int, str, Path]]]:
    """``{q: [(p, refinement, csv)]}`` from the history files of one preset folder."""
    found: dict[float, list[tuple[int, str, Path]]] = {}
    for f in sorted(folder.glob("history_q*_p*_*.csv")):
        m = re.match(r"history_q([\d.]+)_p(\d+)_(\w+)\.csv", f.name)
        found.setdefault(float(m[1]), []).append((int(m[2]), m[3], f))
    return found


def plot_errors(folder: Path, q: float, files: list[tuple[int, str, Path]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for row, (suffix, label) in zip(axes, TARGETS):
        for ax, base in zip(row, ("Vh", "L2")):
            for p, mode, f in files:
                h = read(f)
                x = h["ndofs"] ** (1 / 3)
                ax.loglog(x, h[base + suffix], "o-" if mode == "adaptive" else "x--", ms=3, color=COLORS.get(p, "k"),
                          label=f"p={p} {mode}")
                if mode == "adaptive":  # reference slope -p (Vh) or -(p+1) (L2) below the last point
                    s = p + (base == "L2")
                    xr = np.array([x[-1] / 2, x[-1]])
                    ax.loglog(xr, h[base + suffix][-1] / 3 * (xr / x[-1]) ** -s, "k:", lw=0.8)
                    ax.text(xr[0], h[base + suffix][-1] / 3 * 2**s, f"-{s}", fontsize=7)
            ax.set_ylabel(f"{base} error of ubar ({label})")
    for ax in axes[1]:
        ax.set_xlabel("DoF$^{1/3}$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Fichera 3D, q = {q:.3g}, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"errors_q{q:.2f}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cell_size(folder: Path, q: float, files: list[tuple[int, str, Path]]) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = np.logspace(-2, np.log10(1.8), 14)
    for p, mode, f in files:
        if mode != "adaptive":
            continue
        d = np.load(f.with_name(f.name.replace("history_", "cells_").replace(".csv", ".npz")))
        idx = np.digitize(d["r"], bins)
        mean = [d["h"][idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))]
        ax.loglog(np.sqrt(bins[1:] * bins[:-1]), mean, "o-", ms=3, color=COLORS.get(p, "k"), label=f"p={p}, adaptive")
    ax.set_xlabel("distance r to the corner")
    ax.set_ylabel("mean cell size $|K|^{1/3}$ (last level)")
    ax.legend(fontsize=7)
    ax.set_title(f"q = {q:.3g}, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"cell_size_q{q:.2f}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    written = []
    for folder in sorted(p for p in OUT.glob("*") if p.is_dir()):
        for q, files in runs(folder).items():
            written += [plot_errors(folder, q, files), plot_cell_size(folder, q, files)]
    print("wrote " + ", ".join(str(w) for w in written) if written else "no runs in out/: run python run.py first")


if __name__ == "__main__":
    main()
