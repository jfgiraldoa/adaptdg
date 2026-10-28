"""Plot ./out/<preset>: error versus DoF^(1/d) (L2 and Vh, both targets, adaptive versus uniform, reference
slopes -p and -(p+1)), the last adaptive mesh (2D) and the cell size versus the distance to the outflow x = 1."""

import csv
import re
from math import factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"
COLORS = {1: "C0", 2: "C1", 3: "C2"}
TARGETS = (("", "exact solution"), ("_proj", "DG(p) L2 projection"))


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def runs(folder: Path) -> dict[str, list[tuple[int, str, Path]]]:
    """``{kappa: [(p, refinement, csv)]}`` from the history files of one preset folder."""
    found: dict[str, list[tuple[int, str, Path]]] = {}
    for f in sorted(folder.glob("history_k*_p*_*.csv")):
        m = re.match(r"history_k([\d.e+-]+)_p(\d+)_(\w+)\.csv", f.name)
        found.setdefault(m[1], []).append((int(m[2]), m[3], f))
    return found


def plot_errors(folder: Path, kappa: str, files: list[tuple[int, str, Path]], dim: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for row, (suffix, label) in zip(axes, TARGETS):
        for ax, base in zip(row, ("Vh", "L2")):
            for p, mode, f in files:
                h = read(f)
                x = h["ndofs"] ** (1 / dim)
                ax.loglog(x, h[base + suffix], "o-" if mode == "adaptive" else "x--", ms=3,
                          color=COLORS.get(p, "k"), label=f"p={p} {mode}")
                if mode == "adaptive":  # reference slope -p (Vh) or -(p+1) (L2) below the last point
                    s = p + (base == "L2")
                    xr = np.array([x[-1] / 2, x[-1]])
                    ax.loglog(xr, h[base + suffix][-1] / 3 * (xr / x[-1]) ** -s, "k:", lw=0.8)
                    ax.text(xr[0], h[base + suffix][-1] / 3 * 2**s, f"-{s}", fontsize=7)
            ax.set_ylabel(f"{base} error of ubar ({label})")
    for ax in axes[1]:
        ax.set_xlabel(f"DoF$^{{1/{dim}}}$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Eriksson-Johnson {dim}D, kappa = {kappa}, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"errors_k{kappa}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mesh(folder: Path, kappa: str, files: list[tuple[int, str, Path]], dim: int) -> Path:
    """Last adaptive mesh of the lowest degree (2D only) and the cell size versus 1 - x for every adaptive run."""
    fig, (ax_m, ax_h) = plt.subplots(1, 2, figsize=(11, 4.5))
    adaptive = [(p, f) for p, mode, f in files if mode == "adaptive"]
    for p, f in adaptive:
        d = np.load(f.with_name(f.name.replace("history_", "mesh_").replace(".csv", ".npz")))
        xyz, cells = d["x"][:, :dim], d["cells"]
        if dim == 2 and p == adaptive[0][0]:
            ax_m.triplot(mtri.Triangulation(xyz[:, 0], xyz[:, 1], cells), lw=0.2, color="k")
            ax_m.set_aspect("equal")
            ax_m.set_title(f"last adaptive mesh, p={p}")
        e = xyz[cells[:, 1:]] - xyz[cells[:, :1]]
        h = np.abs(np.linalg.det(e)) ** (1.0 / dim) / factorial(dim) ** (1.0 / dim)
        dist = 1.0 - xyz[cells][:, :, 0].mean(axis=1)
        bins = np.logspace(-4, 0, 17)
        idx = np.digitize(dist, bins)
        mean = [h[idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))]
        ax_h.loglog(np.sqrt(bins[1:] * bins[:-1]), mean, "o-", ms=3, color=COLORS.get(p, "k"), label=f"p={p}")
    ax_h.set_xlabel("distance 1 - x to the outflow boundary")
    ax_h.set_ylabel("mean cell size (last adaptive level)")
    ax_h.legend(fontsize=7)
    fig.suptitle(f"kappa = {kappa}, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"mesh_k{kappa}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    written = []
    for folder in sorted(p for p in OUT.glob("*") if p.is_dir()):
        dim = 3 if folder.name.endswith("_3d") else 2
        for kappa, files in runs(folder).items():
            written += [plot_errors(folder, kappa, files, dim), plot_mesh(folder, kappa, files, dim)]
    print("wrote " + ", ".join(str(w) for w in written) if written else "no runs in out/: run python run.py first")


if __name__ == "__main__":
    main()
