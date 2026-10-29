"""Plot ./out/<preset>: ubar field and last mesh, error versus DoF^(1/2) (L2 and Vh, both targets, reference slopes -p and
-(p+1)), the estimator E, and the minimum and maximum of ubar per level."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"
COLORS = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
TARGETS = (("", "exact solution"), ("_proj", "DG(p) L2 projection"))


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def runs(folder: Path) -> dict[str, list[tuple[int, Path]]]:
    found: dict[str, list[tuple[int, Path]]] = {}
    for f in sorted(folder.glob("history_k*_p*.csv")):
        m = re.match(r"history_k([\d.e+-]+)_p(\d+)\.csv", f.name)
        found.setdefault(m[1], []).append((int(m[2]), f))
    return found


def plot_errors(folder: Path, kappa: str, files: list[tuple[int, Path]]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for row, (suffix, label) in zip(axes, TARGETS):
        for ax, base in zip(row, ("Vh", "L2")):
            for p, f in files:
                h = read(f)
                x = h["ndofs"] ** 0.5
                bad = h["newton_converged"] == 0
                ax.loglog(x, h[base + suffix], "o-", ms=3, color=COLORS.get(p, "k"), label=f"p={p} ubar")
                ax.loglog(x[bad], h[base + suffix][bad], "x", ms=6, color="r")  # Newton did not converge
                if f"{base}_phi{suffix}" in h:
                    ax.loglog(x, h[f"{base}_phi{suffix}"], "--", color=COLORS.get(p, "k"), lw=0.8, label=f"p={p} phi")
                s = p + (base == "L2")
                xr = np.array([x[-1] / 2, x[-1]])
                ax.loglog(xr, h[base + suffix][-1] / 3 * (xr / x[-1]) ** -s, "k:", lw=0.8)
                ax.text(xr[0], h[base + suffix][-1] / 3 * 2**s, f"-{s}", fontsize=7)
            ax.set_ylabel(f"{base} error ({label})")
    for ax in axes[1]:
        ax.set_xlabel("DoF$^{1/2}$")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"Isotropic Burgers, kappa = {kappa}, {folder.name} preset (red x: Newton not converged)")
    fig.tight_layout()
    path = folder / f"errors_k{kappa}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_fields(folder: Path, kappa: str, files: list[tuple[int, Path]]) -> Path:
    p = files[0][0]
    tag = f"k{kappa}_p{p}"
    mesh, field = np.load(folder / f"mesh_{tag}.npz"), np.load(folder / f"field_{tag}.npz")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    tri = mtri.Triangulation(field["x"][:, 0], field["x"][:, 1])
    cs = axes[0, 0].tricontourf(tri, field["u"], levels=np.linspace(-0.1, 1.1, 25), cmap="viridis", extend="both")
    fig.colorbar(cs, ax=axes[0, 0])
    axes[0, 0].set_title(f"ubar, p={p} (min {field['u'].min():.3f}, max {field['u'].max():.3f})")
    axes[0, 1].triplot(mtri.Triangulation(mesh["x"][:, 0], mesh["x"][:, 1], mesh["cells"]), lw=0.2, color="k")
    axes[0, 1].set_title(f"last adaptive mesh ({len(mesh['cells'])} cells)")
    for ax in axes[0]:
        ax.set_aspect("equal")
    for p, f in files:
        h = read(f)
        axes[1, 0].semilogy(h["level"] + 1, h["E"], "o-", ms=3, color=COLORS.get(p, "k"), label=f"p={p}")
        axes[1, 1].plot(h["level"] + 1, h["umin"], "o-", ms=3, color=COLORS.get(p, "k"), label=f"p={p} min")
        axes[1, 1].plot(h["level"] + 1, h["umax"], "s--", ms=3, color=COLORS.get(p, "k"), label=f"p={p} max")
    axes[1, 0].set_ylabel("estimator E")
    axes[1, 1].set_ylabel("min / max of ubar (exact: 0 < u < 1)")
    for ax in axes[1]:
        ax.set_xlabel("level")
        ax.legend(fontsize=6)
    fig.suptitle(f"Isotropic Burgers, kappa = {kappa}, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"fields_k{kappa}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    written = []
    for folder in sorted(p for p in OUT.glob("*") if p.is_dir()):
        for kappa, files in runs(folder).items():
            written += [plot_errors(folder, kappa, files), plot_fields(folder, kappa, files)]
    print("wrote " + ", ".join(str(w) for w in written) if written else "no runs in out/: run python run.py first")


if __name__ == "__main__":
    main()
