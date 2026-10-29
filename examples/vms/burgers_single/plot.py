"""Plot ./out/<preset>: ubar field, last adaptive mesh, profile along y = 0.25 and 0.75 (shock and min/max), and the
estimator E and the Newton iterations per level."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"


def read(path: Path) -> dict[str, list]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: [r[k] for r in rows] for k in rows[0]}


def plot_run(folder: Path, hist: Path) -> Path:
    m = re.match(r"history_k([\d.e+-]+)_p(\d+)\.csv", hist.name)
    kappa, p = m[1], m[2]
    tag = f"k{kappa}_p{p}"
    h = read(hist)
    mesh, field = np.load(folder / f"mesh_{tag}.npz"), np.load(folder / f"field_{tag}.npz")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    tri = mtri.Triangulation(field["x"][:, 0], field["x"][:, 1])
    cs = ax.tricontourf(tri, field["u"], levels=np.linspace(-1.1, 1.1, 23), cmap="RdBu_r", extend="both")
    fig.colorbar(cs, ax=ax)
    ax.set_title(f"ubar, kappa={kappa}, p={p} (min {field['u'].min():.3f}, max {field['u'].max():.3f})")
    ax.set_aspect("equal")
    ax = axes[0, 1]
    xyz, cells = mesh["x"][:, :2], mesh["cells"]
    ax.triplot(mtri.Triangulation(xyz[:, 0], xyz[:, 1], cells), lw=0.2, color="k")
    ax.set_aspect("equal")
    ax.set_title(f"last adaptive mesh ({len(cells)} cells)")
    ax = axes[0, 2]
    for key, yy in (("line_y25", 0.25), ("line_y75", 0.75)):
        ax.plot(field["line_x"], field[key], label=f"y={yy}")
    ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("ubar")
    ax.set_title("profile along x (shock at x = 1/2)")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    nd = np.array(h["ndofs"], dtype=float)
    ax.loglog(nd ** 0.5, np.array(h["E"], dtype=float), "o-", ms=3)
    ax.set_xlabel("DoF$^{1/2}$")
    ax.set_ylabel("estimator E")
    ax = axes[1, 1]
    its = np.array(h["newton_iterations"], dtype=int)
    ok = np.array(h["newton_converged"], dtype=int) == 1
    ax.bar(np.arange(1, len(its) + 1), its, color=np.where(ok, "C0", "C3"))
    ax.set_xlabel("level")
    ax.set_ylabel("Newton iterations (red: not converged)")
    ax = axes[1, 2]
    ax.plot(np.arange(1, len(nd) + 1), np.array(h["umin"], dtype=float), "o-", ms=3, label="min ubar")
    ax.plot(np.arange(1, len(nd) + 1), np.array(h["umax"], dtype=float), "s-", ms=3, label="max ubar")
    ax.axhline(-1, color="gray", ls=":", lw=0.8)
    ax.axhline(1, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("level")
    ax.legend(fontsize=8)
    fig.suptitle(f"Single-component Burgers, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"burgers_single_k{kappa}_p{p}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    written = []
    for folder in sorted(p for p in OUT.glob("*") if p.is_dir()):
        written += [plot_run(folder, f) for f in sorted(folder.glob("history_k*_p*.csv"))]
    print("wrote " + ", ".join(str(w) for w in written) if written else "no runs in out/: run python run.py first")


if __name__ == "__main__":
    main()
