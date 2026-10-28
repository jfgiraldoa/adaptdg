"""Plot ./out/<preset>: error versus DoF^(1/2) of CG(p) Galerkin and of the residual-minimisation ubar, theta, phi
(L2 and Vh, both targets) on the adaptive and on the uniform meshes of the residual-minimisation loop, plus the
Galerkin-only uniform sequence."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"
TARGETS = (("", "exact solution"), ("_proj", "DG(p) L2 projection"))
CURVES = (("", "RM ubar", "C0", "o-"), ("_theta", "RM theta", "C2", "s-"), ("_phi", "RM phi", "C3", "^-"),
          ("_cg", "CG Galerkin", "k", "x--"))


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def plot_degree(folder: Path, p: int) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for col, (tag, title) in enumerate((("rm_adaptive", "adaptive meshes"), ("rm_uniform", "uniform meshes"))):
        h = read(folder / f"history_p{p}_{tag}.csv")
        x = h["ndofs"] ** 0.5
        for row, (suffix, label) in enumerate(TARGETS):
            for j, base in enumerate(("Vh", "L2")):
                ax = axes[row, 2 * col + j]
                for field, name, color, style in CURVES:
                    ax.loglog(x, h[f"{base}{field}{suffix}"], style, ms=3, color=color, label=name)
                s = p + (base == "L2")
                ax.loglog(x[[-2, -1]], h[f"{base}{suffix}"][-1] / 2 * (x[[-2, -1]] / x[-1]) ** -s, "k:", lw=0.8)
                ax.set_title(f"{base} vs {label[:5]}, {title}", fontsize=8)
    gp = folder / f"history_p{p}_galerkin_uniform.csv"
    if gp.exists():  # Galerkin-only sequence (reaches larger DoF) on the Vh/L2 exact panels of the uniform meshes
        g = read(gp)
        for j, base in enumerate(("Vh", "L2")):
            axes[0, 2 + j].loglog(g["ndofs"] ** 0.5, g[base], "x:", color="gray", label="CG uniform (own loop)")
    for ax in axes[1]:
        ax.set_xlabel("DoF$^{1/2}$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Re-entrant corner, p = {p}: CG(p) Galerkin vs residual minimisation, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"compare_p{p}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    written = []
    for folder in sorted(p for p in OUT.glob("*") if p.is_dir()):
        degrees = sorted({int(m[1]) for f in folder.glob("history_p*_rm_adaptive.csv")
                          if (m := re.match(r"history_p(\d+)_", f.name))})
        written += [plot_degree(folder, p) for p in degrees]
    print("wrote " + ", ".join(str(w) for w in written) if written else "no runs in out/: run python run.py first")


if __name__ == "__main__":
    main()
