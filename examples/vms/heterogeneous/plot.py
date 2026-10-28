"""Plot ./out: error versus DoF^(1/2) for ubar, u (= dG) and phi (L2 and Vh) and the final adaptive mesh."""

import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"
QUANTITIES = {"": ("ubar", "o-"), "_u": ("u = ubar + utilde (dG)", "x--"), "_phi": ("phi", "s:")}


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main() -> None:
    written = []
    for mode in ("uniform", "adaptive"):
        files = sorted(OUT.glob(f"history_p*_{mode}.csv"))
        if not files:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, norm in zip(axes, ("L2", "Vh")):
            for color, f in zip(plt.cm.tab10.colors, files):
                p = int(f.stem.split("_")[1][1:])
                h = read(f)
                x = np.sqrt(h["ndofs"])
                for suffix, (label, style) in QUANTITIES.items():
                    ax.loglog(x, h[norm + suffix], style, color=color, ms=3, lw=0.9,
                              label=f"p={p} {label}" if p == 1 else None)
                slope = p + (norm == "L2")  # reference slope -p (Vh) or -(p+1) (L2)
                xr = np.array([x[-1] / 2, x[-1]])
                ax.loglog(xr, h[norm][-1] / 3 * (xr / x[-1]) ** -slope, "k:", lw=0.8)
                ax.text(xr[0], h[norm][-1] / 3 * 2**slope, f"-{slope}", fontsize=7)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("DoF$^{1/2}$")
            ax.set_ylabel(f"{norm} error")
            ax.legend(fontsize=6)
        fig.suptitle(f"heterogeneous diffusion, {mode} refinement")
        fig.tight_layout()
        fig.savefig(OUT / f"errors_{mode}.png", dpi=150)
        written.append(f"errors_{mode}.png")
    meshes = sorted(OUT.glob("mesh_p*_adaptive.npz"))
    if meshes:
        data = np.load(meshes[0])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.triplot(data["x"][:, 0], data["x"][:, 1], data["cells"], lw=0.2, color="k")
        ax.axvline(0.5, color="r", lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(meshes[0].stem)
        fig.savefig(OUT / "final_mesh.png", dpi=200)
        written.append("final_mesh.png")
    print("wrote", ", ".join(written))


if __name__ == "__main__":
    main()
