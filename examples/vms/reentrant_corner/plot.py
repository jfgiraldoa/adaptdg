"""Plot ./out: error versus DoF^(1/2) (L2 and Vh, reference slopes) and the final adaptive mesh."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main() -> None:
    files = sorted(OUT.glob("history_p*_*.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("L2", "Vh")):
        for f in files:
            h = read(f)
            p, mode = f.stem.split("_")[1:3]
            x = np.sqrt(h["ndofs"])
            ax.loglog(x, h[key], "o-" if mode == "adaptive" else "x--", ms=3, label=f"{p} {mode}")
            if mode == "adaptive":  # reference slope -p (Vh) or -(p+1) (L2), below the curve
                slope = int(p[1:]) + (key == "L2")
                xr = np.array([x[-1] / 2, x[-1]])
                ax.loglog(xr, h[key][-1] / 3 * (xr / x[-1]) ** -slope, "k:", lw=0.8)
                ax.text(xr[0], h[key][-1] / 3 * 2**slope, f"-{slope}", fontsize=7)
        ax.set_xlabel("DoF$^{1/2}$")
        ax.set_ylabel("L2 error" if key == "L2" else "Vh error")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "errors.png", dpi=150)
    meshes = sorted(OUT.glob("mesh_p*_adaptive.npz"))
    if meshes:
        data = np.load(meshes[-1])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.triplot(data["x"][:, 0], data["x"][:, 1], data["cells"], lw=0.2, color="k")
        ax.set_aspect("equal")
        ax.set_title(meshes[-1].stem)
        fig.savefig(OUT / "final_mesh.png", dpi=200)
    print(f"wrote {OUT / 'errors.png'}" + (f" and {OUT / 'final_mesh.png'}" if meshes else ""))


if __name__ == "__main__":
    main()
