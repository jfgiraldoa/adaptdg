"""Plot ./out/<preset>: error versus time and DoF versus time for BE and BDF2 (both targets), final-time error versus DoF^(1/2)
(adaptive versus uniform meshes) and the adaptive meshes at the first, middle and last step."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"
TARGETS = (("", "exact solution"), ("_proj", "DG(p) L2 projection"))
STYLE = {"be": ("C0", "BE"), "bdf2": ("C3", "BDF2")}


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def runs(folder: Path) -> dict[int, list[tuple[str, str, Path]]]:
    """``{p: [(integrator, mode, steps csv)]}`` from the ``steps_*.csv`` files of one preset folder."""
    found: dict[int, list[tuple[str, str, Path]]] = {}
    for f in sorted(folder.glob("steps_p*_*.csv")):
        m = re.match(r"steps_p(\d+)_(be|bdf2)_(\w+)\.csv", f.name)
        found.setdefault(int(m[1]), []).append((m[2], m[3], f))
    return found


def plot_time(folder: Path, p: int, files: list[tuple[str, str, Path]]) -> Path:
    """Error of ubar and phi versus time (L2 and Vh, both targets) and DoF per step, adaptive runs only."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for integ, mode, f in files:
        if mode != "adaptive":
            continue
        h, (color, name) = read(f), STYLE[integ]
        for j, (suffix, label) in enumerate(TARGETS):
            axes[j, 0].semilogy(h["t"], h["L2" + suffix], "o-", ms=3, color=color, label=f"{name} ubar")
            axes[j, 0].semilogy(h["t"], h["L2_phi" + suffix], "s--", ms=3, color=color, alpha=0.6, label=f"{name} phi")
            axes[j, 1].semilogy(h["t"], h["Vh" + suffix], "o-", ms=3, color=color, label=f"{name} ubar")
            axes[j, 1].semilogy(h["t"], h["Vh_phi" + suffix], "s--", ms=3, color=color, alpha=0.6, label=f"{name} phi")
            axes[j, 0].set_ylabel(f"L2 error ({label})")
            axes[j, 1].set_ylabel(f"Vh error ({label})")
        axes[0, 2].plot(h["t"], h["ndofs"], "o-", ms=3, color=color, label=name)
        axes[1, 2].plot(h["t"], h["levels"], "o-", ms=3, color=color, label=name)
    axes[0, 2].set_ylabel("DoF at the end of the step")
    axes[1, 2].set_ylabel("solves (levels) per step")
    for ax in axes[1]:
        ax.set_xlabel("t")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Unsteady Eriksson-Johnson, p = {p}, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"time_p{p}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_final(folder: Path, p: int, files: list[tuple[str, str, Path]]) -> Path | None:
    """Final-time error versus DoF^(1/2) of every run (adaptive: one point; uniform: one point per mesh)."""
    if not any(mode.startswith("uniform") for _, mode, _ in files):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, base in zip(axes, ("L2", "Vh")):
        for integ, color_name in STYLE.items():
            color, name = color_name
            uni = sorted((read(f) for i, m, f in files if i == integ and m.startswith("uniform")), key=lambda h: h["ndofs"][-1])
            if uni:
                ax.loglog([np.sqrt(h["ndofs"][-1]) for h in uni], [h[base][-1] for h in uni], "x--", color=color, label=f"{name} uniform")
            for i, m, f in files:
                if i == integ and m == "adaptive":
                    h = read(f)
                    ax.loglog(np.sqrt(h["ndofs"][-1]), h[base][-1], "o", ms=7, color=color, label=f"{name} adaptive")
        ax.set_xlabel("DoF$^{1/2}$ at the final time")
        ax.set_ylabel(f"{base} error of ubar at the final time")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Unsteady Eriksson-Johnson, p = {p}, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"final_p{p}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_meshes(folder: Path, p: int, integ: str) -> Path | None:
    files = sorted(folder.glob(f"mesh_p{p}_{integ}_adaptive_s*.npz"), key=lambda f: int(re.search(r"_s(\d+)", f.name)[1]))
    if not files:
        return None
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 4.2), squeeze=False)
    for ax, f in zip(axes[0], files):
        d = np.load(f)
        ax.triplot(mtri.Triangulation(d["x"][:, 0], d["x"][:, 1], d["cells"]), lw=0.2, color="k")
        ax.set_aspect("equal")
        ax.set_title(f"t = {float(d['t']):.3f}, {len(d['cells'])} cells", fontsize=9)
    fig.suptitle(f"{STYLE[integ][1]}, p = {p}, {folder.name} preset")
    fig.tight_layout()
    path = folder / f"meshes_p{p}_{integ}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    for folder in sorted(d for d in OUT.glob("*") if d.is_dir()):
        for p, files in runs(folder).items():
            made = [plot_time(folder, p, files), plot_final(folder, p, files)]
            made += [plot_meshes(folder, p, integ) for integ in STYLE]
            print(folder.name, p, [m.name for m in made if m])


if __name__ == "__main__":
    main()
