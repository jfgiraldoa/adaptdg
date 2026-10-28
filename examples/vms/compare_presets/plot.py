"""Plot ./out: errors of ubar, theta, phi (rows Vh/L2, columns exact/projection target; solid = paper
preset, dashed = legacy preset), one figure per case, and a summary of Vh(phi)/Vh(theta)."""

import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path(__file__).parent / "out"
PRESETS = {"paper": "-", "legacy": "--"}
FIELDS = {"": ("ubar", "tab:blue"), "_theta": ("theta (dG)", "tab:green"), "_phi": ("phi", "tab:red")}
MARKERS = {1: "o", 2: "^"}
TARGETS = {"exact": "", "projection": "_proj"}


def read(path: Path) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def case_figure(case: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, norm in enumerate(("Vh", "L2")):
        for col, (target, suffix) in enumerate(TARGETS.items()):
            ax = axes[row, col]
            for preset, style in PRESETS.items():
                for f in sorted((case / preset).glob("history_p*_*.csv")):
                    p = int(f.stem.split("_")[1][1:])
                    h = read(f)
                    for fsuffix, (_, color) in FIELDS.items():
                        ax.loglog(np.sqrt(h["ndofs"]), h[f"{norm}{fsuffix}{suffix}"], style, color=color,
                                  marker=MARKERS[p], ms=3, lw=0.9)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(f"{norm} error, target: {'exact solution' if target == 'exact' else 'L2 projection of the exact solution'}",
                         fontsize=9)
            ax.set_ylabel(f"{norm} error")
            if row == 1:
                ax.set_xlabel("DoF$^{1/2}$")
    handles = [Line2D([], [], color=c, lw=1.5, label=n) for n, c in FIELDS.values()]
    handles += [Line2D([], [], color="k", ls=s, lw=1.2, label=f"{k} preset") for k, s in PRESETS.items()]
    handles += [Line2D([], [], color="k", ls="", marker=m, ms=4, label=f"p={p}") for p, m in MARKERS.items()]
    axes[0, 0].legend(handles=handles, fontsize=7)
    fig.suptitle(case.name.replace("_", " "))
    fig.tight_layout()
    name = f"errors_{case.name}.png"
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)
    return name


def summary_figure() -> str:
    with open(OUT / "summary.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["norm"] == "Vh" and r["field"] == "phi"]
    groups = sorted({(r["case"], int(r["p"])) for r in rows})
    combos = [(pr, tg) for pr in PRESETS for tg in TARGETS]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(combos)
    for j, (pr, tg) in enumerate(combos):
        vals = [float(next(r["ratio_Vh_phi_over_theta"] for r in rows if (r["case"], int(r["p"])) == g
                           and r["preset"] == pr and r["target"] == tg)) for g in groups]
        ax.bar(np.arange(len(groups)) + (j - 1.5) * width, vals, width, label=f"{pr} preset, {tg} target",
               hatch="//" if pr == "legacy" else None, color=plt.cm.tab10.colors[j % 4])
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(groups)), [f"{c.replace('_', ' ')}\np={p}" for c, p in groups], fontsize=7)
    ax.set_ylabel("Vh(phi) / Vh(theta), finest level")
    ax.set_ylim(0, 1.7)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    name = "ratio_Vh_phi_theta.png"
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)
    return name


def main() -> None:
    written = [case_figure(c) for c in sorted(OUT.iterdir()) if c.is_dir()]
    written.append(summary_figure())
    print("wrote", ", ".join(written))


if __name__ == "__main__":
    main()
