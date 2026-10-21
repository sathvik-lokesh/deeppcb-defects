"""Data-curve figure for the README: test mAP50-95 vs training images, v1 (mosaic on) next to v2 (mosaic off).

Reads runs/compare.json and runs/compare_v2.json (test 5 curve = mean over the seeds that exist).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = ["25", "50", "100", "225", "450", "900"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"A": "#2a78d6", "T": "#eb6834"}
NAME = {"A": "A: image only", "T": "T: + template"}
PANELS = [("runs/compare.json", "v1: mosaic on (pre-registered recipe)"),
          ("runs/compare_v2.json", "v2: mosaic off")]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=160, facecolor=SURF, sharey=True)
x = list(range(len(SIZES)))
for ax, (path, title) in zip(axes, PANELS):
    curve = json.load(open(path))["test5"]["curve"]
    ax.set_facecolor(SURF)
    for arm in ("A", "T"):
        y = [curve[arm][n] for n in SIZES]
        ax.plot(x, y, color=COL[arm], lw=2, marker="o", ms=7, mec=SURF, mew=2, zorder=3)
        ax.annotate(NAME[arm], (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    ax.set_xticks(x, SIZES)
    ax.set_xlabel("training images (each set repeated to 900 per epoch)", color=INK2, fontsize=9)
    ax.set_xlim(-0.3, len(SIZES) + 0.9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
axes[0].set_ylabel("test mAP50-95 (points)", color=INK2)
axes[0].set_ylim(0, 88)
fig.tight_layout()
fig.savefig("runs/curve.png", facecolor=SURF)
print("runs/curve.png")
