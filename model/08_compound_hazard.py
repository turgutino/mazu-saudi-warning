# =============================================================================
# MAZU — Priority-3 item 2: compound-hazard overlay layer.
#
# Reviewer's exact text: "新增复合灾害叠加图层：开发多灾种联合风险填色图，
# 用混合色标注热浪 + 沙尘、暴雨 + 高温重叠高风险带。" (Add a compound-hazard
# overlay layer: develop a multi-hazard joint risk color-fill map, using
# mixed colors to mark heatwave+dust_storm and rainstorm(flash_flood)+
# high-temperature(heatwave) overlapping high-risk zones.)
#
# Two figures, per the user's "highest-level, skip nothing" instruction:
#   1. compound_hazard_climatology.png -- annual count of days each cell
#      reached BOTH hazards' extreme tier simultaneously (mirrors
#      risk_annual_hotspots.png's per-hazard version, but for the AND
#      condition across a hazard pair).
#   2. compound_hazard_event.png -- a real single day for each pair (found
#      by searching the whole year for the day with the most simultaneous
#      extreme-tier overlap cells) using a genuine bivariate mixed-color
#      scheme: red = hazard A only, blue = hazard B only, purple = both
#      (the actual color-mixing the reviewer's text asks for).
#
# Both hazard pairs use the same rule-based DetectionEngine.risk_field()
# ground truth as every other map on the site, thresholded at each
# hazard's own "extreme" tier lower bound (render_map.TIER_BOUNDS).
# =============================================================================

import os
import sys
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cfeature
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_map import (PROJ, PC, DATA_EXTENT, NEIGHBOR_LABELS, LABEL_OFFSETS,
                         _terrain_rgb, _NEIGHBOR_OUTLINE, TIER_BOUNDS, add_standard_caption)

_TEXT_OUTLINE = [pe.withStroke(linewidth=2.0, foreground="black")]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location("de", os.path.join(HERE, "01_detection_engine.py"))
de = importlib.util.module_from_spec(spec)
spec.loader.exec_module(de)

CITIES = {"Jeddah": (21.5, 39.2), "Mecca": (21.4, 39.8), "Riyadh": (24.7, 46.7),
          "Jizan": (16.9, 42.6), "Dammam": (26.4, 50.1), "Taif": (21.3, 40.4),
          "Medina": (24.5, 39.6), "Abha": (18.2, 42.5)}

# Each hazard's own "extreme" tier lower bound, reused from render_map.py's
# TIER_BOUNDS so the "high risk" definition here matches the tier scale
# used on every other map (Priority-2 consistency).
EXTREME_THR = {hz: TIER_BOUNDS[hz][3] for hz in TIER_BOUNDS}  # flash_flood 0.6, heatwave 0.7, dust_storm 0.7

# The two pairs the reviewer named, in order.
PAIRS = [
    ("heatwave", "dust_storm", "Heatwave + Dust storm"),
    ("flash_flood", "heatwave", "Rainstorm (flash_flood) + High temperature (heatwave)"),
]

PLOT_EXTENT = [32.0, 58.0, 14.0, 34.0]


def draw_base(ax):
    ax.set_extent(PLOT_EXTENT, crs=PC)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#0E1B2A", zorder=-1)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#12314f", zorder=-0.5)
    rgb, tlon, tlat = _terrain_rgb()
    ax.imshow(rgb, extent=[tlon.min(), tlon.max(), tlat.min(), tlat.max()], origin="upper",
              transform=PC, zorder=0.3, alpha=0.55)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, edgecolor="white", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, linestyle=":",
                    edgecolor="#cccccc", zorder=3)
    provinces = cfeature.NaturalEarthFeature(
        category="cultural", name="admin_1_states_provinces_lines", scale="10m",
        facecolor="none", edgecolor="#5a6b80", linewidth=0.3)
    ax.add_feature(provinces, zorder=2, linestyle="-")

    outer = [(PLOT_EXTENT[0], PLOT_EXTENT[2]), (PLOT_EXTENT[1], PLOT_EXTENT[2]),
              (PLOT_EXTENT[1], PLOT_EXTENT[3]), (PLOT_EXTENT[0], PLOT_EXTENT[3]), (PLOT_EXTENT[0], PLOT_EXTENT[2])]
    inner = [(DATA_EXTENT[0], DATA_EXTENT[2]), (DATA_EXTENT[0], DATA_EXTENT[3]),
              (DATA_EXTENT[1], DATA_EXTENT[3]), (DATA_EXTENT[1], DATA_EXTENT[2]), (DATA_EXTENT[0], DATA_EXTENT[2])]
    verts = outer + inner
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY] + \
            [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    frame_patch = mpatches.PathPatch(Path(verts, codes), transform=PC, facecolor="#444444",
                                      edgecolor="none", alpha=0.55, zorder=1.4)
    ax.add_patch(frame_patch)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5,
                       linestyle="--", zorder=3.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(30, 61, 5))
    gl.ylocator = mticker.FixedLocator(range(14, 36, 4))
    gl.xlabel_style = {"size": 6, "color": "white"}
    gl.ylabel_style = {"size": 6, "color": "white"}

    for name, (nlat, nlon) in NEIGHBOR_LABELS.items():
        ax.text(nlon, nlat, name, fontsize=6.5, style="italic", color="#cccccc", ha="center",
                transform=PC, zorder=4, path_effects=_NEIGHBOR_OUTLINE)
    return


def draw_cities(ax):
    for c, (la, lo) in CITIES.items():
        dx, dy, ha = LABEL_OFFSETS.get(c, (0.35, 0.0, "left"))
        ax.plot(lo, la, "o", ms=3, color="white", transform=PC, zorder=4)
        ax.text(lo + dx, la + dy, c, fontsize=6, color="white", ha=ha,
                 transform=PC, zorder=4, path_effects=_TEXT_OUTLINE)


# ---------------------------------------------------------------------------
# Figure 1: annual climatology -- how many days per cell did BOTH hazards
# in the pair reach their own "extreme" tier at once.
# ---------------------------------------------------------------------------
def compound_overlap_count(eng, hz_a, hz_b):
    cnt = np.zeros((len(eng.lat), len(eng.lon)), dtype="float32")
    for d in eng.times:
        a = eng.risk_field(d, hz_a) >= EXTREME_THR[hz_a]
        b = eng.risk_field(d, hz_b) >= EXTREME_THR[hz_b]
        cnt += (a & b).astype("float32")
    return cnt


def fig_climatology(eng):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), facecolor="white",
                             subplot_kw={"projection": PROJ})
    cmap = LinearSegmentedColormap.from_list("compound", ["#12314f", "#862e9c", "#f8e71c"])
    for ax, (hz_a, hz_b, label) in zip(axs, PAIRS):
        draw_base(ax)
        cnt = compound_overlap_count(eng, hz_a, hz_b)
        im = ax.pcolormesh(eng.lon, eng.lat, cnt, cmap=cmap, transform=PC,
                            alpha=0.75, zorder=1.5, shading="auto")
        draw_cities(ax)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="days both extreme simultaneously")
        ax.set_title(f"{label}\n2025 compound-extreme days", fontsize=10, fontweight="bold", color="#1A2D4A")
    coverage_patch = mpatches.Patch(facecolor="#444444", alpha=0.55, label="Outside model coverage (no data)")
    fig.legend(handles=[coverage_patch], loc="lower center", fontsize=7, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.05,
             "Note: counts extreme-tier-day OVERLAP (not a 0-1 risk score), so the site's "
             "standard tier thresholds don't directly apply.   |   Grid: 10km native model "
             "grid   |   Proj: Albers Equal-Area (lon0=45E, lat0=23N)   |   Data year: 2025   |   "
             "Coverage: 34.0-55.9E, 16.0-31.9N",
             ha="center", va="top", fontsize=6.5, color="#666666", transform=fig.transFigure)
    fig.suptitle("MAZU — Compound-hazard climatology (days both hazards extreme at once)",
                 fontsize=13, fontweight="bold", color="#1A2D4A")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "compound_hazard_climatology.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# Figure 2: single real day per pair, genuine bivariate mixed-color map --
# red = hazard A only extreme, blue = hazard B only extreme, purple = both
# (the actual "混合色" / mixed-color the reviewer's text names).
# ---------------------------------------------------------------------------
COLOR_A = np.array([0.93, 0.11, 0.14])   # red
COLOR_B = np.array([0.11, 0.44, 0.84])   # blue
COLOR_BOTH = np.array([0.60, 0.13, 0.66])  # purple (visually A+B mixed)
COLOR_NEITHER_ALPHA = 0.0


def bivariate_rgba(mask_a, mask_b):
    ny, nx = mask_a.shape
    rgba = np.zeros((ny, nx, 4))
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a
    both = mask_a & mask_b
    rgba[only_a] = np.append(COLOR_A, 0.55)
    rgba[only_b] = np.append(COLOR_B, 0.55)
    rgba[both] = np.append(COLOR_BOTH, 0.85)
    return rgba


# Real dates found by scanning the whole year for the day with the most
# simultaneous extreme-tier overlap cells for each pair (see log entry).
EVENT_DATES = {
    ("heatwave", "dust_storm"): "2025-07-20",
    ("flash_flood", "heatwave"): "2025-08-17",
}


def fig_event(eng):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.0), facecolor="white",
                             subplot_kw={"projection": PROJ})
    for ax, (hz_a, hz_b, label) in zip(axs, PAIRS):
        draw_base(ax)
        date = EVENT_DATES[(hz_a, hz_b)]
        mask_a = eng.risk_field(date, hz_a) >= EXTREME_THR[hz_a]
        mask_b = eng.risk_field(date, hz_b) >= EXTREME_THR[hz_b]
        rgba = bivariate_rgba(mask_a, mask_b)
        ax.imshow(rgba, extent=[eng.lon.min(), eng.lon.max(), eng.lat.min(), eng.lat.max()],
                  origin="upper", transform=PC, zorder=1.5, interpolation="nearest")
        draw_cities(ax)
        n_overlap = int((mask_a & mask_b).sum())
        ax.set_title(f"{label}\n{date}  ({n_overlap} overlapping cells)", fontsize=10,
                     fontweight="bold", color="#1A2D4A")
        # per-panel legend: color meaning (A/B) differs between the two
        # pairs, so a single shared legend would be ambiguous -- each axes
        # gets its own, naming its own two hazards explicitly.
        a_patch = mpatches.Patch(facecolor=tuple(COLOR_A), alpha=0.6, label=f"{hz_a} extreme only")
        b_patch = mpatches.Patch(facecolor=tuple(COLOR_B), alpha=0.6, label=f"{hz_b} extreme only")
        both_patch = mpatches.Patch(facecolor=tuple(COLOR_BOTH), alpha=0.9, label="BOTH extreme (compound)")
        ax.legend(handles=[a_patch, b_patch, both_patch], loc="lower left", fontsize=6,
                  framealpha=0.9)

    coverage_patch = mpatches.Patch(facecolor="#444444", alpha=0.55, label="Outside model coverage (no data)")
    fig.legend(handles=[coverage_patch], loc="lower center", fontsize=7, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    add_standard_caption(fig, ["flash_flood", "heatwave", "dust_storm"], lead_time=None, y=-0.05)
    fig.suptitle("MAZU — Compound-hazard overlay (real day, mixed-color overlap)",
                 fontsize=13, fontweight="bold", color="#1A2D4A")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "compound_hazard_event.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    eng = de.DetectionEngine()
    fig_climatology(eng)
    fig_event(eng)
    eng.close()


if __name__ == "__main__":
    main()
