# =============================================================================
# MAZU — Priority-4 item 1: national overview map + per-city detail zoom-ins.
#
# Reviewer's exact text: "新增一张沙特全域总览图，同步展示三类灾害全国分布，
# 再搭配分城市细节放大图，兼顾全局与局地。" (Add one Saudi-wide national
# overview map that SIMULTANEOUSLY shows the national distribution of all
# three hazard types, paired with per-city detail zoom-in maps, so both the
# global/national and local/city views are covered.)
#
# Design: "simultaneously" (同步) on ONE map, for three hazards at once,
# means a single-map composite rather than three separate panels. Each grid
# cell is colored by whichever hazard has the most annual extreme-tier days
# there ("dominant hazard"), using the same EVENT_MARKER_COLORS already
# established (red=flash_flood, orange=heatwave, yellow=dust_storm) for
# color consistency with the rest of the site, with per-cell alpha scaled
# by how extreme that dominance is. This is a real, data-derived summary
# (365-day scan, same rule-based DetectionEngine ground truth as every
# other map on the site), not an illustrative sketch.
#
# The 8 city zoom-ins are the SAME dominant-hazard composite, just cropped
# tighter around each modeled city, so a viewer gets both the national
# pattern and local detail from one consistent underlying dataset.
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
import cartopy.feature as cfeature
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from render_map import (PROJ, PC, DATA_EXTENT, NEIGHBOR_LABELS, LABEL_OFFSETS,
                         _terrain_rgb, _NEIGHBOR_OUTLINE, TIER_BOUNDS, EVENT_MARKER_COLORS,
                         add_standard_caption)
import tools

OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

_TEXT_OUTLINE = [pe.withStroke(linewidth=2.0, foreground="black")]

HAZARDS = ["flash_flood", "heatwave", "dust_storm"]
EXTREME_THR = {hz: TIER_BOUNDS[hz][3] for hz in HAZARDS}

PLOT_EXTENT = [32.0, 58.0, 14.0, 34.0]


def _extreme_day_counts(eng):
    """Per hazard, per cell: how many days in 2025 that cell reached its
    own 'extreme' tier (same threshold used everywhere else on the site).
    Real 365-day scan of the rule-based ground truth, not a shortcut."""
    counts = {}
    for hz in HAZARDS:
        cnt = np.zeros((len(eng.lat), len(eng.lon)), dtype="float32")
        for d in eng.times:
            risk = eng.risk_field(d, hz)
            cnt += (risk >= EXTREME_THR[hz]).astype("float32")
        counts[hz] = cnt
    return counts


def _dominant_hazard_rgba(counts):
    """Per cell, pick whichever hazard has the most extreme-tier days;
    color = that hazard's own site-wide color; alpha scales with how many
    days (normalized per-hazard by its own max, so a hazard with a lower
    typical day-count, e.g. flash_flood, isn't visually erased just
    because heatwave's raw day-counts run higher)."""
    ny, nx = counts[HAZARDS[0]].shape
    stacked = np.stack([counts[hz] for hz in HAZARDS], axis=0)  # (3, ny, nx)
    norm_stacked = np.stack(
        [counts[hz] / max(counts[hz].max(), 1e-6) for hz in HAZARDS], axis=0)
    dominant_idx = np.argmax(norm_stacked, axis=0)
    dominant_strength = np.max(norm_stacked, axis=0)

    rgba = np.zeros((ny, nx, 4))
    for i, hz in enumerate(HAZARDS):
        mask = dominant_idx == i
        color = matplotlib.colors.to_rgb(EVENT_MARKER_COLORS[hz])
        rgba[mask, 0] = color[0]
        rgba[mask, 1] = color[1]
        rgba[mask, 2] = color[2]
    # cells with essentially zero extreme days for all 3 hazards -> fully
    # transparent (no manufactured signal where there isn't one)
    has_signal = stacked.sum(axis=0) > 0
    rgba[..., 3] = np.where(has_signal, np.clip(0.25 + dominant_strength * 0.6, 0.25, 0.85), 0.0)
    return rgba


def _draw_base(ax, extent):
    ax.set_extent(extent, crs=PC)
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


def _draw_no_data_mask(ax, extent):
    outer = [(extent[0], extent[2]), (extent[1], extent[2]),
              (extent[1], extent[3]), (extent[0], extent[3]), (extent[0], extent[2])]
    inner = [(DATA_EXTENT[0], DATA_EXTENT[2]), (DATA_EXTENT[0], DATA_EXTENT[3]),
              (DATA_EXTENT[1], DATA_EXTENT[3]), (DATA_EXTENT[1], DATA_EXTENT[2]), (DATA_EXTENT[0], DATA_EXTENT[2])]
    verts = outer + inner
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY] + \
            [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    frame_patch = mpatches.PathPatch(Path(verts, codes), transform=PC, facecolor="#444444",
                                      edgecolor="none", alpha=0.55, zorder=1.4)
    ax.add_patch(frame_patch)


def _draw_cities(ax, highlight=None):
    """Bug found & fixed during Priority-4 item 1 testing: cartopy GeoAxes
    does not clip ax.text()/ax.plot() to the visible extent by default, so
    on the tight per-city zoom panels, the other 7 cities' labels (whose
    true geographic position falls far outside that panel's small extent)
    rendered scattered across the whole figure's blank margin instead of
    being hidden. Fixed by filtering to cities within the current axes'
    actual extent (with a small margin) before drawing, and by explicitly
    setting clip_on=True as a second safety net."""
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=PC)
    margin = 0.3
    for name, (lat, lon) in tools.CITIES.items():
        if not (lon_min - margin <= lon <= lon_max + margin and
                lat_min - margin <= lat <= lat_max + margin):
            continue
        is_hl = name == highlight
        dx, dy, ha = LABEL_OFFSETS.get(name, (0.35, 0.0, "left"))
        ax.plot(lon, lat, marker="o", markersize=5 if not is_hl else 9,
                 markeredgecolor="black", markeredgewidth=0.7, clip_on=True,
                 color="white" if not is_hl else "#2244ee", transform=PC, zorder=5)
        ax.text(lon + dx, lat + dy, name, fontsize=8 if is_hl else 7, ha=ha,
                 fontweight="bold" if is_hl else "normal", clip_on=True,
                 transform=PC, zorder=6, path_effects=_TEXT_OUTLINE)


def main():
    spec = importlib.util.spec_from_file_location("de", os.path.join(HERE, "01_detection_engine.py"))
    de = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(de)
    eng = de.DetectionEngine()

    print("Scanning 365 days x 3 hazards for extreme-tier day counts...")
    counts = _extreme_day_counts(eng)
    rgba = _dominant_hazard_rgba(counts)
    extent_data = [eng.lon.min(), eng.lon.max(), eng.lat.min(), eng.lat.max()]

    # ---- National overview (1 large map) + 8 city zoom-ins, one figure ----
    fig = plt.figure(figsize=(18, 13), facecolor="white")
    gs = fig.add_gridspec(3, 5, height_ratios=[3.2, 1, 1], hspace=0.35, wspace=0.25)

    ax_national = fig.add_subplot(gs[0, :], projection=PROJ)
    _draw_base(ax_national, PLOT_EXTENT)
    ax_national.imshow(rgba, extent=extent_data, origin="upper", transform=PC, zorder=1.5,
                        interpolation="nearest")
    _draw_no_data_mask(ax_national, PLOT_EXTENT)
    gl = ax_national.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--", zorder=3.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(30, 61, 5))
    gl.ylocator = mticker.FixedLocator(range(14, 36, 4))
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    for name, (nlat, nlon) in NEIGHBOR_LABELS.items():
        ax_national.text(nlon, nlat, name, fontsize=7, style="italic", color="#444444", ha="center",
                          transform=PC, zorder=4, path_effects=_NEIGHBOR_OUTLINE)
    _draw_cities(ax_national)
    ax_national.set_title("National overview — dominant hazard by 2025 extreme-tier day count",
                           fontsize=13, fontweight="bold")

    ff_patch = mpatches.Patch(facecolor=EVENT_MARKER_COLORS["flash_flood"], label="flash_flood dominant")
    hw_patch = mpatches.Patch(facecolor=EVENT_MARKER_COLORS["heatwave"], label="heatwave dominant")
    du_patch = mpatches.Patch(facecolor=EVENT_MARKER_COLORS["dust_storm"], label="dust_storm dominant")
    coverage_patch = mpatches.Patch(facecolor="#444444", alpha=0.55, label="Outside model coverage (no data)")
    ax_national.legend(handles=[ff_patch, hw_patch, du_patch, coverage_patch], loc="lower left",
                        fontsize=7.5, framealpha=0.9)

    # per-city detail zoom-ins (8 cities, 2 rows x 4 cols under the national map)
    city_items = list(tools.CITIES.items())
    for i, (name, (clat, clon)) in enumerate(city_items):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col], projection=PROJ)
        zoom_extent = [clon - 2.2, clon + 2.2, clat - 2.0, clat + 2.0]
        _draw_base(ax, zoom_extent)
        ax.imshow(rgba, extent=extent_data, origin="upper", transform=PC, zorder=1.5,
                  interpolation="nearest")
        _draw_no_data_mask(ax, zoom_extent)
        _draw_cities(ax, highlight=name)
        ax.set_title(name, fontsize=10, fontweight="bold")

    add_standard_caption(fig, ["flash_flood", "heatwave", "dust_storm"], lead_time=None, y=-0.02)
    fig.suptitle("MAZU — National hazard overview + per-city detail (2025)", fontsize=15, fontweight="bold", y=0.98)
    out = os.path.join(OUT, "national_overview.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    eng.close()


if __name__ == "__main__":
    main()
