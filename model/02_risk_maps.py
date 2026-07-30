# =============================================================================
# MAZU Saudi Arabia — Risk Map Visualisation + spatial verification
#
# Renders the detection engine's continuous risk field over the Saudi grid,
# overlays detected event clusters and city markers, and builds an ANNUAL
# HOTSPOT map (how often each cell reaches the extreme tier) — a spatial
# climatology that verifies the physics:
#   flash-flood hotspots -> SW Asir mountains / Red Sea coast
#   heatwave  hotspots    -> interior desert / SE Empty Quarter
#
# Sea/land context is drawn from the SST validity mask (no cartopy needed).
# Output: outputs/*.png
# =============================================================================

import os
import sys
import importlib.util
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_map import (PROJ, PC, DATA_EXTENT, NEIGHBOR_LABELS, LABEL_OFFSETS,
                         _terrain_rgb, _NEIGHBOR_OUTLINE, tiered_norm_cmap, tiered_rgba,
                         EVENT_MARKER_COLORS, add_standard_caption)

_TEXT_OUTLINE = [pe.withStroke(linewidth=2.0, foreground="black")]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# import the detection engine (file starts with a digit -> load via importlib)
spec = importlib.util.spec_from_file_location("de", os.path.join(HERE, "01_detection_engine.py"))
de = importlib.util.module_from_spec(spec)
spec.loader.exec_module(de)

CITIES = {"Jeddah": (21.5, 39.2), "Mecca": (21.4, 39.8), "Riyadh": (24.7, 46.7),
          "Jizan": (16.9, 42.6), "Dammam": (26.4, 50.1), "Taif": (21.3, 40.4),
          "Medina": (24.5, 39.6), "Abha": (18.2, 42.5)}

# Continuous, hazard-appropriate colormap for the ANNUAL HOTSPOT figure only
# (day-count of extreme-tier cells, not a 0-1 risk score, so Priority-2's
# tiered risk-score thresholds don't apply to it -- see plot_event() below
# for the risk_field() 0-1 figure, which DOES use the tiered scale).
CMAPS = {
    "flash_flood": LinearSegmentedColormap.from_list("ff", ["#0E1B2A", "#1C5D99", "#2BC8E2", "#B7F5D8", "#FFFFFF"]),
    "heatwave":    LinearSegmentedColormap.from_list("hw", ["#0E1B2A", "#7A1F1F", "#FF5A00", "#FFC300", "#FFFFFF"]),
}


def sea_mask(eng):
    """Cells that are ocean (SST valid most of the year)."""
    sst = eng.ds["sst_celsius"].values
    return np.isfinite(sst).mean(axis=0) > 0.5   # (lat, lon) bool


PLOT_EXTENT = [32.0, 58.0, 14.0, 34.0]  # includes neighbor-country buffer zone


def draw_base(ax, eng, sea):
    ax.set_extent(PLOT_EXTENT, crs=PC)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#0E1B2A", zorder=-1)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#12314f", zorder=-0.5)

    # real elevation hillshade over the actual model data extent
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

    # no-data mask: gray frame = plot extent minus real data extent
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

    ax.tick_params(labelsize=7)
    return PLOT_EXTENT


def plot_event(ax, eng, date, hazard, sea):
    """Priority-2: hazard-specific tiered risk scale (same 4-tier thresholds
    as the ML-model maps) so Low/Moderate/High/Extreme are visually
    distinguishable at a glance. No confidence-fade here: risk_field() is
    the deterministic rule engine, not the ML ensemble, so there is no
    per-cell model uncertainty to visualize."""
    extent = draw_base(ax, eng, sea)
    risk = eng.risk_field(date, hazard)
    lon, lat = eng.lon, eng.lat
    rgba = tiered_rgba(risk, hazard, base_alpha=0.65)
    im = ax.imshow(rgba, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    origin="upper", transform=PC, zorder=1.5, interpolation="nearest")
    # detected cluster peaks -- Priority-3 item 1: hazard-type color, not
    # severity (reviewer's exact spec: red=flash_flood, orange=heatwave,
    # yellow=dust_storm), so marker color is consistent with the rest of
    # the site's event maps/GIFs.
    for e in eng.detect(date, hazard):
        ms = 40 + e["cluster_size"] ** 0.5 * 3
        ax.scatter(e["lon"], e["lat"], s=ms, facecolors=EVENT_MARKER_COLORS[hazard],
                   edgecolors="black", linewidths=1.0, transform=PC, zorder=5.5)
    # cities
    for c, (la, lo) in CITIES.items():
        dx, dy, ha = LABEL_OFFSETS.get(c, (0.35, 0.0, "left"))
        ax.plot(lo, la, "o", ms=3, color="white", transform=PC, zorder=4)
        ax.text(lo + dx, la + dy, c, fontsize=6, color="white", ha=ha,
                 transform=PC, zorder=4, path_effects=_TEXT_OUTLINE)
    ax.set_title(f"{hazard}  {date}", fontsize=9, color="#1A2D4A", fontweight="bold")
    return im


def annual_hotspot(eng, hazard, tier=("extreme", "emergency")):
    """Count, per cell, how many days it belongs to an extreme-tier cluster."""
    rule = de.RULES[hazard]
    thr = [lo for name, lo in rule["severity"] if name in tier]
    thr = min(thr) if thr else 0.85
    cnt = np.zeros((len(eng.lat), len(eng.lon)), dtype="float32")
    for d in eng.times:
        risk = eng.risk_field(d, hazard)
        cnt += (risk >= thr).astype("float32")
    return cnt


def main():
    eng = de.DetectionEngine()
    sea = sea_mask(eng)

    # ── Figure 1: known events (2x2) ────────────────────────────────────
    events = [("2025-08-23", "flash_flood"), ("2025-08-19", "flash_flood"),
              ("2025-07-25", "heatwave"), ("2025-08-16", "heatwave")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), facecolor="white",
                             subplot_kw={"projection": PROJ})
    for ax, (date, hz) in zip(axs.ravel(), events):
        plot_event(ax, eng, date, hz, sea)
        cmap, norm, bounds = tiered_norm_cmap(hz)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, ticks=bounds)
        cbar.set_label(f"risk tier ({hz})", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    coverage_patch1 = mpatches.Patch(facecolor="#444444", alpha=0.55,
                                      label="Outside model coverage (no data)")
    ff_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                            markerfacecolor=EVENT_MARKER_COLORS["flash_flood"], markeredgecolor="black",
                            label="Dot = real detected flash_flood event")
    hw_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                            markerfacecolor=EVENT_MARKER_COLORS["heatwave"], markeredgecolor="black",
                            label="Dot = real detected heatwave event")
    fig.legend(handles=[coverage_patch1, ff_handle, hw_handle], loc="lower center", fontsize=7,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.03), ncol=3)
    add_standard_caption(fig, ["flash_flood", "heatwave"], lead_time=None, y=-0.06)
    fig.suptitle("MAZU — Detected extreme-event risk (known 2025 events)",
                 fontsize=13, fontweight="bold", color="#1A2D4A")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    f1 = os.path.join(OUT, "risk_known_events.png")
    fig.savefig(f1, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ── Figure 2: annual hotspots (spatial verification) ────────────────
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white",
                             subplot_kw={"projection": PROJ})
    verify = {}
    for ax, hz in zip(axs, ["flash_flood", "heatwave"]):
        extent = draw_base(ax, eng, sea)
        hot = annual_hotspot(eng, hz)
        im = ax.pcolormesh(eng.lon, eng.lat, hot, cmap=CMAPS[hz],
                            transform=PC, alpha=0.65, zorder=1.5, shading="auto")
        for c, (la, lo) in CITIES.items():
            dx, dy, ha = LABEL_OFFSETS.get(c, (0.35, 0.0, "left"))
            ax.plot(lo, la, "o", ms=3, color="white", transform=PC, zorder=4)
            ax.text(lo + dx, la + dy, c, fontsize=6, color="white", ha=ha,
                     transform=PC, zorder=4, path_effects=_TEXT_OUTLINE)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="extreme-tier days")
        ax.set_title(f"{hz} — 2025 annual hotspot", fontsize=10, fontweight="bold", color="#1A2D4A")
        # verification: where is the peak hotspot?
        yi, xi = np.unravel_index(np.argmax(hot), hot.shape)
        verify[hz] = (eng._region(float(eng.lat[yi]), float(eng.lon[xi])),
                      round(float(eng.lat[yi]), 1), round(float(eng.lon[xi]), 1), int(hot.max()))
    coverage_patch2 = mpatches.Patch(facecolor="#444444", alpha=0.55,
                                      label="Outside model coverage (no data)")
    fig.legend(handles=[coverage_patch2], loc="lower center", fontsize=7,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, -0.06,
             "Note: this map counts extreme-tier DAYS (not a 0-1 risk score), so the site's "
             "standard tier thresholds don't directly apply here.   |   Grid: 10km native model "
             "grid   |   Proj: Albers Equal-Area (lon0=45E, lat0=23N)   |   Data year: 2025   |   "
             "Coverage: 34.0-55.9E, 16.0-31.9N",
             ha="center", va="top", fontsize=6.5, color="#666666", transform=fig.transFigure)
    fig.suptitle("MAZU — Annual hazard hotspots (spatial climatology check)",
                 fontsize=13, fontweight="bold", color="#1A2D4A")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f2 = os.path.join(OUT, "risk_annual_hotspots.png")
    fig.savefig(f2, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ── verification report ─────────────────────────────────────────────
    print("=" * 60)
    print("Risk maps saved:")
    print(f"  {f1}")
    print(f"  {f2}")
    print("\nSPATIAL VERIFICATION — peak annual hotspot location:")
    for hz, (reg, la, lo, n) in verify.items():
        print(f"  {hz:12s}: {reg} ({la}N,{lo}E)  {n} extreme-tier days")
    print("\nExpected: flash_flood -> SW mountains (Asir/Abha/Taif/Jizan) or Red Sea;")
    print("          heatwave    -> interior desert / SE (Riyadh/Empty Quarter).")
    eng.close()


if __name__ == "__main__":
    main()
