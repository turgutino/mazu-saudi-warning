# =============================================================================
# MAZU — Priority-3 item 3: optional toggle layers (terrain slope, wadi/
# watershed network, town population), "风险 + 暴露度" integrated display.
#
# Reviewer's exact text: "下垫面 / 脆弱性叠加可选开关：支持一键切换图层：
# 地形坡度、水系流域、城镇人口密度，实现"风险 + 暴露度"一体化空间展示。"
# (Underlying-surface / vulnerability overlay optional toggle: support
# one-click layer switching: terrain slope, watershed/river system, town
# population density, achieving an integrated "risk + exposure" display.)
#
# This is fundamentally different from Priority-3 items 1-2: "一键切换"
# (one-click switching) requires genuine client-side interactivity, so this
# script produces plain-lat/lon (PlateCarree, not the site's usual
# AlbersEqualArea) transparent-background PNGs sized to exact geographic
# bounds, for use as Leaflet.js ImageOverlay layers in a new interactive
# HTML page (deploy/compound_layers.html) with real toggle checkboxes.
#
# Data honesty notes (each layer's real source/limitation, disclosed):
#  - Slope: computed (np.gradient) from the project's own real orography
#    variable (10km grid) -- a genuine derived quantity, not invented.
#  - Wadi/watershed network: HydroRIVERS v1.0 (Europe+Middle East region),
#    a real, citable global hydrography product (Lehner & Grill 2013),
#    filtered to Strahler stream order >=4 to keep the map readable.
#    Downloaded 2026-07-29 from https://www.hydrosheds.org/products/hydrorivers
#  - Town population: real GASTAT 2022 census figures for the system's 8
#    modeled cities (agent/city_population.json) -- point totals, NOT a
#    gridded density surface (no such dataset exists in this project),
#    shown as proportionally-sized markers, disclosed as such in the UI.
# =============================================================================

import os
import sys
import json
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\Turqut\Desktop\Competation\mazu-system\agent")
import tools
from render_map import DATA_EXTENT, TIER_BOUNDS, tiered_rgba

OUT = os.path.join(HERE, "..", "img", "layers")
os.makedirs(OUT, exist_ok=True)

PC = ccrs.PlateCarree()
# Exact bounds every layer PNG is rendered to -- must match 1:1 so Leaflet's
# ImageOverlay bounds line up across all layers without any manual offset.
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = DATA_EXTENT


def _figure(figsize=(10, 8)):
    fig = plt.figure(figsize=figsize, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1], projection=PC)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=PC)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    return fig, ax


def _save_transparent(fig, out_path):
    fig.savefig(out_path, dpi=150, transparent=True, pad_inches=0, bbox_inches="tight",
                pil_kwargs={"compress_level": 6})
    plt.close(fig)
    print("saved", out_path)


# ---------------------------------------------------------------------------
# Layer 1: terrain slope, from the real orography variable.
# ---------------------------------------------------------------------------
def make_slope_layer():
    ds = xr.open_dataset(tools.DATASET) if hasattr(tools, "DATASET") else \
        xr.open_dataset(os.path.join(HERE, "..", "data", "mazu_dataset.nc"))
    elev = ds["orography"].values
    lat = ds.latitude.values
    lon = ds.longitude.values

    # degrees lat/lon -> approx meters, so slope is a physically meaningful
    # gradient magnitude (m elevation change per m horizontal distance),
    # not just raw degree-based gradient.
    dlat_m = 111320.0
    dlon_m = 111320.0 * np.cos(np.deg2rad(lat.mean()))
    dzdy, dzdx = np.gradient(elev, dlat_m, dlon_m)
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)  # rise/run, dimensionless (m/m)
    slope_pct = slope * 100
    # At this dataset's native 10km grid spacing, cell-to-cell elevation
    # gradients are inherently small (real observed max here is ~0.7%,
    # far below a "real-world local slope" percentage) -- normalize against
    # the ACTUAL observed range, not an arbitrary steep-terrain percentage,
    # so the map shows real relative variation instead of rendering
    # near-blank.
    vmax = float(np.nanpercentile(slope_pct, 99))
    slope_norm = np.clip(slope_pct / vmax, 0, 1)

    fig, ax = _figure()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "slope", ["#8B5E34", "#5A2E0A"])
    rgba = cmap(slope_norm)
    rgba[..., 3] = np.clip(0.15 + slope_norm * 0.65, 0.15, 0.8)  # flattest cells still faintly visible
    ax.imshow(rgba, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
              origin="upper" if lat[0] > lat[-1] else "lower", transform=PC, zorder=1)
    _save_transparent(fig, os.path.join(OUT, "slope.png"))
    ds.close()


# ---------------------------------------------------------------------------
# Layer 2: wadi / watershed network, from real HydroRIVERS data.
# ---------------------------------------------------------------------------
def make_wadi_layer(min_order=4):
    shp = os.path.join(HERE, "hydro_data", "HydroRIVERS_v10_eu_shp", "HydroRIVERS_v10_eu.shp")
    r = Reader(shp)
    geoms = []
    for rec in r.records():
        if rec.attributes["ORD_STRA"] < min_order:
            continue
        b = rec.geometry.bounds
        if b[2] < LON_MIN or b[0] > LON_MAX or b[3] < LAT_MIN or b[1] > LAT_MAX:
            continue
        geoms.append(rec.geometry)
    print(f"wadi layer: {len(geoms)} segments (ORD_STRA >= {min_order})")

    fig, ax = _figure()
    feat = ShapelyFeature(geoms, PC, facecolor="none", edgecolor="#1c7ed6")
    ax.add_feature(feat, linewidth=1.1, alpha=0.85, zorder=1)
    _save_transparent(fig, os.path.join(OUT, "wadi.png"))


# ---------------------------------------------------------------------------
# Layer 3: town population (real GASTAT 2022 totals, point markers -- NOT a
# gridded density surface, since no such dataset exists in this project).
# ---------------------------------------------------------------------------
def make_population_layer():
    with open(os.path.join(HERE, "..", "..", "agent", "city_population.json"), encoding="utf-8") as f:
        pop = json.load(f)["cities"]

    fig, ax = _figure()
    max_pop = max(pop.values())
    for city, p in pop.items():
        lat, lon = tools.CITIES[city]
        r = 6 + 34 * (p / max_pop) ** 0.5  # sqrt scale so area, not radius, tracks population
        ax.scatter(lon, lat, s=r ** 2, transform=PC, zorder=2,
                   facecolor="#e64980", edgecolor="black", linewidth=0.8, alpha=0.75)
    _save_transparent(fig, os.path.join(OUT, "population.png"))


# ---------------------------------------------------------------------------
# Base risk layer: flash_flood tiered risk field (real_grid, same rule-based
# ground truth as the rest of the site), for a fixed reference date.
# ---------------------------------------------------------------------------
BASE_DATE = "2025-08-23"  # Jizan 254.9mm flash-flood event -- real, high-signal day


def make_base_risk_layer():
    import importlib.util
    spec = importlib.util.spec_from_file_location("de", os.path.join(HERE, "01_detection_engine.py"))
    de = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(de)
    eng = de.DetectionEngine()
    risk = eng.risk_field(BASE_DATE, "flash_flood")
    rgba = tiered_rgba(risk, "flash_flood", base_alpha=0.75)

    fig, ax = _figure()
    ax.imshow(rgba, extent=[eng.lon.min(), eng.lon.max(), eng.lat.min(), eng.lat.max()],
              origin="upper", transform=PC, zorder=0, interpolation="nearest")
    _save_transparent(fig, os.path.join(OUT, "base_risk_flash_flood.png"))
    eng.close()


def main():
    make_base_risk_layer()
    make_slope_layer()
    make_wadi_layer()
    make_population_layer()
    print("\nAll layers saved to", OUT)
    print(f"Shared geographic bounds for Leaflet: south={LAT_MIN}, west={LON_MIN}, north={LAT_MAX}, east={LON_MAX}")


if __name__ == "__main__":
    main()
