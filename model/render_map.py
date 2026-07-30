import os
import sys
sys.path.insert(0, r"C:\Users\Turqut\Desktop\Competation\mazu-system\agent")
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.colors import LightSource
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import tools
from render_build_grid import predicted_grid, real_grid, predicted_grid_uncertainty

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "..", "data", "mazu_dataset.nc")

# Real dataset coverage (verified against mazu_dataset.nc: lat 16.0-31.9N,
# lon 34.0-55.9E). PLOT_EXTENT is wider -- it includes a neighbor-country
# buffer zone (Jordan/Iraq/Kuwait/north Yemen, per the reviewer's Priority-1
# requirement) so the map doesn't look like the data was cut off at the
# border; NO_DATA_EXTENT vs DATA_EXTENT drives the gray "outside model
# coverage" mask below.
DATA_EXTENT = [34.0, 55.9, 16.0, 31.9]
EXTENT = [32.0, 58.0, 14.0, 34.0]  # kept as `EXTENT` for backward compatibility

PROJ = ccrs.AlbersEqualArea(central_longitude=45, central_latitude=23, standard_parallels=(18, 30))
PC = ccrs.PlateCarree()

CITIES = tools.CITIES
NEIGHBOR_LABELS = {
    "JORDAN": (31.0, 36.5), "IRAQ": (32.5, 44.0), "KUWAIT": (29.3, 47.7),
    "YEMEN": (15.5, 44.5), "OMAN": (20.5, 56.0), "UAE": (23.5, 54.5),
}

# Label placement overrides for the Jeddah/Mecca/Medina/Taif cluster
# (all within ~1.2 deg lat/lon of each other) so text doesn't overlap.
# (dx, dy, ha) offsets in degrees from the marker.
LABEL_OFFSETS = {
    "Jeddah": (-0.35, -0.55, "right"),
    "Mecca":  (0.4, 0.25, "left"),
    "Medina": (0.35, 0.35, "left"),
    "Taif":   (0.4, -0.5, "left"),
    "Abha":   (0.35, -0.15, "left"),
    "Jizan":  (0.35, -0.45, "left"),
    "Riyadh": (0.35, 0.25, "left"),
    "Dammam": (0.35, 0.25, "left"),
}

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "risk", ["#1a7a3c", "#f4d03f", "#e67e22", "#c0392b"]
)
TERRAIN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "terrain_real", ["#8fae7a", "#c9b686", "#b08a5a", "#8a5a3c", "#6b4530", "#e8e4d8"]
)

# ---------------------------------------------------------------------------
# Priority-2: hazard-specific, non-linear risk tiers (reviewer's exact
# thresholds) -- flash_flood's real base rate is far lower than
# heatwave/dust_storm's, so sharing one 0-1 linear scale made a flash-flood
# "extreme" look identical to a heatwave "ordinary hot day". Four discrete
# colors (not a continuous ramp) so Moderate vs Extreme are distinguishable
# at a glance, per the reviewer's explicit complaint.
# ---------------------------------------------------------------------------
TIER_BOUNDS = {
    "flash_flood": [0.0, 0.1, 0.3, 0.6, 1.0],
    "heatwave":    [0.0, 0.2, 0.4, 0.7, 1.0],
    "dust_storm":  [0.0, 0.2, 0.4, 0.7, 1.0],
}
TIER_LABELS = ["Low", "Moderate", "High", "Extreme"]
TIER_COLORS = ["#2b8a3e", "#ffd43b", "#fd7e14", "#c92a2a"]


def tiered_norm_cmap(hazard):
    bounds = TIER_BOUNDS[hazard]
    cmap = mcolors.ListedColormap(TIER_COLORS)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


# Ensemble-confidence -> auxiliary grayscale mask layer (reviewer's exact
# spec: "主图层：灾害风险概率；辅助半透明图层：模型置信度（灰度遮罩），
# 低置信网格自动淡化" -- a *separate* semi-transparent gray layer drawn on
# top of the risk layer, not an alpha modulation of the risk layer itself).
# STD_LOW/STD_HIGH were set from the 5-member production ensemble's actual
# observed std range (~0.0003-0.17 across sampled dates/hazards): below
# STD_LOW the members agree closely (confident, mask fully transparent);
# above STD_HIGH they disagree substantially (uncertain, mask near-opaque
# gray covers that cell).
STD_LOW, STD_HIGH = 0.02, 0.12
_MASK_COLOR = "#888888"
_MASK_MAX_ALPHA = 0.55


def confidence_mask_alpha(std_grid):
    """Alpha for the auxiliary gray confidence-mask layer: 0 where the
    ensemble agrees (mask fully transparent, risk layer shows through
    unobscured) rising to _MASK_MAX_ALPHA where the ensemble disagrees
    (mask grays out that cell)."""
    t = np.clip((std_grid - STD_LOW) / (STD_HIGH - STD_LOW), 0, 1)
    return t * _MASK_MAX_ALPHA


def confidence_mask_rgba(std_grid):
    """Build the auxiliary gray RGBA mask layer (separate imshow call, drawn
    on top of the tiered risk layer) from a per-cell ensemble-std grid."""
    rgba = np.zeros(std_grid.shape + (4,))
    rgba[..., 0:3] = mcolors.to_rgb(_MASK_COLOR)
    rgba[..., 3] = confidence_mask_alpha(std_grid)
    return rgba


def tiered_rgba(grid, hazard, base_alpha=0.65):
    """Build the main RGBA array for imshow from a risk grid using the
    hazard's tiered colormap, at a uniform alpha. Model-confidence, where
    applicable, is drawn as a separate overlay layer (confidence_mask_rgba),
    not baked into this array."""
    cmap, norm, _ = tiered_norm_cmap(hazard)
    rgba = cmap(norm(grid))
    rgba[..., 3] = base_alpha
    return rgba


# ---------------------------------------------------------------------------
# Priority-4 item 3: "预测风险场 vs 规则基准真值场...方便直接对比模型偏差的
# 空间分布规律" (predicted risk field vs rule-based ground-truth field...to
# directly compare the spatial pattern of model bias) -- a 3rd bias/
# difference panel (Predicted - Real) added alongside the existing 2-panel
# Real|Predicted comparison, since the doc's stated PURPOSE is comparing
# bias's spatial pattern, which a raw side-by-side view only shows
# indirectly. real_grid() is on the dataset's native (fine) grid,
# predicted_grid() is on a strided (coarser) grid -- they are NOT pixel-
# aligned, so a naive subtraction would be wrong; _nearest_indices resamples
# the real grid onto the predicted grid's coordinates first.
# ---------------------------------------------------------------------------
def _nearest_indices(source_coords, target_coords):
    """For each value in target_coords, return the index into source_coords
    of its nearest value. Works regardless of ascending/descending order."""
    order = np.argsort(source_coords)
    sorted_source = source_coords[order]
    idx = np.searchsorted(sorted_source, target_coords)
    idx = np.clip(idx, 1, len(sorted_source) - 1)
    left = sorted_source[idx - 1]
    right = sorted_source[idx]
    choose_right = np.abs(target_coords - right) < np.abs(target_coords - left)
    idx_final = idx - 1 + choose_right.astype(int)
    return order[idx_final]


def resample_nearest(grid, src_lat, src_lon, dst_lat, dst_lon):
    """Resample a 2D grid from its native (src_lat, src_lon) coordinates
    onto (dst_lat, dst_lon) via nearest-neighbor lookup (no smoothing --
    each output cell takes the value of its closest native-grid cell)."""
    yi = _nearest_indices(src_lat, dst_lat)
    xi = _nearest_indices(src_lon, dst_lon)
    return grid[np.ix_(yi, xi)]


BIAS_CMAP = mcolors.LinearSegmentedColormap.from_list("bias", ["#1c7ed6", "#f5f5f5", "#c92a2a"])
BIAS_NORM = mcolors.Normalize(vmin=-1.0, vmax=1.0)


def bias_rgba(rgrid_resampled, pgrid, base_alpha=0.8):
    """Predicted - Real, on matching grids. Blue = model under-predicts
    (real risk higher than predicted), red = model over-predicts, white =
    agreement. Both inputs must already be on the same grid (see
    resample_nearest)."""
    diff = pgrid - rgrid_resampled
    rgba = BIAS_CMAP(BIAS_NORM(diff))
    rgba[..., 3] = base_alpha
    return rgba, diff


_TEXT_OUTLINE = [pe.withStroke(linewidth=2.2, foreground="white")]
_NEIGHBOR_OUTLINE = [pe.withStroke(linewidth=2.0, foreground="white")]

# ---------------------------------------------------------------------------
# Priority-5 item 1: "每张图统一标准化图注：包含预报时效、网格分辨率、绘图
# 投影、分级阈值、数据年份、覆盖范围说明" -- every map gets a standardized
# caption footer with: forecast lead time, grid resolution, map projection,
# tier thresholds, data year, coverage range. One shared builder so the
# wording/format is identical everywhere, not re-typed per script.
# ---------------------------------------------------------------------------
DATA_YEAR = 2025
GRID_RESOLUTION = "10km native model grid"
PROJECTION_NAME = "Albers Equal-Area (lon0=45E, lat0=23N)"
COVERAGE_DESC = f"{DATA_EXTENT[0]:.1f}-{DATA_EXTENT[1]:.1f}E, {DATA_EXTENT[2]:.1f}-{DATA_EXTENT[3]:.1f}N"


def _tier_str(hazard):
    b = TIER_BOUNDS[hazard]
    return f"{hazard} {b[0]:.1f}/{b[1]:.1f}/{b[2]:.1f}/{b[3]:.1f}"


def standard_caption(hazards, lead_time=None):
    """hazards: a hazard-name string, or a list of hazard-name strings for
    figures that combine more than one (e.g. compound-hazard, national
    overview). lead_time: e.g. 't-1->t' for a forecast/predicted panel;
    omitted for ground-truth-only or climatology figures."""
    if isinstance(hazards, str):
        hazards = [hazards]
    tiers = "  ".join(_tier_str(h) for h in hazards)
    parts = [f"Grid: {GRID_RESOLUTION}", f"Proj: {PROJECTION_NAME}",
             f"Tiers (low/mod/high/extreme): {tiers}",
             f"Data year: {DATA_YEAR}", f"Coverage: {COVERAGE_DESC}"]
    if lead_time:
        parts.insert(0, f"Lead time: {lead_time}")
    return "   |   ".join(parts)


def add_standard_caption(fig, hazards, lead_time=None, y=-0.03):
    fig.text(0.5, y, standard_caption(hazards, lead_time), ha="center", va="top",
              fontsize=6.5, color="#666666", transform=fig.transFigure)


# ---------------------------------------------------------------------------
# Real elevation (orography) hillshade, built once from the actual model
# dataset (native 0.1deg grid -- same resolution as the risk data, so it
# lines up exactly, no coarse/generic background raster). This is what lets
# the Asir mountain range read as visually distinct from the interior
# plateau/plain, per the reviewer's Priority-1 requirement.
# ---------------------------------------------------------------------------
_TERRAIN_RGB = None
_TERRAIN_LON = None
_TERRAIN_LAT = None


def _terrain_rgb():
    global _TERRAIN_RGB, _TERRAIN_LON, _TERRAIN_LAT
    if _TERRAIN_RGB is None:
        ds = xr.open_dataset(DATASET_PATH)
        _TERRAIN_LAT = ds.latitude.values
        _TERRAIN_LON = ds.longitude.values
        elev = ds["orography"].values
        ls = LightSource(azdeg=315, altdeg=45)
        _TERRAIN_RGB = ls.shade(elev, cmap=TERRAIN_CMAP, vert_exag=8,
                                 blend_mode="soft", vmin=-300, vmax=2800)
        ds.close()
    return _TERRAIN_RGB, _TERRAIN_LON, _TERRAIN_LAT


def _draw_base(ax):
    ax.set_extent(EXTENT, crs=PC)

    # generic land/ocean first (covers the neighbor-buffer area outside the
    # real data extent, where we don't have elevation data)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#cfe3f0", zorder=-1)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#e8e4d8", zorder=-0.5)

    # real elevation hillshade over the actual model data extent
    rgb, tlon, tlat = _terrain_rgb()
    ax.imshow(rgb, extent=[tlon.min(), tlon.max(), tlat.min(), tlat.max()], origin="upper",
              transform=PC, zorder=0.5)

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, linestyle=":",
                    edgecolor="#555555", zorder=3)
    provinces = cfeature.NaturalEarthFeature(
        category="cultural", name="admin_1_states_provinces_lines", scale="10m",
        facecolor="none", edgecolor="#9a9a9a", linewidth=0.35,
    )
    ax.add_feature(provinces, zorder=2, linestyle="-")

    # no-data mask: gray frame = plot extent minus real data extent
    outer = [(EXTENT[0], EXTENT[2]), (EXTENT[1], EXTENT[2]),
              (EXTENT[1], EXTENT[3]), (EXTENT[0], EXTENT[3]), (EXTENT[0], EXTENT[2])]
    inner = [(DATA_EXTENT[0], DATA_EXTENT[2]), (DATA_EXTENT[0], DATA_EXTENT[3]),
              (DATA_EXTENT[1], DATA_EXTENT[3]), (DATA_EXTENT[1], DATA_EXTENT[2]),
              (DATA_EXTENT[0], DATA_EXTENT[2])]
    verts = outer + inner
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY] + \
            [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    frame_patch = mpatches.PathPatch(Path(verts, codes), transform=PC, facecolor="#6b6b6b",
                                      edgecolor="none", alpha=0.45, zorder=1.4)
    ax.add_patch(frame_patch)

    # gridlines with lat/lon tick labels (sparse so they read as a reference
    # frame, not a crosshatch over the data)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.6,
                       linestyle="--", zorder=3.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(30, 61, 5))
    gl.ylocator = mticker.FixedLocator(range(14, 36, 4))
    gl.xlabel_style = {"size": 6.5}
    gl.ylabel_style = {"size": 6.5}

    # neighbor country labels (buffer zone context)
    for name, (lat, lon) in NEIGHBOR_LABELS.items():
        ax.text(lon, lat, name, fontsize=7, style="italic", color="#444444", ha="center",
                transform=PC, zorder=4, path_effects=_NEIGHBOR_OUTLINE)


# Priority-3 item 1: real (rule-based) detected-event marker colors, per the
# reviewer's exact spec -- "红点 = 山洪、橙点 = 热浪、黄点 = 沙尘" (red dot =
# flash_flood, orange dot = heatwave, yellow dot = dust_storm).
EVENT_MARKER_COLORS = {
    "flash_flood": "#e03131",
    "heatwave": "#f76707",
    "dust_storm": "#fcc419",
}


def _draw_event_markers(ax, date, hazard):
    """Overlay that day's real detected-event peak point(s) (from the same
    rule-based DetectionEngine.risk_field() the 'Real' panel's color comes
    from) so missed-detection / false-alarm spatial patterns are visible at
    a glance against the model's predicted-risk panel. Drawn on both the
    Real and Predicted panels per the reviewer's "每张空间图" (every spatial
    map) wording."""
    eng = tools._get_detection_engine()
    try:
        events = eng.detect(date, hazard)
    except (KeyError, IndexError, ValueError):
        return
    color = EVENT_MARKER_COLORS[hazard]
    for e in events:
        ax.plot(e["lon"], e["lat"], marker="o", markersize=8,
                 markerfacecolor=color, markeredgecolor="black",
                 markeredgewidth=1.0, transform=PC, zorder=5.5)


def _draw_cities(ax, highlight=None, extra_markers=None):
    """highlight: None, a city-name string, or a list of city-name strings
    (all drawn in blue/bold). extra_markers: optional dict of
    {label: (lat, lon)} for real locations outside the 8 modeled cities
    (e.g. Hail, Buraidah) -- drawn in a distinct orange marker so it's
    visually clear these are not part of the system's own city set."""
    if highlight is None:
        highlight_set = set()
    elif isinstance(highlight, str):
        highlight_set = {highlight}
    else:
        highlight_set = set(highlight)

    for name, (lat, lon) in CITIES.items():
        is_hl = name in highlight_set
        dx, dy, ha = LABEL_OFFSETS.get(name, (0.35, 0.0, "left"))
        ax.plot(lon, lat, marker="o", markersize=5 if not is_hl else 8,
                 markeredgecolor="black", markeredgewidth=0.6,
                 color="white" if not is_hl else "#2244ee",
                 transform=PC, zorder=5)
        ax.text(lon + dx, lat + dy, name, fontsize=7.5, ha=ha,
                 fontweight="bold" if is_hl else "normal",
                 transform=PC, zorder=6,
                 path_effects=_TEXT_OUTLINE)

    if extra_markers:
        for name, (lat, lon) in extra_markers.items():
            ax.plot(lon, lat, marker="^", markersize=8,
                     markeredgecolor="black", markeredgewidth=0.6,
                     color="#e67e22",
                     transform=PC, zorder=5)
            ax.text(lon + 0.35, lat, name, fontsize=7.5,
                     fontweight="bold", style="italic",
                     transform=PC, zorder=6,
                     path_effects=_TEXT_OUTLINE)


def render_pair(date, hazard, out_path, title_prefix="", highlight_city=None, extra_markers=None):
    """Three-panel: left = real (ground-truth, indicator-derived) risk
    field, middle = model-predicted probability (t-1 -> t), right =
    Predicted-Real bias (Priority-4 item 3 -- "预测风险场 vs 规则基准真值场
    ...方便直接对比模型偏差的空间分布规律", i.e. directly visualize model
    bias's spatial pattern, not just eyeball two side-by-side panels). Real
    and Predicted panels use the hazard-specific non-linear tier color
    scale (Priority-2 item 1+2); the Predicted panel additionally draws a
    separate semi-transparent gray confidence-mask layer on top, from the
    5-member production ensemble's spread (Priority-2 item 3)."""
    rlat, rlon, rgrid = real_grid(date, hazard)
    plat, plon, pgrid = predicted_grid(date, hazard)
    _, _, std_grid = predicted_grid_uncertainty(date, hazard)
    mask_rgba = confidence_mask_rgba(std_grid)
    rgrid_resampled = resample_nearest(rgrid, rlat, rlon, plat, plon)
    bias_rgba_arr, _ = bias_rgba(rgrid_resampled, pgrid)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2),
                              subplot_kw={"projection": PROJ})

    panels = [
        (rlat, rlon, rgrid, "Real (ground-truth indicators)", None, "tiered"),
        (plat, plon, pgrid, "Model prediction (t−1→t)", mask_rgba, "tiered"),
        (plat, plon, bias_rgba_arr, "Predicted − Real (model bias)", None, "bias"),
    ]
    for ax, (lat, lon, grid_or_rgba, label, mask, kind) in zip(axes, panels):
        _draw_base(ax)
        extent_data = [lon.min(), lon.max(), lat.min(), lat.max()]
        rgba = tiered_rgba(grid_or_rgba, hazard, base_alpha=0.65) if kind == "tiered" else grid_or_rgba
        # Priority-5 item 2: explicit interpolation="nearest" (not left to
        # matplotlib's default) guarantees each cell renders as a sharp,
        # un-blended pixel block -- no cross-cell blur that could smear a
        # local extreme into its neighbors or create a false contiguous
        # high-risk patch spanning a terrain boundary.
        ax.imshow(rgba, extent=extent_data, origin="upper", transform=PC, zorder=1.5,
                  interpolation="nearest")
        if mask is not None:
            ax.imshow(mask, extent=extent_data, origin="upper", transform=PC, zorder=1.6,
                      interpolation="nearest")
        _draw_cities(ax, highlight=highlight_city, extra_markers=extra_markers)
        _draw_event_markers(ax, date, hazard)
        ax.set_title(label, fontsize=11)

    cmap, norm, bounds = tiered_norm_cmap(hazard)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes[:2], orientation="horizontal", ticks=bounds,
                          fraction=0.05, pad=0.1, shrink=0.7)
    cbar.set_label(f"Risk tier ({hazard})")
    cbar.ax.set_xticklabels([f"{b:.1f}" for b in bounds])

    bias_sm = plt.cm.ScalarMappable(cmap=BIAS_CMAP, norm=BIAS_NORM)
    bias_cbar = fig.colorbar(bias_sm, ax=axes[2], orientation="horizontal",
                              fraction=0.05, pad=0.1, shrink=0.9)
    bias_cbar.set_label("Predicted − Real")

    coverage_patch = mpatches.Patch(facecolor="#6b6b6b", alpha=0.45,
                                     label="Outside model coverage (no data)")
    conf_patch = mpatches.Patch(facecolor="#888888", alpha=0.3,
                                 label="Gray mask = low ensemble confidence (model panel only)")
    event_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                               markerfacecolor=EVENT_MARKER_COLORS[hazard], markeredgecolor="black",
                               label=f"Dot = real detected {hazard} event (that day)")
    fig.legend(handles=[coverage_patch, conf_patch, event_handle], loc="lower center", fontsize=6.5,
               framealpha=0.9, ncol=3, bbox_to_anchor=(0.5, -0.1))
    add_standard_caption(fig, hazard, lead_time="t-1->t (model panel only)", y=-0.15)

    fig.suptitle(f"{title_prefix}{hazard} — {date}", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    render_pair("2025-05-17", "dust_storm", "test_pair.png",
                title_prefix="Dammam dust storm — ", highlight_city="Dammam")
