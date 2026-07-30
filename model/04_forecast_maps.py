# =============================================================================
# MAZU — Layer 2 visual verification: forecast probability maps for known
# events (predicted the day BEFORE they happened), vs the actual outcome.
# =============================================================================
import os, sys, importlib.util
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_map import (PROJ, PC, DATA_EXTENT, NEIGHBOR_LABELS, LABEL_OFFSETS,
                         _terrain_rgb, _NEIGHBOR_OUTLINE, tiered_norm_cmap, tiered_rgba,
                         confidence_mask_rgba, EVENT_MARKER_COLORS, BIAS_CMAP, BIAS_NORM,
                         add_standard_caption)
import tools

_TEXT_OUTLINE = [pe.withStroke(linewidth=2.0, foreground="black")]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

spec = importlib.util.spec_from_file_location("fb", os.path.join(HERE, "03_forecast_baseline.py"))
fb = importlib.util.module_from_spec(spec)
src = open(os.path.join(HERE, "03_forecast_baseline.py"), encoding="utf-8").read()
src = src.replace('if __name__ == "__main__":\n    main()', "")
exec(compile(src, "fb", "exec"), fb.__dict__)

CITIES = {"Jeddah": (21.5, 39.2), "Mecca": (21.4, 39.8), "Riyadh": (24.7, 46.7),
          "Jizan": (16.9, 42.6), "Dammam": (26.4, 50.1), "Taif": (21.3, 40.4),
          "Medina": (24.5, 39.6), "Abha": (18.2, 42.5)}
_ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]

EVENTS = [
    ("flash_flood", "2025-08-22", "2025-08-23", "flash_flood_risk", 2, "Jizan 254.9mm event"),
    ("flash_flood", "2025-08-18", "2025-08-19", "flash_flood_risk", 2, "Arabian Sea IVT 728 event"),
    ("heatwave",    "2025-07-24", "2025-07-25", "heatwave_day_flag", 1, "Riyadh Tmax 53.7C event"),
    ("heatwave",    "2025-08-15", "2025-08-16", "heatwave_day_flag", 1, "Persian Gulf heat-index 54.7C event"),
]


def main():
    ds = xr.open_dataset(fb.DATASET)
    times = np.array([str(t)[:10] for t in ds.time.values])
    lat_full, lon_full = ds.latitude.values, ds.longitude.values

    # Priority-2 confidence layer: this script trains its own model (a
    # different feature set than the saved production ensemble in
    # agent/saved_models/ensemble/, so that one can't be reused directly).
    # For a genuine per-cell model-confidence signal here, train a real
    # 5-member bootstrap ensemble of the same architecture -- same idea as
    # the production ensemble, just scoped to this script's own model --
    # rather than fabricating an uncertainty estimate.
    models = {}
    for hz in ["flash_flood", "heatwave"]:
        X, y, dates, lat_flat, lon_flat = fb.build_supervised(ds, hz)
        tr = dates <= fb.TRAIN_END
        Xtr, ytr = X[tr], y[tr]
        rng = np.random.default_rng(0)
        members = []
        for seed in _ENSEMBLE_SEEDS:
            boot = rng.integers(0, len(Xtr), size=len(Xtr))
            clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                                 class_weight="balanced", random_state=seed, early_stopping=True)
            clf.fit(Xtr[boot], ytr[boot])
            members.append(clf)
        models[hz] = members

    stride = 2
    yi = np.arange(0, len(lat_full), stride)
    xi = np.arange(0, len(lon_full), stride)
    lat_s, lon_s = lat_full[yi], lon_full[xi]

    fig, axs = plt.subplots(3, len(EVENTS), figsize=(4.6 * len(EVENTS), 12.2), facecolor="white",
                             subplot_kw={"projection": PROJ})
    plot_extent = [32.0, 58.0, 14.0, 34.0]  # includes neighbor-country buffer zone

    def _draw_base(ax):
        ax.set_extent(plot_extent, crs=PC)
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

        outer = [(plot_extent[0], plot_extent[2]), (plot_extent[1], plot_extent[2]),
                  (plot_extent[1], plot_extent[3]), (plot_extent[0], plot_extent[3]), (plot_extent[0], plot_extent[2])]
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
        gl.xlabel_style = {"size": 5.5, "color": "white"}
        gl.ylabel_style = {"size": 5.5, "color": "white"}

        for name, (nlat, nlon) in NEIGHBOR_LABELS.items():
            ax.text(nlon, nlat, name, fontsize=5.5, style="italic", color="#cccccc", ha="center",
                    transform=PC, zorder=4, path_effects=_NEIGHBOR_OUTLINE)

    def _draw_cities(ax, color):
        for c, (la, lo) in CITIES.items():
            dx, dy, ha = LABEL_OFFSETS.get(c, (0.35, 0.0, "left"))
            ax.plot(lo, la, "o", ms=2.5, color=color, transform=PC, zorder=4)
            ax.text(lo + dx, la + dy, c, fontsize=6, color=color, ha=ha,
                     transform=PC, zorder=4, path_effects=_TEXT_OUTLINE)

    def _draw_event_markers(ax, date, hazard):
        """Priority-3 item 1: real detected-event peak point(s) for the date
        this panel represents (d_to -- the date being forecast/verified),
        same rule-based DetectionEngine used everywhere else on the site."""
        eng = tools._get_detection_engine()
        try:
            events = eng.detect(date, hazard)
        except (KeyError, IndexError, ValueError):
            return
        color = EVENT_MARKER_COLORS[hazard]
        for e in events:
            ax.plot(e["lon"], e["lat"], marker="o", markersize=7,
                     markerfacecolor=color, markeredgecolor="black",
                     markeredgewidth=0.9, transform=PC, zorder=5.5)

    for col, (hz, d_from, d_to, label_var, thr, title) in enumerate(EVENTS):
        ti = int(np.where(times == d_from)[0][0])
        feat = np.stack([ds[v].values[ti][yi][:, xi] for v in fb.FEATURE_VARS], axis=-1)
        n_cells = feat.shape[0] * feat.shape[1]
        LAT2, LON2 = np.meshgrid(lat_s, lon_s, indexing="ij")
        doy = int(np.datetime64(d_from).astype("datetime64[D]").item().timetuple().tm_yday)
        extra = np.stack([LAT2, LON2, np.full_like(LAT2, doy)], axis=-1)
        X_map = np.concatenate([feat, extra], axis=-1).reshape(n_cells, -1)
        member_proba = np.stack([m.predict_proba(X_map)[:, 1] for m in models[hz]], axis=0)
        proba = member_proba.mean(axis=0).reshape(len(yi), len(xi))
        std = member_proba.std(axis=0).reshape(len(yi), len(xi))
        mask_rgba = confidence_mask_rgba(std)

        actual = ds[label_var].values[int(np.where(times == d_to)[0][0])][yi][:, xi] >= thr

        ax = axs[0, col]
        _draw_base(ax)
        rgba = tiered_rgba(proba, hz, base_alpha=0.7)
        extent_f = [lon_s.min(), lon_s.max(), lat_s.min(), lat_s.max()]
        im = ax.imshow(rgba, extent=extent_f, origin="upper", transform=PC, zorder=1.5,
                       interpolation="nearest")
        ax.imshow(mask_rgba, extent=extent_f, origin="upper", transform=PC, zorder=1.6,
                  interpolation="nearest")
        _draw_cities(ax, "white")
        _draw_event_markers(ax, d_to, hz)
        ax.set_title(f"forecast p({hz}) | {d_from}\n->predicting {d_to}", fontsize=8, fontweight="bold")
        cmap, norm, bounds = tiered_norm_cmap(hz)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, ticks=bounds)
        cbar.ax.tick_params(labelsize=6)

        ax2 = axs[1, col]
        _draw_base(ax2)
        ax2.pcolormesh(lon_s, lat_s, np.where(actual, 1.0, np.nan), cmap="Reds", vmin=0, vmax=1,
                        transform=PC, alpha=0.85, zorder=1.5, shading="auto")
        _draw_cities(ax2, "black")
        _draw_event_markers(ax2, d_to, hz)
        ax2.set_title(f"ACTUAL {hz} on {d_to}\n({title})", fontsize=8)

        # Priority-4 item 3: Predicted - Real bias panel. proba and actual
        # are already on the identical (yi, xi) stride grid, so no
        # resampling is needed here (unlike render_map.py's render_pair,
        # where the real/predicted grids come from different resolutions).
        ax3 = axs[2, col]
        _draw_base(ax3)
        bias = proba - actual.astype(float)
        bias_rgba_arr = BIAS_CMAP(BIAS_NORM(bias))
        bias_rgba_arr[..., 3] = 0.8
        ax3.imshow(bias_rgba_arr, extent=extent_f, origin="upper", transform=PC, zorder=1.5,
                  interpolation="nearest")
        _draw_cities(ax3, "black")
        _draw_event_markers(ax3, d_to, hz)
        ax3.set_title(f"BIAS: forecast − actual\n{hz} | {d_to}", fontsize=8)

    bias_sm = plt.cm.ScalarMappable(cmap=BIAS_CMAP, norm=BIAS_NORM)
    bias_cbar = fig.colorbar(bias_sm, ax=axs[2, :], orientation="horizontal",
                              fraction=0.04, pad=0.12, shrink=0.4)
    bias_cbar.set_label("Predicted − Actual")

    coverage_patch = mpatches.Patch(facecolor="#444444", alpha=0.55, label="Outside model coverage (no data)")
    conf_patch = mpatches.Patch(facecolor="#888888", alpha=0.3,
                                 label="Gray mask = low ensemble confidence (forecast row only)")
    ff_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                            markerfacecolor=EVENT_MARKER_COLORS["flash_flood"], markeredgecolor="black",
                            label="Dot = real detected flash_flood event")
    hw_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                            markerfacecolor=EVENT_MARKER_COLORS["heatwave"], markeredgecolor="black",
                            label="Dot = real detected heatwave event")
    fig.legend(handles=[coverage_patch, conf_patch, ff_handle, hw_handle], loc="lower center",
               fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.04), ncol=2)
    add_standard_caption(fig, ["flash_flood", "heatwave"], lead_time="t-1->t (forecast row only)", y=-0.08)

    fig.suptitle("MAZU Layer 2 — one-day-ahead forecast probability vs actual outcome", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "forecast_vs_actual.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[SAVED] {out}")
    ds.close()


if __name__ == "__main__":
    main()
