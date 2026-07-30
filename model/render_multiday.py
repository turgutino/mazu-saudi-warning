import sys
sys.path.insert(0, r"C:\Users\Turqut\Desktop\Competation\mazu-system\agent")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import tools
from render_build_grid import predicted_grid, real_grid, predicted_grid_uncertainty
from render_map import (EXTENT, PROJ, PC, _draw_base, _draw_cities, _draw_event_markers,
                         EVENT_MARKER_COLORS, tiered_norm_cmap, tiered_rgba, confidence_mask_rgba,
                         resample_nearest, bias_rgba, BIAS_CMAP, BIAS_NORM, add_standard_caption)
import matplotlib.patches as mpatches


def render_multiday(dates, hazard, out_path, event_label, highlight_city=None, extra_markers=None):
    """Small-multiples: row 1 = real (ground-truth), row 2 = predicted, row
    3 = Predicted-Real bias (Priority-4 item 3), one column per date.
    Hazard-specific tiered color scale throughout (Priority-2 item 1+2);
    predicted row additionally draws a separate semi-transparent gray
    confidence-mask layer on top, from the 5-member ensemble spread
    (Priority-2 item 3)."""
    n = len(dates)
    fig, axes = plt.subplots(3, n, figsize=(3.1 * n, 9.2),
                              subplot_kw={"projection": PROJ})
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, date in enumerate(dates):
        rlat, rlon, rgrid = real_grid(date, hazard)
        plat, plon, pgrid = predicted_grid(date, hazard)
        _, _, std_grid = predicted_grid_uncertainty(date, hazard)
        mask_rgba = confidence_mask_rgba(std_grid)
        rgrid_resampled = resample_nearest(rgrid, rlat, rlon, plat, plon)
        bias_rgba_arr, _ = bias_rgba(rgrid_resampled, pgrid)

        ax_r = axes[0, col]
        _draw_base(ax_r)
        rgba_r = tiered_rgba(rgrid, hazard, base_alpha=0.65)
        # Priority-5 item 2: interpolation="nearest" everywhere a real risk/
        # bias/confidence grid is drawn -- sharp per-cell pixels, no
        # cross-cell blending that could fabricate a smooth contiguous
        # high-risk patch across a real terrain boundary.
        ax_r.imshow(rgba_r, extent=[rlon.min(), rlon.max(), rlat.min(), rlat.max()],
                    origin="upper", transform=PC, zorder=1.5, interpolation="nearest")
        _draw_cities(ax_r, highlight=highlight_city, extra_markers=extra_markers)
        _draw_event_markers(ax_r, date, hazard)
        ax_r.set_title(date, fontsize=10)

        extent_p = [plon.min(), plon.max(), plat.min(), plat.max()]
        ax_p = axes[1, col]
        _draw_base(ax_p)
        rgba_p = tiered_rgba(pgrid, hazard, base_alpha=0.65)
        ax_p.imshow(rgba_p, extent=extent_p, origin="upper", transform=PC, zorder=1.5, interpolation="nearest")
        ax_p.imshow(mask_rgba, extent=extent_p, origin="upper", transform=PC, zorder=1.6, interpolation="nearest")
        _draw_cities(ax_p, highlight=highlight_city, extra_markers=extra_markers)
        _draw_event_markers(ax_p, date, hazard)

        ax_b = axes[2, col]
        _draw_base(ax_b)
        ax_b.imshow(bias_rgba_arr, extent=extent_p, origin="upper", transform=PC, zorder=1.5, interpolation="nearest")
        _draw_cities(ax_b, highlight=highlight_city, extra_markers=extra_markers)
        _draw_event_markers(ax_b, date, hazard)

    axes[0, 0].text(-0.12, 0.5, "REAL\n(ground-truth)", transform=axes[0, 0].transAxes,
                     fontsize=10, fontweight="bold", ha="center", va="center", rotation=90)
    axes[1, 0].text(-0.12, 0.5, "PREDICTED\n(model, t-1→t)", transform=axes[1, 0].transAxes,
                     fontsize=10, fontweight="bold", ha="center", va="center", rotation=90)
    axes[2, 0].text(-0.12, 0.5, "BIAS\n(Predicted−Real)", transform=axes[2, 0].transAxes,
                     fontsize=10, fontweight="bold", ha="center", va="center", rotation=90)

    cmap, norm, bounds = tiered_norm_cmap(hazard)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes[0:2, :], orientation="horizontal", ticks=bounds,
                         fraction=0.035, pad=0.08, shrink=0.4)
    cbar.set_label(f"Risk tier ({hazard})")
    cbar.ax.set_xticklabels([f"{b:.1f}" for b in bounds])

    bias_sm = plt.cm.ScalarMappable(cmap=BIAS_CMAP, norm=BIAS_NORM)
    bias_cbar = fig.colorbar(bias_sm, ax=axes[2, :], orientation="horizontal",
                              fraction=0.035, pad=0.15, shrink=0.4)
    bias_cbar.set_label("Predicted − Real")

    coverage_patch = mpatches.Patch(facecolor="#6b6b6b", alpha=0.45,
                                     label="Outside model coverage (no data)")
    conf_patch = mpatches.Patch(facecolor="#888888", alpha=0.3,
                                 label="Gray mask = low ensemble confidence (predicted row only)")
    event_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                               markerfacecolor=EVENT_MARKER_COLORS[hazard], markeredgecolor="black",
                               label=f"Dot = real detected {hazard} event (that day)")
    fig.legend(handles=[coverage_patch, conf_patch, event_handle], loc="lower center", fontsize=6.5,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.03))
    add_standard_caption(fig, hazard, lead_time="t-1->t (PREDICTED/BIAS rows only)", y=-0.07)

    fig.suptitle(event_label, fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    render_multiday(
        ["2025-05-16", "2025-05-17", "2025-05-18", "2025-05-19"],
        "dust_storm", "test_multiday_dust.png",
        "Dammam dust storm, 16-19 May 2025 -- national event, city-level peak on 17th",
        highlight_city="Dammam",
    )
