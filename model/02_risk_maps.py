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
import importlib.util
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")

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

CMAPS = {  # perceptual, hazard-appropriate
    "flash_flood": LinearSegmentedColormap.from_list("ff", ["#0E1B2A", "#1C5D99", "#2BC8E2", "#B7F5D8", "#FFFFFF"]),
    "heatwave":    LinearSegmentedColormap.from_list("hw", ["#0E1B2A", "#7A1F1F", "#FF5A00", "#FFC300", "#FFFFFF"]),
}


def sea_mask(eng):
    """Cells that are ocean (SST valid most of the year)."""
    sst = eng.ds["sst_celsius"].values
    return np.isfinite(sst).mean(axis=0) > 0.5   # (lat, lon) bool


def draw_base(ax, eng, sea):
    lon, lat = eng.lon, eng.lat
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    # sea shading
    ax.imshow(np.where(sea, 1.0, np.nan), extent=extent, origin="upper",
              cmap="Blues", alpha=0.18, vmin=0, vmax=1, aspect="auto", zorder=0)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude E", fontsize=8); ax.set_ylabel("Latitude N", fontsize=8)
    ax.tick_params(labelsize=7)
    return extent


def plot_event(ax, eng, date, hazard, sea):
    extent = draw_base(ax, eng, sea)
    risk = eng.risk_field(date, hazard)
    im = ax.imshow(risk, extent=extent, origin="upper", cmap=CMAPS[hazard],
                   vmin=0, vmax=1, aspect="auto", alpha=0.92, zorder=1)
    # detected cluster peaks
    for e in eng.detect(date, hazard):
        ms = 40 + e["cluster_size"] ** 0.5 * 3
        ax.scatter(e["lon"], e["lat"], s=ms, facecolors="none",
                   edgecolors="#FF3355" if e["severity"] in ("extreme", "emergency") else "#FFD166",
                   linewidths=1.4, zorder=3)
    # cities
    for c, (la, lo) in CITIES.items():
        ax.plot(lo, la, "o", ms=3, color="white", zorder=4)
        ax.text(lo + 0.2, la + 0.2, c, fontsize=6, color="white", zorder=4)
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
    fig, axs = plt.subplots(2, 2, figsize=(12, 9), facecolor="white")
    for ax, (date, hz) in zip(axs.ravel(), events):
        im = plot_event(ax, eng, date, hz, sea)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="risk")
    fig.suptitle("MAZU — Detected extreme-event risk (known 2025 events)",
                 fontsize=13, fontweight="bold", color="#1A2D4A")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    f1 = os.path.join(OUT, "risk_known_events.png")
    fig.savefig(f1, dpi=130); plt.close(fig)

    # ── Figure 2: annual hotspots (spatial verification) ────────────────
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white")
    verify = {}
    for ax, hz in zip(axs, ["flash_flood", "heatwave"]):
        extent = draw_base(ax, eng, sea)
        hot = annual_hotspot(eng, hz)
        im = ax.imshow(hot, extent=extent, origin="upper", cmap=CMAPS[hz],
                       aspect="auto", alpha=0.92)
        for c, (la, lo) in CITIES.items():
            ax.plot(lo, la, "o", ms=3, color="white"); ax.text(lo + 0.2, la + 0.2, c, fontsize=6, color="white")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="extreme-tier days")
        ax.set_title(f"{hz} — 2025 annual hotspot", fontsize=10, fontweight="bold", color="#1A2D4A")
        # verification: where is the peak hotspot?
        yi, xi = np.unravel_index(np.argmax(hot), hot.shape)
        verify[hz] = (eng._region(float(eng.lat[yi]), float(eng.lon[xi])),
                      round(float(eng.lat[yi]), 1), round(float(eng.lon[xi]), 1), int(hot.max()))
    fig.suptitle("MAZU — Annual hazard hotspots (spatial climatology check)",
                 fontsize=13, fontweight="bold", color="#1A2D4A")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    f2 = os.path.join(OUT, "risk_annual_hotspots.png")
    fig.savefig(f2, dpi=130); plt.close(fig)

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
