# =============================================================================
# MAZU — Layer 2 visual verification: forecast probability maps for known
# events (predicted the day BEFORE they happened), vs the actual outcome.
# =============================================================================
import os, importlib.util
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

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
CMAP = LinearSegmentedColormap.from_list("p", ["#0E1B2A", "#1C5D99", "#2BC8E2", "#FFD166", "#FF3355"])

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

    models = {}
    for hz in ["flash_flood", "heatwave"]:
        X, y, dates, lat_flat, lon_flat = fb.build_supervised(ds, hz)
        tr = dates <= fb.TRAIN_END
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08,
                                             class_weight="balanced", random_state=42, early_stopping=True)
        clf.fit(X[tr], y[tr])
        models[hz] = clf

    stride = 2
    yi = np.arange(0, len(lat_full), stride)
    xi = np.arange(0, len(lon_full), stride)
    lat_s, lon_s = lat_full[yi], lon_full[xi]

    fig, axs = plt.subplots(2, len(EVENTS), figsize=(4.6 * len(EVENTS), 8.4), facecolor="white")
    for col, (hz, d_from, d_to, label_var, thr, title) in enumerate(EVENTS):
        ti = int(np.where(times == d_from)[0][0])
        feat = np.stack([ds[v].values[ti][yi][:, xi] for v in fb.FEATURE_VARS], axis=-1)
        n_cells = feat.shape[0] * feat.shape[1]
        LAT2, LON2 = np.meshgrid(lat_s, lon_s, indexing="ij")
        doy = int(np.datetime64(d_from).astype("datetime64[D]").item().timetuple().tm_yday)
        extra = np.stack([LAT2, LON2, np.full_like(LAT2, doy)], axis=-1)
        X_map = np.concatenate([feat, extra], axis=-1).reshape(n_cells, -1)
        proba = models[hz].predict_proba(X_map)[:, 1].reshape(len(yi), len(xi))

        actual = ds[label_var].values[int(np.where(times == d_to)[0][0])][yi][:, xi] >= thr

        extent = [lon_full.min(), lon_full.max(), lat_full.min(), lat_full.max()]
        ax = axs[0, col]
        im = ax.imshow(proba, extent=extent, origin="upper", cmap=CMAP, vmin=0, vmax=1, aspect="auto")
        for c, (la, lo) in CITIES.items():
            ax.plot(lo, la, "o", ms=2.5, color="white"); ax.text(lo + 0.3, la + 0.3, c, fontsize=6, color="white")
        ax.set_title(f"forecast p({hz}) | {d_from}\n->predicting {d_to}", fontsize=8, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        ax2 = axs[1, col]
        ax2.imshow(np.where(actual, 1.0, np.nan), extent=extent, origin="upper", cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax2.set_facecolor("#0E1B2A")
        for c, (la, lo) in CITIES.items():
            ax2.plot(lo, la, "o", ms=2.5, color="black")
        ax2.set_title(f"ACTUAL {hz} on {d_to}\n({title})", fontsize=8)

    fig.suptitle("MAZU Layer 2 — one-day-ahead forecast probability vs actual outcome", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "forecast_vs_actual.png")
    fig.savefig(out, dpi=130)
    print(f"[SAVED] {out}")
    ds.close()


if __name__ == "__main__":
    main()
