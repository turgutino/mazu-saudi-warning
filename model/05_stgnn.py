# =============================================================================
# MAZU Saudi Arabia — Layer 2b: Spatiotemporal GNN (t -> t+1)
#
# Same forecasting task and train/test split as the Layer-2a baseline
# (HistGradientBoosting, per-cell independent), but now each grid cell's
# prediction can use MESSAGE PASSING from its spatial neighbours — testing
# whether explicit spatial structure improves skill, especially flash-flood
# recall, over the baseline's per-cell-independent view.
#
# Graph: static 4-connected grid graph over the same stride-2 subsampled
# grid used in Layer 2a (80x110 = 8800 nodes), so results are directly
# comparable.
#
# Missing values: NaN indicators are median-imputed (median computed from
# TRAINING data only, no leakage) and a companion "was_missing" binary
# feature is added per variable, so the model can learn to discount
# imputed values rather than being silently biased.
#
# Loss: masked, class-weighted BCE — loss is computed only on nodes with a
# valid t+1 label; weighting compensates severe class imbalance (same
# philosophy as Layer 2a's class_weight="balanced").
# =============================================================================

import os
import importlib.util
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spec = importlib.util.spec_from_file_location("fb", os.path.join(HERE, "03_forecast_baseline.py"))
fb = importlib.util.module_from_spec(spec)
src = open(os.path.join(HERE, "03_forecast_baseline.py"), encoding="utf-8").read()
src = src.replace('if __name__ == "__main__":\n    main()', "")
exec(compile(src, "fb", "exec"), fb.__dict__)

STRIDE = 2
VAL_START = "2025-06-01"   # carve out June from train for model selection (early stopping)


def build_grid_edges(n_lat, n_lon):
    """4-connected undirected grid graph over an (n_lat, n_lon) node array,
    flattened row-major (lat-major, lon-minor), matching build_supervised()."""
    idx = np.arange(n_lat * n_lon).reshape(n_lat, n_lon)
    src_list, dst_list = [], []
    # horizontal neighbours
    src_list += list(idx[:, :-1].ravel()); dst_list += list(idx[:, 1:].ravel())
    src_list += list(idx[:, 1:].ravel());  dst_list += list(idx[:, :-1].ravel())
    # vertical neighbours
    src_list += list(idx[:-1, :].ravel()); dst_list += list(idx[1:, :].ravel())
    src_list += list(idx[1:, :].ravel());  dst_list += list(idx[:-1, :].ravel())
    edge_index = torch.tensor(np.stack([src_list, dst_list]), dtype=torch.long)
    return edge_index


def build_daily_frames(ds, hazard):
    """Return per-day (X_raw, y, valid_label_mask) at stride-2 resolution,
    aligned exactly like build_supervised() in Layer 2a for comparability."""
    times = np.array([str(t)[:10] for t in ds.time.values])
    lat, lon = ds.latitude.values, ds.longitude.values
    yi = np.arange(0, len(lat), STRIDE); xi = np.arange(0, len(lon), STRIDE)
    n_lat_s, n_lon_s = len(yi), len(xi)
    LAT2, LON2 = np.meshgrid(lat[yi], lon[xi], indexing="ij")
    lat_flat, lon_flat = LAT2.ravel().astype("float32"), LON2.ravel().astype("float32")
    doy_all = np.array([d.timetuple().tm_yday for d in
                        ds.time.values.astype("datetime64[D]").astype(object)], dtype="float32")

    cfg = fb.TARGETS[hazard]
    label_all = ds[cfg["label_var"]].values
    feat_stack = np.stack([ds[v].values[:, yi][:, :, xi] for v in fb.FEATURE_VARS], axis=-1)  # (T,y,x,F)

    frames = []
    for ti in range(len(times) - 1):
        X_raw = feat_stack[ti].reshape(-1, feat_stack.shape[-1]).astype("float32")
        y_next = label_all[ti + 1][yi][:, xi].reshape(-1)
        valid = np.isfinite(y_next)
        y_bin = (y_next >= cfg["label_thr"]).astype("float32")
        frames.append({
            "date": times[ti + 1], "X_raw": X_raw, "y": y_bin, "valid": valid,
            "lat": lat_flat, "lon": lon_flat, "doy": doy_all[ti],
        })
    return frames, (n_lat_s, n_lon_s)


def compute_train_medians(frames, train_end):
    """Per-feature median from TRAINING frames only (no leakage)."""
    train_X = np.concatenate([f["X_raw"] for f in frames if f["date"] <= train_end], axis=0)
    med = np.nanmedian(train_X, axis=0)
    return med


def frame_to_data(f, median, edge_index):
    X_raw = f["X_raw"]
    missing = ~np.isfinite(X_raw)
    X_imp = np.where(missing, median[None, :], X_raw)
    extra = np.stack([f["lat"], f["lon"], np.full_like(f["lat"], f["doy"])], axis=-1)
    X = np.concatenate([X_imp, missing.astype("float32"), extra], axis=-1)
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(f["y"], dtype=torch.float32),
        valid=torch.tensor(f["valid"], dtype=torch.bool),
    )
    return data


class STGNN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.15)

    def forward(self, x, edge_index):
        h = self.act(self.conv1(x, edge_index))
        h = self.drop(h)
        h = self.act(self.conv2(h, edge_index)) + h        # residual
        h = self.drop(h)
        h = self.act(self.conv3(h, edge_index)) + h
        return self.out(h).squeeze(-1)                      # logits


def masked_weighted_bce(logits, y, valid, pos_weight):
    loss_all = nn.functional.binary_cross_entropy_with_logits(
        logits, y, pos_weight=pos_weight, reduction="none")
    loss_all = loss_all[valid]
    return loss_all.mean() if loss_all.numel() > 0 else logits.sum() * 0.0


def run_hazard(ds, hazard, edge_index, grid_shape):
    print(f"\n{'='*60}\n{hazard}\n{'='*60}")
    frames, _ = build_daily_frames(ds, hazard)
    median = compute_train_medians(frames, fb.TRAIN_END)

    # train = Jan..May, val = June (for early stopping), test = Jul..Dec (unseen, same as Layer 2a)
    train_frames = [f for f in frames if f["date"] < VAL_START]
    val_frames = [f for f in frames if VAL_START <= f["date"] <= fb.TRAIN_END]
    test_frames = [f for f in frames if f["date"] > fb.TRAIN_END]
    print(f"  train days={len(train_frames)}  val days={len(val_frames)}  test days={len(test_frames)}")

    train_data = [frame_to_data(f, median, edge_index) for f in train_frames]
    val_data = [frame_to_data(f, median, edge_index) for f in val_frames]
    test_data = [frame_to_data(f, median, edge_index) for f in test_frames]

    pos = sum(f["y"][f["valid"]].sum() for f in train_frames)
    neg = sum(f["valid"].sum() for f in train_frames) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=DEVICE)
    print(f"  train positive rate: {100*pos/(pos+neg):.3f}%  pos_weight={pos_weight.item():.1f}")

    in_dim = train_data[0].x.shape[1]
    model = STGNN(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)

    BATCH_DAYS = 12
    best_val_prauc, best_state, patience, bad_epochs = -1, None, 8, 0

    for epoch in range(1, 61):
        model.train()
        np.random.shuffle(train_data)
        tot_loss = 0.0
        for i in range(0, len(train_data), BATCH_DAYS):
            batch = Batch.from_data_list(train_data[i:i + BATCH_DAYS]).to(DEVICE)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = masked_weighted_bce(logits, batch.y, batch.valid, pos_weight)
            loss.backward(); opt.step()
            tot_loss += loss.item() * len(train_data[i:i + BATCH_DAYS])
        tot_loss /= len(train_data)

        # validation (threshold-free PR-AUC)
        model.eval()
        with torch.no_grad():
            vb = Batch.from_data_list(val_data).to(DEVICE)
            vlogits = model(vb.x, vb.edge_index)
            vproba = torch.sigmoid(vlogits).cpu().numpy()
            vy = vb.y.cpu().numpy(); vvalid = vb.valid.cpu().numpy()
        val_prauc = average_precision_score(vy[vvalid], vproba[vvalid])
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}  train_loss={tot_loss:.4f}  val_PR-AUC={val_prauc:.4f}")

        if val_prauc > best_val_prauc:
            best_val_prauc, best_state, bad_epochs = val_prauc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch} (best val PR-AUC={best_val_prauc:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tb = Batch.from_data_list(test_data).to(DEVICE)
        tlogits = model(tb.x, tb.edge_index)
        tproba = torch.sigmoid(tlogits).cpu().numpy()
        ty = tb.y.cpu().numpy(); tvalid = tb.valid.cpu().numpy()

    roc = roc_auc_score(ty[tvalid], tproba[tvalid])
    prauc = average_precision_score(ty[tvalid], tproba[tvalid])
    base_rate = ty[tvalid].mean()
    print(f"\n  TEST  ROC-AUC={roc:.3f}  PR-AUC={prauc:.3f}  (base rate {base_rate:.4f})")

    # per-day slice for known-event + negative-control checks
    n_nodes = grid_shape[0] * grid_shape[1]
    proba_by_date = {}
    for i, f in enumerate(test_frames):
        proba_by_date[f["date"]] = tproba[i * n_nodes:(i + 1) * n_nodes]

    return {"roc": roc, "prauc": prauc, "base_rate": float(base_rate), "proba_by_date": proba_by_date}


def main():
    ds = xr.open_dataset(fb.DATASET)
    lat = ds.latitude.values; lon = ds.longitude.values
    yi = np.arange(0, len(lat), STRIDE); xi = np.arange(0, len(lon), STRIDE)
    grid_shape = (len(yi), len(xi))
    edge_index = build_grid_edges(*grid_shape).to(DEVICE)
    print(f"Graph: {grid_shape[0]}x{grid_shape[1]} = {grid_shape[0]*grid_shape[1]} nodes, "
          f"{edge_index.shape[1]} directed edges")
    print(f"Device: {DEVICE}")

    known = {
        "flash_flood": [("2025-08-23", "Jizan 254.9mm"), ("2025-08-19", "Arabian Sea IVT 728")],
        "heatwave":    [("2025-07-25", "Riyadh Tmax 53.7C"), ("2025-08-16", "Persian Gulf heat-index 54.7C")],
    }
    controls = ["2025-11-05"]

    results = {}
    lines = ["=" * 70, "MAZU Layer 2b — Spatiotemporal GNN (t -> t+1), verified",
             "=" * 70, f"Graph: {grid_shape} = {grid_shape[0]*grid_shape[1]} nodes (4-connected, stride={STRIDE})",
             f"Train: Jan-May | Val: June | Test: Jul-Dec (same split as Layer 2a baseline)", ""]

    for hazard in ["heatwave", "flash_flood"]:
        r = run_hazard(ds, hazard, edge_index, grid_shape)
        results[hazard] = r
        lines.append(f"--- {hazard} ---")
        lines.append(f"TEST ROC-AUC={r['roc']:.3f}  PR-AUC={r['prauc']:.3f}  base_rate={r['base_rate']:.4f}")
        for date, note in known[hazard]:
            if date in r["proba_by_date"]:
                p = r["proba_by_date"][date]
                lines.append(f"  known event {date} ({note}): mean_p={p.mean():.3f} max_p={p.max():.3f} "
                            f"cells_p>0.5={int((p>0.5).sum())}/{len(p)}")
        for date in controls:
            if date in r["proba_by_date"]:
                p = r["proba_by_date"][date]
                lines.append(f"  negative control {date}: mean_p={p.mean():.3f} max_p={p.max():.3f}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("COMPARISON vs Layer 2a baseline (HistGradientBoosting, per-cell)")
    lines.append("=" * 70)
    lines.append(f"{'hazard':14s} {'metric':10s} {'baseline':>10s} {'ST-GNN':>10s} {'delta':>8s}")
    baseline_scores = {"heatwave": {"roc": 0.958, "prauc": 0.756}, "flash_flood": {"roc": 0.873, "prauc": 0.089}}
    for hz in ["heatwave", "flash_flood"]:
        for metric in ["roc", "prauc"]:
            b, g = baseline_scores[hz][metric], results[hz][metric]
            lines.append(f"{hz:14s} {metric.upper():10s} {b:10.3f} {g:10.3f} {g-b:+8.3f}")

    rpt = "\n".join(lines)
    with open(os.path.join(HERE, "stgnn_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(rpt)
    print("\n" + rpt)
    ds.close()


if __name__ == "__main__":
    main()
