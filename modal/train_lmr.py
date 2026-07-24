"""Learned-LMR model (Phase 2): a tiny MLP that predicts P(a move raises alpha)
from the search context the telemetry emits, trained on the `gen-search-telemetry`
dataset. The engine turns that probability into a clamped reduction correction
(`reduction = classical + clamp(learned, -1, +2)`): low P(raise alpha) -> reduce
more (save nodes), borderline -> leave it.

Model: 18 standardized features -> `hidden` ReLU -> 1 logit (sigmoid). ~320 params at
hidden 16; inference is a handful of FLOPs, cheap enough for the hot loop.

Export format RFLM v1 (little-endian):
  magic b"RFLM" | u32 version=1 | u32 input_dim | u32 hidden
  | feature_mean[input_dim] f32 | feature_scale[input_dim] f32   (standardization)
  | w1[hidden*input_dim] f32 | b1[hidden] f32 | w2[hidden] f32 | b2 f32
The Rust loader normalizes with mean/scale then runs the same forward, so training
and inference agree by construction.

TSV schema (0-based columns) the reader consumes:
  0 pos_id 1 depth 2 ply 3 move_index 4 is_quiet 5 is_priority 6 pv_node
  7 gives_check 8 static_eval 9 extension 10 reduction 11 lmp_pruned
  12 raised_alpha 13 caused_cutoff 14 needed_lmr_research 15 needed_pvs_research
  16 subtree_nodes 17 history_score 18 is_tt_move 19 is_killer 20 is_counter
  21 is_capture 22 is_promotion 23 node_in_check 24 tt_depth

Columns 17..24 are the v2 additions, appended so every v1 index (notably the target
and the `lmp_pruned` filter) is unchanged.
"""

import struct

MAGIC = b"RFLM"
VERSION = 1
# Feature columns (order defines the model input vector) and the target/filter cols.
# v1 context is 1..10 (depth..reduction); v2 appended 17..24 (history_score,
# is_tt_move, is_killer, is_counter, is_capture, is_promotion, node_in_check,
# tt_depth). The v2 columns are APPENDED in the TSV, so the target and filter column
# indices below are unchanged from v1.
#
# Why: the v1 model saturated at val AUC ~0.94 across both more data and more
# capacity — it was feature-limited, not data- or capacity-limited. These columns are
# the cheap high-signal additions available at the same hook.
FEATURE_COLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24]
TARGET_COL = 12       # raised_alpha
LMP_PRUNED_COL = 11   # exclude pruned (unsearched) moves — their outcome is not real
INPUT_DIM = len(FEATURE_COLS)
STATIC_EVAL_COL = 8   # clamped before standardization so mate scores don't blow the scale
HISTORY_COL = 17      # clamped too: history scores are unbounded and would skew the scale
# Rows shorter than this are a different (older) schema and are skipped.
MIN_COLUMNS = max(FEATURE_COLS + [TARGET_COL, LMP_PRUNED_COL]) + 1


def load_telemetry_sample(path, stride=24, max_rows=10_000_000):
    """Stride-sample the (huge) telemetry TSV into standardization-ready float arrays.

    Returns (X, y) as numpy float32/[N] arrays. Skips the header, filters out
    `lmp_pruned` rows (their `raised_alpha` is not a searched-move outcome), and
    stops once `max_rows` samples are collected. `stride` decorrelates rows from the
    same search (records are grouped by position)."""
    import numpy as np

    feats = []
    targets = []
    with open(path, "r", encoding="utf-8") as handle:
        handle.readline()  # header
        for i, line in enumerate(handle):
            if i % stride:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < MIN_COLUMNS:
                continue
            if parts[LMP_PRUNED_COL] == "1":
                continue
            try:
                row = [float(parts[c]) for c in FEATURE_COLS]
                y = float(parts[TARGET_COL])
            except ValueError:
                continue
            # Clamp the unbounded scalars so one extreme row can't dominate the scale.
            idx = FEATURE_COLS.index(STATIC_EVAL_COL)
            row[idx] = max(-2000.0, min(2000.0, row[idx]))
            hidx = FEATURE_COLS.index(HISTORY_COL)
            row[hidx] = max(-20000.0, min(20000.0, row[hidx]))
            feats.append(row)
            targets.append(y)
            if len(targets) >= max_rows:
                break
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X, y


def train(X, y, hidden=16, epochs=12, batch_size=8192, lr=1e-3, device="cpu"):
    """Fit the tiny MLP (BCE on raised_alpha). Returns (model, mean, scale, metrics)
    where metrics reports base rate, val accuracy, and val AUC — AUC well above 0.5 is
    the signal that the reduction decision is learnable."""
    import numpy as np
    import torch
    from torch import nn

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0  # guard constant columns (e.g. a bool that never varied)
    scale = 1.0 / std
    Xn = (X - mean) * scale

    n = len(y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xt = torch.from_numpy(Xn).to(device)
    yt = torch.from_numpy(y).to(device)
    tr = torch.from_numpy(train_idx).to(device)
    va = torch.from_numpy(val_idx).to(device)

    model = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(device)
    # Class weight for the ~20% positive rate so the model doesn't collapse to "never".
    base = float(yt.mean())
    pos_weight = torch.tensor([(1 - base) / max(base, 1e-6)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        idx = tr[torch.randperm(tr.numel(), device=device)]
        for start in range(0, idx.numel(), batch_size):
            b = idx[start:start + batch_size]
            opt.zero_grad()
            logit = model(Xt[b]).squeeze(1)
            loss_fn(logit, yt[b]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_logit = model(Xt[va]).squeeze(1)
        val_prob = torch.sigmoid(val_logit).cpu().numpy()
    y_val = y[val_idx]
    acc = float(((val_prob > 0.5).astype("float32") == y_val).mean())
    auc = _auc(y_val, val_prob)
    metrics = {"n": int(n), "base_rate": base, "val_acc": acc, "val_auc": auc}
    return model, mean.astype("float32"), scale.astype("float32"), metrics


def _auc(y_true, scores):
    """ROC AUC via the rank formula (no sklearn dependency)."""
    import numpy as np

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1.0
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def export_rflm(model, mean, scale, out_path):
    """Serialize the model to the RFLM v1 format (see module docstring)."""
    import numpy as np

    layers = [m for m in model if hasattr(m, "weight")]
    w1 = layers[0].weight.detach().cpu().numpy().astype("<f4")  # [hidden, input]
    b1 = layers[0].bias.detach().cpu().numpy().astype("<f4")     # [hidden]
    w2 = layers[1].weight.detach().cpu().numpy().reshape(-1).astype("<f4")  # [hidden]
    b2 = float(layers[1].bias.detach().cpu().numpy()[0])
    hidden = b1.shape[0]
    with open(out_path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<III", VERSION, INPUT_DIM, hidden))
        handle.write(np.asarray(mean, dtype="<f4").tobytes())
        handle.write(np.asarray(scale, dtype="<f4").tobytes())
        handle.write(w1.reshape(-1).tobytes())  # row-major [hidden, input]
        handle.write(b1.tobytes())
        handle.write(w2.tobytes())
        handle.write(struct.pack("<f", b2))
    return out_path
