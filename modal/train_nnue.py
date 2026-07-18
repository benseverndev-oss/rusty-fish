"""GPU NNUE trainer that exports the engine's `RFNN` network format.

Reads the TSV emitted by `engine-bench gen-data` (one sample per line:
``target_cp<TAB>own_feature_csv<TAB>opp_feature_csv``), trains a perspective
network whose forward pass mirrors the quantised inference in
``engine-search/src/nnue.rs`` exactly, using the win-probability (WDL) loss, and
writes a quantised ``RFNN`` file that the Rust engine can load with
``setoption name EvalFile`` or ``engine-bench nnue-sprt``.

It is dependency-light (PyTorch only) and runs anywhere a GPU is available —
locally or on Modal. It is intentionally standalone so it can be unit-checked
without Modal:

    python train_nnue.py data.tsv out.rfnn --hidden 128 --epochs 30

Validate an exported network against the Rust engine:

    engine-bench nnue-sprt out.rfnn 5
"""

from __future__ import annotations

import argparse
import struct
import sys

# These MUST match engine-search/src/nnue.rs.
INPUT_DIMENSION = 2 * 6 * 64  # 768
MAGIC = b"RFNN"
FORMAT_VERSION = 1
ACTIVATION_CLIP = 127.0  # clipped-ReLU upper bound
OUTPUT_SCALE = 64.0      # inference divides the integer output by this
EVAL_CLAMP = 20000       # inference clamps the centipawn score to +/- this
WDL_SCALE = 400.0        # centipawns -> win-probability steepness


def target_win_prob(target: float, wdl_target: bool) -> float:
    """The training target as a win-probability in [0, 1].

    In WDL mode the target already IS a win-probability (a game outcome
    0.0/0.5/1.0), so it is used directly. In centipawn mode it is squashed
    through the WDL sigmoid. Encoding a 0/1 outcome as centipawns would be
    degenerate (logit(1) is infinite), which is why WDL data needs this mode.
    """
    if wdl_target:
        return min(1.0, max(0.0, target))
    import math

    return 1.0 / (1.0 + math.exp(-target / WDL_SCALE))


def _load_samples(path: str):
    """Yields (own_indices, opp_indices, target_cp) parsed from a gen-data TSV."""
    owns, opps, targets = [], [], []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            target_str, own_str, opp_str = line.split("\t")
            targets.append(float(target_str))
            owns.append([int(x) for x in own_str.split(",") if x != ""])
            opps.append([int(x) for x in opp_str.split(",") if x != ""])
    return owns, opps, targets


def _ragged_to_bag(rows):
    """Flattens ragged index lists into (values, offsets) for nn.EmbeddingBag."""
    import torch

    offsets = [0]
    flat = []
    for row in rows:
        flat.extend(row)
        offsets.append(len(flat))
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets[:-1], dtype=torch.long),
    )


def train(
    data_path: str,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    wdl_target: bool = False,
):
    import torch
    from torch import nn

    owns, opps, targets = _load_samples(data_path)
    if not owns:
        raise SystemExit(f"no training samples in {data_path}")

    own_values, own_offsets = _ragged_to_bag(owns)
    opp_values, opp_offsets = _ragged_to_bag(opps)
    target = torch.tensor(targets, dtype=torch.float32)

    class Nnue(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            # weight[feature] is the hidden-vector added to the accumulator, which
            # is exactly the RFNN feature_weights row-major (feature*hidden + i)
            # layout. mode="sum" reproduces the accumulator's summed columns.
            self.transformer = nn.EmbeddingBag(INPUT_DIMENSION, hidden, mode="sum")
            self.feature_bias = nn.Parameter(torch.zeros(hidden))
            self.output = nn.Linear(2 * hidden, 1)
            nn.init.uniform_(self.transformer.weight, -0.1, 0.1)
            nn.init.uniform_(self.output.weight, -0.1, 0.1)
            nn.init.zeros_(self.output.bias)

        def forward(self, own_v, own_o, opp_v, opp_o):
            acc_own = self.transformer(own_v, own_o) + self.feature_bias
            acc_opp = self.transformer(opp_v, opp_o) + self.feature_bias
            a_own = torch.clamp(acc_own, 0.0, ACTIVATION_CLIP)
            a_opp = torch.clamp(acc_opp, 0.0, ACTIVATION_CLIP)
            features = torch.cat([a_own, a_opp], dim=1)
            # pred is centipawns (inference divides the integer output by 64).
            return self.output(features).squeeze(1) / OUTPUT_SCALE

    model = Nnue(hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    own_values, own_offsets = own_values.to(device), own_offsets.to(device)
    opp_values, opp_offsets = opp_values.to(device), opp_offsets.to(device)
    target = target.to(device)
    # In WDL mode the target already IS a win-probability (0.0/0.5/1.0 game
    # outcome) and is used directly; in centipawn mode it is squashed through the
    # WDL sigmoid. See target_win_prob for why the two paths cannot be merged.
    if wdl_target:
        target_wp = torch.clamp(target, 0.0, 1.0)
    else:
        target_wp = torch.sigmoid(target / WDL_SCALE)

    count = len(owns)
    for epoch in range(epochs):
        permutation = torch.randperm(count, device=device)
        total = 0.0
        for start in range(0, count, batch_size):
            idx = permutation[start:start + batch_size]
            # Rebuild ragged bags for the minibatch on CPU indices.
            ov, oo = _ragged_to_bag([owns[i] for i in idx.tolist()])
            pv, po = _ragged_to_bag([opps[i] for i in idx.tolist()])
            pred_cp = model(ov.to(device), oo.to(device), pv.to(device), po.to(device))
            pred_wp = torch.sigmoid(pred_cp / WDL_SCALE)
            loss = ((pred_wp - target_wp[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{epochs}: wdl_loss {total / count:.6f}", file=sys.stderr)

    return model


def quantize_and_write(model, hidden: int, out_path: str):
    import torch

    with torch.no_grad():
        w1 = model.transformer.weight.detach().cpu()          # [INPUT, hidden]
        b1 = model.feature_bias.detach().cpu()                # [hidden]
        w2 = model.output.weight.detach().cpu().squeeze(0)    # [2*hidden]
        b2 = float(model.output.bias.detach().cpu().item())

    def to_i16(t):
        return torch.clamp(torch.round(t), -32768, 32767).to(torch.int16)

    fw = to_i16(w1).reshape(-1).tolist()   # row-major feature*hidden + i
    fb = to_i16(b1).tolist()
    ow = to_i16(w2).tolist()               # own (hidden) then opp (hidden)
    ob = int(round(b2))

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<I", FORMAT_VERSION)
    blob += struct.pack("<I", hidden)
    blob += struct.pack(f"<{len(fw)}h", *fw)
    blob += struct.pack(f"<{len(fb)}h", *fb)
    blob += struct.pack(f"<{len(ow)}h", *ow)
    blob += struct.pack("<i", ob)
    with open(out_path, "wb") as handle:
        handle.write(blob)
    print(f"wrote {out_path} ({len(blob)} bytes, hidden={hidden})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Train and export an RFNN NNUE network.")
    parser.add_argument("data", help="gen-data TSV path")
    parser.add_argument("out", help="output .rfnn path")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--wdl-target",
        action="store_true",
        help="targets are game-outcome win-probabilities (0/0.5/1), used directly "
        "instead of squashed through the centipawn sigmoid",
    )
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = train(
        args.data, args.hidden, args.epochs, args.batch_size, args.lr, device,
        wdl_target=args.wdl_target,
    )
    quantize_and_write(model, args.hidden, args.out)


if __name__ == "__main__":
    main()
