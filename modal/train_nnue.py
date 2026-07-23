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
MAX_FEATURES = 32        # >= max active features per perspective (<=16 pieces/side)
PAD_INDEX = INPUT_DIMENSION  # 768: a dedicated padding row, frozen to zero
VAL_EVERY = 50           # 1-in-50 samples (~2%) held out for validation


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


def _load_padded(path: str):
    """Parse a gen-data TSV straight into padded numpy arrays.

    Returns ``(own, opp, targets)`` where ``own``/``opp`` are ``[N, MAX_FEATURES]``
    int32 arrays padded with PAD_INDEX and ``targets`` is a ``[N]`` float32 array.

    The naive path — building a Python list-of-lists for every sample and then
    flattening it — holds ~40 GB for the 19M-position corpus (each feature is a
    boxed Python int inside a per-row list), which is what forces the trainer's
    huge memory request and costs minutes of pure-Python parsing. Instead we count
    the rows in one cheap pass, preallocate the two int32 matrices, and scatter each
    row's features in place with `np.fromstring` (C-level CSV parsing) — no boxed
    ints, no list-of-lists, no giant flatten. On a 1M-row corpus this is ~6x faster
    and ~7x lighter (274 MB vs ~2 GB peak) with byte-identical output.

    A perspective has <=16 pieces so a legal sample never exceeds MAX_FEATURES; a
    malformed public FEN can parse into an illegal >32-piece board, so such rows are
    dropped (not aborted on) exactly as before, keeping surviving rows in order."""
    import numpy as np

    # Pass 1: count non-empty rows so the output matrices can be preallocated.
    with open(path, "rb") as handle:
        total = sum(1 for line in handle if line.strip())

    own = np.full((total, MAX_FEATURES), PAD_INDEX, dtype=np.int32)
    opp = np.full((total, MAX_FEATURES), PAD_INDEX, dtype=np.int32)
    targets = np.empty(total, dtype=np.float32)

    # Pass 2: parse each row directly into row `write` of the preallocated arrays.
    # `write` trails the read index so over-MAX_FEATURES rows are skipped in place.
    write = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            target_str, own_str, opp_str = line.split("\t")
            own_feat = np.fromstring(own_str, dtype=np.int32, sep=",")
            opp_feat = np.fromstring(opp_str, dtype=np.int32, sep=",")
            if own_feat.size > MAX_FEATURES or opp_feat.size > MAX_FEATURES:
                continue
            own[write, : own_feat.size] = own_feat
            opp[write, : opp_feat.size] = opp_feat
            targets[write] = float(target_str)
            write += 1

    if write < total:
        dropped = total - write
        print(
            f"dropped {dropped} samples over MAX_FEATURES "
            f"({100 * dropped / total:.4f}%)",
            flush=True,
        )
    return own[:write], opp[:write], targets[:write]


def _load_padded_shards(shard_paths):
    """Parse several gen-data TSV shards into one set of padded arrays.

    Same preallocated in-place scatter as `_load_padded`, but over a list of shard
    files, so the caller never has to concatenate them into one multi-GB temp file
    first. Rows are kept in shard order (shards processed in the given order)."""
    import numpy as np

    total = 0
    for path in shard_paths:
        with open(path, "rb") as handle:
            total += sum(1 for line in handle if line.strip())

    own = np.full((total, MAX_FEATURES), PAD_INDEX, dtype=np.int32)
    opp = np.full((total, MAX_FEATURES), PAD_INDEX, dtype=np.int32)
    targets = np.empty(total, dtype=np.float32)

    write = 0
    for path in shard_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                target_str, own_str, opp_str = line.split("\t")
                own_feat = np.fromstring(own_str, dtype=np.int32, sep=",")
                opp_feat = np.fromstring(opp_str, dtype=np.int32, sep=",")
                if own_feat.size > MAX_FEATURES or opp_feat.size > MAX_FEATURES:
                    continue
                own[write, : own_feat.size] = own_feat
                opp[write, : opp_feat.size] = opp_feat
                targets[write] = float(target_str)
                write += 1

    if write < total:
        dropped = total - write
        print(
            f"dropped {dropped} samples over MAX_FEATURES "
            f"({100 * dropped / total:.4f}%)",
            flush=True,
        )
    return own[:write], opp[:write], targets[:write]


def _shard_manifest(shard_paths):
    """A cache key identifying exactly these shard contents.

    The store is append-only (shards are added, never rewritten in place), so a
    shard's (full path, byte size) pins its content: if any shard is added, removed,
    or changed, the manifest changes and the cache is rebuilt. The full path (not
    the basename) keeps same-named shards in different datasets distinct."""
    import os

    return sorted(
        (p, os.path.getsize(p)) for p in shard_paths
    )


def load_corpus(shard_paths, cache_dir=None):
    """Parse `shard_paths` into padded arrays, memoized under `cache_dir`.

    A sweep spins a fresh container per config, each re-parsing the same store
    corpus from scratch. When `cache_dir` is given, the parsed arrays and the shard
    manifest are written there on the first run; later runs whose manifest matches
    load the arrays back (seconds) instead of re-parsing (minutes).

    Safe by construction: the cache is used ONLY when its stored manifest exactly
    matches the current shards, and ANY problem — missing files, manifest mismatch,
    a read error — silently falls through to a fresh parse. A stale or corrupt cache
    can never feed the trainer the wrong data; the worst case is re-parsing."""
    import json
    import os

    import numpy as np

    manifest = _shard_manifest(shard_paths)
    own_path = opp_path = tgt_path = man_path = None
    if cache_dir is not None:
        own_path = os.path.join(cache_dir, "own.npy")
        opp_path = os.path.join(cache_dir, "opp.npy")
        tgt_path = os.path.join(cache_dir, "targets.npy")
        man_path = os.path.join(cache_dir, "manifest.json")
        # manifest.json is written LAST as a commit marker, so its presence with a
        # matching payload means the three .npy files are complete and current.
        if os.path.exists(man_path):
            try:
                with open(man_path, "r", encoding="utf-8") as handle:
                    cached = [tuple(entry) for entry in json.load(handle)]
                if cached == manifest:
                    own = np.load(own_path, allow_pickle=False)
                    opp = np.load(opp_path, allow_pickle=False)
                    targets = np.load(tgt_path, allow_pickle=False)
                    print(f"loaded parsed corpus from cache {cache_dir}", flush=True)
                    return own, opp, targets
            except Exception as error:  # noqa: BLE001 - cache is best-effort
                print(f"cache miss ({error}); reparsing", flush=True)

    own, opp, targets = _load_padded_shards(shard_paths)

    if cache_dir is not None:
        # Write each artifact to a private temp path and os.replace it into place:
        # replace is atomic on one filesystem, so a reader (or a sibling sweep
        # container writing the same deterministic bytes) never sees a torn file.
        def atomic_save(path, arr):
            tmp = f"{path}.tmp.{os.getpid()}"
            with open(tmp, "wb") as handle:
                np.save(handle, arr)
            os.replace(tmp, path)

        try:
            os.makedirs(cache_dir, exist_ok=True)
            atomic_save(own_path, own)
            atomic_save(opp_path, opp)
            atomic_save(tgt_path, targets)
            tmp_man = f"{man_path}.tmp.{os.getpid()}"
            with open(tmp_man, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            os.replace(tmp_man, man_path)  # commit marker, replaced last
            print(f"wrote parsed corpus cache to {cache_dir}", flush=True)
        except Exception as error:  # noqa: BLE001 - caching is an optimization only
            print(f"could not write corpus cache ({error}); continuing", flush=True)

    return own, opp, targets


def train(
    data_path: str,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    wdl_target: bool = False,
):
    # Parse the TSV straight into preallocated padded matrices — over-MAX_FEATURES
    # rows (a malformed FEN parsing into a >32-piece board) are dropped in place.
    own_np, opp_np, target_np = _load_padded(data_path)
    return train_arrays(
        own_np, opp_np, target_np, hidden, epochs, batch_size, lr, device, wdl_target
    )


def train_arrays(
    own_np,
    opp_np,
    target_np,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    wdl_target: bool = False,
):
    import torch
    from torch import nn

    if own_np.shape[0] == 0:
        raise SystemExit("no training samples")

    own_t = torch.from_numpy(own_np).to(device)    # [N, 32] int32
    opp_t = torch.from_numpy(opp_np).to(device)
    target = torch.from_numpy(target_np).to(device)
    # In WDL mode the target already IS a win-probability (0.0/0.5/1.0 game
    # outcome) and is used directly; in centipawn mode it is squashed through the
    # WDL sigmoid. See target_win_prob for why the two paths cannot be merged.
    target_wp = torch.clamp(target, 0.0, 1.0) if wdl_target else torch.sigmoid(target / WDL_SCALE)

    class Nnue(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            # 769 rows: 0..767 are real features, 768 is a frozen zero pad row.
            # weight[feature] is the hidden-vector added to the accumulator, exactly
            # the RFNN feature_weights row-major (feature*hidden + i) layout. The
            # padding row is dropped on export, so the block stays byte-identical.
            #
            # EmbeddingBag(mode="sum") fuses the per-bag gather-and-sum: it returns
            # the [B, hidden] accumulator directly instead of materializing the
            # [B, MAX_FEATURES, hidden] intermediate that Embedding + .sum(dim=1)
            # would. Mathematically identical (same rows summed, padding_idx rows
            # contribute zero and take no gradient), but far cheaper in compute and
            # activation memory — the dominant cost for a net this shallow.
            self.transformer = nn.EmbeddingBag(
                INPUT_DIMENSION + 1, hidden, mode="sum", padding_idx=PAD_INDEX
            )
            self.feature_bias = nn.Parameter(torch.zeros(hidden))
            self.output = nn.Linear(2 * hidden, 1)
            nn.init.uniform_(self.transformer.weight[:INPUT_DIMENSION], -0.1, 0.1)
            nn.init.uniform_(self.output.weight, -0.1, 0.1)
            nn.init.zeros_(self.output.bias)

        def forward(self, own_rows, opp_rows):
            # own_rows/opp_rows: [B, MAX_FEATURES] int; EmbeddingBag sums each bag's
            # feature rows in one fused gather -> [B, hidden] (padding rows add zero).
            # Both perspectives share the same table, so they go through a single
            # call over the stacked [2B, MAX_FEATURES] batch (one kernel launch) and
            # are split back out — identical results, half the embedding launches.
            batch = own_rows.shape[0]
            stacked = self.transformer(torch.cat([own_rows, opp_rows], dim=0))
            a_own = torch.clamp(stacked[:batch] + self.feature_bias, 0.0, ACTIVATION_CLIP)
            a_opp = torch.clamp(stacked[batch:] + self.feature_bias, 0.0, ACTIVATION_CLIP)
            features = torch.cat([a_own, a_opp], dim=1)
            # pred is centipawns (inference divides the integer output by 64).
            return self.output(features).squeeze(1) / OUTPUT_SCALE

    count = own_t.shape[0]
    all_idx = torch.arange(count, device=device)
    is_val = (all_idx % VAL_EVERY) == 0
    train_idx = all_idx[~is_val]
    val_idx = all_idx[is_val]

    model = Nnue(hidden).to(device)
    # On CUDA, run Adam's whole parameter update in one fused kernel — the
    # optimizer step is a meaningful slice of each batch for a net this shallow.
    # fused=True needs CUDA tensors, so off-GPU (the CLI/CPU path) keeps the
    # standard update, which is also what keeps that path bit-for-bit reproducible.
    on_cuda = str(device).startswith("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=on_cuda)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def loss_on(idx):
        pred_cp = model(own_t[idx], opp_t[idx])
        pred_wp = torch.sigmoid(pred_cp / WDL_SCALE)
        return ((pred_wp - target_wp[idx]) ** 2).mean()

    if on_cuda:
        # Fuse the forward+loss into fewer kernels. torch.compile falls back to
        # eager on any graph break, and a compile-time failure leaves the eager
        # loss_on in place, so this only ever changes speed, never correctness.
        try:
            loss_on = torch.compile(loss_on)
        except Exception as error:  # noqa: BLE001 - compile is an optimization only
            print(f"torch.compile unavailable ({error}); using eager", file=sys.stderr)

    def mean_loss_batched(idx):
        # Evaluate a mean loss over `idx` in minibatches. Forwarding a large index
        # set at once materializes a [len(idx), MAX_FEATURES, hidden] activation
        # (~12 GB for a ~180k validation set at hidden 512), which OOMs the GPU;
        # batching keeps the peak at one minibatch.
        if not idx.numel():
            return float("nan")
        total = 0.0
        for start in range(0, idx.numel(), batch_size):
            chunk = idx[start:start + batch_size]
            total += loss_on(chunk).item() * chunk.numel()
        return total / idx.numel()

    val = float("nan")
    for epoch in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        # Accumulate the epoch's loss on-device and read it back once, after the
        # loop. A per-batch loss.item() forces a GPU->CPU sync every step, which
        # stalls the pipeline and stops the host from queueing the next batch's
        # kernels ahead of time; the running device total avoids that.
        total = torch.zeros((), device=device)
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start:start + batch_size]
            loss = loss_on(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach() * idx.numel()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val = mean_loss_batched(val_idx)
        train_mean = (total / train_idx.numel()).item() if train_idx.numel() else float("nan")
        print(
            f"epoch {epoch + 1}/{epochs}: train_wdl_loss {train_mean:.6f} "
            f"val_wdl_loss {val:.6f} lr {scheduler.get_last_lr()[0]:.2e}",
            file=sys.stderr,
        )

    return model, val


def quantize_and_write(model, hidden: int, out_path: str):
    import torch

    with torch.no_grad():
        w1 = model.transformer.weight.detach().cpu()[:INPUT_DIMENSION]   # [768, hidden]
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
    model, _ = train(
        args.data, args.hidden, args.epochs, args.batch_size, args.lr, device,
        wdl_target=args.wdl_target,
    )
    quantize_and_write(model, args.hidden, args.out)


if __name__ == "__main__":
    main()
