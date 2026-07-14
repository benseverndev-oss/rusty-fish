# Manifest-backed Modal NNUE experiments

This is the v1 NNUE capacity ladder: a reproducible corpus is built first,
labelled against an explicit Stockfish configuration, then the same manifest is
used to train 128, 256, and 512-wide `EmbeddingBag(..., mode="sum")` networks.
Each stage is immutable and stored in the Modal Volume as
`runs/<run-id>/<stage>-<sha256>`. Repeating the identical input reuses its
artifact; a conflicting result at that address is rejected.

## Prerequisites

- A Modal account and authenticated CLI (`pip install modal`, then `modal token new`).
- A Stockfish configuration calibrated against Stockfish **15.1-4**. The Modal
  image installs that pinned package at `/usr/games/stockfish`, verifies its
  SHA-256 against `binary_sha256` in the supplied config, then rewrites only the
  config's binary path for the remote container. This prevents a caller-local
  path from silently selecting a different engine remotely.
- Run from the repository root so the Modal images can build `engine-bench`.

## Run

```bash
modal run modal/app.py --run-id smoke-v1 --smoke --schema v1 --widths 128,256,512 \
  --stockfish-config stockfish-config.tsv --seed 1
```

`--smoke` creates a 1,000-position corpus; omit it for the 1,000,000-position
corpus. The manifest and its split shards are passed explicitly between every
stage—there is no module-level run state.

Every width receives the supplied deterministic seed (default `1`) before model
initialization and minibatch permutation, and the report records it. For each
width, the GPU stage writes a `report.json` artifact containing the
train/validation/test WDL losses, checksum, schema, input dimension, epochs,
learning rate, and maximum sealed-test prediction delta between float and
quantized networks in centipawns. A candidate screens only after
it beats the 128-wide control by 2% validation and 1% test loss and stays within
the quantization bound. The screen is 12 deterministic shards of 16 openings
(384 games) with `gate-file ... 100`, and promotion additionally needs at least
192 score points (`W + 0.5D`); the entrypoint prints the resulting promotion
decision.

## Local checks

```bash
python -m pytest modal -q
```

The Python test verifies that a mixed schema (or out-of-range feature index) is
rejected before GPU training. Modal execution requires valid Modal credentials,
the configured Stockfish binary, and an available GPU; it is intentionally not
part of a local unit-test run.
