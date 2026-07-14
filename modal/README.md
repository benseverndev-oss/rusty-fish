# Manifest-backed Modal NNUE experiments

This is the v1 NNUE capacity ladder: a reproducible corpus is built first,
labelled against an explicit Stockfish configuration, then the same manifest is
used to train 128, 256, and 512-wide `EmbeddingBag(..., mode="sum")` networks.
Each stage is immutable and stored in the Modal Volume as
`runs/<run-id>/<stage>-<sha256>`. Repeating the identical input reuses its
artifact; a conflicting result at that address is rejected.

## Prerequisites

- A Modal account and authenticated CLI (`pip install modal`, then `modal token new`).
- A Stockfish configuration produced by `engine-bench stockfish-calibrate`; it
  includes the binary hash, node budget, and response deadline used for labels.
- Run from the repository root so the Modal images can build `engine-bench`.

## Run

```bash
modal run modal/app.py --run-id smoke-v1 --smoke --schema v1 --widths 128,256,512 \
  --stockfish-config stockfish-config.tsv
```

`--smoke` creates a 1,000-position corpus; omit it for the 1,000,000-position
corpus. The manifest and its split shards are passed explicitly between every
stage—there is no module-level run state.

For each width, the GPU stage writes a `report.json` artifact containing the
train/validation/test WDL losses, checksum, schema, input dimension, epochs,
learning rate, and maximum quantization error. A candidate screens only after
it beats the 128-wide control by 2% validation and 1% test loss and stays within
the quantization bound. The screen is 12 deterministic shards of 16 openings
(384 games) with `gate-file ... 100`, and promotion additionally needs at least
192 score points (`W + 0.5D`).

## Local checks

```bash
python -m pytest modal -q
```

The Python test verifies that a mixed schema (or out-of-range feature index) is
rejected before GPU training. Modal execution requires valid Modal credentials,
the configured Stockfish binary, and an available GPU; it is intentionally not
part of a local unit-test run.
