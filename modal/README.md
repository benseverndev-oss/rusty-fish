# Manifest-backed Modal NNUE experiments

This pipeline first runs the v1 NNUE capacity ladder, then trains eligible
64-bucket HalfKA v2 candidates from the same reproducible corpus. Positions are
labelled against an explicit Stockfish configuration, and the v1 ladder trains
128, 256, and 512-wide `EmbeddingBag(..., mode="sum")` networks.
Each stage is immutable and stored in the Modal Volume as
`runs/<run-id>/<stage>-<sha256>`. Repeating the identical input reuses its
artifact; a conflicting result at that address is rejected.

## Prerequisites

- A Modal account and authenticated CLI (`pip install modal`, then `modal token new`).
- A Stockfish configuration calibrated against the exact Stockfish **18**
  Ubuntu x86-64 release artifact. The Modal image downloads
  `stockfish-ubuntu-x86-64.tar`, verifies the pinned archive SHA-256
  `5c6f38b02a4da5f3ffe763f27da6c3e743eebefd92b50cb3661623b96696adff`, and
  verifies the extracted executable against `binary_sha256` in the supplied
  config before rewriting only the caller-local binary path for the remote
  container.
- Run from the repository root so the Modal images can build `engine-bench`.

## Run

First create the run-specific config with the direct remote function. It builds
the corpus and calibrates the deterministic 1,000-position training sample in
one Modal execution against the pinned Linux Stockfish 18 binary, then writes
the returned config to the requested local path:

```bash
modal run --write-result stockfish-config.tsv modal/app.py::calibrate_run --run-id stockfish18-v1
```

Then use that exact config for the matching run. Do not create it with a local
Windows Stockfish binary: label jobs verify the executable hash inside Modal.

For a production v1 campaign that survives a Windows client disconnect, start
the durable Modal coordinator directly and inspect its persisted state later:

```bash
modal run --detach modal/app.py::run_v1_pipeline --run-id full-v1
modal run modal/app.py::pipeline_status --run-id full-v1
```

The coordinator owns corpus/calibration, bounded parallel labels, GPU training,
eligibility, screens, and full gates. It writes immutable status artifacts at
each stage; restarting the same run reuses completed content-addressed stages.

```bash
modal run modal/app.py --run-id smoke-v1 --smoke --schema v1 --widths 128,256,512 \
  --stockfish-config stockfish-config.tsv --seed 1
```

After a capacity selection report names `selected_width` (`128`, `256`, or
`512`), run HalfKA with its deterministic ladder:

```bash
modal run modal/app.py --run-id smoke-halfka --smoke --schema halfka-v2-64 \
  --capacity-selection runs/smoke-v1/capacity-selection.json \
  --stockfish-config stockfish-config.tsv --seed 1
```

The mapping is `128 -> [128, 256]`, `256 -> [256, 512]`, and `512 -> [512]`.
The first HalfKA candidate that fails promotion stops the ladder. Exported v2
networks carry RFNN version 2, schema tag 1, bucket count 64, and input
dimension 40,960; a Rust one-opening gate confirms the file loads before the
384-game screen.

`--smoke` creates a 1,000-position corpus; omit it for the 1,000,000-position
corpus. The manifest and its split shards are passed explicitly between every
stage—there is no module-level run state.

## Labeling at production scale

Each train/validation/test split is deterministically cut into contiguous
1,000-row batches. Modal runs those CPU label batches in deterministic waves of
at most 80 workers (leaving headroom below the 100-container workspace limit);
each batch
checks the pinned Stockfish config and writes an immutable content-addressed
artifact before the coordinator aggregates them. Re-running the same run/schema
reuses completed batch artifacts. Aggregation sorts by split and batch index,
then emits one schema header followed by the original split row order.

Individual label workers are capped at 30 minutes. The coordinator only
dispatches, waits for, and validates batches, so a large corpus is not searched
inside one 60-minute Stockfish process. A failed batch leaves its completed
siblings reusable; rerun the identical command to resume from those artifacts.

Every width receives the supplied deterministic seed (default `1`) before model
initialization and minibatch permutation, and the report records it. For each
width, the GPU stage writes a `report.json` artifact containing the
train/validation/test WDL losses, checksum, schema, input dimension, epochs,
learning rate, and maximum sealed-test prediction delta between float and
quantized networks in centipawns. A candidate screens only after
it beats the 128-wide control by 2% validation and 1% test loss and stays within
the quantization bound. The screen is 12 deterministic shards of 16 openings
(384 games) with `gate-file ... 100`, and promotion additionally needs at least
192 score points (`W + 0.5D`). The screen and full gate each write an immutable,
input-addressed `report.json` with the run ID, network/candidate/control/
manifest/config checksums, W/D/L, Elo and SPRT evidence, and promotion decision.
`AcceptH0` and `Continue` are recorded as non-adoption outcomes.

Only a promoted HalfKA candidate runs the full 2,304-game gate: 12 shards of
96 openings, depth 4, and `gate-file ... 100`. The GitHub Actions workflow is
dispatch-only: provide the immutable network URL, its SHA-256, and the corpus
manifest SHA-256. It verifies the network before playing the same 12-shard gate
and prints aggregate W/D/L plus the engine SPRT (Elo, LLR, and decision).
`AcceptH0` is a failed branch; `AcceptH1` requires a separate adoption design.

## Local checks

```bash
python -m pytest modal -q
```

The Python test verifies that a mixed schema (or out-of-range feature index) is
rejected before GPU training and that the HalfKA RFNN v2 header matches Rust's
contract. Modal execution requires valid Modal credentials,
the configured Stockfish binary, and an available GPU; it is intentionally not
part of a local unit-test run.
