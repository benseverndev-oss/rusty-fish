# Modal NNUE training + gating

Runs the slow, parallelizable parts of the NNUE loop on [Modal](https://modal.com):
labeling and the SPRT gate fan out across many containers, and training runs on a
GPU. This escapes the single-machine / CI-timeout limits and — most importantly —
lets the SPRT gate play enough games to reach a **decisive** verdict, which one
machine cannot.

```
                 ┌── gen-data (seed 1) ─┐
 label (parallel)├── gen-data (seed 2) ─┤→ samples.tsv ─→ train_net (GPU) ─→ net.rfnn
                 └── gen-data (seed N) ─┘                                        │
                 ┌── gate-file (shard 1) ─┐                                      │
 gate  (parallel)├── gate-file (shard 2) ─┤←── openings ─── net.rfnn ◄──────────┘
                 └── gate-file (shard M) ─┘→ Σ W/D/L ─→ sprt verdict
```

## What runs where

- **Rust engine** (`engine-bench`) does labeling, opening generation, and game
  play through four shardable sub-commands added for this:
  - `gen-data <plies> <label_depth> <seed>` → labelled samples TSV.
  - `gen-openings <count> <plies> <seed>` → random opening FENs.
  - `gate-file <net> <depth> <openings_file>` → `W<TAB>D<TAB>L`.
  - `sprt <W> <D> <L>` → SPRT verdict.
- **`train_nnue.py`** (PyTorch) trains a `768→hidden→1` perspective network with
  the win-probability (WDL) loss and exports the engine's `RFNN` format. Its
  forward pass mirrors `engine-search/src/nnue.rs` exactly.
- **`app.py`** is the Modal orchestration.

## Prerequisites

- A Modal account: `pip install modal` then `modal token new`.
- Run from the repository root so the image build can copy the source.

## Run

```bash
modal run modal/app.py
# scale it up:
modal run modal/app.py --label-shards 64 --hidden 256 --epochs 60 \
                       --gate-openings 4096 --gate-depth 6
```

The entrypoint prints the sample count, the trained network size, and the final
`W/D/L` + SPRT decision. To keep the trained network, add a
[Modal Volume](https://modal.com/docs/guide/volumes) and write `net.rfnn` to it
(the scaffold passes bytes in memory, which is fine for modest sizes).

## Validate the exported network against the Rust engine

The PyTorch export is quantised to the same integer format the engine reads, so
you can gate a downloaded `net.rfnn` locally without Modal:

```bash
cargo run --release -p engine-bench -- nnue-sprt net.rfnn 5
# or play with it directly:
#   position startpos / go ... after:  setoption name EvalFile value net.rfnn
```

If `nnue-sprt` loads and plays, the format is compatible. Adopting a network as
the engine default is a separate step, gated on the network **passing** this SPRT
against the hand-crafted baseline.

## Notes

- Written against the Modal Python API; pin the `modal` version you install and
  adjust image/`gpu=` details to taste (`A10G` is a small, cheap GPU — plenty for
  this tiny network).
- Determinism: the engines are deterministic, so gate diversity comes from the
  many generated openings, not from randomness during play.
- Cost scales with games and epochs; start small and scale once a run looks sane.
