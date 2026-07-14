# Modal Training Pipeline Design

## Goal

Run the slow, parallelizable parts of the NNUE loop — position labeling and the
SPRT gate — across many cloud containers, and training on a GPU, so the loop is
fast and the gate can play enough games to reach a **decisive** verdict (which a
single machine cannot).

## Scope

- Four shardable `engine-bench` sub-commands so an external orchestrator can
  parallelise the pipeline:
  - `gen-data <plies> <label_depth> <seed>` — labelled samples as TSV.
  - `gen-openings <count> <plies> <seed>` — random opening FENs (via a new
    `random_opening_fens`).
  - `gate-file <net> <depth> <openings_file>` — NNUE-candidate vs baseline over
    a file of openings, emitting `W<TAB>D<TAB>L`.
  - `sprt <W> <D> <L>` — SPRT verdict from aggregated counts.
- A `modal/` directory (not a workspace crate):
  - `train_nnue.py` — a PyTorch trainer for the `768→hidden→1` perspective
    network with the WDL loss, exporting the engine's `RFNN` format.
  - `app.py` — the Modal orchestration (parallel label → GPU train → parallel
    gate → SPRT).
  - `README.md` — setup, run, and validation instructions.

Out of scope: running Modal here (it needs the user's account/token); adopting a
trained network as the engine default.

## Rationale

Labeling (independent searches) and gating (independent games) are
embarrassingly parallel; training benefits from a GPU. Modal maps these over
containers and provides a GPU, and — because the engines are deterministic — a
large, diverse opening set gives the gate the game volume an SPRT needs to
conclude. The Rust engine stays the single source of truth for move generation,
labeling, and play; Python only orchestrates and trains.

## Architecture

The `train_nnue.py` forward pass mirrors `engine-search/src/nnue.rs` exactly
(summed feature accumulator via `EmbeddingBag`, clipped-ReLU in `[0, 127]`,
integer output divided by `OUTPUT_SCALE`), and quantises to the same `RFNN`
byte layout, so an exported network loads unchanged in the engine. The Modal app
builds `engine-bench` into a Rust image for the labeling/gating/opening
functions and uses a PyTorch GPU image for training; the SPRT is aggregated from
the summed shard counts.

## Safety rules

- The new sub-commands only read/compute; they never change the default engine.
- Gate candidate and baseline search at equal depth.
- The PyTorch export is quantised to the engine's exact format and is validated
  by loading it back through `engine-bench nnue-sprt`.
- Adopting a trained network as default remains gated on a passing SPRT.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including a test that generated
openings are legal and varied. The sub-commands are exercised locally; the
Python is syntax-checked and the `RFNN` byte contract is verified by having the
Rust engine load a Python-written network. Running the Modal app itself requires
the user's Modal account.
