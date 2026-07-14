# NNUE Training + Incremental Accumulator Design

## Goal

Make the NNUE evaluation real and fast: a self-contained bootstrap **trainer**
that produces a loadable network, and an **incremental accumulator** so NNUE
evaluation is efficiently updatable in the search tree rather than rebuilt from
scratch at every node.

## Scope

Follow-up to the NNUE foundation. Two coupled pieces:

- **Bootstrap trainer (`engine-bench`).** Generates training positions by
  seeded random self-play, labels each with the hand-crafted evaluation (the
  teacher), fits a float network by SGD using the same arithmetic structure as
  the quantised inference, and exports a quantised `RFNN` network. Exposed as a
  `train` sub-command and a `workflow_dispatch` workflow.
- **Incremental accumulator (`engine-search`).** The search maintains the NNUE
  accumulator across `make`/`unmake`, applying only the handful of feature
  changes a move causes (and reversing them on unmake), instead of a full
  refresh per evaluation.

Enablers added to `engine-search`: a public `hand_crafted_evaluation`
(teacher signal), `Nnue::from_parameters` (assemble a net from trained
weights), `active_features` (per-position features), and `Nnue::evaluate_with`
(evaluate a prebuilt accumulator).

Out of scope (documented follow-ups): training against stronger targets (deep
search, game outcomes, an external engine) to exceed the hand-crafted
evaluation; and adopting a trained net as the default via an SPRT-gated swap.

## Alternatives considered

1. **External PyTorch trainer.** The eventual production path, but it needs an
   out-of-repo toolchain and dataset. Deferred.
2. **Refresh-only NNUE (no incremental).** Simple but O(pieces) per node —
   leaves the main NNUE speed win on the table.
3. **In-repo Rust SGD trainer + incremental accumulator (chosen).**
   Dependency-free, deterministic, testable, and delivers both a real network
   and the efficient-update path the architecture is named for.

## Architecture

- **Trainer.** A float mirror of the quantised net (feature transformer →
  clipped ReLU → output) trained by per-sample SGD, then rounded to `i16`/`i32`
  and assembled via `Nnue::from_parameters`. Data comes from seeded random
  legal playouts so it is reproducible.
- **Incremental accumulator.** `nnue_changed_squares` returns the ≤4 squares a
  move touches (from/to, plus castling rook squares or the en-passant captured
  square). `nnue_make` snapshots those squares, makes the move, applies the
  add/remove feature deltas to the accumulator, and pushes the delta;
  `nnue_unmake` pops and reverses it. The accumulator is refreshed once at the
  root of each search (and by each Lazy SMP helper). Null moves change no
  pieces, so they need no update. A `debug_assert` in `evaluate` checks the
  maintained accumulator equals a full refresh at every node.

## Safety rules

- With no network loaded, the search path and evaluation are unchanged;
  `nnue_make`/`nnue_unmake` reduce to plain `make`/`unmake`.
- Every `make` is balanced by an `unmake`, keeping the delta stack balanced.
- The `debug_assert` guarantees any incremental/refresh divergence fails tests.
- The trainer is deterministic given its seed; all randomness is seeded.
- Trained weights are clamped into `i16`/`i32` range on export.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including: the trainer tests
(labelled-sample generation; loss reduction beating a zero-predictor; export
round-trip) and the incremental-search test that drives castling, en passant,
and promotion under the accumulator debug-assert. The tactical suite and
fixed-opponent gauntlet must not regress (default evaluation is unchanged). A
trained network is adopted as the default only behind an external Stockfish
SPRT.
