# NNUE Stockfish-Eval Teacher Design

## Goal

Test whether a cleaner per-position teacher — Stockfish's own evaluation — beats
the game-outcome (WDL) teacher, which got the outcome-trained NNUE to −37.4 Elo
(SPRT AcceptH0) against the tuned hand-crafted eval. Raw game outcomes are a noisy
per-position label (many positions in a won game were actually equal or losing);
a fixed-budget Stockfish search gives a low-noise value for the *specific*
position. This slice swaps only the label — same architecture, same trainer, same
gate — so the gate result isolates the teacher change.

Nothing about the shipped engine changes unless the powered gate returns
**AcceptH1**. NNUE remains opt-in (`EvalFile` / `nnue-sprt`); the tuned
hand-crafted eval (+10.5 Elo, shipped) stays the default and the bar to beat.

## The key reuse: the trainer needs no change

The GPU trainer (`modal/train_nnue.py`) already has a **centipawn mode**: with
`wdl_target=False` it fits `sigmoid(pred_cp / 400)` to `sigmoid(target / 400)`.
A Stockfish evaluation *is* a centipawn target, so pure SF-eval training is the
trainer's original (non-WDL) path — the batched padded-tensor forward, cosine LR,
validation-loss report, and RFNN export are all unchanged and already merged. The
opening-book corpus, the `gate_net`/`nnue_gate_run` gate, and the Modal Volume
plumbing all carry over. The only new work is producing the labels.

## Teacher and target

Each sampled position is labelled with Stockfish's centipawn evaluation from a
**fixed 100,000-node search** (`go nodes 100000`, ~depth 12–14 equivalent). Fixed
nodes — not fixed depth — gives deterministic per-position cost (predictable Modal
fan-out sizing) and consistent label quality, and is reproducible. The score is
taken **side-to-move-relative**, which is exactly what Stockfish's `score cp`
reports and exactly the perspective the net's target uses, so no sign flip is
needed. `score mate N` is clamped to a large ±cp (e.g. ±3000) so the training
sigmoid saturates near 0/1 rather than seeing an undefined centipawn value.

Pure SF-eval only (no blend with the game outcome). This reuses the cp trainer
mode unchanged and cleanly isolates "does a better teacher help?" against the WDL
run. A λ·outcome + (1−λ)·SF blend is the documented next lever if pure SF-eval
wins but does not beat the hand-crafted eval.

## Data pipeline: two new `engine-bench` commands

The labelling splits into position generation (search-free, cheap) and Stockfish
scoring (the expensive new step), piped in one container so nothing intermediate
is materialised.

### `gen-eval-positions`

Reuses the existing `gen-wdl-data` sampling (the `WdlBuilder` PGN visitor, the
rated/standard/≥2200 filter, `--shard i/n`, `--per-game N`, the whole-game board
tracking) but emits, per sampled position, `fen<TAB>own_csv<TAB>opp_csv` — the
position's FEN plus the already-computed `active_features(board, side_to_move)` /
`active_features(board, opponent)` — and **no outcome label**. `--per-game 4`
across the six pinned months (`assets/nnue/wdl-corpus.toml`) yields on the order of
~3M positions.

This requires the `engine_core::Board` to emit a FEN string for the current
position (Stockfish is fed `position fen <fen>`). The design assumes a
board→FEN capability exists (the external SF gauntlet already communicates
mid-game positions to Stockfish); the implementation plan will confirm it and add
a `Board::to_fen()` if missing (a self-contained, unit-testable addition —
round-trip `parse_fen(to_fen(b)) == b` on a few positions).

### `label-sf <positions_or_-> <nodes>`

Reads `fen<TAB>own_csv<TAB>opp_csv` lines from a path or `-` (stdin), drives **one
persistent Stockfish process** (spawned once, reused for every position in the
shard — spawning per position would dominate cost), and for each line:

- `position fen <fen>` then `go nodes <nodes>`;
- read UCI output until `bestmove`, keeping the last `info … score cp X` (or
  `score mate M`);
- convert to a centipawn target (`mate` → clamped ±cp);
- emit `<cp><TAB>own_csv<TAB>opp_csv` — the exact format `train_nnue.py` reads.

It reuses the Stockfish UCI-subprocess plumbing already in `engine-bench` (the
external SF18 gauntlet spawns and talks to Stockfish over UCI). Malformed or
illegal FENs (should not occur — the sampler only emits legal positions) are
skipped with a count to stderr rather than aborting the shard.

## Modal `train_sf` pipeline

Mirrors `train_wdl`, swapping the labeller and the trainer's teacher flag:

- **Stockfish in the image:** a pinned Stockfish binary is downloaded and
  SHA-256-verified into `rust_image` (a reproducible pinned release build, not an
  unpinned apt package). The binary path is passed to `label-sf`.
- **`label_sf_shard(name, i, n, per_game, nodes)`** runs the one-container pipe
  under `bash -c 'set -euo pipefail'`:
  `zstdcat /vol/export-<name>.pgn.zst | engine-bench gen-eval-positions - --shard i/n --per-game <p> | engine-bench label-sf - <nodes> > /vol/samples-sf-<name>-<i>.tsv`,
  commits, returns the line count. Fanned across `(month × shard)` pairs (~64–128
  containers so each labels a tractable slice at 100k nodes/pos).
- **`train_sf` entrypoint** fans `prepare_export` (reusing the same cached exports
  and the atomic verified download), then `label_sf_shard`, then a training run
  with **`wdl_target=False`** (cp mode) on the `samples-sf-*-*.tsv` shards, then
  the existing powered movetime gate via `nnue_gate_run` / the `gate_net` path.
  The trainer selects exactly this run's shards (the same explicit-shard-set
  mechanism `train_wdl_run` uses, generalised to a shard-name prefix so
  `samples-sf-*` and `samples-*` don't collide on the shared Volume).

The gate (`gate_net` / `nnue_gate_run`) is unchanged — the 30-min `gate_shard`
timeout and movetime binding already handle a hidden-512 net.

## Architecture

Unchanged from the WDL slice: `768 → hidden → 1` side-to-move-relative perspective
net, trained at hidden 512, exported in the byte-identical RFNN format. HalfKA is
out of scope.

## Adoption

If the powered SPRT gate (4096 games vs the tuned hand-crafted baseline) returns
**AcceptH1**, a follow-up commits the `.rfnn` as an asset, wires it as the default
evaluation, and re-gates under CI. Otherwise NNUE stays opt-in and we record:

- **Beats −37 but not the eval:** the cleaner teacher helped; next is the λ-blend
  or more positions / higher nodes.
- **Flat vs −37:** SF-eval at this scale/architecture is not the lever; reconsider
  (HalfKA, or far more data).

## Verification

- **Rust (CI, `cargo test --workspace`):**
  - `gen-eval-positions` over a tiny fixture PGN emits well-formed
    `fen<TAB>own<TAB>opp` lines: FENs parse back to legal positions, feature CSVs
    are valid indices `< 768`, and the sampled positions match the `gen-wdl-data`
    sampler (same count/positions for the same config, minus the label).
  - `Board::to_fen()` round-trips (`parse_fen(to_fen(b)) == b`) if it is added.
  - `label-sf` parsing is unit-tested against **captured UCI transcripts** (fixed
    `info … score cp` / `score mate` lines) so the score-extraction and
    mate-clamp logic are tested without spawning Stockfish; a single optional
    end-to-end test may drive a real Stockfish if one is available on the runner,
    but the core logic is transcript-tested.
- **Python:** no trainer change, so the existing `target_win_prob` / cp-mode
  behaviour already covers it; the batched trainer's parity check stands.
- **End-to-end:** validated by the Modal run — a short config first (one month,
  tiny per-game, low nodes, small gate) to confirm the SF binary, the
  gen→label pipe, the cp-mode training, and the gate all work, then the real run
  (~3M positions, 100k nodes, hidden 512, powered gate).
- Cargo is never run locally; all Rust validation is GitHub Actions.

## Out of scope

- The λ-blend (outcome + SF) teacher — the next lever if pure SF-eval wins but
  does not beat the hand-crafted eval.
- HalfKA / king-buckets.
- Any change to the trainer, the RFNN format, or the inference path.
- Multi-PV or WDL-model Stockfish output (single side-to-move cp only).
- Tuning the node budget or position count beyond this first attempt.
