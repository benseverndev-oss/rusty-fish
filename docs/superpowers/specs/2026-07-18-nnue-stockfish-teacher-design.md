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
fan-out sizing) and consistent label quality. The score is taken
**side-to-move-relative**, which is exactly what Stockfish's `score cp` reports and
exactly the perspective the net's target uses, so no sign flip is needed. `score
mate N` is clamped to a large ±cp (`MATE_CP = 30000`, i.e. ±30000) so the training sigmoid saturates
near 0/1 rather than seeing an undefined centipawn value.

**Reproducibility requires two things the labeller must do explicitly:** the
persistent Stockfish process is single-threaded (`setoption Threads 1`, already in
the engine's UCI handshake — fixed node counts are only deterministic
single-threaded), and a `ucinewgame` + `isready`/`readyok` is sent **before each
position** to clear the transposition table. Without the TT clear, a warm table
from position N−1 makes position N's fixed-node result depend on scan order,
breaking reproducibility.

Pure SF-eval only (no blend with the game outcome). This reuses the cp trainer
mode unchanged and cleanly isolates "does a better teacher help?" against the WDL
run. A λ·outcome + (1−λ)·SF blend is the documented next lever if pure SF-eval
wins but does not beat the hand-crafted eval.

## Data pipeline: two new `engine-bench` commands

The labelling splits into position generation (search-free, cheap) and Stockfish
scoring (the expensive new step), piped in one container so nothing intermediate
is materialised.

### `gen-eval-positions`

Reuses the same sampling as `gen-wdl-data` (the `WdlBuilder` PGN visitor, the
rated/standard/≥2200 filter, `--shard i/n`, `--per-game N`, the whole-game board
tracking) but emits, per sampled position, `fen<TAB>own_csv<TAB>opp_csv` — the
position's FEN plus the already-computed `active_features(board, side_to_move)` /
`active_features(board, opponent)` — and **no outcome label**. `--per-game 4`
across the six pinned months (`assets/nnue/wdl-corpus.toml`) yields on the order of
~3M positions.

**This is not a drop-in "same visitor minus the label."** The WDL visitor's
per-ply snapshot (`WdlGame.positions`, a `Vec<(own, opp, stm, ply)>`) and
`WdlSample` store no FEN and clone no board, so a **FEN field must be threaded
through**: capture `game.board.to_fen()` at the same pre-move point where `own`/`opp`
features are captured (in the visitor's `san`, where `stm = game.board.side_to_move`
— so the FEN's side-to-move equals the `stm` used for the features and the SF
score's perspective), add it to the snapshot tuple and the sample struct, and emit
it. The plan may add the field to the shared structs or use a parallel
FEN-capturing builder; either way the same **result-tag filter is kept
deliberately** (`end_game` early-returns on a missing/odd `Result`), because that
is what makes the "same sampled positions as `gen-wdl-data`" verification test
hold — eval positions do not use the outcome, but they use the same game set.

`Board::to_fen()` already exists (`engine-core/src/lib.rs`) and is already how the
external SF gauntlet feeds Stockfish (`position fen {board.to_fen()}`), so board→FEN
is zero new work; `gen-eval-positions` matches that existing convention rather than
emitting a move list. A `to_fen`/`from_fen` round-trip test (`Board::from_fen(&b.to_fen())`,
comparing the re-emitted FEN string) can characterise it if not already covered.

### `label-sf <positions_or_-> <nodes>`

Reads `fen<TAB>own_csv<TAB>opp_csv` lines from a path or `-` (stdin), drives **one
persistent Stockfish process** (spawned once, reused for every position in the
shard — spawning per position would dominate cost), and for each line:

- `ucinewgame` + `isready`/`readyok` (clear the TT — see Reproducibility above);
- `position fen <fen>` then `go nodes <nodes>`;
- read UCI output until `bestmove`, keeping the **last** `info … score cp X` (or
  `score mate M`) seen before it;
- convert to a centipawn target (`mate` → clamped ±cp);
- emit `<cp><TAB>own_csv<TAB>opp_csv` — the exact format `train_nnue.py` reads.

**What is reused vs new:** `engine-bench` already has a `UciProcess` helper (from
the external SF18 gauntlet) that spawns Stockfish, runs the `uci`/`uciok` +
`setoption Threads 1` + `setoption Hash 16` + `isready` handshake, and does
line-buffered `send`/`wait_for` IO with a timeout and `Drop` cleanup — that
spawn/handshake/IO scaffolding is reused. But the existing `best_move` path issues
`go movetime`, reads only the `bestmove` token, and **discards every `info … score
cp` line** — there is no score parsing, no `go nodes`, and no `score mate` handling
anywhere in `engine-bench` today. So `label-sf` adds a genuinely new
`score_position(fen, nodes) -> cp` primitive (an `impl` on the existing
`UciProcess`, taking the FEN string directly rather than a `&Board`): send
`ucinewgame`/`position fen`/`go nodes`, parse the last score line, apply the
mate-clamp. Malformed or illegal FENs (should not occur — the sampler only emits
legal positions) are skipped with a count to stderr rather than aborting the shard.

## Modal `train_sf` pipeline

Mirrors `train_wdl`, swapping the labeller and the trainer's teacher flag:

- **Stockfish in the image:** Stockfish is added to `rust_image` via
  `apt_install("stockfish")` — a build-time image layer at `/usr/games/stockfish`.
  Debian ships a **baseline x86-64** build, so there is no AVX/BMI2 `SIGILL` risk on
  Modal's CPUs and no per-microarch tar to extract, which is why apt is the
  pragmatic choice for this first attempt. The apt version (pinned-ish by the base
  image digest) is a strong-enough teacher at 100k nodes; a specific
  SHA-pinned release build is a documented later refinement (out of scope) if exact
  version reproducibility is needed. The binary path is passed to `label-sf`.
- **`label_sf_shard(name, i, n, per_game, nodes)`** runs the one-container pipe
  under `bash -c 'set -euo pipefail'`:
  `zstdcat /vol/export-<name>.pgn.zst | engine-bench gen-eval-positions - --shard i/n --per-game <p> | engine-bench label-sf - <nodes> > /vol/sf/samples-<name>-<i>.tsv`,
  commits, returns the line count. Fanned across `(month × shard)` pairs (~64–128
  containers so each labels a tractable slice at 100k nodes/pos).
- **Shard separation (two-sided):** the SF labels are the single most expensive
  artifact in the pipeline, and the WDL trainer's cleanup globs
  `/vol/samples-*.tsv` and deletes anything outside its own run. Because `samples-`
  is a prefix of any `samples-sf-*` name, a same-directory prefix rename is **not**
  disjoint (the WDL glob would still match and delete SF shards, and vice-versa).
  So SF shards live in a **separate `/vol/sf/` subdirectory**: `train_sf_run` globs
  and cleans only `/vol/sf/samples-*-*.tsv`, and the existing `train_wdl_run`'s
  top-level `glob("/vol/samples-*.tsv")` does not recurse into `/vol/sf/`, so the
  two families are mutually invisible and neither run wipes the other's cache.
- **`train_sf` entrypoint** fans `prepare_export` (reusing the same cached exports
  and the atomic verified download), then `label_sf_shard`, then a `train_sf_run`
  training with **`wdl_target=False`** (cp mode) on this run's `/vol/sf/samples-*-*.tsv`
  shards (same explicit-shard-set selection `train_wdl_run` uses, scoped to
  `/vol/sf/`), then the existing powered movetime gate via `nnue_gate_run` /
  `gate_net`.

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
  - `Board::to_fen()` (already present) round-trips: `Board::from_fen(&b.to_fen())`
    re-emits the same FEN string on a few positions (a characterisation test; only
    needed if not already covered).
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
