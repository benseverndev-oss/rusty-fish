# NNUE Trained on Lichess Game Outcomes Design

## Goal

Train an NNUE that beats the (now SPSA-tuned) hand-crafted evaluation by learning
from a teacher stronger than itself: the eventual game result of real 2200+-rated
Lichess games. The previously rejected net (−373 Elo) failed because its teacher
was the engine's own hand-crafted evaluation (positions labelled with a depth-N
search whose leaves are that evaluation) — a student cannot beat its teacher, so
mimicking a weak eval imperfectly lands below it. Game outcomes are a genuine
strength signal that can exceed the hand-crafted eval, so they are the essential
change; the capacity bump (hidden 64 → 256) is secondary.

Nothing about the shipped engine changes unless a powered gate shows the net wins.

## Teacher and target

Each training sample's target is the game's final result **relative to the side
to move at that position**: 1.0 for a win, 0.5 for a draw, 0.0 for a loss. This
is the Leela-style outcome signal. It is not a centipawn score.

The GPU trainer (`modal/train_nnue.py`) today squashes a centipawn target through
`sigmoid(target / 400)` to produce a win-probability, then fits the net's
`sigmoid(pred_cp / 400)` to it. For outcome training the target already *is* a
win-probability, so encoding it as centipawns would be degenerate (`logit(1)` is
infinite). The trainer therefore gains a `--wdl-target` mode: `target_wp =
clamp(target, 0, 1)` used directly, skipping the centipawn sigmoid. The loss
(squared error on win-probabilities), the perspective architecture, and the RFNN
quantisation are otherwise unchanged, so the exported net loads in the Rust
engine exactly as today. The flag is threaded as a `train(..., wdl_target=True)`
**function parameter**, not only an argparse flag, because Modal's `train_net`
calls the Python `train()` function directly rather than the CLI. It must be
passed on WDL data: if it were omitted, `sigmoid(1.0/400) ≈ 0.5006` would collapse
every win/draw/loss target to ~0.5 and train the net on noise silently — so the
data producer's outcome targets and the trainer's `wdl_target` mode are one
coupled decision.

## Data pipeline: `gen-wdl-data`

A new `engine-bench gen-wdl-data` command produces outcome-labelled training
samples from PGN, emitting the **same TSV format** the trainer already reads
(`target<TAB>own_feature_csv<TAB>opp_feature_csv`), with `target` set to the
game outcome and the feature CSVs from `engine_search::nnue::active_features`
(which is `pub` and already used by `engine-bench`).

This is a **new PGN visitor in engine-bench**, not a reuse of `book-tool`.
`book-tool` is a binary-only crate: its `pgn-reader` visitor and rated/standard/
≥2200 filter are private items in `book-tool/src/main.rs`, `engine-bench` does not
depend on `pgn-reader`, and — decisively — `book-tool`'s visitor only advances the
`engine_core::Board` for the first sixteen plies, whereas the WDL sampler needs
the real board state deep into the middlegame. So the plan adds `pgn-reader` as an
`engine-bench` dependency and writes a purpose-built visitor that (a) applies the
same tag filter (standard, rated, both ≥ 2200) as book-tool but re-implemented
here, and (b) keeps the `engine_core::Board` current for the **whole game** so
positions past ply sixteen can be sampled. It reads a PGN from a path or `-`
(stdin), so the pinned export streams on Modal as the book refresh does
(`zstdcat export.zst | gen-wdl-data -`).

For Modal fan-out over the single monolithic export, `gen-wdl-data` takes a
`--shard i/n` option: container `i` of `n` emits samples only for games whose
zero-based index satisfies `index % n == i`, where the index counts games in
**stream order before the rating filter** (so shard assignment is
filter-independent and the partition is trivially disjoint; each shard then
applies the filter to its assigned games). Each shard still decompresses and
scans the whole stream (cheap relative to the search-free feature extraction) but
labels a disjoint set of games, so the shards concatenate to the full dataset
with no overlap.

Selection and sampling, chosen to give the net diverse, decisive-signal
positions rather than correlated or trivial ones:

- **Games:** the same filter the book uses — standard, rated, both players ≥ 2200
  — for teacher quality, and only decisive-or-drawn games with a real result
  (`1-0`/`0-1`/`1/2-1/2`).
- **Positions per game:** skip the opening (roughly the first 8 plies, which the
  book already covers and which carry little per-position outcome signal) and the
  final few plies (where the result is mechanical), then subsample the remaining
  middlegame positions to a small fixed number per game (about 10–15) so adjacent,
  highly-correlated positions do not dominate. Sampling is deterministic from a
  seed so the dataset is reproducible.
- **Target:** for each sampled position, the game result mapped to the side to
  move (White-to-move in a `1-0` game → 1.0; Black-to-move in that game → 0.0;
  any draw → 0.5).
- **Features:** `active_features(board, side_to_move)` for the own CSV and
  `active_features(board, opponent)` for the opp CSV — matching how the trainer
  and inference build the two perspective accumulators.

The pinned source is the same `2017-01` export already in
`assets/opening-book/manifest.toml`; its ~132k accepted games at ~10–15 sampled
positions each yield on the order of one to two million samples — enough for a
first 256-wide net (real NNUE uses far more, so this is a first attempt, not a
ceiling).

## Architecture

The current `768 → hidden → 1` side-to-move-relative perspective network is kept.
Inputs are the 768 piece-type × square × colour-relative features. The campaign
trains at **hidden = 256** (up from the rejected net's 64). **HalfKA king-bucketed
features are explicitly out of scope** — they are a much larger input and feature
change that belongs in a later slice, only after outcome-trained NNUE is shown to
work at this architecture.

## Modal `train_wdl` pipeline

Mirrors the existing NNUE train→gate loop in `modal/app.py`, swapping only the
labeller:

- **Label** from Lichess instead of self-play: download and SHA-verify the pinned
  export (as the book refresh does), then fan `gen-wdl-data --shard i/n` across
  `n` containers. Because the concatenated dataset (~1–2M samples ≈ several
  hundred MB) is too large to pass as a Python function argument the way the
  small self-play dataset is, the shards write into a **Modal Volume** (a shared
  file) that the trainer reads, rather than being `"".join`-ed in the launching
  client's memory.
- **Train** one GPU container (`train()` with `wdl_target=True`, hidden 256),
  reading the dataset from the Volume and exporting the RFNN bytes.
- **Gate** the net with the existing powered self-play gate: NNUE candidate vs the
  tuned hand-crafted baseline over thousands of parallel games, summed to an SPRT
  verdict — the same rig that gated the eval tune. Launched and retrieved the same
  way (`infisical run … modal run … modal app logs`), with the verdict emitted
  from a remote function so a detached run stays retrievable.

## Adoption

NNUE is opt-in today (loaded via `EvalFile` / `nnue-sprt`), so this ships nothing
by default. If the gate shows the outcome-trained net beats the tuned hand-crafted
eval under SPRT, a follow-up commits the net as an asset and wires it in as the
default evaluation, re-gated by normal CI. If the result is flat or negative, the
net stays opt-in and we have learned that the 768 / hidden-256 / Lichess-WDL
recipe is not enough — the next levers are a stronger teacher (Stockfish eval
labels) or HalfKA, not more of the same.

## Verification

- A unit test that `gen-wdl-data` over a tiny fixture PGN emits well-formed TSV:
  the right number of samples, targets in `{0.0, 0.5, 1.0}`, feature CSVs that
  parse to valid indices, and side-to-move-relative targets correct for a known
  game (a White-win fixture yields 1.0 for White-to-move positions, 0.0 for
  Black-to-move).
- A unit test that the sampler is deterministic from its seed and respects the
  opening/endgame skips and the per-game cap.
- A unit test that `--shard i/n` partitions games disjointly: the union of all `n`
  shards over a fixture PGN equals the unsharded output, with no game sampled
  twice.
- A trainer check (the standalone `python train_nnue.py` path) that `--wdl-target`
  uses the target as the win-probability directly — a couple of samples with
  targets `0.0`/`1.0` drive the prediction toward those, and the exported RFNN
  round-trips through `engine-bench nnue-sprt`.
- The end-to-end pipeline (label → train → gate) is validated by running it on
  Modal, a short campaign first (few games, small sample) then the real one, the
  same way the eval tune was.
- All in-repo Rust validation runs in GitHub Actions; Cargo is never run locally.

## Out of scope

- HalfKA / king-bucketed features (a later architecture slice).
- Stockfish-eval labels or a hybrid teacher (the fallback if Lichess-WDL is
  insufficient).
- Data augmentation, deduplication beyond the per-game cap, and multi-month
  datasets.
- Incremental / second-order training refinements beyond the existing Adam loop.
- Changing the RFNN format or the inference path.
