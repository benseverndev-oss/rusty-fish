# Rusty Fish — Handoff

A snapshot of where the engine is, what was built, what was learned, and what to
do next. The north star is a **grandmaster-strength** engine.

## TL;DR

`rusty-fish` is a Rust chess-engine workspace with a strong alpha-beta searcher,
**Lazy SMP** multithreading, **SPSA** parameter tuning, and a complete **NNUE**
evaluation stack (features → accumulator → quantised inference → incremental
updates → file format → trainer → SPRT gate) plus a **parallel training/gating
pipeline** on GitHub Actions (and Modal). The hand-crafted evaluation is still
the default; NNUE is opt-in until a trained network **beats it through the SPRT
gate**. Getting a network to pass that gate is the remaining work between here
and real strength — the whole loop to do it now exists.

## Workspace

| Crate | Role |
|-------|------|
| `engine-core` | Board, bitboards, legal move-gen, incremental Zobrist, FEN, perft. |
| `engine-search` | Iterative-deepening PVS search, evaluation, `nnue` module, Syzygy. |
| `engine-uci` | UCI binary. Options: `Hash`, `Threads`, `Move Overhead`, `Max Depth`, `SyzygyPath`/`SyzygyProbeDepth`/`SyzygyProbeLimit`, `EvalFile`. |
| `engine-bench` | Benchmarks, tactical suite, SPRT/match harness, SPSA tuner, NNUE trainer + gate, shardable pipeline sub-commands. |
| `app-desktop` | `egui` desktop GUI. |

## Search (engine-search)

Negamax/PVS with a transposition table (sharded, lock-based, shared across Lazy
SMP threads), aspiration windows, null-move, razoring, reverse-futility, LMR,
LMP, singular extensions, passed-pawn extensions, killers/history/counter-moves,
quiescence, and Syzygy WDL + root probing. The tunable scalars live in
`SearchParams` (aspiration window, razor/RFP base+scale, LMP base+scale,
null-move reduction) and are optimised by the SPSA tuner.

- **Lazy SMP**: `Threads` option (default 1). The primary thread runs the normal
  loop; `Threads-1` helpers deepen over the shared table. `Threads <= 1` is the
  exact old path.
- **Evaluation**: `hand_crafted_evaluation` (tapered material/PST/pawns/king
  safety…) is the default. If an NNUE network is loaded, `evaluate` uses it.

## NNUE (engine-search/src/nnue.rs)

- **Features**: 768 = own/their × 6 kinds × 64 side-relative squares.
- **Accumulator**: `refresh(board)` plus incremental `add/remove_feature`. The
  search maintains it across `make`/`unmake` (`nnue_make`/`nnue_unmake`,
  `nnue_changed_squares`), with a `debug_assert` that the incremental value
  equals a full refresh **at every node** (so any desync fails tests).
- **Inference**: clipped-ReLU `[0,127]`, integer output ÷ 64, clamped to
  `±20000` so it can never look like a mate score.
- **Format**: `RFNN` — magic + `u32` version + `u32` hidden + `W1`/`b1`/`W2`
  (i16 LE) + `b2` (i32 LE). Loaders: `from_file`, `from_bytes`, `from_parameters`
  (trainer), `from_seed` (tests).
- **Wiring**: opt-in via `Searcher::set_nnue` / the `EvalFile` UCI option. With
  no network loaded the engine is byte-for-byte the hand-crafted engine.

## Training + gating (engine-bench)

- **Trainer** (`train.rs`): generates positions by seeded random self-play,
  labels them with either the static hand-crafted eval or a **depth-N search**
  score (a stronger teacher), fits a float network whose forward pass mirrors the
  quantised inference, and exports `RFNN`. Loss is the **win-probability (WDL)
  sigmoid** loss (`sigmoid(cp/400)`), which is why it fits well (see Findings).
  Config: `hidden`, `epochs`, `learning_rate`, `seed`.
- **Gate** (`run_nnue_gauntlet`): plays the NNUE candidate vs the hand-crafted
  baseline at **equal depth**, both colours, and feeds the result into the
  existing `sprt`. This is the decision procedure: a network becomes the default
  only if it **passes**.
- **Shardable sub-commands** (for parallel orchestration):
  `gen-data`, `gen-openings`, `gate-file`, `sprt`, plus `train` and `nnue-sprt`.
  `gate-file` bounds each search to 100 ms by default (or an explicit final
  `move_time_ms` argument), so one pathological position cannot stall a shard.

## Parallel pipeline

Labeling and gating are embarrassingly parallel; training wants a GPU. Two
orchestrations exist:

- **GitHub Actions** — `.github/workflows/nnue-campaign.yml`: a `workflow_dispatch`
  job that trains a WDL net, then a **matrix-parallel gate** (12 shards × N
  openings × 2 colours) aggregated into one **decisive SPRT**. This is the
  working path from CI; runners have open egress and real CPUs.
  (`nnue-train.yml` is the smaller single-runner train→gate.)
- **Modal** — `modal/` (`train_nnue.py` PyTorch WDL trainer exporting `RFNN`;
  `app.py` fan-out labeling → GPU train → fan-out gate → SPRT; `README.md`).
  **Note:** Modal's gRPC control-plane is blocked by the agent sandbox's egress
  policy, so it must be launched from a normal machine (`modal run modal/app.py`).
  The `RFNN` byte contract is verified — a Python-written net loads and plays in
  the Rust engine.

## How to run things

```bash
# Play / analyse (UCI):  build then feed UCI commands
cargo run --release -p engine-uci
#   setoption name Threads value 4
#   setoption name EvalFile value net.rfnn        # load an NNUE net
#   position startpos / go depth 12

# Train a network locally (CPU):  train <out> <plies> <label_depth> <epochs> [lr]
cargo run --release -p engine-bench -- train net.rfnn 64 6 150

# Grade a network vs the hand-crafted baseline:
cargo run --release -p engine-bench -- nnue-sprt net.rfnn 5

# Decisive, parallel campaign (from CI): dispatch nnue-campaign.yml
#   (inputs: plies, label_depth, epochs, gate_depth, openings_per_shard, gate_plies)

# GPU training / massive gate (from your machine, not the sandbox):
pip install modal && modal token new && modal run modal/app.py
```

Other workflows: `engine-core-perft`, `external-stockfish-sprt`,
`fixed-opponent-gauntlet`, `syzygy-corpus`, `tactical-suite`,
`throughput-benchmark`, `spsa-tuning`.

## Findings & decisions (the useful learnings)

1. **Raw-centipawn MSE plateaus.** Labels clamp to ±10000 for tactical
   positions; MSE is dominated by those extremes, which a small net can't
   represent. A larger campaign (more plies/epochs) did **not** improve the gate
   (~−364 Elo). *Fix:* the **WDL sigmoid loss** — bounds targets to `[0,1]` so a
   decisive-but-not-huge edge and a mate-in-20 contribute comparable gradients.
   It fits far better (win-prob loss → ~0 on training data).
2. **Gradient clipping was a stepping stone.** Before WDL, deep-search labels
   made SGD diverge (loss increased); a Huber-style error clip fixed divergence
   and roughly halved the deficit. WDL then superseded the clip.
3. **Deterministic engines need opening diversity.** Same opening ⇒ same game,
   so a big SPRT needs many distinct openings (`gen-openings`), not repeated
   runs. This is why the gate is sharded over generated openings.
4. **Incremental NNUE is guarded, not hoped.** The per-node `debug_assert`
   against a full refresh is what makes the make/unmake accumulator trustworthy
   across castling / en passant / promotion.
5. **The sandbox is ephemeral and network-restricted.** Background jobs die on
   container restart (it happened repeatedly), and Modal's gRPC can't leave the
   sandbox. GitHub Actions is the reliable cloud lever from here.

## Campaign result (2026-07-14)

**No: the first properly-trained WDL NNUE did not beat the hand-crafted
baseline.**

- Decisive GitHub Actions campaign:
  [`29342974072`](https://github.com/benseverndev-oss/rusty-fish/actions/runs/29342974072)
  (64 plies, depth-6 labels, 150 epochs, depth-4 gate, 12 shards × 96
  openings × two colours).
- Training fit 1,024 samples from WDL loss 0.06 to 0.00, but the 2,304-game
  gate scored **23W / 435D / 1846L**, estimated **−373.40 Elo**, with SPRT
  **AcceptH0** (LLR −32.636; threshold ±2.944). Do **not** adopt this net.
- All twelve gate shards completed in about two minutes after the bounded
  search-time fix in PR #38, so this is a valid verdict rather than the old
  stalled campaign.

## Next steps (prioritised)

1. **Raise NNUE capacity and teacher quality.** The WDL objective is working,
   but the current net is decisively weak. In order of leverage, try:
   a larger hidden layer (256+), **HalfKA king-bucketed** features, and stronger
   teachers — WDL game-outcome labels and **external Stockfish** labels (the UCI
   match harness already exists). Run these on a GPU via Modal (from your
   machine) for turnaround, then repeat the same decisive gate.
2. **Only after an SPRT pass**, embed the `RFNN` via `include_bytes!` and load it
   in `Searcher::default()` / the UCI binary behind a clean toggle.
3. **Engine wins independent of NNUE**: aggregate node reporting + `nps` across
   Lazy SMP threads; a **lockless atomic** TT (replaces the sharded-mutex one)
   for better scaling; **Ponder** and **MultiPV** UCI options; an SPSA campaign
   to tune `SearchParams`; time-management tuning.

## HalfKA promotion records (2026-07-14)

No HalfKA campaign has been run from this checkout. The reproducible promotion
path is now `modal run modal/app.py --run-id <id> --schema halfka-v2-64
--capacity-selection <capacity-selection.json>` with the same calibrated
Stockfish configuration used by its v1-128 control.

For every real run, append: run ID; corpus and Stockfish-config hashes;
`halfka-v2-64` architecture and attempted widths; model checksum and
quantization maximum error; 384-game screen W/D/L; then (only if promoted) the
2,304-game W/D/L, Elo, LLR, SPRT decision, and campaign URL. An `AcceptH0` is a
failed branch. An `AcceptH1` authorizes a separate adoption design only; it does
not embed the net automatically.

## Gotchas

- **`workspace` CI status** can show a combined "pending" because there are no
  required-status contexts; trust the individual check runs.
- **`engine-core` tests** need rustc ≥ 1.95 (dev-dep `shakmaty`), so they run
  only on CI, not on the local 1.94 toolchain — the other crates test locally.
- **Cargo.lock churn**: release builds sometimes reorder a `shlex` entry; revert
  it (`git checkout Cargo.lock`) — it is not a real change.
- **Modal auth**: needs your account/token and a network that permits Modal's
  gRPC; if you ever pass a token where it could be logged, rotate it afterward.
- **Gate liveness**: keep the bounded `gate-file` move time enabled. The original
  2,304-game run stalled because a depth-only search could take arbitrarily long;
  PR #38 fixes that with a 100 ms default.
- One harmless pre-existing warning: `mut callback` at `engine-search/src/lib.rs`
  (predates this work).

## Merged this session

PRs **#29–#38** on `main`: Lazy SMP; SPSA tuning; NNUE foundation; incremental
accumulator + bootstrap trainer; deep-search training target; NNUE-vs-baseline
SPRT gate; trainer gradient-clipping fix; configurable epochs/LR; WDL sigmoid
loss; the Modal + Actions parallel pipeline; and bounded gate search time. Design
specs and plans for each are under `docs/superpowers/`.
