# Phase 4 (Learned Move Ordering) — Session Handoff

**Date:** 2026-07-25 · **Branch:** `claude/work-in-progress-fevs11`

## Resume here

```bash
git fetch origin
git checkout claude/work-in-progress-fevs11
git log --oneline -4   # expect: 7bdf5ac wiring, c1fcb83 trainer+RFPO, db4b631 v3 telemetry
```

Everything below is pushed. The previous session had a broken GitHub MCP token that
was **not a repo collaborator**, so it could neither merge nor open PRs. That's the
first thing to clear now that the MCP is fixed.

## Immediate actions (need the working MCP)

1. **Merge PR #96** — `fix/lmr-v2-thresholds`, "re-sweep LMR thresholds for the v2
   model (+50.6 → +57.2 Elo)". It was green and `mergeable_state: clean`. Squash-merge
   (repo convention: one commit per PR, `(#NN)` suffix).

2. **Open the Phase 4 PR** for `claude/work-in-progress-fevs11` → `main`, as a **draft**.
   Title and body to use:

   > **Title:** `feat: Phase 4 learned move ordering — telemetry, Rust trainer, wiring`
   >
   > **Body:** three commits, all search-neutral by default:
   > - `db4b631` v3 telemetry — append-only move-ordering columns (`order_score`, `see`,
   >   `mover_piece`, `captured_piece`); byte-identical telemetry invariant holds.
   > - `c1fcb83` in-process Rust trainer + `RFPO` format — no Python/Modal; standardize →
   >   class-weighted BCE-with-logits → Adam → val AUC (port of `train_lmr.py`); exports
   >   `RFPO` via `PolicyModel` so training/inference agree; `train-policy` CLI; no new dep.
   > - `7bdf5ac` wiring behind a `None`-by-default toggle — `order = classical +
   >   clamp(learned_correction)`; applied only in main `negamax` (root/qsearch stay
   >   classical); neutral with no policy, and a zero-correction policy is byte-identical
   >   (both tested).
   >
   > Full workspace suite green on stable (1.95). No bundled policy asset yet, so a
   > default search is unchanged; adoption waits on an SPRT gate. Next steps in
   > `docs/superpowers/plans/2026-07-25-learned-move-ordering.md`.

   Then `subscribe_pr_activity` on it and keep it green.

## What's on the branch vs main

`main` ends at `55e0759` (#95). The branch adds three Phase 4 commits on top. Nothing
here changes a **default** search — the policy is opt-in (`Searcher::set_policy_model`)
and there is no bundled `RFPO` asset. The `PolicyModel` is **not** referenced by
`Searcher::default()`.

## The real next task: is the ordering signal actually there?

Steps 1–3 of the arc (instrument → train → wire) are done, but only proven on synthetic
data. Before spending a gate, get evidence that "search this move first" is learnable
from ordering-time features — i.e. that `train-policy` reaches a val **AUC well above
0.5** on real telemetry. All Rust, all in-sandbox:

```bash
# 0) engine-bench needs rustc >= 1.95 (dev-dep shakmaty). Local default may be 1.94:
rustup install 1.95.0            # then prefix cargo with +1.95.0
cargo +1.95.0 build --release -p engine-bench

BIN=./target/release/engine-bench

# 1) FENs to search (opening positions; or bring your own FEN list, one per line):
$BIN gen-openings 2000 8 1 > /tmp/openings.fens

# 2) Telemetry TSV (per-move-decision rows, depth-8 search over each FEN):
$BIN gen-search-telemetry /tmp/openings.fens 8 > /tmp/telemetry.tsv

# 3) Train the policy in-process. Args: <tsv|-> <out.rfpo> [hidden] [epochs] [lr]
#    [stride] [max_rows] [seed]. stride>1 decorrelates rows from the same search.
$BIN train-policy /tmp/telemetry.tsv /tmp/policy.rfpo 16 20 1e-3 8 5000000 0
# -> prints: POLICY_TRAIN_DONE ... base_rate=.. val_acc=.. val_auc=..
```

**Decision gate:** if `val_auc` is ~0.5, ordering isn't learnable from these features —
stop and rethink the feature set / target (`caused_cutoff` is sparse; `raised_alpha` is
denser, or move to a pairwise/listwise ranking loss). If it's clearly > 0.6, proceed:

```bash
# 4) SPRT-gate the policy vs classical ordering at equal movetime. There is NOT yet a
#    ready-made "policy gate" subcommand analogous to gate-file (which gates NNUE nets).
#    Add one: a bench-compare / match variant that installs the RFPO on the candidate
#    via set_policy_model(Some(..)) and leaves the baseline as set_policy_model(None),
#    equal movetime, sharded over gen-openings, fed into `sprt`. Model this on how
#    gate-file + the nnue-campaign workflow gate NNUE nets. Report Elo AND node count
#    (ordering wins should shrink the tree even when Elo is flat).

# 5) Only on an SPRT pass: bundle the RFPO (include_bytes! like the LMR asset in
#    assets/lmr/), make Searcher::default() install it, and sweep policy_order_bound
#    (SearchParams) by gated A/B — thresholds/bounds are per-model, exactly like the
#    LMR thresholds that PR #96 had to re-sweep.
```

## Design facts you must preserve

- **Feature order is load-bearing.** `PolicyModel`'s feature vector, the trainer's
  column selection, and `telemetry::POLICY_FEATURE_COLUMNS` must stay in the same order.
  `Searcher::policy_features` builds it field-for-field like the v3 telemetry so
  inference features == training features. A cross-check test in `telemetry.rs` and the
  clamp-index test in `policy_model.rs` guard this — keep them green.
- **`RFPO` = `RFLM` shape, different magic** (`b"RFPO"`). `input_dim` is checked on load;
  widening the feature set and swapping any bundled asset must happen in one commit.
- **Two neutrality guarantees, both tested — do not break:**
  `telemetry_never_perturbs_the_search` (telemetry off==on) and
  `policy_model_with_zero_correction_is_byte_identical_to_policy_off` (p=0.5 ⇒ no change).
- **The correction is a residual on the classical score**, magnitude in
  `SearchParams::policy_order_bound` (default 4000, tunable). The model is a pure
  predictor; the policy/scale lives in the search — same split as learned LMR.
- Policy applies **only in main `negamax`** ordering, not root, not quiescence (that's
  where the training telemetry comes from).

## Repo gotchas (from the earlier handoff, still true)

- **rustc 1.95** required for `engine-bench`/`engine-core` (dev-dep `shakmaty`);
  `engine-search` alone builds on 1.94. CI uses `dtolnay/rust-toolchain@stable`.
- **CI = `cargo test --workspace` only** (`.github/workflows/engine-core-perft.yml`,
  job `workspace`). **No `cargo fmt --check`, no clippy** — local `cargo fmt` disagrees
  with the repo's formatter across the board, so it is *not* authoritative; match
  surrounding style by hand.
- The `workspace` check can show combined "pending"; trust individual check runs.
- One harmless pre-existing warning: `mut callback` at `engine-search/src/lib.rs`.

## File map (this session's work)

- `engine-search/src/telemetry.rs` — v3 columns + `POLICY_FEATURE_COLUMNS` /
  `POLICY_TARGET_COLUMN` (`caused_cutoff`) / `POLICY_FEATURE_CLAMPS`.
- `engine-search/src/policy_model.rs` — `RFPO` format, `PolicyModel`, `cutoff_prob`,
  `order_correction`, `DEFAULT_POLICY_ORDER_BOUND`.
- `engine-search/src/lib.rs` — v3 telemetry capture, `Searcher::policy_model` +
  `set_policy_model`, `SearchParams::policy_order_bound`, `PolicyOrderContext`,
  `policy_features`, the ordering correction in `order_moves`, byte-identical test.
- `engine-bench/src/policy_train.rs` — the trainer.
- `engine-bench/src/main.rs` — `train-policy` CLI.
- `docs/superpowers/plans/2026-07-25-learned-move-ordering.md` — the arc.
