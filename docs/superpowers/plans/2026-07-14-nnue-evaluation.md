# NNUE Evaluation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the NNUE inference machinery (features, accumulator, quantised forward pass, file format) as an opt-in evaluator that defaults to the current hand-crafted evaluation, leaving clean seams for a trained network and make/unmake integration.

**Architecture:** A `nnue` module in `engine-search` owns immutable network weights and a per-evaluation accumulator. `Searcher` carries an optional `Arc<Nnue>` and uses it in `evaluate` only when present. A UCI `EvalFile` option loads a network file.

**Tech Stack:** Rust 2024, existing Rusty Fish search/eval, GitHub Actions.

## Global Constraints

- With no network loaded, `evaluate` must equal the current hand-crafted evaluation exactly.
- All network reads are validated (magic, version, length); malformed data is rejected, never trusted.
- NNUE output is clamped to a non-mate range.
- Workspace tests, tactical suite, and fixed-opponent gauntlet are required remote evidence.

---

### Task 1: NNUE module (engine-search)

**Files:**
- Add: `engine-search/src/nnue.rs`
- Modify: `engine-search/src/lib.rs` (`mod nnue;`, re-export)

**Interfaces:**
- Produces: `Nnue` (weights + `evaluate`, `from_bytes`, `to_bytes`, `from_file`, `from_seed`), an `Accumulator` with `refresh`/`add_feature`/`remove_feature`, and `feature_index`.

- [ ] **Step 1: Write failing tests** for feature-index bounds, incremental == refresh, forward determinism, bytes round-trip, and loader rejection.
- [ ] **Step 2: Implement** the feature set, accumulator, quantised forward pass, and `RFNN` format.
- [ ] **Step 3: Verify green and commit.**

### Task 2: Optional NNUE in Searcher + UCI (engine-search, engine-uci)

**Files:**
- Modify: `engine-search/src/lib.rs`, `engine-uci/src/main.rs`

**Interfaces:**
- Produces: `Searcher::set_nnue`, NNUE-aware `evaluate`, helper inheritance of the network, and an `EvalFile` UCI option.

- [ ] **Step 1: Write failing tests** that a loaded network changes `evaluate` and that `EvalFile` errors on a missing file.
- [ ] **Step 2: Implement** the `Option<Arc<Nnue>>` field, the `evaluate` branch, helper threading, and the UCI option.
- [ ] **Step 3: Verify green and commit.**

### Task 3: Review and merge

- [ ] Update the open PR to describe the NNUE foundation.
- [ ] Require workspace tests, tactical suite, and fixed-opponent gauntlet to pass.
- [ ] Record follow-ups: offline trainer + real network, incremental make/unmake accumulator, HalfKA features, and an SPRT-gated default swap.

## Self-review

The plan keeps the default engine unchanged, validates all network reads, clamps NNUE output away from mate scores, shares immutable weights across threads, and proves the incremental accumulator correct so the make/unmake hook is a clean follow-up.
