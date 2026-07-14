# NNUE Training + Incremental Accumulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a self-contained NNUE bootstrap trainer and an incremental accumulator so NNUE evaluation is both trainable and efficiently updatable.

**Architecture:** A Rust SGD trainer in `engine-bench` distils the hand-crafted evaluation into a quantised `RFNN` network. The search in `engine-search` maintains the NNUE accumulator across make/unmake via small per-move deltas, guarded by a refresh debug-assert.

**Tech Stack:** Rust 2024, existing Rusty Fish search/eval/match harness, GitHub Actions.

## Global Constraints

- With no network loaded, the search and evaluation are byte-for-byte unchanged.
- The incremental accumulator must equal a full refresh at every node (debug-asserted).
- The trainer must be deterministic given its seed.
- Workspace tests, tactical suite, and fixed-opponent gauntlet are required remote evidence.

---

### Task 1: Training enablers (engine-search)

**Files:** Modify `engine-search/src/lib.rs`, `engine-search/src/nnue.rs`.

- [ ] Expose `hand_crafted_evaluation`, `active_features`, `INPUT_DIMENSION`, `Nnue::from_parameters`, and `Nnue::evaluate_with`.
- [ ] Verify green and commit.

### Task 2: Bootstrap trainer (engine-bench)

**Files:** Add `engine-bench/src/train.rs`; modify `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`.

- [ ] **Step 1: Write failing tests** for labelled-sample generation and for training reducing loss below a zero-predictor.
- [ ] **Step 2: Implement** data generation, a float SGD trainer, quantised export, and a `train` sub-command.
- [ ] **Step 3: Verify green and commit.**

### Task 3: Incremental accumulator (engine-search)

**Files:** Modify `engine-search/src/lib.rs`, `engine-search/src/nnue.rs`.

- [ ] **Step 1: Write a failing test** that searches castling/en-passant/promotion positions with a network loaded (the accumulator debug-assert guards each node).
- [ ] **Step 2: Implement** `nnue_changed_squares`, `nnue_make`/`nnue_unmake`, root refresh, and the debug-assert in `evaluate`; wire the make/unmake sites in negamax, negamax_root, and quiescence.
- [ ] **Step 3: Verify green and commit.**

### Task 4: Workflow + review

- [ ] Add a `workflow_dispatch` `nnue-train` workflow that trains and uploads a network.
- [ ] Open a PR describing the trainer and incremental accumulator; require workspace tests, tactical suite, and fixed-opponent gauntlet to pass.
- [ ] Record follow-ups: stronger training targets and an SPRT-gated default-network swap.

## Self-review

The plan keeps the default engine unchanged, proves the incremental accumulator equals a refresh at every node, keeps the trainer deterministic, and reuses the existing match/eval infrastructure.
