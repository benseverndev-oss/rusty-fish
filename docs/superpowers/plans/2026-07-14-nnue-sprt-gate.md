# NNUE SPRT Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a trained NNUE network beats the hand-crafted-eval baseline via SPRT, so adoption is gated on real strength.

**Architecture:** An NNUE-vs-baseline self-play gauntlet in `engine-bench` feeds the existing SPRT. A `nnue-sprt` sub-command and a gate step in the `nnue-train` workflow turn training into a train → gate loop.

**Tech Stack:** Rust 2024, existing Rusty Fish search + match/SPRT harness, GitHub Actions.

## Global Constraints

- The gate only measures; it never changes the default engine.
- Candidate and baseline search at equal depth.
- Adopting a network as default is a separate change contingent on a passing verdict.
- Workspace tests are required remote evidence.

---

### Task 1: NNUE-vs-baseline gauntlet + sub-command

**Files:** Modify `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`, `.github/workflows/nnue-train.yml`.

- [ ] **Step 1: Write a failing test** that the gauntlet plays both colours for each position and produces a scored result.
- [ ] **Step 2: Implement** `run_nnue_gauntlet`/`play_nnue_game`, the `nnue-sprt <network> [depth]` sub-command, and a workflow gate step.
- [ ] **Step 3: Verify green and commit.**

### Task 2: Review and merge

- [ ] Open a PR describing the gate and that adoption is gated on a passing verdict.
- [ ] Require workspace tests, tactical suite, and fixed-opponent gauntlet to pass.
- [ ] Record the adoption follow-up: embed a passing network via `include_bytes!` and load it by default.

## Self-review

The gate reuses the existing SPRT, keeps candidate/baseline depths equal, never touches the default engine, and is covered by a focused test plus the train → gate workflow.
