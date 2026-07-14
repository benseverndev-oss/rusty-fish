# Deep-Search Training Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the NNUE trainer label positions with a depth-N search score instead of the static evaluation, a stronger teacher.

**Architecture:** `generate_training_samples` gains an optional `label_depth`; when set it labels each position with a fixed-depth hand-crafted search. Surfaced through the `train` sub-command and the `nnue-train` workflow.

**Tech Stack:** Rust 2024, existing Rusty Fish search + trainer, GitHub Actions.

## Global Constraints

- `label_depth = None` reproduces the previous static-label behaviour exactly.
- The labeller uses the hand-crafted evaluation (no NNUE), avoiding self-reference.
- Search-score targets are clamped away from mate values.
- Workspace tests are required remote evidence; the default engine is unaffected.

---

### Task 1: Deep-search labels

**Files:** Modify `engine-bench/src/train.rs`, `engine-bench/src/main.rs`, `.github/workflows/nnue-train.yml`.

- [ ] **Step 1: Write a failing test** that depth-labelled targets differ from static targets on some positions.
- [ ] **Step 2: Implement** the optional `label_depth`, the sub-command argument, and the workflow input.
- [ ] **Step 3: Verify green and commit.**

### Task 2: Review and merge

- [ ] Open a PR describing the deep-search teacher and its defaults.
- [ ] Require workspace tests, tactical suite, and fixed-opponent gauntlet to pass.
- [ ] Record follow-ups: WDL and external-engine teachers, hyperparameter tuning, and an SPRT-gated default-network swap.

## Self-review

The change is additive and defaults to the prior behaviour, keeps the labeller non-self-referential, clamps targets, and is covered by a focused test plus the existing loss-reduction test.
