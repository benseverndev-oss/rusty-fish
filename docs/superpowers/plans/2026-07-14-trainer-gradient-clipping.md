# Trainer Gradient Clipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the NNUE trainer diverging on large deep-search labels by clipping the gradient error (Huber-style).

**Architecture:** In `sgd_step`, clip the error used for the gradient to `±GRADIENT_ERROR_CLIP` while reporting the true squared error. Everything else in the trainer is unchanged.

**Tech Stack:** Rust 2024, existing Rusty Fish trainer, GitHub Actions.

## Global Constraints

- Static-label training behaviour is unchanged (its errors are within the clip).
- The reported loss stays the true mean squared error.
- The clip affects only the gradient, not the network shape or export.
- Workspace tests are required remote evidence.

---

### Task 1: Gradient error clipping

**Files:** Modify `engine-bench/src/train.rs`.

- [ ] **Step 1: Write a failing test** that depth-labelled training reduces the loss.
- [ ] **Step 2: Implement** the `GRADIENT_ERROR_CLIP` constant and clip the gradient error in `sgd_step`.
- [ ] **Step 3: Verify green and commit.**

### Task 2: Review and merge

- [ ] Open a PR describing the divergence, the Huber-style fix, and the end-to-end result.
- [ ] Require workspace tests, tactical suite, and fixed-opponent gauntlet to pass.
- [ ] Record follow-ups: adaptive optimizer / LR schedule / win-probability loss, and larger training + gate campaigns toward a passing network.

## Self-review

The change is a minimal, well-understood robustness fix, leaves static-label training and the export path untouched, keeps honest loss reporting, and is guarded by a deep-label convergence regression test.
