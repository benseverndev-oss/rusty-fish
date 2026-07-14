# WDL Sigmoid Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the NNUE network with a win-probability (sigmoid) loss so extreme evaluations stop dominating the fit.

**Architecture:** Squash prediction and target through `sigmoid(cp / WDL_SCALE)` and take the MSE in win-probability space; the bounded gradient replaces the Huber-clipped raw-cp gradient, with a larger default learning rate to match its scale.

**Tech Stack:** Rust 2024, existing Rusty Fish trainer + SPRT gate, GitHub Actions.

## Global Constraints

- Confined to the trainer's loss and learning-rate default; network shape, inference, and export are unchanged.
- `win_probability` is bounded and numerically stable.
- The learning rate stays configurable.
- Workspace tests are required remote evidence.

---

### Task 1: Win-probability loss

**Files:** Modify `engine-bench/src/train.rs`.

- [ ] Replace the raw-cp MSE gradient with the WDL-sigmoid gradient; remove the Huber clip; add `win_probability`; raise the default learning rate.
- [ ] Update the trainer tests to win-probability units (loss reduction, no divergence, beats a 0.5 predictor).
- [ ] Verify green and commit.

### Task 2: Campaign and review

- [ ] Train a depth-6 campaign with the WDL loss and record the SPRT-gate verdict.
- [ ] Open a PR describing the objective change and the before/after gate result.
- [ ] Record follow-ups if the gate still fails: larger network, activation reparameterisation / adaptive optimizer, and stronger (WDL game-outcome / external) teachers.

## Self-review

The change swaps in the standard NNUE objective, removes the now-unnecessary clip, keeps the learning rate configurable, leaves inference/export untouched, and is guarded by trainer tests in the new loss units plus the SPRT gate.
