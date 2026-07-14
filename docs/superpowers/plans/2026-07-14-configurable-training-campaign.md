# Configurable Training Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make training epochs and the learning rate configurable so an NNUE campaign can be scaled up to fit deep-search labels.

**Architecture:** Optional `train` sub-command arguments override the epoch count and learning rate; the `nnue-train` workflow gains an `epochs` input and larger campaign defaults, feeding the existing SPRT gate.

**Tech Stack:** Rust 2024, existing Rusty Fish trainer + gate, GitHub Actions.

## Global Constraints

- Omitting the new arguments reproduces the previous defaults.
- Epochs floored at 1; a non-positive learning rate is ignored.
- Plumbing only — the trainer maths, gate, and export paths are unchanged.
- Workspace tests are required remote evidence.

---

### Task 1: Configurable epochs / learning rate

**Files:** Modify `engine-bench/src/main.rs`, `.github/workflows/nnue-train.yml`.

- [ ] Parse optional epochs (5th arg) and learning rate (6th arg) in the `train` sub-command.
- [ ] Add an `epochs` workflow input and raise campaign defaults; pass it through to `train`.
- [ ] Verify the workspace builds and tests pass; commit.

### Task 2: Run a campaign and review

- [ ] Dispatch / run a larger campaign (more plies + epochs, depth-6 labels) and record the SPRT verdict.
- [ ] Open a PR describing the knobs and the campaign result.
- [ ] Record follow-ups: adaptive optimizer / LR schedule and larger networks if the gate still fails.

## Self-review

The change is additive and defaults to the prior behaviour, floors the new knobs sensibly, leaves the trainer maths untouched, and turns the existing train → gate workflow into a scalable campaign runner.
