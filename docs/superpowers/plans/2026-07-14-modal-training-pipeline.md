# Modal Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelise NNUE labeling and the SPRT gate across containers and train on a GPU via Modal, for a fast and decisive train → gate loop.

**Architecture:** Shardable `engine-bench` sub-commands do labeling/openings/gating/SPRT; a PyTorch trainer exports the engine's RFNN format; a Modal app orchestrates the fan-out and GPU training.

**Tech Stack:** Rust 2024, Python + PyTorch + Modal, existing Rusty Fish trainer/gate/SPRT.

## Global Constraints

- The Rust engine stays the source of truth for labeling and play.
- The PyTorch export must match `engine-search/src/nnue.rs` byte-for-byte.
- Sub-commands only measure/compute; the default engine is unchanged.
- Workspace tests are required remote evidence; Modal execution needs the user's account.

---

### Task 1: Shardable Rust primitives

**Files:** Modify `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`.

- [ ] Add `random_opening_fens` (+ test) and the `gen-data`, `gen-openings`, `gate-file`, `sprt` sub-commands.
- [ ] Verify green and commit.

### Task 2: PyTorch trainer + Modal app

**Files:** Add `modal/train_nnue.py`, `modal/app.py`, `modal/README.md`.

- [ ] Write the PyTorch trainer (WDL loss) exporting RFNN, mirroring the inference math.
- [ ] Write the Modal orchestration (parallel label → GPU train → parallel gate → SPRT) and the README.
- [ ] Syntax-check the Python and verify the RFNN byte contract by loading a Python-written network through the Rust engine.
- [ ] Commit.

### Task 3: Review and merge

- [ ] Open a PR describing the pipeline and that Modal execution needs the user's token.
- [ ] Require workspace tests, tactical suite, and fixed-opponent gauntlet to pass.
- [ ] Record follow-ups: a Modal Volume for artifacts, larger networks, and adopting a passing network as the default.

## Self-review

The Rust primitives are testable and validated locally; the Python is syntax-checked and its RFNN output is proven loadable by the engine; nothing changes the default engine; and adoption stays gated on a passing SPRT.
