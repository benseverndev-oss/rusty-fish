# SPSA Parameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, reproducible SPSA tuner over the existing self-play match harness, plus a runtime-configurable `SearchParams` whose default reproduces today's constants.

**Architecture:** `SearchParams` on `Searcher` carries the tunable scalars. `engine-bench` gains an SPSA optimizer that perturbs a parameter vector, plays `θ+` vs `θ-` self-play matches, and steps the parameters, all driven by a seeded PRNG.

**Tech Stack:** Rust 2024, existing Rusty Fish search + match/SPRT harness, GitHub Actions.

## Global Constraints

- `SearchParams::default()` must equal the current constants exactly.
- The tuner must be deterministic given its seed; no unseeded randomness or wall-clock in the parameter path.
- All produced parameters are clamped to spec bounds before use.
- Workspace tests, tactical suite, and fixed-opponent gauntlet are required remote evidence; a real campaign runs behind `workflow_dispatch` and a tuned set must clear an external SPRT before adoption.

---

### Task 1: Tunable SearchParams (engine-search)

**Files:**
- Modify: `engine-search/src/lib.rs`

**Interfaces:**
- Produces: `pub struct SearchParams` (+ `Default`), `Searcher::set_search_params`, `Searcher::search_params`.
- Consumes: `SearchParams` in `razor_margin`, `reverse_futility_margin`, `late_move_pruning_limit`, the aspiration window, and the null-move reduction.

- [ ] **Step 1: Write a failing test** that `SearchParams::default()` matches the current constants and that the margin helpers scale with a supplied `SearchParams`.
- [ ] **Step 2: Implement** the struct, the `Searcher` field/accessors, and thread the params through the call sites and Lazy SMP helpers.
- [ ] **Step 3: Verify green and commit.**

### Task 2: SPSA optimizer (engine-bench)

**Files:**
- Modify: `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`

**Interfaces:**
- Produces: `SpsaSpec`/`SPSA_SPECS`, a seeded PRNG, `SearchParams` <-> vector helpers, `spsa_update`, `run_spsa_campaign`, `spsa_tsv_report`, and a `spsa` sub-command.

- [ ] **Step 1: Write failing tests** for PRNG reproducibility, `spsa_update` direction and clamping, and a fast smoke campaign that returns in-bounds parameters.
- [ ] **Step 2: Implement** the optimizer and the parameter self-play match (`play_parameter_game`), reusing `summarize`/`MatchScore`.
- [ ] **Step 3: Verify green and commit.**

### Task 3: Tuning workflow

**Files:**
- Add: `.github/workflows/spsa-tuning.yml`

- [ ] **Step 1: Add** a `workflow_dispatch` job that runs `engine-bench spsa` and uploads the TSV artifact.
- [ ] **Step 2: Verify** the workflow is valid and commit.

### Task 4: Review and merge

- [ ] Update the open PR to describe SPSA alongside the shared-table/Lazy SMP work.
- [ ] Require workspace tests, tactical suite, and fixed-opponent gauntlet to pass.
- [ ] Record follow-ups: eval-weight (Texel) tuning, UCI exposure of the parameters for external tuners, and running a real campaign + SPRT to adopt a tuned default.

## Self-review

The plan keeps the untuned engine unchanged, isolates all randomness behind a seed, clamps every produced parameter, reuses the existing match/SPRT harness, and gates CI on a fast smoke campaign while leaving real campaigns to `workflow_dispatch`.
