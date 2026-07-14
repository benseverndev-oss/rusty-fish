# SPSA Parameter Tuning Design

## Goal

Give the engine a reproducible way to tune its search parameters by playing
games, using Simultaneous Perturbation Stochastic Approximation (SPSA) over the
existing self-play match harness, so future strength gains come from measured
tuning rather than hand-set constants.

## Scope

This slice adds tuning to `engine-bench` and makes a curated set of search
constants runtime-configurable in `engine-search`:

- A public `SearchParams` struct holding the tunable scalars, with a `Default`
  that reproduces today's hard-coded values exactly. `Searcher` reads its
  parameters from this struct.
- The tuned parameters are the clean scalar search knobs already threaded
  through `Searcher` methods: the aspiration window, the razoring and
  reverse-futility margin base/scale pairs, the late-move-pruning base/scale,
  and the null-move reduction.
- A deterministic SPSA optimizer in `engine-bench`: a fixed spec table
  (name/min/max/perturbation per parameter), a seeded PRNG for the perturbation
  directions, a `SearchParams` <-> vector mapping, a parameter self-play match,
  a campaign runner, and a TSV report.
- A `spsa` sub-command and a `workflow_dispatch` GitHub Actions workflow that
  runs a campaign and uploads the tuned parameters.

Eval-weight tuning is out of scope here: those flow through free functions and
a global `EVAL_PARAMS` const, so parameterising them is a separate slice.

## Alternatives considered

1. **Texel tuning.** Fits eval weights to game outcomes from a labelled
   position set. Excellent for eval but needs a large labelled corpus and only
   touches evaluation, not search. Deferred.
2. **Expose parameters over UCI and tune with an external harness
   (OpenBench/fishtest).** The most powerful long-term path, but depends on
   external infrastructure. Worth doing later; noted as a follow-up.
3. **In-tree deterministic SPSA over the existing match harness (chosen).**
   Self-contained, reproducible, unit-testable, and reuses the self-play match
   and SPRT plumbing already in `engine-bench`. Bounded and independently
   verifiable.

## Architecture

- `SearchParams` is `Copy` with public integer fields and a `Default` equal to
  the current constants. `Searcher` gains a `params` field plus
  `set_search_params`/`search_params`. The former free functions
  (`razor_margin`, `reverse_futility_margin`, `late_move_pruning_limit`) take a
  `&SearchParams`; the aspiration window and null-move reduction read from it
  directly. Helper (Lazy SMP) searchers inherit the primary searcher's params.
- The optimizer works on a fixed-length `f64` vector. `SPSA_SPECS` defines each
  dimension's name, bounds, and perturbation size; conversion helpers round and
  clamp between the vector and `SearchParams`.
- Each iteration draws a Rademacher direction vector from a seeded xorshift
  PRNG, forms `θ+ = clamp(θ + step·Δ)` and `θ- = clamp(θ - step·Δ)`, plays a
  self-play match of `θ+` (candidate) versus `θ-` (baseline) over the supplied
  positions and both colours, and updates each parameter by
  `θ_i += learning_rate · (2·score − 1) · Δ_i · step_i`, clamped to bounds.
- The campaign records per-iteration match scores and the running parameters and
  emits a TSV report; the final tuned `SearchParams` is returned.

## Safety rules

- `SearchParams::default()` must equal the current constants, so an untuned
  engine is byte-for-byte unchanged.
- The optimizer is deterministic given its seed and inputs; no wall-clock or
  unseeded randomness enters the parameter path.
- Every produced parameter vector is clamped to its spec bounds before being
  applied to a searcher.
- CI runs only a fast smoke campaign (few iterations, shallow depth); a real
  campaign runs behind `workflow_dispatch`.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including the SPSA math tests
(PRNG reproducibility, update direction, clamping) and a smoke-campaign test.
The tactical suite and fixed-opponent gauntlet must not regress. The
`workflow_dispatch` SPSA campaign is the mechanism that produces tuned
parameters, and any tuned set must clear an external Stockfish SPRT before it is
adopted as the new default.
