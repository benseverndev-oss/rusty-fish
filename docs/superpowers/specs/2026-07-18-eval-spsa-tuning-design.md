# SPSA Eval Tuning Design

## Goal

Find hand-crafted evaluation weights that gain Elo, by SPSA self-play tuning a
bounded set of eval weights (mobility plus never-tuned existing terms), run on
Modal for game volume, and gated against today's evaluation. If the tune gains
under SPRT, ship the tuned weights; if it is flat, that is itself a signal that
the hand-crafted eval is near its ceiling at this structure, and the next lever
is NNUE or new terms rather than more tuning.

This continues the eval effort: the mobility term shipped neutral at hand-set
weights, and the hypothesis is that calibration, not the idea, was the problem.

## Making eval weights tunable: `EvalParams`

The weights to tune move out of hardcoded constants in `engine-search` into an
`EvalParams` struct, threaded into `evaluate_position` the same way
`mobility_scale` already is (the searcher owns it and passes it in; the public
`hand_crafted_evaluation` uses the default). `EvalParams::default()` reproduces
today's exact values, so a default build's evaluation is **byte-identical** and
nothing ships until a gate proves a change. A regression test pins that
`evaluate_position` with default `EvalParams` equals today's output on a corpus
of positions.

The first tunable set is deliberately small (~18 weights); larger sets tune
noisier under SPSA. Each weight has a lower/upper bound and an SPSA step:

- **Tapered piece values** for knight, bishop, rook, queen — middlegame and
  endgame each (8). Today these are untapered round numbers (320/330/500/900
  with mg == eg), so both the values and the taper are open. **Pawn is fixed at
  100/100 and king at 0**, anchoring the scale so self-play SPSA cannot simply
  inflate every value together (self-play is blind to absolute scale).
- **Mobility weights** for knight, bishop, rook, queen — middlegame and endgame
  each (8). This is the direct retry of the neutral mobility gate: does
  *calibrated* mobility help? The mobility offsets stay fixed (the neutral
  result points at the per-square weights, not the offsets).
- **Bishop pair** and **passed-pawn base bonus** (2).

Mobility during a tuning run is enabled (its `mobility_scale` is set to 100 for
both perturbed sides) so the mobility weights are actually exercised; the
baseline that the final gate compares against keeps mobility off, matching
today's shipped default.

## Extending SPSA to an eval vector

The tuner already optimizes a fixed-size `[f64; N]` vector with a per-dimension
`SpsaSpec` (name, min, max, step), projecting to and from `SearchParams`. This
design adds an analogous eval vector: `eval_params_to_vector` /
`vector_to_eval_params` over `EvalParams`, and an `EVAL_SPSA_SPECS` table with
one spec per tunable weight. The existing `spsa_update`, `SpsaRng` (deterministic
Rademacher directions), and clamping are reused unchanged — they operate on a
vector and a specs table, so they generalize.

This run tunes **eval only**. Search-parameter tuning is a separate, existing
concern and stays off here so the eval signal is not confounded by search-param
perturbation. The two tuners share the same `spsa_update`/RNG code but run over
different vectors and specs.

The per-side plumbing mirrors mobility: `play_parameter_game` already sets
`SearchParams` per side; the tuning campaign additionally sets `EvalParams` per
side (the searcher gains a settable `EvalParams`, defaulting to today's values,
passed into `evaluate_position`).

## Running the campaign on Modal

SPSA is sequential across iterations — each gradient step needs the previous
iteration's match result — but each iteration's `theta+` vs `theta-` match is
embarrassingly parallel, exactly like the gate. A single sequential container is
too slow: the game volume SPSA needs (dozens of iterations, hundreds of games
each) would run for hours, the same single-machine wall the gate hit. So the
campaign fans each iteration's match across containers.

Concretely, mirroring the existing Modal gate:

- A `spsa-match-file` engine-bench command plays one shard of a `theta+` vs
  `theta-` match: it reads a shard of openings and both parameter vectors (as
  values or a small file), plays them color-swapped, and emits `W\tD\tL`. This
  is the SPSA analog of `mobility-gate-file`.
- A Modal `spsa_tune` entrypoint runs the sequential SPSA loop *locally in the
  driver* (cheap: RNG, perturb, gradient, step), but for each iteration fans the
  perturbed match across `spsa_match_shard` containers, sums the score, and steps
  `theta`. It prints the tuned `EvalParams` and the per-iteration trace.

Iterations are sequential and each waits on its parallel match, so wall time is
`iterations * (one parallel match)`. With a parallel match at gate-like speed
(~1 minute) and ~30 iterations, a campaign is ~30 minutes — feasible, and far
beyond what one runner could do.

The campaign is launched the same way as the gate: `infisical run --env dev --
uv run --with modal -- modal run modal/app.py::spsa_tune` (Infisical supplies the
Modal token; `PYTHONUTF8=1` on Windows).

## Gating and shipping

The campaign emits tuned `EvalParams`. Those are fed to a gate — the existing
mobility gate generalized, or a sibling `eval-gate-file` — that plays the tuned
eval (mobility enabled, tuned weights) against today's default eval (mobility
off, default weights) over thousands of parallel games and takes the SPRT
verdict. This is the honest test: the tuned eval must beat what ships today.

- **Gains under SPRT** → a one-line change sets `EvalParams::default()` (and the
  mobility default) to the tuned values, gated again by the normal PR CI.
- **Flat or negative** → the default is unchanged (it already is; tuning writes
  no defaults on its own), and the result says the hand-crafted eval is near its
  ceiling — pivot to NNUE or new terms rather than tuning further.

Nothing about the shipped engine changes until that explicit, gated flip.

## Verification

- A regression test pins that default `EvalParams` reproduces today's evaluation
  byte-for-byte on a position corpus (the guarantee that tuning is inert until a
  flip).
- Unit tests for `eval_params_to_vector` / `vector_to_eval_params` round-tripping
  the default, and that out-of-range weights clamp to bounds.
- A unit test that `EvalParams::default()` matches the current hand-set eval
  constants (the analog of the existing `default_search_params_match...` test).
- A small self-play smoke test that `spsa-match-file` runs and emits `W\tD\tL`.
- The Modal SPSA campaign and the eval gate are validated by running them
  end-to-end on Modal (a short campaign first, then the real one), the same way
  the mobility gate was.
- All in-repo validation runs in GitHub Actions; Cargo is never run locally.

## Out of scope

- Tuning search parameters (a separate existing concern; off for this run).
- Tuning every eval term — king safety internals, pawn-structure penalty
  weights, activity, threats, and piece-square tables stay hardcoded for now;
  they can be added to `EvalParams` in later slices once the rig is proven.
- Second-order SPSA refinements (adaptive learning rate, momentum); the existing
  fixed-step update is reused as-is.
- Any change to the NNUE path.
