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

An `EvalParams` struct already exists in `engine-search` (a private const holding
only the five piece values). This design **extends it** — adds the tunable
mobility weights, bishop pair, and passed-pawn base, tapers the piece values into
middlegame/endgame pairs — and makes it a searcher-held, settable value threaded
into `evaluate_position` (which today takes `(board, mobility_scale)`). The
searcher owns an `EvalParams`, passes it into `evaluate_position`; the public
`hand_crafted_evaluation` uses the default. `EvalParams::default()` reproduces
today's exact values, so a default build's evaluation is **byte-identical** and
nothing ships until a gate proves a change. A regression test pins that
`evaluate_position` with default `EvalParams` equals today's output on a corpus
of positions.

Threading is more invasive than `mobility_scale` (a single scalar). The values
being made tunable are read today by several free functions and inline literals
that must be rewired to take `&EvalParams`: `tapered_piece_value` /
`EVAL_PARAMS` (piece values), the hardcoded match arms in `mobility_score`, the
inline `TaperedScore::equal(35)` bishop-pair adds, and the passed-pawn base `20`
inside `pawn_structure_bonus`. That passed-pawn `20` is entangled with an
`advancement * 10` term and shares the function with unrelated `20` literals
(isolated-pawn, open-file) that must **not** be swept into the tunable weight —
only the passed-pawn base becomes a parameter.

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

The tuner today optimizes a fixed-size `[f64; SPSA_DIMENSIONS]` (=8) vector, but
its primitives — `spsa_update`, `clamp_vector`, `perturb`, and
`SpsaRng::direction` — are **hardcoded to that size and read the global
`SPSA_SPECS` const directly**. They cannot take an ~18-weight eval vector as-is.
The core Rust refactor of this effort is therefore to **generalize those
primitives** to be size-agnostic (a slice, or const-generic `[f64; N]`) and to
take a `&[SpsaSpec]` argument instead of reading the global. Once generalized,
the same primitives serve both the existing 8-dim search vector (passing
`&SPSA_SPECS`) and the new eval vector (passing a new `EVAL_SPSA_SPECS`). This is
a mechanical but real refactor and a regression test pins that the search-param
campaign is unchanged by it.

The design then adds the eval projection: `eval_params_to_vector` /
`vector_to_eval_params` over `EvalParams`, and an `EVAL_SPSA_SPECS` table with one
spec (name, min, max, step) per tunable weight.

This run tunes **eval only**. Search-parameter tuning is a separate, existing
concern and stays off here so the eval signal is not confounded. The two tuners
share the generalized primitives but run over different vectors and specs.

The per-side plumbing mirrors mobility: `play_parameter_game` and the gate's
`play_mobility_game` currently take only `SearchParams` per side; they gain a
per-side `EvalParams` (via a `Searcher::set_eval_params`), so a match can pit two
eval configurations. During tuning, both perturbed sides also run with mobility
enabled (`mobility_scale = 100`) so the mobility weights are exercised.

## Running the campaign on Modal

To keep all the SPSA math in one place (Rust) and avoid a second, drift-prone
Python implementation of the RNG/perturb/step, the **entire SPSA loop stays in
Rust** — the generalized `run_spsa_campaign`, extended to tune `EvalParams`.
Modal's role is simply to run that Rust campaign in a container with a generous
timeout, exactly as the NNUE `train_net` function runs the trainer in one
container. This escapes GitHub Actions' 60-minute wall (Modal timeouts are hours)
without splitting the loop across languages.

The tradeoff, stated plainly: within that one container the per-iteration matches
are **sequential**, not fanned across containers, so campaign size is bounded by
the container's wall time. The first campaign is sized to fit (for example ~40
iterations of a modest per-iteration match, a couple of hours), which is enough
to propose candidate weights. This is acceptable because the campaign only
*proposes* — the definitive verdict comes from the powered, parallel gate below.
Fanning each iteration's match across containers for a larger campaign is a real
future optimization (it would require moving the loop into a Python driver that
calls thin Rust step commands, or an RPC), and is deliberately **out of scope for
this first slice**.

A Modal `spsa_tune` entrypoint (in `modal/app.py`) invokes a new
`engine-bench spsa-eval` command inside the container — it runs the campaign over
a generated opening set and prints the tuned `EvalParams` plus the per-iteration
trace. Launched the same way as the gate: `infisical run --env dev -- uv run
--with modal -- modal run modal/app.py::spsa_tune` (Infisical supplies the Modal
token; `PYTHONUTF8=1` on Windows).

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
  flip), plus a unit test that `EvalParams::default()` matches the current
  hand-set eval constants (the analog of `default_search_params_match...`).
- Unit tests for `eval_params_to_vector` / `vector_to_eval_params` round-tripping
  the default, and that out-of-range weights clamp to bounds.
- A regression test that the generalized SPSA primitives leave the existing
  search-param campaign unchanged (same tuned output for a fixed seed as before
  the refactor).
- A short in-process `spsa-eval` smoke test (a couple of iterations over a few
  positions) that returns in-bounds `EvalParams`, and an `eval-gate-file` smoke
  test that emits `W\tD\tL`.
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
