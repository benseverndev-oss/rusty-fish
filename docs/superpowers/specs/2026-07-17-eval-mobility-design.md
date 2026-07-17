# Mobility Evaluation Term Design

## Goal

Add a piece-mobility term to the hand-crafted evaluation — the largest hand-crafted
term still missing after material and piece-square bonuses. Ship it behind a
runtime scale defaulting to off, and gate it with a self-play SPRT so `main`
never carries an unvalidated active eval change. This is the first slice of the
broader hand-crafted-eval-plus-tuning effort; SPSA-tuning the mobility weights is
a deliberate follow-up sub-project.

## Enabler: expose attacks from `engine_core`

`engine_core` computes piece attack bitboards (`knight_attacks`, `bishop_attacks`,
`rook_attacks`, `queen_attacks`) but keeps them private, and no eval term uses
them today. Add a public accessor:

```
pub fn attacks(&self, square: Square, piece: Piece) -> Bitboard
```

It returns the piece's pseudo-legal attack bitboard given the current all-piece
occupancy (`occupancy(White) | occupancy(Black)`), dispatching to the existing
private attack routines. Knight and king use their step tables; bishop, rook, and
queen use the sliding routines with occupancy. The mobility term only ever calls
it for knight, bishop, rook, and queen, so those four are the accessor's
contract; king is included because its step table is already at hand. Pawns are
out of the accessor's scope — there is no existing `pawn_attacks(square)` helper
(pawn captures are inlined per color in move generation), the term does not use
pawn mobility, and adding per-color pawn-attack code is unrequested. The accessor
either excludes the pawn kind from its contract or returns an empty bitboard for
it; it is never called with a pawn. This is a pure read-only accessor and changes
no existing behavior.

## The mobility term

In `evaluate_position`, for each knight, bishop, rook, and queen, mobility is the
number of pseudo-legal attack squares not occupied by a friendly piece:

```
mob = popcount(board.attacks(square, piece) & !board.occupancy(piece.color))
```

The term is a per-piece **tapered** bonus, centered by a per-piece offset so an
average-mobility piece scores near zero (the offset keeps the term from silently
re-rewarding mere piece presence, which material already covers):

```
score(kind) = TaperedScore {
    middlegame: weight_mg[kind] * (mob - offset[kind]),
    endgame:    weight_eg[kind] * (mob - offset[kind]),
}
```

Starting hand-set values (these are tuned in the follow-up SPSA sub-project; they
only need to be sane here):

| piece  | weight_mg | weight_eg | offset |
|--------|-----------|-----------|--------|
| knight | 4         | 4         | 4      |
| bishop | 3         | 3         | 6      |
| rook   | 2         | 4         | 7      |
| queen  | 1         | 2         | 13     |

King and pawn mobility are not counted. Mobility is pseudo-legal (attack squares
minus friendly occupancy); it is **not** "safe mobility" (excluding squares
attacked by enemy pawns). Both are natural later refinements and are out of scope.

The term is added through the existing `EvalFeatures`/`TaperedScore` interpolation
so it participates in tapering exactly like the other terms.

## The runtime scale and default-off

Evaluation gains a mobility scale in the range 0–100. The mobility contribution is
multiplied by `scale` and divided by 100 (integer), so `scale = 100` is full
strength (`x * 100 / 100 = x`) and `scale = 0` removes the term entirely
(`x * 0 / 100 = 0`).

**Where the scale lives — `SearchParams`.** The scale is a new
`SearchParams.mobility_scale` field. This is the deliberate, load-bearing choice:
the self-play harness's `play_parameter_game` sets `SearchParams` per side (via
`set_search_params`) and nothing else, so `SearchParams` is the only per-side
lever, and putting the scale there is what lets the gate pit `scale = 100` against
`scale = 0` without changing the harness. It must **not** go in `SearchOptions`
(which `play_parameter_game` leaves at default for both sides, so it could not
differ per game).

`SearchParams` is also the SPSA-tunable struct, projected onto a fixed
`[f64; SPSA_DIMENSIONS]` by `search_params_to_vector` / `vector_to_search_params`.
To keep SPSA-tuning of mobility out of this slice:

- `SPSA_DIMENSIONS` stays at its current value and `search_params_to_vector` is
  unchanged — `mobility_scale` is **excluded from the SPSA vector**, so the SPSA
  search-parameter tuner never perturbs it.
- `vector_to_search_params` reconstructs `SearchParams` from the vector; since the
  vector does not carry `mobility_scale`, that function sets it to the default
  (0). This is correct: the vector round-trip is used only inside SPSA
  search-parameter tuning, where mobility is off anyway, and the existing
  `spsa_vector_round_trips_default_params` test still passes because the default
  is 0. The mobility gate never round-trips through the vector — it sets
  `mobility_scale` directly on each side's `SearchParams`.

**Threading and default.** `Searcher::evaluate` passes `self.params.mobility_scale`
into `evaluate_position`, which gains a scale parameter. The public
`hand_crafted_evaluation(board)` keeps its signature by evaluating at scale 0 (its
callers — tests, NNUE labeling — want the stable hand-crafted baseline).

The default `mobility_scale` is **0**. Mobility therefore ships disabled:
`SearchParams::default()` stays byte-identical (preserving the existing
"Default reproduces the hand-set constants exactly" invariant and its guarding
test), and `main`'s engine behavior is unchanged until the gate proves the term,
at which point a one-line change flips the default to 100. When the scale is 0,
evaluation is byte-for-byte identical to today's, which a test pins.

Exposing the scale over UCI is out of scope; the harness sets it in-process.

## The gate: self-play SPRT

A new `engine-bench` command plays mobility-on against mobility-off and reports
the existing SPRT TSV. Both sides are the same engine at the same search budget;
the only difference is the mobility scale (100 vs 0), so the SPRT isolates the
term's Elo. It plays the existing opening suite color-swapped, summarizes into a
`MatchScore`, and emits `sprt_tsv_report` (wins/draws/losses, Elo estimate, LLR,
bounds, decision) plus a per-game TSV, matching the other campaigns.

Reuse the existing self-play machinery: `play_parameter_game` already plays a
candidate `SearchParams` against a baseline `SearchParams` and the surrounding
code summarizes into a `MatchScore` and runs `sprt`. The mobility gate is that
machinery with the candidate params equal to `SearchParams::default()` but
`mobility_scale = 100` and the baseline equal to `SearchParams::default()`
(`mobility_scale = 0`) — identical in every other field, so the SPRT isolates the
mobility term.

A dispatch-only workflow runs it, like the other strength campaigns — self-play
over many games is CPU-heavy and must not run on ordinary PR CI. The acceptance
criterion is an SPRT that accepts H1 (or, if inconclusive after the budgeted
games, a clearly positive Elo estimate that a follow-up run can confirm).

## Sequence

1. This slice: the attacks accessor, the mobility term, the default-off scale, the
   gate command, and the dispatch-only workflow. Merged with mobility off.
2. Dispatch the gate; read the SPRT result.
3. If the term gains, a one-line PR flips the default scale to 100. If it does not,
   the term stays off (or its weights are revisited) — no revert needed, because
   the default-off scale already neutralizes it.

## Verification

- `engine_core`: `attacks()` on known positions — a knight on d4 attacks 8
  squares, a knight on a1 attacks 2, a rook's slide stops at the first blocker,
  and a bishop's diagonal is masked by occupancy.
- `engine-search`: the mobility term produces the expected sign and magnitude on a
  position with a lopsided mobility difference; and `scale = 0` reproduces today's
  evaluation exactly (a regression pin that mobility is truly inert when off).
- `engine-bench`: a small self-play smoke test that the gate command runs and
  produces a well-formed SPRT report.
- The throughput benchmark quantifies the NPS cost of computing slider attacks per
  piece each evaluation; a measurable but bounded drop is expected and acceptable.
- All validation runs in GitHub Actions; Cargo is never run locally.

## Out of scope

- SPSA-tuning the mobility weights (the next sub-project).
- Safe mobility, king mobility, pawn mobility.
- Exposing the mobility scale over UCI.
- Any other new evaluation term.
