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
queen use the sliding routines with occupancy; pawns return their capture-attack
squares. This is a pure read-only accessor and changes no existing behavior.

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
multiplied by `scale` and divided by 100, so `scale = 100` is full strength and
`scale = 0` removes the term entirely.

- The scale lives where the self-play harness can set it per side, so a game can
  pit `scale = 100` against `scale = 0` with everything else identical. It is
  threaded from the searcher into `evaluate_position`; the public
  `hand_crafted_evaluation` keeps its current signature by evaluating at the
  default scale.
- The default scale is **0**. Mobility therefore ships disabled: `main`'s engine
  behavior is unchanged until the gate proves the term, at which point a one-line
  change flips the default to 100. When the scale is 0, evaluation is byte-for-byte
  identical to today's, which a test pins.

Exposing the scale over UCI is out of scope; the harness sets it in-process.

## The gate: self-play SPRT

A new `engine-bench` command plays mobility-on against mobility-off and reports
the existing SPRT TSV. Both sides are the same engine at the same search budget;
the only difference is the mobility scale (100 vs 0), so the SPRT isolates the
term's Elo. It plays the existing opening suite color-swapped, summarizes into a
`MatchScore`, and emits `sprt_tsv_report` (wins/draws/losses, Elo estimate, LLR,
bounds, decision) plus a per-game TSV, matching the other campaigns.

Reuse the existing self-play machinery: the harness already plays a candidate
configuration against a baseline configuration and summarizes/SPRTs the result.
The mobility gate is that machinery with the two configurations differing only in
mobility scale.

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
