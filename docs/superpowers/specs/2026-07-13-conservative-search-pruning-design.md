# Conservative Search Pruning Design

## Goal

Reduce search work in clearly non-tactical interior positions while preserving
the engine's existing tactical and fixed-opponent regression gates.

## Scope

This slice adds three guarded alpha-beta heuristics to `Searcher::negamax`:

- Razoring at depth one: when a static evaluation is far below alpha, enter
  quiescence instead of expanding ordinary moves.
- Reverse futility pruning at shallow depth: when static evaluation is safely
  above beta, return that lower bound without generating moves.
- Late-move pruning: after ordered quiet moves have failed at shallow depth,
  skip remaining quiet moves.

These heuristics never run at the root, in check, on a mate-adjacent window,
or for tactical moves. Existing null-move pruning, LMR, check extensions,
passed-pawn extensions, PVS re-searches, and quiescence remain unchanged.

## Alternatives considered

1. Add unguarded pruning thresholds throughout the move loop. This yields the
   largest node reduction but risks tactical omissions and makes regressions
   difficult to attribute.
2. Add only reverse futility pruning. This is smallest but leaves the late
   quiet-move branch factor untouched.
3. Add all three, limited to shallow non-checking positions with material and
   tactical guards. Chosen: it is a bounded, independently testable search
   quality increment and matches the tracker priority.

## Architecture

Pure helpers calculate the fixed depth-scaled margins and the quiet-move
threshold. `negamax` uses one static evaluation per eligible node. It first
applies razoring or reverse futility before legal-move generation, then uses
late-move pruning only after at least one ordered move has been searched.

The helpers are private and unit-tested directly. Existing tactical and
gauntlet workflows remain the acceptance evidence; no Rust commands run
locally.

## Safety rules

- Do not prune when `in_check`, `depth > 3`, or either alpha/beta is within
  `MATE_SCORE - 1_024`.
- Do not late-prune captures, promotions, killer moves, the transposition-table
  move, counter moves, or passed-pawn extensions.
- Do not use pruning when the side to move has no non-pawn material; retain
  pawn endgame precision.
- A pruning return supplies no principal variation and is stored only through
  the existing normal search path when appropriate.

## Verification

Remote `Rusty Fish Tests` must pass the focused helper coverage. The tactical
suite must not regress from its committed baseline, and the fixed-opponent
gauntlet must complete and preserve its TSV evidence. CodeQL must remain green.
