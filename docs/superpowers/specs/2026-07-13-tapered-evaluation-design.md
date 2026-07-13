# Tapered Evaluation Design

## Goal

Replace the scalar hand-tuned evaluation accumulator with a parameterized,
tapered middlegame/endgame model while preserving the existing evaluator API
and search behavior at its boundary.

## Design

`TaperedScore` stores centipawn components for middlegame and endgame. An
`EvalParams` constant supplies separate piece values and piece-square weights.
Existing scalar positional features are represented as equal components unless
they are phase-specific: king safety contributes only to middlegame and king
centralization only to endgame. At the end of evaluation, material phase in
the existing 0..24 range linearly interpolates White-minus-Black and returns
the result from the side-to-move perspective.

## Alternatives considered

1. Add an NNUE dependency immediately. It would need a training corpus,
incremental feature state, and a reproducible model pipeline that do not yet
exist.
2. Keep scalar terms and tune constants in place. This does not establish the
phase-aware parameter interface needed for later Texel tuning or NNUE.
3. Recommended: pair-score tapering now, preserving feature extraction and
creating a small, deterministic parameter seam for later work.

## Validation

- Unit tests prove phase interpolation selects the middlegame/endgame endpoint
  and handles an equal blend.
- Regression tests prove an opening position prefers the middlegame piece
  balance and a bare king-and-pawn endgame values king centralization.
- GitHub-only workspace, tactical, gauntlet, throughput, and CodeQL checks
  must pass before merge.
