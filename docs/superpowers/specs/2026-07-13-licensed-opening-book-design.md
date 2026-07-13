# Licensed Opening Book Design

## Goal

Ship a reproducible, configurable opening book built from openly licensed,
high-quality games, with source provenance and measurable coverage.

## Source and provenance

The pipeline uses the Lichess Open Database's CC0 standard-game export. A
versioned manifest pins the export URL, release month, SHA-256, license URL,
and selection rules: standard chess, both players rated at least 2200, rated
games only, and the first 16 plies. The repository includes a compact,
attributed extraction fixture generated from that manifest for CI; full exports
are downloaded only by an explicit refresh command or manually dispatched
workflow, never ordinary PR CI.

## Data flow

The generator parses legal PGN mainlines, starts from the standard position,
and aggregates every reached position by canonical FEN signature. This makes
transpositions share statistics naturally. Each legal next move receives a
weight of `frequency * (0.5 + score_fraction)`, where score fraction is 1,
0.5, or 0 for White wins, draws, or Black wins from the moving side's
perspective. Moves with fewer than three observations are excluded. Entries
are sorted by descending weight, then UCI move for deterministic ties.

The emitted `rusty-fish-book v2` file records each FEN with `uci:weight`
alternatives. The loader remains backward-compatible with v1 and validates all
moves against the FEN. `Book Variety` is a UCI spin option from 0 to 100:
zero always selects the highest-weight move; larger values deterministically
select among cumulative weights using the position hash and configured
variety, so repeated searches remain reproducible.

## Verification

- Unit tests cover PGN filtering, transposition aggregation, side-relative
  scoring, stable serialization, v1 compatibility, and variety boundaries.
- A committed manifest and fixture produce a byte-identical v2 book in CI.
- The book workflow reports source count, accepted games, positions, entries,
  alternatives, and hit rate over a fixed opening-position suite.
- UCI protocol tests verify `BookPath` and `Book Variety`; missing/invalid
  books safely preserve ordinary search.
