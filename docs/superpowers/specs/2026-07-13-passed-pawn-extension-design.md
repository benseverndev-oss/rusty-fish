# Passed-Pawn Extension Design

## Goal

Extend a search by one ply when an actually passed pawn advances to the seventh
rank for White or the second rank for Black, without changing extension policy
for ordinary pawn moves, captures, promotions, or checks.

## Design

Extract the passed-pawn geometry currently embedded in `pawn_structure_bonus`
into `is_passed_pawn(board, square, color)`. The predicate is true only when
no enemy pawn is ahead of the square on its own or adjacent files.

`passed_pawn_extension(board, mv)` will return one only when the moving piece
is a pawn, the destination is rank six for White or rank one for Black
(zero-based ranks), the move is not a promotion, and the pawn is passed before
it moves. Search will calculate this before `make_move`, then combine it with
the existing check extension using `max`, never adding more than one ply.

## Alternatives considered

1. Extend every pawn push to the penultimate rank. This is inexpensive but
   wastes depth on blocked or capturable pawns.
2. Duplicate the evaluation loops in search. This creates two definitions of
   passed-pawn status that can drift.
3. Recommended: a shared predicate used by evaluation and the extension
   policy. It is deterministic, bounded, and independently testable.

## Validation

- A test proves qualifying white and black pawn pushes extend.
- Tests prove a pawn blocked by an opposing pawn on an adjacent forward file,
  a non-pawn move, and a promotion do not extend.
- Existing tactical, gauntlet, throughput, workspace, and CodeQL workflows run
  on GitHub only; no local Cargo command is used.
