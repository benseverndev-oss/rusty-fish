# Singular Extension Design

## Goal

Extend only a transposition-table move that is demonstrably singular at a deep
interior node, improving tactical depth without broadening speculative search.

## Contract

`Searcher::negamax` will accept an optional excluded move used only by a
verification search. A singular candidate requires an exact TT entry with a
move that is close enough to the requested depth to order the node, but still
shallower than that depth so the existing TT cutoff has not already resolved
the node. It also requires depth at least six, no check, no mate-adjacent TT
score, and non-pawn material.

Before searching that TT move normally, the engine searches all *other* moves
at `depth / 2` with a null window below `tt_score - 32`. If no alternative
reaches that bound, the TT move receives one extra ply. The verification search
does not use singular extension recursively and does not store an incomplete
excluded-move result as an exact TT entry.

## Safety

- Root, quiescence, check, pawn-only, mate-adjacent, and shallow nodes never
  enter singular verification.
- The excluded move is skipped after ordering, preserving all other move
  ordering and normal alpha-beta behavior.
- Stop signals terminate verification through the existing cancellation path.
- Existing extension caps, PVS, LMR, null move, tablebases, and pruning remain
  in effect for ordinary searches.

## Verification

Private policy tests cover eligibility and score-margin behavior. Existing
workspace, tactical-suite, fixed-opponent gauntlet, throughput, and CodeQL
GitHub checks provide remote-only acceptance evidence.
