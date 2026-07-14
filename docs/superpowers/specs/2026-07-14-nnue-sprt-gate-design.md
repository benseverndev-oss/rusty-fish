# NNUE SPRT Gate Design

## Goal

Close the loop from "we can train networks" to "we know whether a network is
actually stronger": a mechanism that plays the NNUE-equipped engine against the
current hand-crafted-eval engine and returns an SPRT verdict, so a trained net
is only ever adopted as the default if it measurably beats the baseline.

## Scope

- `run_nnue_gauntlet` (`engine-bench`): plays the candidate (NNUE loaded) versus
  the baseline (hand-crafted eval) over each position and both colours, both
  sides searching at the same depth.
- A `nnue-sprt <network> [depth]` sub-command that loads a network, plays the
  gauntlet over the standard position set, and prints the SPRT TSV plus a
  win/draw/loss and decision summary.
- A gate step in the `nnue-train` workflow so a run trains a network and then
  reports its SPRT verdict against the baseline in one pass.

Out of scope: **adopting** a network as the default (embedding a net in the
binary and loading it automatically). That happens only after a network passes
this gate, and shipping a non-passing net would regress the engine, so no net is
embedded here.

## Rationale

The engine already had an SPRT implementation and a self-play match harness, but
only for hand-crafted-vs-hand-crafted (parameter) comparisons. Strength from
NNUE can only be trusted through a head-to-head measurement, so the missing
piece is an NNUE-vs-baseline match feeding the existing SPRT. This is the
decision procedure the whole training effort needs.

## Architecture

`play_nnue_game` mirrors `play_parameter_game` but sets a network on the
candidate searcher and leaves the baseline on the hand-crafted evaluation; both
search at the same depth for a fair comparison. `run_nnue_gauntlet` schedules
both colours per position. The sub-command feeds `summarize` into the existing
`sprt`/`sprt_tsv_report`. The workflow runs train then gate.

## Safety rules

- The gate never mutates the default engine; it only measures and reports.
- Candidate and baseline search at equal depth, so the verdict reflects
  evaluation quality, not a depth handicap.
- Adoption of a network as the default is contingent on a passing verdict and is
  a separate, explicit change.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including a test that the
gauntlet plays both colours for each position and produces a scored result. The
`nnue-train` workflow demonstrates the end-to-end train → gate loop. A small
in-CI gate is inconclusive by design (few games); a decisive verdict needs a
larger campaign, which the workflow can scale up.
