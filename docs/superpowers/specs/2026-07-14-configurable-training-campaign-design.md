# Configurable Training Campaign Design

## Goal

Let an NNUE training run be scaled up — more epochs and a tunable learning rate,
not just more data — so a campaign can actually fit deep-search labels well
enough to challenge the baseline through the SPRT gate.

## Problem

The `train` sub-command and `nnue-train` workflow only exposed the amount of
self-play (`plies`) and the label depth. Epochs (20) and the learning rate were
fixed, so a "larger campaign" only added data and could not fit the harder
deep-search targets tightly — the produced network stayed a poor approximation.

## Scope

- Add optional `train` arguments: epochs (5th) and learning rate (6th),
  overriding the defaults when supplied.
- Add an `epochs` input to the `nnue-train` workflow and raise the campaign
  defaults (`plies` 32 → 64, `epochs` 20 → 150) so a dispatched run is a real
  campaign that then feeds the existing SPRT gate.

Out of scope: adaptive optimizers, LR schedules, and network-size changes.

## Rationale

Fitting a depth-`N` search is a harder regression than the static evaluation; it
needs many more passes over the data. Making epochs and the learning rate
first-class knobs turns the existing train → gate workflow into a scalable
campaign runner without touching the trainer's maths.

## Safety rules

- Omitting the new arguments reproduces the previous defaults exactly.
- Epochs is floored at 1 and a non-positive learning rate is ignored.
- The change is plumbing only; the trainer, gate, and export paths are unchanged.

## Verification

Remote `Rusty Fish Tests / workspace` must pass (the existing trainer tests
cover the training maths). The larger campaign is exercised by dispatching the
`nnue-train` workflow, whose SPRT-gate step reports the verdict.
