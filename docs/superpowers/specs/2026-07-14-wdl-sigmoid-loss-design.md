# WDL Sigmoid Loss Design

## Goal

Replace the trainer's raw-centipawn mean-squared-error objective with the
standard NNUE **win-probability (WDL) loss**, so extreme evaluations stop
dominating the fit and the network can approach — and ideally beat — the
hand-crafted baseline through the SPRT gate.

## Problem

A larger campaign (more plies, epochs) did not improve the gated strength (it
plateaued ~350 Elo below baseline). The bottleneck is the objective: raw
centipawn MSE is dominated by a few extreme, ±10000-clamped tactical targets
that a small network cannot represent, so training chases outliers instead of
learning useful positional structure.

## Scope

- Squash prediction and target through `sigmoid(cp / WDL_SCALE)` and take the
  mean-squared error in win-probability space. The gradient is naturally bounded
  (sigmoid output in `(0, 1)`, derivative ≤ 0.25), so no ad-hoc error clipping is
  needed — the Huber-style clip is removed.
- Because the WDL gradient is small, raise the default learning rate to match its
  scale (the rate is already configurable per campaign).

Out of scope: reparameterising activations, adaptive optimizers, larger
networks, and non-search teachers — separate levers if the gate still fails.

## Rationale

The WDL-sigmoid loss is the objective every strong NNUE trainer uses precisely
because a chess evaluation's usefulness is its win/draw/loss implication, not its
exact centipawn value. Bounding targets to `[0, 1]` makes a decisive-but-not-huge
advantage and a mate-in-20 contribute comparable, well-conditioned gradients, so
the fit reflects what actually matters for play.

## Safety rules

- The change is confined to the trainer's loss and learning-rate default; the
  network shape, quantised-inference arithmetic, and export path are unchanged.
- `win_probability` is numerically stable and bounded in `(0, 1)`.
- The learning rate stays configurable, so campaigns can tune it.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including the trainer tests
updated to win-probability units: training reduces the loss, deep-search labels
do not diverge, and the fitted network beats a zero-centipawn (0.5 win
probability) predictor. End-to-end, a depth-6 campaign is trained and graded by
the SPRT gate; the resulting verdict is recorded on the pull request.
