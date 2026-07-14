# Trainer Gradient Clipping Design

## Goal

Make the NNUE trainer converge on large, high-variance deep-search labels
instead of diverging, so the deep-search teacher can actually produce a usable
network.

## Problem

The bootstrap trainer uses per-sample SGD with a fixed learning rate. Static
hand-crafted labels are small, so it converges. But depth-`N` search labels are
large (up to the `±10000` target clamp) and high-variance; the gradient scales
with the error magnitude, so a single large-target step explodes the output and
feature weights and the loss *increases* (observed: `735967 → 765240` on
depth-6 labels, and the resulting network lost 31/32 games in the SPRT gate).

## Scope

- Clip the error that drives the gradient to `±GRADIENT_ERROR_CLIP`
  (centipawns) — a Huber-style robust loss. Beyond the clip the loss is linear,
  so an extreme tactical target contributes a bounded step. The reported loss
  still uses the true squared error.
- Add a regression test that depth-labelled training reduces the loss.

Out of scope: adaptive optimizers (Adam), learning-rate schedules, and
win-probability (sigmoid) loss — larger changes that can follow if needed.

## Rationale

Clipping the gradient error puts every update — whether from a small static
target or a huge tactical one — in the same well-behaved regime that already
converges for static labels. It is a one-line, well-understood robustness fix
(Huber loss / gradient clipping) that needs no optimizer rewrite and no change
to the quantised-inference arithmetic.

## Safety rules

- The reported training loss remains the true mean squared error, so progress is
  measured honestly.
- Static-label training is unaffected: its errors are already within the clip.
- The clip changes only the gradient, not the network shape or export path.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including a new test that
deep-search-labelled training reduces the loss (guarding against the divergence
regression) alongside the existing static-label convergence test. End-to-end,
depth-6 training now reduces the loss instead of increasing it, and the produced
network is graded by the existing SPRT gate.
