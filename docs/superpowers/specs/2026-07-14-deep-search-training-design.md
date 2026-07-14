# Deep-Search Training Target Design

## Goal

Give the NNUE trainer a stronger teacher than the static hand-crafted
evaluation, so a trained network can begin to *exceed* hand-crafted play rather
than merely match it — the first step whose payoff is real strength.

## Scope

- Add an optional `label_depth` to `generate_training_samples`. With `None` the
  label is the static hand-crafted evaluation (unchanged). With `Some(depth)`
  the label is a depth-`depth` search score (hand-crafted evaluation at the
  leaves), clamped away from mate values.
- Surface it through the `train` sub-command (optional 4th argument) and the
  `nnue-train` workflow (a `label_depth` input, default 6).

Out of scope: game-outcome (WDL) and external-engine teachers; network
architecture changes; adopting a trained net as the default (SPRT-gated).

## Rationale

Distilling a depth-`N` search into a static network teaches it tactics and
short-term dynamics the static evaluation cannot see in one ply. It reuses the
existing search unchanged as the labeller and needs no external data, so it is
the highest-value, lowest-risk upgrade to the training signal.

## Architecture

`generate_training_samples` keeps a single reusable `Searcher` (hand-crafted
evaluation, so there is no teacher/student circularity) and, per recorded
position, either reads the static evaluation or runs a fixed-depth search and
takes its score. Targets are clamped to a well-conditioned range. Everything
else — data generation by seeded random self-play, the float SGD trainer, and
quantised `RFNN` export — is unchanged.

## Safety rules

- `label_depth = None` (and workflow value `0`) reproduces the previous static
  behaviour exactly.
- The labeller uses the hand-crafted evaluation, never an NNUE, so training is
  not self-referential.
- Search-score targets are clamped so mate scores cannot distort the fit.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including a test that
depth-labelled targets differ from static targets on some positions and the
existing loss-reduction test. The default engine is unaffected (training is an
offline tool). A trained network is adopted as the default only behind an
external Stockfish SPRT.
