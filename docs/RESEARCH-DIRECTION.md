# Rusty Fish Research Direction
## Beyond Stockfish Through Learned Search

> Adopted 2026-07-23 as the engine's primary research direction, replacing the
> "scale the NNUE / reach Stockfish parity" track. Motivating evidence: a
> data-scale sweep showed the current NNUE is **evaluation-saturated** — 3M→17M
> Stockfish-labeled positions and a 1024→1536 hidden bump both yielded ~0 Elo
> (the larger net had the *lowest* val loss and played *worst*). Marginal returns
> are no longer in the evaluator; they are in **how search spends computation**.

## Executive Summary

Rusty Fish has already reached the point where incremental "Stockfish parity"
work produces diminishing returns. The engine already contains nearly every major
modern alpha-beta technique: PVS/Negamax, transposition table, aspiration windows,
null-move, razoring, reverse-futility, LMR/LMP, singular extensions, passed-pawn
extensions, killer/history heuristics, Lazy SMP, Syzygy, NNUE, SPSA tuning, and
automated training + SPRT gating.

At this stage, simply implementing the remaining Stockfish heuristics is unlikely
to ever surpass Stockfish itself. The opportunity is to make Rusty Fish
**fundamentally different**.

## North Star

Rather than building a better evaluator, build an engine that **learns how to
search.** Instead of asking "How good is this position?", ask "How should
computation be spent in this position?" This moves machine learning from
evaluating leaves to **controlling the search itself**.

## Research Priorities

### 1. Learned Search Control (Highest Priority)

Train a small neural network that controls search decisions rather than only
evaluating positions. Possible outputs: LMR adjustment, pruning confidence,
extension confidence, probability a move raises alpha, search instability,
expected value of deeper search. Rather than replacing classical heuristics,
correct them:

```
Reduction = ClassicalFormula(...) + LearnedCorrection(...)
```

Clamp corrections initially to `-1 ... +2` plies. This keeps search safe while
allowing measurable improvements.

### 2. Learned Move Ordering

Train a policy network (deep Stockfish searches or self-play) for root ordering,
quiet-move ordering, LMR ordering, singular-extension candidate selection, and
pruning order. Better ordering compounds because good moves are searched first.

### 3. Search Uncertainty

Have the NNUE return `evaluation` **and** `uncertainty` (probability the eval
changes after deeper search, variance across deeper searches, probability the
best move changes, tactical volatility). Use uncertainty to drive aspiration
window size, extensions, reductions, pruning aggressiveness, root time
allocation, and quiescence depth.

### 4. Counterfactual Search Training

Train from search *decisions*, not just positions. For sampled nodes: run the
normal search, then perturb one decision (prune / don't, reduce ±1 ply, reorder)
and compare against deeper verification. Train the model to predict: was the
shortcut safe? how many nodes were saved? what tactical error resulted? This
directly optimizes search efficiency rather than evaluation accuracy.

### 5. Rich Search Dataset

Beyond `position, evaluation`, capture: best move, top-k moves, principal
variation, search depth, nodes searched, move ordering, alpha raises, fail-high
info, tactical volatility, uncertainty, search stability.

### 6. Dual Evaluators

A fast evaluator used everywhere and a strong evaluator used selectively (PV
nodes, tactical/uncertain positions, root moves, unstable evals). A learned gate
decides when the expensive evaluator is worthwhile.

### 7. Root Search as Resource Allocation

Treat each root move as an investment problem (score, uncertainty, node count,
stability) and allocate search budget where extra computation is expected to
matter most — adaptive allocation instead of uniform iterative deepening.

## NNUE Research (secondary to search control)

- **King-relative features** — HalfKA, learned king buckets, adaptive king
  embeddings (better king-safety understanding).
- **Multi-task training** — predict cp, WDL, policy, uncertainty, tacticality,
  phase together; shared representations often beat single-task.
- **Mixture of experts** — opening/middlegame/tactical/endgame specialists with a
  lightweight gate.
- **Quantization-aware training** — fake quantization, per-channel scaling, sparse
  nets, int8 feature tables: larger effective models at fixed inference cost.

## Engine Infrastructure

- **Lock-free transposition table** — replace the lock-based shared TT for better
  scaling and more meaningful learned-search experiments.
- **Modern history heuristics** — continuation / capture / pawn / correction
  history; these become inputs for learned search.
- **Better benchmarking** — equal nodes, equal wall time, multiple time controls,
  SPRT, tactical suite, throughput (not equal-depth alone).
- **Search telemetry** — cutoff source, move index, reduction amount, re-search
  frequency, TT usefulness, pruning mistakes, node distribution, PV stability,
  evaluator latency. Without telemetry, learned search is hard to improve.

## Development Roadmap

- **Phase 1 — Instrumentation.** Collect millions of search decisions: node
  context, move, reduction, pruning, verification result, nodes, alpha raises.
- **Phase 2 — Learned LMR.** Predict safe reductions; bound corrections to small
  values; evaluate Elo, node count, tactical regressions.
- **Phase 3 — Uncertainty-aware search.** Use uncertainty for aspiration windows,
  time management, extensions, reductions.
- **Phase 4 — Policy network.** Ordering only; does not replace alpha-beta.
- **Phase 5 — Unified search controller.** One network predicts policy,
  uncertainty, reduction, pruning confidence. Classical search remains the
  framework; ML controls computation.

## What Not To Prioritize (low expected return)

Replacing alpha-beta with MCTS; transformers evaluated at every node; endless
SPSA tuning; opening-book work; trainer micro-optimizations; continually scaling
identical NNUE architectures.

## Ultimate Vision

Optimize **decision quality per unit of computation.** Instead of learning "How
good is this position?", learn "What computation should happen next?" — prune,
reduce, extend, verify, invoke a stronger evaluator, continue, or stop. This
turns machine learning from an evaluation function into a **search controller**,
Rusty Fish's strongest opportunity to surpass Stockfish rather than reproduce it.
