# Learned Move Ordering (Phase 4) — Plan

**Goal:** Learn a move-ordering policy that corrects the classical order, mirroring
how learned LMR corrects the classical reduction:

```
order_score = classical_move_order_score(...) + clamp(learned_correction, ...)
```

Ordering only — alpha-beta is untouched. Better ordering compounds: good moves
searched first means more cutoffs at the first move, smaller trees at equal strength.

## Why this is the next lever

The high-power (16384-game) LMR sweep (PR #96) showed learned LMR is
**action-limited, not prediction-limited**: the model fires only where classical LMR
already reduces and the correction is clamped to four discrete decisions, so a
better-calibrated probability (v2's val AUC 0.9390 → 0.9463) bought **no measurable
Elo**. Move ordering scores *every* move — a far wider action space — so a better
predictor there has room to express itself. This is the bet Research Direction §2
names and the LMR result points to.

## The arc (each step its own gated PR, as Phases 1–2 were)

1. **Instrument** *(this PR)*. Extend the search telemetry with the move as the
   *orderer* saw it: `order_score` (the classical baseline), `see`, `mover_piece`,
   `captured_piece`. Append-only v3 columns; the byte-identical telemetry invariant
   test proves the search is unchanged. Unblocks a policy dataset — the labels
   (`caused_cutoff`, `raised_alpha`, `move_index`) already exist in v1/v2.
2. **Train** — *done, in Rust.* `engine-bench`'s `train-policy` fits the policy
   in-process (no Python, no Modal): standardize → class-weighted BCE-with-logits →
   Adam → 90/10 val split → AUC, a port of `train_lmr.py`'s math. It reads a
   `gen-search-telemetry` TSV, resolves the ordering-time feature columns by name, and
   exports an `RFPO` binary — the `RFLM` format's sibling (own magic, same shape),
   defined by `engine-search/src/policy_model.rs` so the trainer and the engine's
   inference agree by construction. A first cut uses the pointwise `caused_cutoff`
   target; a pairwise/listwise ranking loss is a later refinement. Keeping this in Rust
   removes Modal from the critical path for the small nets — the whole train→gate loop
   can run in one binary in the sandbox.
3. **Wire** behind a neutral toggle. `PolicyModel` on `Searcher`, `None` by default →
   byte-identical search (same guarantee, same test shape as learned LMR). When
   installed, `move_order_score` adds a clamped learned correction; the correction is
   a *tunable* residual so the policy stays a pure predictor and the search reads
   tunables from `SearchParams`.
4. **Gate.** SPRT vs classical ordering at equal movetime, sharded over generated
   openings (the existing `gate-file` / campaign machinery). Report Elo **and** node
   count at equal strength — ordering wins should show as a smaller tree even when Elo
   is flat. Adopt only on an SPRT pass.

## Feature notes (step 2+)

Ordering-time features only (everything a move has *before* it is searched): the v3
columns above, plus the v1/v2 flags already captured (`is_tt_move`, `is_killer`,
`is_counter`, `is_capture`, `is_promotion`, `gives_check`, `history_score`, `pv_node`,
`depth`, `ply`, `node_in_check`, `static_eval`, `tt_depth`). `move_index` is the
classical rank — usable as a feature but it partly encodes the label, so prefer
`order_score` as the residual baseline and treat ordering as *re-ranking within a
node*, not absolute regression.

## Guardrails (unchanged from Phases 1–2)

- Telemetry never perturbs the search (enforced test).
- New model formats validate dims on load and swap the bundled asset in the same
  commit that changes the feature set.
- Nothing is adopted without an SPRT pass at equal movetime over diverse openings.

## Results so far (2026-07-25)

**Step 2 — the signal is real.** Trained on real telemetry (100.2M depth-8
move-decision rows from 2000 openings; 5M kept at stride 8), `train-policy` reaches
**val AUC ≈ 0.933** (val_acc 0.847, base_rate 0.232), stable across seeds (0.9324 /
0.9330). "Search this move first" is clearly learnable from ordering-time features —
well past the 0.6 bar. Caveat: the feature set includes `order_score`/`move_index`
(the classical rank), so part of that AUC is the model *reproducing* an already-good
ranker; AUC measures prediction, not incremental value over classical.

**Step 4 — the gate exists (`engine-bench gate-policy`).** Candidate installs the
RFPO via `set_policy_model(Some(..))`, baseline `None`, both otherwise the shipped
default engine, equal movetime, color-swapped, sharded over an openings file; emits
`W\tD\tL` on stdout (feed `sprt`) and a per-side avg-depth / avg-nodes / node-ratio
summary on stderr.

**Step 4 — first result: inference cost dominates.** Smoke over 12 openings (24
games) at 20 ms:

| order_bound | W-D-L | Elo (24g, wide CI) | node_ratio (cand/base) | depth Δ |
|---|---|---|---|---|
| 0 (ordering unchanged) | 3-5-16 | −211 | 0.680 | −0.42 |
| 500 | 4-7-13 | −137 | 0.676 | −0.31 |
| 4000 (default) | 2-5-17 | −255 | 0.677 | −0.50 |

The node_ratio is ~0.68 **regardless of bound** — even at `bound=0`, where ordering
is byte-identical to classical. So the regression is dominated not by bad ordering but
by the **per-move inference tax**: `order_moves` builds the 29-feature vector (incl.
`static_exchange_evaluation`) and runs the MLP forward pass for *every move at every
main-`negamax` node*, costing ~32% NPS → ~0.4 plies shallower at equal movetime. A
bound sweep (best at 500) can't overcome that. The Elo numbers are directional (24
games), but node_ratio is essentially deterministic and robust.

**Next before adoption:** make inference cheap enough to pay for itself — e.g. reuse
the classical SEE already computed in `move_order_score` instead of recomputing it,
lazy/partial feature building, or apply the correction only to a top-k slice of moves
rather than every move at every node — then re-gate at a fuller game count. The
offline signal (AUC 0.93) says the policy *knows* something; the open question is
whether re-ranking can be made cheap enough for that to net positive at real TC.
