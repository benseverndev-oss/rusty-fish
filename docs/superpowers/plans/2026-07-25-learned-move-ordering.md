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
