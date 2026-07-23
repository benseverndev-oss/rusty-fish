# Search Telemetry (Phase 1) — Design

Status: approved to build (2026-07-23). First project of the learned-search
program (`docs/RESEARCH-DIRECTION.md`). Foundation for Phase 2 (learned LMR).

## Goal

Emit a per-move-decision dataset from the live alpha-beta search — the training
substrate for learned search control. Phase 1 delivers the **instrumentation** and
a **dataset generator**, nothing learned yet.

## The inviolable invariant

**Telemetry must never change a search decision.** With telemetry disabled the
search is byte-for-byte what it is today; with it enabled the *only* difference is
that records are collected (and wall-clock is slower). Same best move, same score,
same node count. A test enforces this.

This mirrors the existing NNUE `debug_assert` discipline (incremental == refresh):
instrumentation is *guarded*, not hoped.

## Mechanism

- Add `engine-search/src/telemetry.rs`: a `MoveDecision` record struct and a
  `TelemetryCollector` (a `Vec<MoveDecision>` plus a bounded cap to avoid unbounded
  growth on deep searches).
- `Searcher` gains `telemetry: Option<TelemetryCollector>`, default `None`. A
  `with_telemetry()` / `take_telemetry()` pair enables collection and drains records.
- When `Some`, the `negamax` move loop appends one record per move considered.
  When `None`, the collection site is a single predictable branch — no allocation,
  no perturbation.

## Where it hooks (engine-search/src/lib.rs, `negamax`, ~1271-1358)

Per move in the loop, capture:

**Context (model inputs):**
- `depth`, `ply`, `move_index`
- `is_quiet` (line 1290), `is_priority_move` (1292-1299)
- `pv_node` = `beta - alpha > 1`
- `gives_check` = `board.in_check(side)` after `nnue_make` (the position the move
  reaches; the extension at 1309 already computes in-check of the child)
- `static_eval` (available at the node; the value used by razoring/RFP)
- `extension` (1309-1311), `reduction` (1313)

**Decision:**
- `lmp_pruned` — the move hit the late-move-pruning `break` at 1300-1307 (record it
  with outcome fields zeroed / unknown; the counterfactual "was pruning safe?" is
  Phase 4, out of scope here).
- `reduction` applied, `extension` applied.

**Outcome (labels), for searched (non-pruned) moves:**
- `raised_alpha` = `score > alpha` (pre-update alpha)
- `caused_cutoff` = `alpha >= beta` after the update (1352)
- `needed_lmr_research` = the reduced-search re-search at 1331 fired (reduction was
  too aggressive for this move)
- `needed_pvs_research` = the full-window re-search at 1337 fired
- `subtree_nodes` = `self.nodes` delta across this move's search (snapshot before the
  first child search, diff after the last re-search)

Record fields are fixed-width primitives (ints/bools) so the row serializes to one
TSV line cheaply.

## Dataset generator

Add an `engine-bench` subcommand `gen-search-telemetry <positions_or_-> <depth>`
that mirrors `gen-eval-positions` / `label-sf`:
- Reads FENs (one per line; `-` = stdin).
- For each, builds a `Searcher` with telemetry enabled, runs a fixed-depth search,
  drains the records, and prints them as TSV to stdout (one row per `MoveDecision`,
  with a leading position id so rows can be grouped).
- Prints a header row once. Malformed FENs are counted and skipped (same ergonomics
  as `label-fens`).

This reuses the whole existing fan-out story: a later Modal entrypoint shards FENs
across containers exactly like the labeling pipeline, and the `public-full` corpus
(394M FENs) is a ready position source.

## TSV schema (v1)

```
pos_id  depth  ply  move_index  is_quiet  is_priority  pv_node  gives_check
static_eval  extension  reduction  lmp_pruned  raised_alpha  caused_cutoff
needed_lmr_research  needed_pvs_research  subtree_nodes
```

Booleans as 0/1; scores/counts as ints. Documented in the subcommand help.

## Tests (engine-bench + engine-search)

1. **Byte-identical invariant (critical):** for a set of FENs, run a fixed-depth
   search with telemetry `None` and with telemetry `Some`; assert identical
   `SearchResult` (best move, score, depth) **and** identical final node count.
2. **Records well-formed:** every record has `move_index` within the node's move
   count, `reduction`/`extension` in their known ranges, and `caused_cutoff`
   implies `raised_alpha`.
3. **LMP record present:** a position engineered to trigger late-move pruning emits
   at least one `lmp_pruned` record.
4. **Cutoff consistency:** at a node that fails high, exactly the cutoff move has
   `caused_cutoff = 1` and it is the last searched move.

## Non-goals (Phase 1)

- No learned model, no `LearnedCorrection` in the reduction formula (Phase 2).
- No counterfactual verification of pruned/reduced moves (Phase 4).
- No Modal fan-out entrypoint yet (trivial follow-up once the subcommand emits TSV).
- No uncertainty / policy targets (later phases).

## Verification

CI is remote-only for this repo (rustc ≥ 1.95 needed for `engine-core` dev-deps),
so build + test run in GitHub Actions, not locally. The byte-identical test is the
gate: if it fails, the instrumentation is perturbing search and must be fixed
before anything else.
