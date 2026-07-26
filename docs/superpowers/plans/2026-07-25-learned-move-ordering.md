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

## Making inference cheap (2026-07-25, cont.)

The per-move inference tax was the blocker, so it got three fixes, each byte-identical
to the previous ordering (all invariant tests still green):

1. **Hoisted standardization** in `PolicyModel::cutoff_prob` — the
   `(feat − mean) · scale` was recomputed inside the `hidden`-way loop; now computed
   once per feature. Bit-identical (the RFPO round-trip test pins it).
2. **SEE computed once per capture** — `move_order_score` and `policy_features` each
   SEE'd every capture; `order_score_parts` now threads the one SEE (and the capture
   flag) into the features. A drift-guard test pins the byproducts to the standalone
   `is_capture_move` / `static_exchange_evaluation`.
3. **Top-K re-ranking** (`SearchParams::policy_order_top_k`, tunable, `0` = all) — the
   forward pass is the hot loop's dominant cost and cutoffs come from the front of the
   order, so the policy now re-ranks only the top `K` classically-ordered moves and
   leaves the tail classical. This is the lever.

New tooling: `engine-bench policy-overhead <rfpo> <fens> <depth> [top_k]` measures the
tax load-insensitively — it searches each position at fixed depth policy-off vs
policy-on at `bound=0` (byte-identical ordering, so node counts *must* match — a
built-in check) and reports `on/off` time. `gate-policy` gained a `[top_k]` arg.

**Inference tax (40 positions, depth 9, node counts identical throughout):**

| top_k | time_ratio |
|---|---|
| all | 1.82× |
| 16 | 1.45× |
| 8 | 1.29× |
| 4 | **1.16×** |

**Re-gate (60 games, 30 ms, still wide CI but the trend is monotonic):**

| bound | top_k | W-D-L | Elo | node_ratio | depth Δ |
|---|---|---|---|---|---|
| 500 | 8 | 9-19-32 | −140 | 0.779 | −0.29 |
| 1500 | 8 | 14-20-26 | −70 | 0.754 | −0.16 |
| 500 | 4 | 17-20-23 | **−35** | 0.827 | −0.08 |

Lower `K` → lower tax → less depth lost at equal movetime → Elo climbs from the
original −255 toward the −35 point estimate above. At `K=4` the candidate is only ~0.08
plies shallower. **Inference cost is no longer the blocker.**

**Big campaign settles the sign — it's a real regression.** The best config from the
sweep (`K=4`, `bound=500`, 30 ms), run at **1200 games** (4 single-threaded shards,
out-of-sample seeds), sharded through `gate-policy` → `sprt`:

```
309W 393D 498L  →  Elo -55.2  ·  LLR -4.23  ·  SPRT decision = AcceptH0 (elo0=0, elo1=5)
```

The SPRT **decisively accepts H0** (the policy is not a ≥5-Elo gain), and the point
estimate is −55 Elo with a ~±14 Elo CI that clearly excludes 0. The hopeful −35 at 60
games was the optimistic tail of the noise; the tight measurement lands solidly
negative. So even with inference made cheap, **this policy at its best config genuinely
costs ~55 Elo.**

**Conclusion: the lever is the training distribution, not more tuning.** Inference cost
is fixed and threshold/`K` tuning has been swept — the regression is what's left, and it
traces to the policy having little *new* to say: trained on classical-order labels with
`order_score` as a feature, it mostly re-derives the classical rank and the small
residual perturbation is net-negative. Threshold re-sweeps won't fix a model that echoes
the baseline. The two real next moves:

1. **Change what the labels represent** — the *decision-mutation / off-policy relabel*
   idea below (train on labels from orderings the policy itself would take, DAgger-style)
   is the direct attack; this campaign is the "if not, …" that motivates it.
2. **Change what the model can see** — drop `order_score`/`move_index` (the classical
   rank) so the model can't echo it, and/or move from pointwise `caused_cutoff` to a
   pairwise/listwise ranking loss.

Until one of those moves the needle offline *and* through a ≥1000-game gate, the policy
stays opt-in with no bundled asset — the default search is unchanged.

## Ideas / backlog

### Decision-mutation label augmentation (Ben's idea)

Take labelled games and add an element of *decision mutation* to cheaply generate more
labels — perturb the decisions the search made and re-observe the outcome, turning one
search into many labelled decision points instead of paying for a fresh search per
label.

**Reframing the value (the important part).** Labels are *not* currently the binding
constraint — a single depth-8 sweep already emits ~100M rows and the trainer hits val
AUC 0.93. The binding constraint is that every label is recorded under *classical*
ordering, and `caused_cutoff` is order-dependent (a move "causes a cutoff" only relative
to the alpha/beta window left by whatever was searched before it). So the model trains
on the classical distribution and — with `order_score` as a feature — largely echoes
it. The real prize in decision mutation is therefore **off-policy / counterfactual
labels**: labels for orderings the base search never took, which is exactly the
distribution the policy *creates* once it re-ranks. Volume is a side benefit; the
distribution shift is the point.

**Tier 1 — replay (free, approximate).** Record, per move-decision, two more things
besides the ordering-time features: the score the search returned for that move and the
node's alpha/beta on entry (append-only v4 telemetry, same byte-identical discipline).
Then *offline*, replay a node's cutoff logic under a mutated move order using the
recorded scores — walk the moves in the new order maintaining the running alpha; the
first whose recorded score ≥ beta is the counterfactual cutoff. This synthesizes
`caused_cutoff` for any permutation with **zero extra search**. Caveat: recorded scores
come from order-dependent windows (PVS null-window, LMR reductions), so they approximate
a move's order-invariant value — good enough to probe re-orderings near the top,
loosest for deep re-search cases.

**Tier 2 — re-search (cheap-ish, exact).** On a *sampled* subset of nodes, actually
re-run the move loop under a mutated order from the node's saved board state (not from
root), letting the search produce true labels. Bound cost by sampling few nodes at
shallow depth. Use this to (a) get exact labels where Tier 1 is weakest and (b)
*measure* Tier 1's fidelity (compare synthesized vs true cutoff on the same nodes).

**Mutation policies**, roughly in increasing value:
- random small permutations / swap the classical-first move with a later one — cheap
  coverage of "what if we'd tried this quiet move first";
- **order by the current policy's predicted cutoff prob** — a DAgger-style loop:
  train → relabel under the policy's *own* ordering → retrain, converging the training
  distribution onto the deployment one. This is the version that directly attacks the
  "echoes classical" problem.

**Plumbing.** The mutation/replay is a new engine-bench step (e.g. `gen-mutated-labels`)
that reads v4 telemetry and emits extra rows in the *same* TSV schema `train-policy`
already consumes, tagged `synthetic` with a weight so the trainer can down-weight
approximate labels. Trainer and model formats stay put; only the dataset grows/shifts.

**Open questions to firm up with Ben:**
- Off-policy correction vs raw volume — if it's the former (likely), prioritise mutation
  policy (b) (policy-ordered relabel) over cheap random mutation.
- Tier-1 fidelity: how far does the recorded-score proxy drift from a full-window score?
  Measure with a Tier-2 sample before trusting Tier-1 at scale.
- Does the DAgger loop actually move the gate, given inference is now cheap? The clean
  experiment: one round of policy-ordered relabel → retrain → re-gate at K=4/bound≈500.
- Generalises beyond ordering — the same replay trick could densify LMR / pruning labels
  (`raised_alpha`, `needed_lmr_research` are also order/threshold-dependent).

#### Tier 1 — built (2026-07-25)

Shipped: **v4 telemetry** (append-only `node_id`, `move_score`, `node_alpha`, `node_beta`
— byte-identical invariant still green), a **replay engine + `gen-mutated-labels`** step
(`<telemetry_v4|-> <mutations_per_node> [seed]`) that groups rows by `(pos_id, node_id)`,
replays each node's cutoff logic under seeded shuffles, and emits rows in the *same*
schema (so the output concatenates with real telemetry and feeds `train-policy`
unchanged). End-to-end validated: 2.19M telemetry rows → 3.53M mutated rows →
`train-policy` trains on the 5.7M-row union.

**Sharpened finding while building it — Tier 1 cannot add new `caused_cutoff` signal,
and here's the precise reason.** A move cuts iff `move_score ≥ node_beta`. Within a node
only *one* searched move ever records that (the loop breaks at the first cutoff), and
every searched non-cutter genuinely has `move_score < node_beta`. So the recorded
`caused_cutoff` **already is** the counterfactual "would this move cut if ordered first?"
for every *searched* move — reordering them changes nothing. The moves that could be
undiscovered cutters are the *unsearched* tail (after the cutoff, or LMP-pruned), and
those have no recorded score. **Only re-search (Tier 2) can label them.** What Tier 1
*does* add is counterfactual **`raised_alpha`** labels (order-dependent on the running
alpha) — the denser target — plus it is the exact substrate Tier 2 needs.

**Fidelity measured** (the open question above): replaying the *actual* order reproduces
the recorded labels for **97.2%** of nodes on real depth-8 telemetry. The ~3% gap is the
PVS-null-window / LMR approximation (a late fail-low move's recorded score is a bound),
exactly where the caveat predicted — and low enough to trust Tier 1 for `raised_alpha`
augmentation. (One doc-note above is now refined: replay does *not* meaningfully
synthesize `caused_cutoff`; that claim is superseded by the finding here.)

**So the recommended next step is Tier 2, not a Tier-1 volume push:** sampled re-search
under policy-predicted ordering (the DAgger loop) is the only thing that manufactures the
missing "an unsearched move would have cut first" labels — the actual off-policy signal.
Tier 1's `raised_alpha` output is worth one cheap experiment (retrain on the
`raised_alpha` target with the augmented set, re-gate at K=4/bound≈500) but is not
expected to beat the −55 Elo on its own.

#### Tier 2 — built (2026-07-25)

Shipped: a **re-search pass** in `engine-search` (`enable_research(stride, min_depth,
cap)` / `take_research`) that, at sampled main-`negamax` nodes, searches *every* legal
move full-window (no PVS null-window, no LMR reduction) and records an **order-independent
`caused_cutoff`** — the true "would this move cut first?" — with features from the live
searcher state (this node's real history/killers/TT), so a research row is
indistinguishable from a real telemetry row but for its label. It runs after the node's
own result is settled and is guarded so its child searches don't recurse into sampling.
Off by default → byte-identical (all invariants green); it mutates the TT, so it is
offline data-gen only. Exposed as `engine-bench gen-research-telemetry <fens|-> <depth>
[stride] [min_depth]`, same v4 schema, so it concatenates with `gen-search-telemetry` and
feeds `train-policy` unchanged.

**This is the signal Tier 1 couldn't produce, and the data proves it.** On 30 openings at
depth 8 (stride 32): 42.5k research rows, and **78% of cut-nodes have ≥2 moves that each
reach beta.** Normal telemetry records *exactly one* cutter per node — so it was
systematically under-labelling by ~4×, and the extra cutters are exactly the
low-classical-rank moves the policy needs to learn to promote.

**And it's harder to predict — which is the point.** Training on research-only rows
reaches val **AUC ≈ 0.68** (vs 0.93 on normal telemetry). The drop is not a regression:
normal `caused_cutoff` is easy because `order_score` (the classical rank) all but names
the one move classical searched first; the order-independent label deliberately strips
that crutch, so 0.68 is the *real* difficulty of "which move would cut, independent of
how classical ranked it." It confirms the features carry genuine but limited
order-independent signal — motivating **dropping `order_score`/`move_index`** so the model
must learn move quality rather than echo the rank.

**The clean next experiment** (now fully tooled, no new infra): build a training set from
`gen-research-telemetry` (optionally DAgger-style: order by the current policy before
sampling), retrain the policy — ideally without `order_score`/`move_index` — and re-gate
at K=4/bound≈500 against the −55 Elo baseline. This is the first run with a real shot at
beating classical ordering, because it's the first trained on what the policy actually
faces when it re-ranks.

#### Experiment run — order-score-free policy on Tier 2 labels (2026-07-25)

Done. Dropped `order_score` from the policy feature set (`POLICY_FEATURES` 17→16 — the
telemetry *column* stays; it is just no longer a model input; the classical score still
enters the search as the residual base, and `move_index` was already excluded). Generated
a 1.43M-row research corpus (480 openings, depth 8, stride 16), trained the 16-feature
policy on it, and gated it.

- **Offline:** val **AUC ≈ 0.765** on the order-independent target — *up* from ~0.68 with
  `order_score` in, i.e. removing the classical-rank crutch and giving more data let the
  model learn real move quality, not less.
- **Gate (800 games each, K=4, 30 ms, same openings as the −55 baseline):**

  | bound | 1000 | **2000** | 3000 | 4000 |
  |---|---|---|---|---|
  | Elo | −47.6 | **−30.9** | −47.2 | −39.3 |

  Best is `bound=2000` at **−30.9 Elo** — a **~24 Elo improvement over the −55 baseline**
  (the `order_score`-in, telemetry-trained policy). The larger optimal bound is expected:
  without `order_score` as a feature the model's raw prediction needs more scale to move
  the ordering. The curve peaks at 2000 (over-correction beyond).

**Verdict: the thesis is validated directionally but not yet a win.** Tier 2's
order-independent labels + an order-score-free model recover ~24 Elo — the biggest single
step — confirming that "train on which moves would actually cut, with a model that can't
echo the classical rank" is the right axis. But it is still **−31 Elo**, not positive. The
residual gap is the remaining inference tax (~16% NPS at K=4) plus a still-modest predictor
(AUC 0.765). Next levers, in order of expected payoff: (1) a **DAgger loop** — regenerate
research labels under *this* policy's ordering and retrain, iterating the training
distribution onto the deployed one; (2) a bigger/better model or richer move-quality
features now that the rank crutch is gone; (3) drive the K/bound sweep at higher game
counts around K=4/bound=2000. The policy stays opt-in with no bundled asset — default
search unchanged.

#### DAgger iteration — tried, did not help (2026-07-25)

`gen-research-telemetry` gained an optional trailing `[policy.rfpo] [bound] [top_k]`: with
a policy installed the search *orders* moves like the deployed policy, so the re-search
pass samples nodes on the policy's own distribution (labels stay order-independent). Ran
one round: generated a 1.45M-row DAgger corpus with the round-1 policy driving ordering
(K=4/bound=2000), retrained, and gated at K=4, 800 games/config, same openings as round-1.

| policy (K=4, bound=2000, 800g) | Elo |
|---|---|
| round-1 (research, classical distribution) | **−30.9** |
| round-2, replacement (train on DAgger corpus only) | −61.4 (AcceptH0) |
| round-2, aggregated (train on round-1 ∪ DAgger) | −48.1 |

**Both regressed.** Offline AUC was flat across all three (0.765 / 0.764 / 0.769), so this
is purely a distribution effect, and the *right* way round: DAgger's premise is to train on
the distribution the policy induces so it sees its own mistakes — but that only helps when
the policy is *better* than the reference it will run against. Here the round-1 policy is
**−31 Elo, i.e. worse than classical**, so its induced node distribution is a *degraded*
search's, not a better one; training toward it moves away from the useful (≈classical)
distribution the residual actually runs in. Aggregation (−48) beat pure replacement (−61)
but still lost to plain round-1. Compounding it: the correction is a nudge within the quiet
band (bound 2000 ≪ the 100k+ classical bucket gaps), so the policy's node distribution is
already *nearly* classical's — DAgger's shift is small and slightly harmful.

**Conclusion: DAgger is not the lever while the policy is below the reference.** The
round-1 Tier 2 / order-score-free policy at **−30.9 Elo remains the best result**. The
binding constraint is predictor quality (AUC 0.765) plus the residual inference tax, not the
training distribution — so the payoff order is now **(1)** a bigger/better model or richer
move-quality features (the rank crutch is gone, so capacity can finally matter), **(2)**
cheaper inference to shrink the ~16% K=4 tax, and only **(3)** revisit DAgger *after* a
round crosses into positive Elo, where its premise actually holds. Policy stays opt-in;
default search unchanged.
