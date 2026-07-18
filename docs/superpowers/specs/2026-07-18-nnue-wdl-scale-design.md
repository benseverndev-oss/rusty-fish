# NNUE Lichess-WDL v2: Scale and Capacity Design

## Goal

Give the outcome-trained NNUE its best real shot at beating the tuned hand-crafted
evaluation, by removing the two things that most plausibly held the first attempt
back: too little data and too little training throughput. The first Lichess-WDL
net (768 -> hidden 256, ~1.5M samples, 25 epochs) gated at **-325 Elo (SPRT
AcceptH0)** against the tuned hand-crafted eval, with `wdl_loss` only falling
0.231 -> 0.219 and still declining at the last epoch — a weak, under-trained
signal. This slice scales the corpus ~6x, bumps capacity to hidden 512, and
rewrites the trainer so that much data is actually consumable within the GPU
budget.

Nothing about the shipped engine changes unless the powered gate returns
**AcceptH1**. NNUE remains opt-in (loaded via `EvalFile` / `nnue-sprt`); the tuned
hand-crafted eval (+10.5 Elo, already shipped) stays the default and is the bar to
beat.

## What stays the same

This is an extension of the pipeline merged in #56, not a redesign. Unchanged:

- The `engine-bench gen-wdl-data` labeller (streaming PGN visitor, rated/standard/
  >=2200 filter, side-to-move-relative outcome target, `--shard i/n`,
  `--per-game N`), and its TSV output format (`target<TAB>own_csv<TAB>opp_csv`).
- The `768` side-to-move-relative perspective feature set and
  `active_features(board, perspective)`.
- The RFNN quantised i16 export format and the Rust inference path
  (`engine-search/src/nnue.rs`). The exported file must load in the engine exactly
  as today.
- The Modal train -> gate structure (`prepare_export` -> `label_wdl_shard` ->
  `train_wdl_run` -> `nnue_gate_run`), launched via `infisical run … modal run
  --detach` and retrieved from `modal app logs` with `NNUE_GATE_RESULT` markers.

## Change 1: Multi-month WDL corpus

The first run used the single `2017-01` export already pinned by the opening
book's `assets/opening-book/manifest.toml`. This slice trains on **six months**,
`2017-01` through `2017-06`.

- A **new committed manifest** `assets/nnue/wdl-corpus.toml` lists the six months,
  each with its `url` and `sha256`. This is deliberately **separate** from
  `assets/opening-book/manifest.toml`: that file pins `2017-01` as a *shipped
  asset* whose exact SHA gates a committed book artifact, and must not be
  repointed or coupled to training-data churn. The WDL corpus is training input,
  not a shipped asset, and lives on its own.
- The five new months' SHA-256 digests are not yet known. They are obtained
  **once** via a throwaway Modal probe function that downloads each month and
  prints its `sha256sum` (the same approach used when the opening book was
  re-pinned from a runner). Those digests are pinned into `wdl-corpus.toml` before
  the real training run; the probe is not committed.
- Expected scale: the six months hold on the order of ~800K accepted 2200+
  standard-rated games; at `--per-game 12` that is roughly **6-9M samples** — a
  ~6x increase over the first run's ~1.5M, and more importantly a far more diverse
  set of games than re-sampling one month.

Trade-off noted: months are pinned by SHA for reproducibility and integrity, at
the cost of a one-time probe to discover the digests. The alternative (skipping
verification for the extra months) is rejected — it would break the pipeline's
download-integrity guarantee that the book refresh established.

## Change 2: Trainer throughput (the crux)

The current trainer (`modal/train_nnue.py`) rebuilds the ragged `EmbeddingBag`
inputs for **every minibatch** on the CPU via Python list comprehensions
(`_ragged_to_bag([owns[i] for i in idx.tolist()])`). At ~1.5M samples this is
tolerable; at 6-9M samples across ~60 epochs it would dominate wall-clock and blow
the GPU container's timeout. Consuming 6x data is therefore gated on making
batching cheap.

**Approach — fixed-width padded index tensors, tokenised once:**

- Each perspective's active-feature list is a subset of that side's on-board
  pieces (at most 16 per side), so `active_features` returns **at most 16, always
  fewer than 32,** indices per perspective. Width 32 is therefore a safe
  over-provision that no position can overflow — a stated invariant the padded
  layout's correctness rests on. Pre-tokenise the **entire dataset once** into two
  `[N, 32]` integer tensors (own, opp), padding each row to width 32 with a
  dedicated **padding index 768** (real feature indices are `0..=767`). Use
  **int32** for the index tensors (feature indices and N both fit comfortably);
  int64 would double the footprint for no benefit. At N ~= 9M the two index
  tensors are ~2.3 GB combined, plus the `[N]` float32 targets (~36 MB), all moved
  to the GPU once — comfortably within the A10G's 24 GB alongside the embedding
  weights, activations, and Adam state.
- The model's transformer becomes an `nn.Embedding(769, hidden)` whose row `768`
  (padding) is **fixed to zero and frozen** (`padding_idx=768`), so padded slots
  contribute nothing to the summed accumulator. The forward for a minibatch of row
  indices is then a pure GPU gather: `emb(own_rows).sum(dim=1) + feature_bias`,
  masked by padding — mathematically identical to the current bag-sum accumulator,
  but with no per-batch Python work.
- Shuffling is a GPU index permutation; a minibatch is a tensor slice. This is the
  standard sparse-NNUE training layout and removes the CPU bottleneck entirely.

**RFNN export is unchanged.** Only rows `0..=767` of the embedding table are
written to the RFNN feature-weights block (the padding row is excluded); the
`feature_bias`, output weights, and output bias are quantised exactly as today.
The clipped-ReLU + output-linear forward still mirrors quantised inference, so a
net trained this way loads and plays identically to one trained the old way. A
parity check (below) pins this.

## Change 3: Training schedule and validation

- **Learning-rate schedule:** cosine decay over the epochs (replacing the fixed
  `lr=1e-3`), which typically reaches a lower loss floor than a flat rate on a
  longer run. Adam is retained.
- **More epochs:** ~60 (now affordable), configurable.
- **Validation split:** a deterministic hold-out (~2% of samples, split by a fixed
  hash of sample index so it is reproducible and independent of shuffling) whose
  `val_wdl_loss` is reported each epoch to stderr alongside the train loss. This
  distinguishes under-fitting (both losses high — need capacity or better teacher)
  from over-fitting (train low, val rising — need regularisation or more data),
  which a single train-loss curve cannot show. The split is not used for early
  stopping in v1 (just reported), to keep the loop simple.

## Change 4: Capacity

Hidden `256` -> `512`. The user chose to change capacity and data together to
maximise the chance of actually beating the eval this run.

**Stated confound:** because both the corpus (6x) and the width (2x) change at
once, a win or loss at the gate **cannot be cleanly attributed** to data-scale
versus capacity. This is an accepted trade-off (best single shot over clean
attribution). If the result is ambiguous and attribution matters, a follow-up can
isolate one variable.

## Change 5: Modal pipeline

Mirrors the existing `train_wdl` flow, extended for the multi-month corpus and the
faster trainer:

- **`prepare_export`** loops the six corpus entries, downloading and SHA-verifying
  each into the Volume once (idempotent per month, as today for the single file).
- **`label_wdl_shard`** gains a month/export-filename argument; the entrypoint fans
  out over `(month, shard)` pairs, so labeling parallelises across all months.
  Each shard streams `zstdcat /vol/export-<month>.pgn.zst | gen-wdl-data - --shard
  i/n` under `bash -c 'set -euo pipefail'` as today, writing
  `samples-<month>-<i>.tsv`.
- **`train_wdl_run`** concatenates all `samples-*.tsv`, runs the batched trainer at
  hidden 512, and returns the RFNN bytes. Its timeout is raised to accommodate the
  larger dataset (the batched trainer makes this feasible).
- **Gate:** switch to a **movetime-bounded** gate (as the eval gate uses), not the
  effectively depth-bound gate the first run used. The first run's `gate_shard`
  called `gate-file <net> <depth> <openings>` without forwarding a movetime, so it
  silently fell back to the binary's default and fixed-depth-ish self-play produced
  long games and a ~25-minute gate; a bound movetime brings the 4096-game gate back
  to ~5 minutes and is a fairer time-equal comparison. **This is a Modal /
  orchestration change only, not a Rust change:** `gate-file` already accepts an
  optional movetime budget as its 5th argument (`gate-file <net> <depth>
  <openings_file> [move_time_ms]`) and already threads it into the search's limits.
  The change is to `gate_shard` / `nnue_gate_run` / `train_wdl` — forward an
  explicit `move_time_ms` and set `depth` high enough that the movetime cap binds
  first, exactly as the `eval_gate_shard` / `mobility_gate_shard` paths do.
  `nnue_gate_run` keeps the remote-aggregation + `NNUE_GATE_RESULT` marker pattern
  for detached retrieval.

## Adoption

Unchanged from #56. If the powered SPRT gate (4096 games vs the tuned hand-crafted
baseline) returns **AcceptH1**, a follow-up commits the `.rfnn` as an asset, wires
it as the default evaluation, and re-gates under normal CI. On **AcceptH0** or a
flat result, the net stays opt-in and we record the outcome:

- A **material improvement** off -325 toward 0 (even without a win) plus a
  `val_wdl_loss` breaking below ~0.20 justifies pushing the outcome-teacher
  direction further (more months, longer training).
- A **flat** result (still near -325, `val_wdl_loss` stuck near 0.219) is strong
  evidence that raw game-outcome labels are the ceiling at this architecture, and
  the next lever is a **stronger teacher** (Stockfish-eval labels) rather than more
  of the same.

Either way the tracker is updated with the verdict and the reasoning.

## Verification

- **Rust:** this slice needs **no Rust change**. Training is Python; the labeller
  is unchanged; and the movetime gate is orchestration-only because `gate-file`
  already accepts a movetime budget (5th argument). There is therefore no new
  `cargo test` surface here — the existing `Rusty Fish Tests` workflow still guards
  the unchanged engine. Cargo is never run locally regardless.
- **Trainer parity (torch, not CI):** a check that the padded-tensor forward
  produces the same accumulator as the original ragged bag-sum on a few fixed
  samples (padding rows contribute zero), and that a tiny end-to-end train +
  `quantize_and_write` + `engine-bench nnue-sprt` round-trips — so the throughput
  rewrite provably preserves the RFNN semantics. Run via `uv`, recorded in task
  notes.
- **`target_win_prob` / `--wdl-target`** behaviour is unchanged and already
  covered.
- **End-to-end** is validated by the Modal run: a short config first (a couple of
  months, tiny per-game and epochs, small gate) to confirm the multi-month
  download, `(month, shard)` fan-out, batched trainer, and movetime gate all work,
  then the real run (six months, hidden 512, ~60 epochs, 2048-opening gate).

## Out of scope

- HalfKA / king-bucketed features.
- Stockfish-eval or hybrid teachers (the fallback if scaling outcome labels is
  insufficient).
- Incremental accumulator updates and any change to the RFNN format or inference
  path.
- Early stopping, dropout, or other regularisation beyond the cosine schedule
  (v1 reports validation loss but does not act on it).
- Deduplication beyond the per-game cap, and multi-year datasets.
