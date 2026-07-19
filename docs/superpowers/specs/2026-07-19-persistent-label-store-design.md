# Persistent Label Store Design

## Goal

Stop paying the dominant cost of every NNUE experiment — Stockfish labeling —
more than once. Today the Modal training pipeline re-labels from scratch each run
and, worse, its "delete non-expected shards" cleanup will destroy the accumulated
labels the moment a differently-configured run executes. This slice turns the
existing `rusty-fish-labels` volume into an **append-only, tag-keyed label store**:
labeling only ever *adds* the positions not already present, training *reads* the
store and never mutates it, and the destructive cleanup is removed. Once labels
exist, sweeping training configs (architecture, hidden size, loss, epochs) costs
only GPU time — the expensive labeling is amortized to zero.

This is sub-project #1 of the "faster/cheaper iteration" effort. The gate ladder,
the lightweight experiment harness, and cold-start reduction are follow-on specs
that build on this store.

## Current state (verified)

- **Volume `rusty-fish-wdl`** (the working volume): holds the raw Lichess exports
  `export-<month>.pgn.zst` (labeling input), plus — as leftovers — the WDL shards
  at the root and the SF shards under `/sf/`.
- **Volume `rusty-fish-labels`** (just created as a safety backup): holds a copy of
  the ~3M SF labels (192 shards `sf/samples-<month>-<i>.tsv` at 100k nodes,
  per-game 4, months 2017-01…06) plus `sf/data.tsv`, 48 WDL shards under `wdl/`,
  and `net.rfnn`. **The labels are intact; nothing was lost.**
- **`modal/app.py`**: `label_sf_shard`/`label_wdl_shard` write shards to the
  working volume; `train_sf_run`/`train_wdl_run` glob those shards, **delete any not
  in the current run's expected set** (the footgun), concatenate to `data.tsv`, and
  train. `prepare_export` downloads exports into `rusty-fish-wdl`.

## Design

### Volume roles (clean split)

- **`rusty-fish-wdl`** stays the **raw-export** volume (labeling *input*):
  `export-<month>.pgn.zst`, SHA-verified, downloaded once. Unchanged.
- **`rusty-fish-labels`** becomes the **append-only label store** (labeling
  *output*, training *input*). Nothing ever deletes from it.

### Store layout (tag-keyed by data identity, not parallelism)

A labeled dataset is identified by what makes its *positions and labels* distinct —
the teacher/node budget and the sampling density — not by how many shards the
labeling was divided into. Layout:

```
/store/sf/n<nodes>-pg<per_game>/samples-<month>-<i>.tsv   # one dataset per (nodes, per_game)
/store/sf/n<nodes>-pg<per_game>/<month>.complete          # marker: this (dataset, month) is fully labeled
```

e.g. the existing labels become `/store/sf/n100000-pg4/samples-2017-03-11.tsv`. A
future 25k-node run writes under `sf/n25000-pg4/`; a denser run under
`sf/n100000-pg8/`. The `<month>.complete` marker (written only after every shard of
that month finishes and commits) is the idempotency key: a labeling run **skips any
(dataset, month) whose marker exists** and labels only the rest. A crashed run
leaves no marker, so its partial month is re-labeled cleanly next time.

The shard count within a (dataset, month) is a parallelism detail: re-requesting an
already-`.complete` month is a no-op regardless of the requested `shards_per_month`;
only *new* months (or a new `n…-pg…` dataset) are labeled. This makes labeling
**additive and idempotent** — a bigger corpus pays only for its delta.

### Labeling: `label_into_store`

`label_sf_shard` gains a store mount and writes to
`/store/sf/n<nodes>-pg<per_game>/samples-<name>-<i>.tsv`. A new
`ensure_sf_labels(months, per_game, nodes, shards_per_month)` entrypoint:

1. `prepare_export` the requested months into `rusty-fish-wdl` (idempotent, as
   today).
2. For each requested `month`, check `/store/sf/n<nodes>-pg<per_game>/<month>.complete`.
   If present, **skip** (already labeled). Otherwise fan `label_sf_shard` across
   that month's shards, and — only after all its shards return and commit — write
   the `<month>.complete` marker and commit. (Per-month markers, not per-run, so a
   partial run's completed months are still reusable.)
3. Print a summary: months skipped (cache hits) vs labeled (delta), and the store's
   total sample count.

The labeling functions mount both volumes: read the export from `rusty-fish-wdl`,
write labels to `rusty-fish-labels`. They **never delete**.

### Training: `train_from_store`

`train_sf_run` is replaced by a store reader that **never mutates the store**:

- Takes a **dataset selector** — e.g. a list of `n<nodes>-pg<per_game>` dataset
  names (default: all `sf/n*-pg*` present) — plus `hidden`, `epochs`.
- Globs the selected datasets' `samples-*.tsv`, concatenates them into a **scratch**
  `data.tsv` in the container's ephemeral `/tmp` (NOT in the store volume), and
  trains (`wdl_target=False` cp mode, unchanged).
- Returns the RFNN bytes. The **destructive cleanup is deleted** — no `glob + unlink`
  of store shards, ever.
- Dedup note: mixing two datasets that sample the *same* games at different
  densities can double-count positions. v1 keeps this simple — the caller selects
  compatible datasets (typically one `n…-pg…`); a dedup pass is out of scope and
  flagged for a later slice.

### Entrypoint: `train_sf` becomes label-then-train over the store

`train_sf` (and by symmetry a future `train_wdl`) calls `ensure_sf_labels(...)`
(additive) then `train_from_store(...)` (read-only) then the existing gate. The
first real payoff: a run whose months are already `.complete` skips labeling
entirely and goes straight to training — the amortization this whole slice exists
for.

### Migration of the existing labels

A one-time step reorganizes the current flat backup
`rusty-fish-labels/sf/samples-<month>-<i>.tsv` (100k nodes, per-game 4) into
`rusty-fish-labels/sf/n100000-pg4/samples-<month>-<i>.tsv` and writes the six
`<month>.complete` markers, so the ~3M existing SF labels are immediately a
cache-hit dataset that the new `train_sf` reuses with zero re-labeling. (A small
Modal function does the move + marker-write + commit; the `wdl/` and `net.rfnn`
copies are left as-is.)

## Verification

- No Rust change; no CI gate. Verify `modal/app.py` with
  `uv run --python 3.12 python -m py_compile`.
- **Idempotency check (the core property), on Modal:** after migration, run
  `train_sf` for the six already-labeled months at (100k, pg4) — confirm from
  `modal app logs` that it reports **all six months skipped (cache hit), zero
  labeled**, trains, and gates. Then add ONE new month (e.g. 2017-07, pinned in
  `wdl-corpus.toml`) and confirm only that month labels (the delta), the prior six
  still skip, and the store grows.
- **Non-destruction check:** confirm the store's SF shard count is **>= the
  starting count after every run** (a labeling/training run never lowers it) — read
  it via `modal volume ls` before/after.
- **Cheap-sweep check:** run `train_from_store` twice with different `hidden`/`epochs`
  on the same (already-labeled) store and confirm neither labels anything — pure
  train cost.

## Out of scope

- The gate ladder, the experiment harness, and cold-start/warm-image reduction
  (follow-on specs #2–#4).
- Cross-dataset position dedup (v1 selects compatible datasets).
- Migrating the raw exports off `rusty-fish-wdl` or changing the WDL path.
- Any change to `label-sf` / `gen-eval-positions` / the trainer internals or the
  RFNN format.
- Deleting the `rusty-fish-labels` safety copies of `wdl/` and `net.rfnn`.
