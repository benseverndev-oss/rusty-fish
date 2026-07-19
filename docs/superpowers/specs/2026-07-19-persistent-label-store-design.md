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

`label_sf_shard` gains the store mount and writes to
`/store/sf/n<nodes>-pg<per_game>/samples-<name>-<i>.tsv`. Because the local
entrypoint runs **client-side with no volume mounted**, the marker read and write
must be small **volume-mounted Modal functions**, not inline entrypoint code:

- `missing_sf_months(dataset, months) -> list[str]` — mounts the store,
  `labels_volume.reload()`, returns the months whose
  `/store/sf/<dataset>/<month>.complete` marker is absent.
- `mark_sf_month_complete(dataset, month)` — mounts the store, writes the marker,
  `labels_volume.commit()`.

The `ensure_sf_labels(months, per_game, nodes, shards_per_month)` **local
entrypoint** orchestrates:

1. `prepare_export.starmap` the requested months into `rusty-fish-wdl` (idempotent,
   as today).
2. `missing = missing_sf_months.remote("n<nodes>-pg<per_game>", months)`.
3. For each `month` in `missing`:
   a. `wipe_sf_month.remote(dataset, month)` — a single volume-mounted function that
      deletes any existing `samples-<month>-*.tsv` in **this dataset dir and month
      only** (a month-scoped delete, NOT the global footgun) and commits. This runs
      **once, before** the shards, so an earlier crashed attempt at a *different*
      `shards_per_month` cannot leave orphan shards that later get concatenated. It
      must NOT be inside `label_sf_shard` — parallel shards would clobber each
      other's fresh output.
   b. `list(label_sf_shard.starmap([...]))` across that month's shards (blocks on
      all; each shard writes its file and commits).
   c. **Only after** all of the month's shards return: `mark_sf_month_complete.remote(dataset, month)`.
4. Print a summary: months skipped (cache hit) vs labeled (delta), and the store's
   total sample count.

Labeling mounts both volumes (read export from `rusty-fish-wdl`, write labels to
`rusty-fish-labels`) and **never deletes across months or datasets** — the only
deletes are the month-scoped `wipe_sf_month` pre-step, which is bounded to the one
(dataset, month) about to be re-labeled.

### Training: `train_from_store`

`train_sf_run` is replaced by a store reader that **never mutates the store**:

- `@app.function(image=torch_image, gpu="A10G", memory=32768, timeout=60*60*3,
  volumes={"/store": labels_volume})` — carry the `memory`/`timeout`/GPU config
  forward (concatenating multiple datasets is at least as memory-heavy as today).
- Takes a **dataset selector** — a list of `n<nodes>-pg<per_game>` dataset names
  (default: all `sf/n*-pg*` present) — plus `hidden`, `epochs`.
- `labels_volume.reload()` first (so freshly-committed shards from this run's
  labeling step are visible — the current code reloads for exactly this reason),
  then globs the selected datasets' `samples-*.tsv`, concatenates into a **scratch**
  `/tmp/data.tsv` (the container's ephemeral disk, NOT the store — the current code
  writes `/vol/sf/data.tsv`, which must change), and trains (`wdl_target=False` cp
  mode, unchanged).
- **Canonical net write (Gap: `gate_net`):** the current `train_sf_run` writes the
  net to `/vol/net.rfnn` on the working volume, which `read_net`/`gate_net` read
  back. Since `train_from_store` mounts the store instead, write the net to
  `/store/nets/latest.rfnn` and commit, and point `read_net`/`gate_net` at
  `rusty-fish-labels:/nets/latest.rfnn`, so a post-hoc `gate_net` re-gates *this*
  net, not a stale one.
- Returns the RFNN bytes (the primary gate still flows in-memory via
  `nnue_gate_run(net_bytes)`). The **destructive cleanup is deleted** — no
  `glob + unlink` of store shards, ever.
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
  labeled**, trains, and gates. Then add ONE new month (2017-07 — pin it in
  `wdl-corpus.toml`, which needs `sha_probe` run first since `train_sf` asserts
  each month has a `sha256`) and confirm only that month labels (the delta), the
  prior six still skip, and the store grows.
- **Non-destruction check:** confirm the store's SF shard count is **>= the
  starting count after every run** (a labeling/training run never lowers it) — read
  it via `modal volume ls` before/after.
- **Cheap-sweep check:** run `train_from_store` twice with different `hidden`/`epochs`
  on the same (already-labeled) store and confirm neither labels anything — pure
  train cost.

## Out of scope

- **`train_wdl_run`'s identical destructive cleanup** (`glob("/vol/samples-*.tsv")`
  + unlink non-expected on the WDL volume) is a live footgun too, but is
  **explicitly deferred** here — the WDL/outcome labels carry no Stockfish cost, so
  they are far cheaper to regenerate. This slice converts the SF path only; the WDL
  path keeps its current behaviour until a later slice (noted so it is not
  forgotten).
- The gate ladder, the experiment harness, and cold-start/warm-image reduction
  (follow-on specs #2–#4).
- Cross-dataset position dedup (v1 selects compatible datasets).
- Migrating the raw exports off `rusty-fish-wdl` or changing the WDL path.
- Any change to `label-sf` / `gen-eval-positions` / the trainer internals or the
  RFNN format.
- Deleting the `rusty-fish-labels` safety copies of `wdl/` and `net.rfnn`.
