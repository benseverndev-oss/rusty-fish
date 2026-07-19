# Experiment Harness Design

## Goal

Run many NNUE experiments from one command and collect the results comparably, so
sweeping architecture / hyperparameters is a casual `sweep` invocation instead of a
per-experiment spec→plan→review cycle. This is sub-project #3 of the
faster-iteration effort. With the persistent label store (#1) and gate ladder (#2)
already in place, an experiment is cheap — the harness is pure orchestration plus a
results ledger on top of what exists.

## What an experiment already is

`train_from_store(datasets, hidden, epochs, lr)` (GPU) → val-loss pre-check →
`gate_ladder_run(net_bytes, …)` (sequential SPRT vs the bundled champion) →
verdict. Labeling is a cache hit (store), so each experiment is train + gate only.
Nothing about training or gating changes here.

## Components

### `run_experiment` — one experiment, server-side

A lightweight **CPU** `@app.function` that orchestrates a single experiment and
returns a structured result (so nothing needs re-parsing client-side):

1. `net_bytes, val_loss = train_from_store.remote([dataset], hidden, epochs, lr)`.
2. Val pre-check: if `val_loss` is `NaN` or above the sanity ceiling (0.1), return a
   `rejected` result with `val_loss` and no gate.
3. Else `verdict = gate_ladder_run.remote(net_bytes, gate_depth, gate_plies,
   move_time_ms, gate_shard_size, chunk_openings, max_openings)`; parse its
   `NNUE_LADDER_RESULT` summary line. The real line is
   `gate ladder: {W}W {D}D {L}L over {games} games ({played}/{max} openings),
   decision {decision}` — anchor the regex on the `gate ladder:` prefix (or the
   `\d+W \d+D \d+L` triple), which is stable. Compute the Elo estimate from the
   score (`score = (W + 0.5·D)/(W+D+L)`, `elo = -400·log10(1/score − 1)`), with
   explicit edge handling: `games == 0` → `elo` is `NA` (no games); `score == 0` →
   a large negative floor (e.g. `-800.0`); `score == 1` → a large positive ceiling
   (`+800.0`), avoiding the div-by-zero / `log10(0)`.
4. Return a dict: `{dataset, hidden, epochs, lr, val_loss, wins, draws, losses,
   games, elo, decision}`.

It holds only a cheap CPU container while the GPU train and CPU gate run as their
own remote calls. This is the unit `sweep` parallelizes.

The experiment config is passed as a **single dict argument** (so `starmap` over
`[(cfg,) for cfg in configs]` fans them out); the gate parameters
(`gate_depth`/`gate_plies`/`move_time_ms`/`gate_shard_size`/`chunk_openings`/
`max_openings`) are carried in the same dict with the ladder's defaults.

### `sweep` — the cross-product entrypoint (parallel)

A `@app.local_entrypoint()` taking comma-separated axes — e.g.
`--hiddens 256,512,1024 --epochs-list 40,80 --lrs 1e-3,5e-4` — plus a fixed
`--dataset` (default `n100000-pg4`, the current SF labels) and the gate defaults.
It:

1. Parses each axis, forms the **cross-product** (`itertools.product`) into a list
   of config dicts.
2. Ensures the dataset's labels exist is **out of scope** — the sweep trains from
   already-labeled data; if a config's dataset has no shards, `train_from_store`
   fails loudly for that experiment (the harness records the failure, does not
   crash the whole sweep — see failure handling).
3. `results = run_experiment.starmap([(cfg,) for cfg in configs])` — all
   experiments run concurrently (Modal queues past its GPU-concurrency cap, so it is
   as parallel as allowed). `.starmap` blocks until all finish.
4. **Single serial ledger write:** the parallel part is train+gate; the results are
   appended in **one** `append_results.remote(rows, sweep_id)` call after the
   gather, so concurrent containers never race on the shared file.
5. Print a summary table sorted by Elo (best first), so the winner is obvious.

### The results ledger — append-only, in the store

`append_results(rows, sweep_id)` mounts the label store, `reload()`s, appends TSV
rows to `/store/experiments/results.tsv` (writing a header first if the file does
not exist), and commits. Columns: `sweep_id`, `timestamp` (UTC ISO, via `datetime`
— allowed in Modal Python), `dataset`, `hidden`, `epochs`, `lr`, `val_loss`,
`wins`, `draws`, `losses`, `games`, `elo`, `decision`. Append-only in the same
spirit as the labels — every experiment ever run is preserved and comparable across
sessions.

**Sentinel for non-gated rows (keep the TSV column-aligned):** a `rejected` row
(val pre-check failed) and an `error` row (training/gate raised) have no gate — set
`wins`/`draws`/`losses`/`games`/`elo` to the literal `NA`, and `decision` to
`"rejected"` or `"error"` respectively (the error message can be appended to the
`decision` field or dropped). Every row has all columns filled.

A `results` entrypoint reads and prints the ledger sorted by Elo (a
`read_results` volume-mounted function returns the file contents; the entrypoint
formats it).

## Enabling change

`train_from_store` currently hardcodes `lr=1e-3` in its `train(...)` call. Thread an
`lr: float = 1e-3` parameter through it (it already forwards to `train()`). This is
the only change to existing functions; `hidden` and `epochs` are already parameters.

## Failure handling

An experiment whose training or gate errors must not sink the whole sweep. Since
Modal `.starmap` propagates a raised exception, `run_experiment` **catches its own
errors** and returns a result with `decision = "error"` and the message in a field,
so a bad config is logged like any other and the sweep completes. (A config whose
dataset is missing, an OOM, etc. becomes one `error` row, not a crashed sweep.)

## Verification

- No Rust change; no CI gate. Verify `modal/app.py` with `py_compile`.
- **Modal:** a tiny 2-experiment sweep on the existing labels — e.g.
  `sweep --hiddens 8,16 --epochs-list 1 --chunk-openings 128 --max-openings 256` —
  and confirm from `modal app logs` / the printed table that: both experiments run,
  each produces a row (val_loss + a gate verdict, likely `AcceptH0` since tiny nets
  lose to the champion), the ledger `/store/experiments/results.tsv` gains 2 rows
  with a header, and the `results` entrypoint prints them sorted by Elo. Then re-run
  a 1-experiment sweep and confirm the ledger **appends** (grows, header not
  duplicated) — the persistence property.
- A deliberately-broken config (val-precheck path) is covered by the tiny nets
  above if any diverge; the `error`-row path is exercised if `train_from_store`
  raises.

## Out of scope

- Auto-adopting a sweep winner (bundling a new champion stays the manual adoption
  flow).
- Cold-start / warm-image reduction (#4).
- Sweeping the SF-teacher / data axes (node budget, per-game) — those change the
  *labels*, so they run through `ensure_sf_labels` (the label store), a natural
  later extension, not this train-only sweep.
- Cross-dataset dedup, and any change to training, gating, or the RFNN/label formats.
