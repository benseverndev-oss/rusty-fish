# Persistent Label Store Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `rusty-fish-labels` Modal volume into an append-only, tag-keyed label store so Stockfish labeling is paid once: labeling adds only the missing `(dataset, month)` shards, training reads + concatenates and never deletes, and the destructive `train_sf_run` cleanup is removed. The existing ~3M SF labels migrate in as an immediate cache hit.

**Architecture:** `rusty-fish-wdl` stays the raw-export input; `rusty-fish-labels` becomes the store, laid out `sf/n<nodes>-pg<per_game>/samples-<month>-<i>.tsv` with per-month `<month>.complete` markers for idempotency. Labeling writes into the store dataset dir; a read-only `train_from_store` globs the selected datasets and trains. All Python/Modal — no Rust, no CI gate; verified by `py_compile` and a Modal idempotency run.

**Spec:** `docs/superpowers/specs/2026-07-19-persistent-label-store-design.md`

---

## Global constraints

- **Python/Modal only.** No Rust change. Verify each edit with `uv run --python 3.12 python -m py_compile modal/app.py` (do NOT import `modal`/`torch` locally). The end-to-end check is a Modal run (Task 4). Spurious Pyright "modal has no attribute / FunctionType has no remote" warnings are expected — ignore.
- **Modal runs** launch via `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 infisical run --env dev -- uv run --with modal --python 3.12 -- modal run [--detach] modal/app.py::<entrypoint>`, retrieved from `modal app logs <app-id>`.
- **Verify `gh` account is `benzsevern`** before remote ops; stage paths explicitly — never `git add -A`; push to `feat/persistent-label-store` (tokenized URL fallback).
- Conventional Commits. **Branch:** `feat/persistent-label-store` (already created off latest main, spec committed).
- **The store already holds the labels:** `rusty-fish-labels` currently has flat `sf/samples-<month>-<i>.tsv` (192 files, 100k nodes, per-game 4, 2017-01…06), `sf/data.tsv`, `wdl/`, `net.rfnn` (a safety backup made out-of-band). Task 1 reorganizes the flat SF files into the tagged layout. **Never delete the `wdl/` or `net.rfnn` copies.**

## Background: the exact current code (`modal/app.py`)

- `wdl_volume = modal.Volume.from_name("rusty-fish-wdl", create_if_missing=True)` (line 67). `rusty-fish-labels` is NOT yet referenced.
- `label_sf_shard(name, i, n, per_game, nodes)` (234-255): mounts `{"/vol": wdl_volume}`; `os.makedirs("/vol/sf")`; runs `bash -c 'set -euo pipefail; zstdcat /vol/export-{name}.pgn.zst | {BIN} gen-eval-positions - --shard {i}/{n} --per-game {per_game} | {BIN} label-sf - {nodes} > /vol/sf/samples-{name}-{i}.tsv'`; `wdl_volume.commit()`; returns line count.
- `train_sf_run(shard_names, hidden, epochs)` (302-349): `@app.function(image=torch_image, gpu="A10G", timeout=60*60*3, memory=32768, volumes={"/vol": wdl_volume})`; `reload()`; **deletes** `/vol/sf/samples-*.tsv` not in `expected` + stale `/vol/sf/data.tsv`; concats `expected` → `/vol/sf/data.tsv`; `train_nnue.train(..., wdl_target=False)`; writes `/vol/net.rfnn`; returns bytes.
- `train_sf(...)` entrypoint (557-611): loads corpus, `prepare_export.starmap`, `label_sf_shard.starmap` over `(name, i, shards_per_month, per_game, nodes)`, builds `shard_names=[f"sf/samples-{name}-{i}.tsv" ...]`, `train_sf_run.remote(shard_names, hidden, epochs)`, `nnue_gate_run.remote(...)`.
- `read_net()` (614-619): mounts `{"/vol": wdl_volume}`, `reload()`, reads `/vol/net.rfnn`. `gate_net` (622-643) calls `read_net.remote()` then `nnue_gate_run`.
- `prepare_export` (189-213): downloads to `rusty-fish-wdl`, unchanged. `_load_wdl_corpus`, `sha_probe`, `nnue_gate_run` unchanged.
- **`train_wdl_run` (258-299) keeps its destructive cleanup — out of scope, deferred (spec).**

## File structure

Only `modal/app.py` changes. No new files.

---

### Task 1: Add the store volume + migrate the existing labels

**Files:** Modify `modal/app.py`

- [ ] **Step 1: Add the labels volume.** After line 67 add:

```python
labels_volume = modal.Volume.from_name("rusty-fish-labels", create_if_missing=True)


def _sf_dataset(nodes: int, per_game: int) -> str:
    """The store dataset dir keyed on data identity (teacher budget + density)."""
    return f"n{nodes}-pg{per_game}"
```

- [ ] **Step 2: Add a migration function + entrypoint.** The backup left flat `sf/samples-<month>-<i>.tsv` (100k nodes, per-game 4). Move them into `sf/n100000-pg4/` and write per-month `.complete` markers so they're an immediate cache hit:

```python
@app.function(volumes={"/store": labels_volume}, timeout=60 * 30)
def migrate_flat_sf_labels() -> str:
    import os, glob, re, shutil
    labels_volume.reload()
    dataset = _sf_dataset(100000, 4)
    dst = f"/store/sf/{dataset}"
    os.makedirs(dst, exist_ok=True)
    moved, months = 0, set()
    for p in glob.glob("/store/sf/samples-*.tsv"):  # flat, top-level only (no recursion)
        base = os.path.basename(p)
        shutil.move(p, f"{dst}/{base}")
        moved += 1
        m = re.match(r"samples-(\d{4}-\d{2})-\d+\.tsv", base)
        if m:
            months.add(m.group(1))
    for month in sorted(months):
        with open(f"{dst}/{month}.complete", "w") as h:
            h.write("migrated\n")
    if os.path.exists("/store/sf/data.tsv"):
        os.remove("/store/sf/data.tsv")  # stale flat concat; datasets are re-concatenated on train
    labels_volume.commit()
    report = f"moved {moved} shards into sf/{dataset}, markers: {sorted(months)}"
    print(f"MIGRATE_DONE {report}", flush=True)
    return report


@app.local_entrypoint()
def migrate_labels():
    print(migrate_flat_sf_labels.remote())
```

- [ ] **Step 3: py_compile + commit.** `uv run --python 3.12 python -m py_compile modal/app.py`. Commit `feat(store): add rusty-fish-labels store volume + flat-label migration`.

- [ ] **Step 4: Run the migration once + verify.** `modal run modal/app.py::migrate_labels`. Then `modal volume ls rusty-fish-labels sf/n100000-pg4` — confirm **192 `samples-*.tsv` + six `<month>.complete` markers**, and `modal volume ls rusty-fish-labels sf` shows the flat files are gone (moved). Record the counts.

---

### Task 2: Idempotent labeling into the store

**Files:** Modify `modal/app.py`

- [ ] **Step 1: Store-marker functions.** Add three volume-mounted functions (the local entrypoint can't touch a volume):

```python
@app.function(volumes={"/store": labels_volume}, timeout=60 * 10)
def missing_sf_months(dataset: str, months: list[str]) -> list[str]:
    import os
    labels_volume.reload()
    return [m for m in months if not os.path.exists(f"/store/sf/{dataset}/{m}.complete")]


@app.function(volumes={"/store": labels_volume}, timeout=60 * 10)
def wipe_sf_month(dataset: str, month: str) -> int:
    import os, glob
    labels_volume.reload()
    removed = 0
    for p in glob.glob(f"/store/sf/{dataset}/samples-{month}-*.tsv"):
        os.remove(p); removed += 1
    labels_volume.commit()
    return removed


@app.function(volumes={"/store": labels_volume}, timeout=60 * 10)
def mark_sf_month_complete(dataset: str, month: str) -> None:
    import os
    os.makedirs(f"/store/sf/{dataset}", exist_ok=True)
    with open(f"/store/sf/{dataset}/{month}.complete", "w") as h:
        h.write("done\n")
    labels_volume.commit()
```

- [ ] **Step 2: Rewrite `label_sf_shard` to write into the store.** It now mounts BOTH volumes (read export from `/vol`, write labels to `/store`), takes the `dataset`, and writes `/store/sf/{dataset}/samples-{name}-{i}.tsv`:

```python
@app.function(image=rust_image, volumes={"/vol": wdl_volume, "/store": labels_volume}, timeout=60 * 90)
def label_sf_shard(dataset: str, name: str, i: int, n: int, per_game: int, nodes: int) -> int:
    import os
    os.makedirs(f"/store/sf/{dataset}", exist_ok=True)
    out = f"/store/sf/{dataset}/samples-{name}-{i}.tsv"
    cmd = (
        f"set -euo pipefail; zstdcat /vol/export-{name}.pgn.zst | "
        f"{BIN} gen-eval-positions - --shard {i}/{n} --per-game {per_game} | "
        f"{BIN} label-sf - {nodes} > {out}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)
    labels_volume.commit()
    with open(out, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
```

(Note: `label_sf_shard` no longer does the month wipe — that's `wipe_sf_month`, run once per month by the orchestrator, so parallel shards can't clobber each other.)

- [ ] **Step 3: The `ensure_sf_labels` orchestration helper** (a plain function called client-side by `train_sf`, using `.remote`/`.starmap`):

```python
def ensure_sf_labels(corpus, per_game, nodes, shards_per_month) -> None:
    dataset = _sf_dataset(nodes, per_game)
    names = [m["name"] for m in corpus]
    missing = missing_sf_months.remote(dataset, names)
    print(f"SF store {dataset}: {len(names) - len(missing)} months cached, labeling {missing}")
    for month in missing:
        wipe_sf_month.remote(dataset, month)  # once, before the shards
        list(label_sf_shard.starmap(
            [(dataset, month, i, shards_per_month, per_game, nodes) for i in range(shards_per_month)]
        ))
        mark_sf_month_complete.remote(dataset, month)
```

- [ ] **Step 4: py_compile + commit.** Commit `feat(store): idempotent SF labeling into the store (markers, month wipe)`.

---

### Task 3: Read-only training from the store

**Files:** Modify `modal/app.py`

- [ ] **Step 1: Replace `train_sf_run` with `train_from_store`** — no cleanup, reads the store, concats to `/tmp`, writes the net into the store:

```python
@app.function(
    image=torch_image, gpu="A10G", timeout=60 * 60 * 3, memory=32768,
    volumes={"/store": labels_volume},
)
def train_from_store(datasets: list[str], hidden: int, epochs: int) -> bytes:
    """Train (cp mode) on the concatenation of the given store datasets. Read-only
    on the store: it globs + concatenates, NEVER deletes."""
    import os, train_nnue
    labels_volume.reload()
    shard_paths = sorted(
        p for d in datasets for p in glob.glob(f"/store/sf/{d}/samples-*.tsv")
    )
    assert shard_paths, f"no shards found for datasets {datasets}"
    data_path = "/tmp/data.tsv"  # ephemeral container disk, NOT the store
    with open(data_path, "w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
    model = train_nnue.train(
        data_path, hidden, epochs, batch_size=1024, lr=1e-3, device="cuda", wdl_target=False
    )
    os.makedirs("/store/nets", exist_ok=True)
    out_path = "/store/nets/latest.rfnn"
    train_nnue.quantize_and_write(model, hidden, out_path)
    labels_volume.commit()
    with open(out_path, "rb") as handle:
        return handle.read()
```

Delete the old `train_sf_run` (302-349). Leave `train_wdl_run` untouched (deferred).

- [ ] **Step 2: Point `read_net`/`gate_net` at the store net.** Change `read_net` to mount the store and read `/store/nets/latest.rfnn`:

```python
@app.function(volumes={"/store": labels_volume})
def read_net() -> bytes:
    """Read the last store-trained net (/store/nets/latest.rfnn) back off the store."""
    labels_volume.reload()
    with open("/store/nets/latest.rfnn", "rb") as handle:
        return handle.read()
```

`gate_net` still calls `read_net.remote()` — but **update its docstring** (currently
mentions "after a `train_wdl` run" and `/vol/net.rfnn`): it now re-gates the
store-trained net `/store/nets/latest.rfnn` (the SF/store path). Drop the
`train_wdl`/`/vol/net.rfnn` language so it isn't misdocumented — `train_wdl_run`
still writes `/vol/net.rfnn` (deferred), but nothing reads it now, so `gate_net`
after a `train_wdl` run would gate the store net, not the WDL net; the docstring
must not claim otherwise. (Re-adding a WDL re-gate path is out of scope.)

- [ ] **Step 3: py_compile + commit.** Commit `feat(store): read-only train_from_store + store-net gate_net`.

---

### Task 4: Wire `train_sf` + Modal idempotency validation

**Files:** Modify `modal/app.py`

- [ ] **Step 1: Rewrite the `train_sf` body** to label-into-store then train-from-store then gate. **Replace lines 599-611** (the old `label_args`/`shard_names`/`counts`/`label_sf_shard.starmap`/`train_sf_run.remote`/gate block) with the following. The existing `list(prepare_export.starmap(...))` at line 598 stays (shown here for placement — do NOT duplicate it):

```python
    list(prepare_export.starmap([(m["name"], m["url"], m["sha256"]) for m in corpus]))  # existing line 598
    ensure_sf_labels(corpus, per_game, nodes, shards_per_month)
    net_bytes = train_from_store.remote([_sf_dataset(nodes, per_game)], hidden, epochs)
    print(f"trained network: {len(net_bytes)} bytes")
    print(nnue_gate_run.remote(net_bytes, gate_depth, gate_openings, gate_plies,
                               gate_shard_size, move_time_ms))
```

Update the docstring to note labeling is now cached (a fully-labeled corpus skips straight to training). py_compile; commit `feat(store): train_sf uses the persistent label store`.

- [ ] **Step 2: PR + merge.** Verify the diff is `modal/app.py` + spec + plan only. Open the PR (superpowers:finishing-a-development-branch); body: labeling now amortized via an append-only store; SF path only. Merge on green (no Rust changed; `Rusty Fish Tests` passes trivially; the search-based suites are unaffected).

- [ ] **Step 3: Idempotency validation on Modal (the acceptance test).** After merge (or from the branch — Modal builds the local tree):
  - **Cache-hit:** `modal run --detach modal/app.py::train_sf` (defaults: 6 months, 100k, pg4). Retrieve logs: confirm `SF store n100000-pg4: 6 months cached, labeling []` — **zero labeling** — then it trains + gates. This proves the migrated labels are reused with no Stockfish cost.
  - **Non-destruction:** `modal volume ls rusty-fish-labels sf/n100000-pg4` before and after — the 192 shards + 6 markers are unchanged (a train/label run never lowers the count).
  - **Delta:** pin `2017-07` in `assets/nnue/wdl-corpus.toml` (run `sha_probe` first for its SHA — commit the manifest), then `train_sf` and confirm the log shows `labeling ['2017-07']` only (the prior six still cached) and the store grows by that month's shards.
  - **Cheap sweep:** `train_from_store` implicitly re-runs on the cache-hit path with a different `--hidden`/`--epochs` and confirm no labeling happens — pure GPU cost.

- [ ] **Step 4: Update `D:/Work-Tracking/work-tracker-personal.md`:** the label store is live — SF labeling is now paid once and reused; each future NNUE experiment on an already-labeled corpus is train+gate only (the amortization that unlocks cheap architecture/hyperparameter sweeps). Note the follow-on specs (#2 gate ladder, #3 experiment harness, #4 cold-start) and that `train_wdl_run`'s cleanup footgun is still deferred.

---

## Out of scope

The gate ladder / experiment harness / cold-start reduction (specs #2–#4); `train_wdl_run`'s cleanup (deferred); cross-dataset position dedup; moving raw exports off `rusty-fish-wdl`; any `label-sf`/`gen-eval-positions`/trainer/RFNN change.
