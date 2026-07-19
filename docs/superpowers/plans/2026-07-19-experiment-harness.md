# Experiment Harness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run many NNUE experiments from one `sweep` command (parallel) and log results to an append-only ledger, so architecture/hyperparameter sweeps are casual. Pure orchestration + a ledger over the existing label store (#1) and gate ladder (#2).

**Architecture:** `run_experiment(config)` (CPU) does `train_from_store` → val pre-check → `gate_ladder_run` → structured result (catches its own errors). `sweep` fans a cross-product of `--hiddens/--epochs-list/--lrs` across `run_experiment.starmap`, then one serial `append_results` to `/store/experiments/results.tsv`. `results` prints the ledger. Only enabling change: thread `lr` through `train_from_store`.

**Tech Stack:** Python/Modal (`modal/app.py`). No Rust.

**Spec:** `docs/superpowers/specs/2026-07-19-experiment-harness-design.md`

---

## Global constraints

- **Python/Modal only.** No Rust, no CI gate. Verify each edit with `uv run --python 3.12 python -m py_compile modal/app.py`. The end-to-end check is a Modal run (Task 4). Spurious Pyright `.remote`/`.starmap` warnings are expected — ignore.
- **Modal runs** via `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 infisical run --env dev -- uv run --with modal --python 3.12 -- modal run modal/app.py::<entrypoint> ...`.
- **gh account `benzsevern`**; stage `modal/app.py` explicitly — never `git add -A`; push to `feat/experiment-harness` (tokenized URL fallback).
- Conventional Commits. **Branch:** `feat/experiment-harness` (created off latest main, spec committed).

## Background: the exact current code (`modal/app.py`)

- `labels_volume = modal.Volume.from_name("rusty-fish-labels", …)`. Store functions mount `volumes={"/store": labels_volume}`, `reload()` before read, `commit()` after write, default image.
- `train_from_store(datasets: list[str], hidden: int, epochs: int) -> tuple[bytes, float]` (~402): reloads, globs the datasets, concats to `/tmp/data.tsv`, `model, val_loss = train_nnue.train(data_path, hidden, epochs, batch_size=1024, lr=1e-3, device="cuda", wdl_target=False)` (lr hardcoded), writes `/store/nets/latest.rfnn`, returns `(bytes, val_loss)`.
- `gate_ladder_run(net_bytes, gate_depth, gate_plies, move_time_ms, gate_shard_size, chunk_openings=256, max_openings=8192) -> str` (~583): sequential SPRT; returns a string wrapping the summary line
  `gate ladder: {W}W {D}D {L}L over {played*2} games ({played}/{max_openings} openings), decision {decision}`
  inside `NNUE_LADDER_RESULT_BEGIN/END`.
- `math` is only imported locally inside `train_sf`; `itertools`/`datetime` are NOT imported anywhere — add them at the right scope.

## File structure

Only `modal/app.py` changes.

---

### Task 1: Thread `lr` through `train_from_store`

**Files:** Modify `modal/app.py`

- [ ] **Step 1:** Add `lr: float = 1e-3` to `train_from_store`'s signature (after `epochs`) and replace the hardcoded `lr=1e-3` in its `train_nnue.train(...)` call with `lr=lr`. Nothing else changes (its two callers — `train_sf` and the new `run_experiment` — pass `lr` positionally/by-keyword; `train_sf` keeps the default by not passing it, so it's unchanged).
- [ ] **Step 2:** `uv run --python 3.12 python -m py_compile modal/app.py`. Commit `feat(harness): thread lr through train_from_store`.

---

### Task 2: `run_experiment`, `append_results`, `read_results`

**Files:** Modify `modal/app.py`

- [ ] **Step 1: `run_experiment` — one experiment, structured result, error-safe.**

```python
@app.function(timeout=60 * 60 * 3)
def run_experiment(config: dict) -> dict:
    """Train one net from the store, val-precheck it, gate it vs the champion, and
    return a structured result. Catches its own errors so a bad config becomes an
    `error` row rather than sinking the whole sweep."""
    import math
    import re

    base = {
        "dataset": config["dataset"], "hidden": config["hidden"],
        "epochs": config["epochs"], "lr": config["lr"], "val_loss": "NA",
        "wins": "NA", "draws": "NA", "losses": "NA", "games": "NA",
        "elo": "NA", "decision": "error",
    }
    try:
        net_bytes, val_loss = train_from_store.remote(
            [config["dataset"]], config["hidden"], config["epochs"], config["lr"]
        )
        base["val_loss"] = val_loss
        if math.isnan(val_loss) or val_loss > 0.1:
            return {**base, "decision": "rejected"}
        verdict = gate_ladder_run.remote(
            net_bytes, config["gate_depth"], config["gate_plies"], config["move_time_ms"],
            config["gate_shard_size"], config["chunk_openings"], config["max_openings"],
        )
        m = re.search(r"gate ladder: (\d+)W (\d+)D (\d+)L over (\d+) games .* decision (\w+)", verdict)
        if not m:
            return base  # decision stays "error"
        w, d, l, games = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        total = w + d + l
        if total == 0:
            elo = "NA"
        else:
            score = (w + 0.5 * d) / total
            elo = -800.0 if score <= 0 else 800.0 if score >= 1 else round(-400 * math.log10(1 / score - 1), 1)
        return {**base, "wins": w, "draws": d, "losses": l, "games": games, "elo": elo, "decision": m[5]}
    except Exception as error:  # noqa: BLE001 — any failure becomes one error row
        return {**base, "decision": f"error: {error}"[:120]}
```

- [ ] **Step 2: `append_results` — one serial, race-free ledger write.**

```python
@app.function(volumes={"/store": labels_volume}, timeout=60 * 10)
def append_results(rows: list[str]) -> None:
    import os
    labels_volume.reload()
    os.makedirs("/store/experiments", exist_ok=True)
    path = "/store/experiments/results.tsv"
    header = ("sweep_id\ttimestamp\tdataset\thidden\tepochs\tlr\tval_loss\t"
              "wins\tdraws\tlosses\tgames\telo\tdecision")
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as handle:
        if not exists:
            handle.write(header + "\n")
        for row in rows:
            handle.write(row + "\n")
    labels_volume.commit()
```

- [ ] **Step 3: `read_results`.**

```python
@app.function(volumes={"/store": labels_volume})
def read_results() -> str:
    import os
    labels_volume.reload()
    path = "/store/experiments/results.tsv"
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()
```

- [ ] **Step 4:** `py_compile`; commit `feat(harness): run_experiment + append/read results ledger`.

---

### Task 3: `sweep` and `results` entrypoints

**Files:** Modify `modal/app.py`

- [ ] **Step 1: `sweep` — cross-product, parallel, one serial append.**

```python
@app.local_entrypoint()
def sweep(
    hiddens: str = "512", epochs_list: str = "60", lrs: str = "1e-3",
    dataset: str = "n100000-pg4", gate_depth: int = 64, gate_plies: int = 8,
    move_time_ms: int = 50, gate_shard_size: int = 16,
    chunk_openings: int = 256, max_openings: int = 8192,
):
    """Sweep a cross-product of (hidden, epochs, lr) — train each from the store and
    gate it vs the champion, in parallel — and append the results to the ledger.

        modal run modal/app.py::sweep --hiddens 256,512,1024 --epochs-list 40,80 --lrs 1e-3,5e-4
    """
    import datetime
    import itertools

    hs = [int(x) for x in hiddens.split(",") if x.strip()]
    es = [int(x) for x in epochs_list.split(",") if x.strip()]
    ls = [float(x) for x in lrs.split(",") if x.strip()]
    assert hs and es and ls, "each of --hiddens/--epochs-list/--lrs needs >=1 value"
    configs = [
        {"dataset": dataset, "hidden": h, "epochs": e, "lr": l,
         "gate_depth": gate_depth, "gate_plies": gate_plies, "move_time_ms": move_time_ms,
         "gate_shard_size": gate_shard_size, "chunk_openings": chunk_openings, "max_openings": max_openings}
        for h, e, l in itertools.product(hs, es, ls)
    ]
    print(f"sweep: {len(configs)} experiments")
    results = list(run_experiment.starmap([(c,) for c in configs]))

    sweep_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    ts = datetime.datetime.utcnow().isoformat()
    rows = ["\t".join(str(x) for x in [
        sweep_id, ts, r["dataset"], r["hidden"], r["epochs"], r["lr"], r["val_loss"],
        r["wins"], r["draws"], r["losses"], r["games"], r["elo"], r["decision"]]) for r in results]
    append_results.remote(rows)

    def elo_key(r):
        try:
            return float(r["elo"])
        except (TypeError, ValueError):
            return -1e9
    print("=== sweep results (best first) ===")
    for r in sorted(results, key=elo_key, reverse=True):
        print(f"hidden={r['hidden']} epochs={r['epochs']} lr={r['lr']} "
              f"val_loss={r['val_loss']} elo={r['elo']} decision={r['decision']}")
```

- [ ] **Step 2: `results` — print the ledger sorted by Elo.**

```python
@app.local_entrypoint()
def results():
    """Print every logged experiment, best Elo first."""
    content = read_results.remote()
    if not content.strip():
        print("no experiments logged yet")
        return
    lines = content.strip().splitlines()
    header, rows = lines[0], lines[1:]

    def elo_key(line):
        try:
            return float(line.split("\t")[11])  # elo column
        except (IndexError, ValueError):
            return -1e9
    print(header)
    for line in sorted(rows, key=elo_key, reverse=True):
        print(line)
```

- [ ] **Step 3:** `py_compile`; commit `feat(harness): sweep + results entrypoints`.

---

### Task 4: PR, merge, Modal validation

- [ ] **Step 1:** Verify the branch diff is `modal/app.py` + spec + plan only. Open the PR (superpowers:finishing-a-development-branch); body: parallel sweep + append-only results ledger over the store + gate ladder.
- [ ] **Step 2: Merge on green** (no Rust; Python isn't CI-gated).
- [ ] **Step 3: Modal validation** (from the branch or after merge — Modal builds the local tree):
  - **Tiny sweep:** `modal run modal/app.py::sweep --hiddens 8,16 --epochs-list 1 --chunk-openings 128 --max-openings 256`. Confirm: 2 experiments run in parallel; the printed table shows 2 rows (each with a `val_loss` and a gate `decision` — likely `AcceptH0`, tiny nets lose to the champion); `modal run modal/app.py::results` prints a header + 2 rows sorted by Elo; and `modal volume ls rusty-fish-labels experiments` shows `results.tsv`.
  - **Append/persistence:** run a 1-experiment sweep (`--hiddens 8 --epochs-list 1 --chunk-openings 128 --max-openings 256`), then `results` — confirm the ledger **grew to 3 rows** (append, header NOT duplicated).
- [ ] **Step 4: Update `D:/Work-Tracking/work-tracker-personal.md`:** the experiment harness is live — one `sweep` command runs a parallel cross-product of configs (train-from-store + champion gate) and logs to an append-only `/store/experiments/results.tsv`; `results` reviews everything tried. With #1/#2/#3 done, sweeping architectures/hyperparameters is now a one-liner. Note the last follow-on (#4 cold-start) and that adopting a sweep winner stays the manual bundle-and-ship flow.

---

## Out of scope

Auto-adopting a sweep winner; cold-start (#4); sweeping the SF-teacher/data axes (they change the labels — go through `ensure_sf_labels`); cross-dataset dedup; any training/gating/RFNN/label-format change.

**Known caveat (not fixed here):** every parallel experiment writes `/store/nets/latest.rfnn` (pre-existing `train_from_store` behavior), so after a sweep that file is whichever training finished last — an arbitrary sweep member. This does NOT affect sweep correctness (each `run_experiment` gates the `net_bytes` returned to it, never the store path), but a standalone `gate_net`/`gate_ladder` run *after* a sweep would re-gate an arbitrary net, not a chosen one. Adopting a specific sweep winner is the manual bundle-and-ship flow, which re-trains/loads the intended net explicitly.
