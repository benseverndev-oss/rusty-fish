# Gate Ladder Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cheap NNUE gating: gate a candidate net vs the bundled **champion** net (net-vs-net), with a free val-loss pre-check and a sequential SPRT that early-stops (clearly-worse nets reject in ~1–2 chunks). Resolves the "gates need an NNUE baseline now" caveat.

**Architecture:** Rust — a `BaselineMode { Champion, Handcrafted }` threaded through the NNUE gauntlet so the baseline uses `Searcher::default()`'s bundled champion (default) or hand-crafted (opt-in); `gate-file` gains an optional mode token. Python — `train_nnue.train()` returns `(model, val_loss)`. Modal — `train_from_store` returns `(net_bytes, val_loss)`; a `gate_ladder_run` does the chunk→sum→SPRT→short-circuit loop; `train_sf` does the val pre-check then the ladder.

**Tech Stack:** Rust 2024 (`engine-bench`, `engine-search`), Python/PyTorch (`train_nnue.py`), Modal (`app.py`).

**Spec:** `docs/superpowers/specs/2026-07-19-gate-ladder-design.md`

---

## Global constraints

- **Never run Cargo locally.** Rust tests run in the `Rusty Fish Tests` GHA workflow (`cargo test --workspace`; `engine-bench/**` is in its filters). Python/Modal has no CI — verify with `uv run --with torch` / `py_compile`; the end-to-end check is a Modal run (Task 4). Four-space format by hand.
- **Modal runs** via `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 infisical run --env dev -- uv run --with modal --python 3.12 -- modal run [--detach] modal/app.py::<entrypoint>`, retrieved from `modal app logs`.
- **gh account `benzsevern`** before remote ops; stage paths explicitly — never `git add -A`; push to `feat/gate-ladder` (tokenized URL fallback).
- Conventional Commits. **Branch:** `feat/gate-ladder` (created off latest main, spec committed).

## Background: the exact current code

- **`engine-bench/src/lib.rs`:** `play_nnue_game(fen, candidate_color, net, config, move_time)` (~429) builds `candidate = Searcher::default(); candidate.set_nnue(Some(Arc::clone(net)))` and `baseline = Searcher::default(); baseline.set_nnue(None) // hand-crafted`. The call chain: `run_nnue_gauntlet(positions, net, config)` (352) → `run_nnue_gauntlet_with_move_time(…, move_time)` (362) → `run_nnue_gauntlet_with_optional_move_time(…, move_time: Option<Duration>)` (371) → loops `play_nnue_game`. `Searcher::default()` installs the bundled champion net (adoption flipped `Default`; `has_nnue()` is true). `nnue-sprt` command (`main.rs:176`) uses `run_nnue_gauntlet`.
- **`engine-bench/src/main.rs`:** `gate-file <net> <depth> <openings_file> [move_time_ms]` (216-247): parses net/depth/openings, `move_time = Duration::from_millis(arg_u64(5).unwrap_or(100))`, calls `run_nnue_gauntlet_with_move_time`, prints `W\tD\tL`.
- **`engine-bench` `sprt` command / `sprt_tsv_report`** emits a TSV whose last column `decision` is the bare token `AcceptH0`/`AcceptH1`/`Continue` (config `elo0=0, elo1=5, alpha=beta=0.05`).
- **`modal/train_nnue.py`:** `train(...)` (ends ~185) returns only `model`; `val` (the per-epoch `mean_loss_batched(val_idx)`) is printed to stderr. Callers: `train_net` (`app.py:~220`), `train_wdl_run` (`app.py:~386`), `train_from_store` (`app.py`) all do `model = train(...)`.
- **`modal/app.py`:** `gate_shard(net_bytes, depth, openings_text, move_time_ms=0)` (~130) calls `[BIN, "gate-file", net_path, str(depth), openings_path]` + `str(move_time_ms)` when truthy. `nnue_gate_run(net_bytes, gate_depth, gate_openings, gate_plies, gate_shard_size, move_time_ms)` (~473): `make_openings` → `_chunks` → `gate_shard.starmap` → sum → `sprt_verdict` → prints `NNUE_GATE_RESULT`. `sprt_verdict` returns `stdout+stderr`. `make_openings(count, plies, seed)`. `train_from_store(datasets, hidden, epochs) -> bytes`. `train_sf` calls `net_bytes = train_from_store.remote(...)` then `nnue_gate_run.remote(...)`. `gate_net`/`read_net` read `/store/nets/latest.rfnn`. Stale "vs the hand-crafted baseline" docstrings on `gate_shard`, `nnue_gate_run`, `train_sf`, `train_wdl`, `gate_net`.

---

### Task 1: Rust net-vs-champion gate (`BaselineMode`)

**Files:** Modify `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`

- [ ] **Step 1: Write the failing test.** Factor the baseline construction into a testable helper and test it:

```rust
#[test]
fn baseline_mode_selects_champion_net_or_handcrafted() {
    assert!(baseline_searcher(BaselineMode::Champion).has_nnue(), "champion baseline keeps the bundled net");
    assert!(!baseline_searcher(BaselineMode::Handcrafted).has_nnue(), "handcrafted baseline disables NNUE");
}
```

- [ ] **Step 2: Push, confirm it fails to compile (`BaselineMode`/`baseline_searcher` missing).**

- [ ] **Step 3: Implement.** In `lib.rs`:

```rust
#[derive(Clone, Copy)]
pub enum BaselineMode { Champion, Handcrafted }

fn baseline_searcher(mode: BaselineMode) -> Searcher {
    let mut baseline = Searcher::default(); // Default installs the bundled champion net.
    if matches!(mode, BaselineMode::Handcrafted) {
        baseline.set_nnue(None);
    }
    baseline
}
```

Thread `mode: BaselineMode` into `run_nnue_gauntlet_with_optional_move_time` and `play_nnue_game`; in `play_nnue_game` build `let mut baseline = baseline_searcher(mode);` (replacing the inline `Searcher::default()` + `set_nnue(None)`). The two public wrappers `run_nnue_gauntlet` and `run_nnue_gauntlet_with_move_time` keep their current signatures and pass `BaselineMode::Champion`. Add one public entry `run_nnue_gauntlet_with_move_time_and_baseline(positions, net, config, move_time, mode)` for `gate-file` to call.

- [ ] **Step 4: `gate-file` mode arg.** In `main.rs`, **add `BaselineMode` and `run_nnue_gauntlet_with_move_time_and_baseline` to the `use engine_bench::{…}` import block** (lines 3-14), and **drop the now-unused `run_nnue_gauntlet_with_move_time` import** (gate-file was its only main.rs user; leaving it is an unused-import warning). Parse the mode by **token match** (robust to position): `let mode = if std::env::args().any(|a| a == "handcrafted") { BaselineMode::Handcrafted } else { BaselineMode::Champion };`. Call `run_nnue_gauntlet_with_move_time_and_baseline(&fens, Arc::new(net), config, move_time, mode)`. Update the usage string to `gate-file <net> <depth> <openings_file> [move_time_ms] [champion|handcrafted]` and the command's comment (it now defaults to **champion**, not hand-crafted).

- [ ] **Step 5: Push, confirm green.** (`nnue-sprt` still compiles — `run_nnue_gauntlet` signature unchanged.)

---

### Task 2: `train()` returns the final validation loss

**Files:** Modify `modal/train_nnue.py`, `modal/app.py`

- [ ] **Step 1: `train()` returns `(model, val_loss)`.** In `train_nnue.py`, init `val = float("nan")` before the epoch loop (so it is defined even for `epochs == 0`), keep the last epoch's `val`, and change `return model` → `return model, val`.

- [ ] **Step 2: Update ALL FOUR callers to unpack.** There are four, not three:
  - `modal/train_nnue.py`'s own `main()` (~line 238): `model, _ = train(...)` before `quantize_and_write(model, ...)` — else `model` is a tuple and the standalone `python train_nnue.py` path breaks. (Same file as Step 1.)
  - `app.py` `train_net` and `train_wdl_run` → `model, _ = train_nnue.train(...)` (they don't need the loss).
  - `app.py` `train_from_store` → `model, val_loss = train_nnue.train(...)` and change its return to `return (handle.read(), val_loss)` (net bytes AND loss); its return annotation becomes `-> tuple[bytes, float]`.

- [ ] **Step 3: Update `train_sf` to unpack `train_from_store`'s new return** (keep the OLD `nnue_gate_run` gate for now — the ladder is Task 3): `net_bytes, val_loss = train_from_store.remote([_sf_dataset(nodes, per_game)], hidden, epochs)` (the `val_loss` is used in Task 3). `nnue_gate_run.remote(net_bytes, ...)` unchanged this task.

- [ ] **Step 4: Verify + commit.** `uv run --with torch --python 3.12 python -c "..."` — a tiny 3-sample `train(...)` now returns a 2-tuple `(model, float)` and `quantize_and_write(model, ...)` still works (assert `isinstance(t[1], float)`). `uv run --python 3.12 python -m py_compile modal/app.py`. Commit `feat(nnue): train() returns final val loss for the gate ladder`.

---

### Task 3: Modal sequential SPRT gate ladder (vs champion)

**Files:** Modify `modal/app.py`

- [ ] **Step 1: `gate_shard` forwards the baseline mode.** Add `baseline: str = "champion"` param; append it to the `gate-file` command **after** the movetime (so the positional order is `… [move_time_ms] [baseline]`): ensure a movetime is always passed when a non-default baseline is used (the ladder always passes movetime). Update `gate_shard`'s docstring (no longer "vs hand-crafted"; now "vs the bundled champion net by default").

- [ ] **Step 2: `gate_ladder_run` — the sequential SPRT.**

```python
@app.function(image=rust_image, timeout=60 * 60)
def gate_ladder_run(
    net_bytes: bytes, gate_depth: int, gate_plies: int, move_time_ms: int,
    gate_shard_size: int, chunk_openings: int = 256, max_openings: int = 8192,
) -> str:
    """Sequential SPRT of a candidate net vs the bundled champion: play openings in
    chunks, check the SPRT on the cumulative W/D/L after each, stop on a decision."""
    wins = draws = losses = 0
    played = 0
    decision = "Continue"
    chunk = 0
    while played < max_openings:
        chunk += 1
        openings = [l for l in make_openings.remote(chunk_openings, gate_plies, chunk).splitlines() if l]
        shard_texts = list(_chunks(openings, gate_shard_size))
        for w, d, l in gate_shard.starmap(
            [(net_bytes, gate_depth, text, move_time_ms, "champion") for text in shard_texts]
        ):
            wins += w; draws += d; losses += l
        played += len(openings)
        verdict = sprt_verdict.remote(wins, draws, losses)
        # Defensively find the TSV values line: the last line whose final
        # tab-field is a decision token (avoids off-by-one if stderr is empty).
        tokens = {"AcceptH0", "AcceptH1", "Continue"}
        decision = next(
            (ln.split("\t")[-1] for ln in reversed(verdict.splitlines())
             if ln.split("\t")[-1] in tokens),
            "Continue",
        )
        if decision in ("AcceptH0", "AcceptH1"):
            break
    summary = (f"gate ladder: {wins}W {draws}D {losses}L over {played * 2} games "
               f"({played}/{max_openings} openings), decision {decision}")
    out = f"NNUE_LADDER_RESULT_BEGIN\n{summary}\nNNUE_LADDER_RESULT_END"
    print(out, flush=True)
    return out
```

(Confirm the exact index for the TSV `decision` column against `sprt_verdict`'s output — it returns `stdout+stderr`; the TSV values line is the one with tab-separated numbers ending in the decision token. Parse defensively: find the line whose last tab-field is one of the three tokens.)

- [ ] **Step 3: Wire `train_sf` — val pre-check then ladder.** Replace the `nnue_gate_run.remote(...)` call:

```python
    net_bytes, val_loss = train_from_store.remote([_sf_dataset(nodes, per_game)], hidden, epochs)
    print(f"trained network: {len(net_bytes)} bytes, val_loss {val_loss:.6f}")
    import math
    if math.isnan(val_loss) or val_loss > 0.1:
        print(f"NNUE_LADDER_RESULT_BEGIN\nrejected: val_loss {val_loss} failed the pre-check\nNNUE_LADDER_RESULT_END")
    else:
        print(gate_ladder_run.remote(net_bytes, gate_depth, gate_plies, move_time_ms,
                                     gate_shard_size, chunk_openings, max_openings))
```

Add `chunk_openings: int = 256, max_openings: int = 8192` to `train_sf`'s params (replace the old `gate_openings` param, which the ladder no longer uses — or keep it unused; prefer replacing). Keep `gate_depth`/`gate_shard_size`/`move_time_ms`.

- [ ] **Step 4: `gate_ladder` entrypoint** (mirror `gate_net`) — re-run the ladder on the stored net without retraining:

```python
@app.local_entrypoint()
def gate_ladder(gate_depth: int = 64, gate_plies: int = 8, move_time_ms: int = 50,
                gate_shard_size: int = 16, chunk_openings: int = 256, max_openings: int = 8192):
    net_bytes = read_net.remote()
    print(f"loaded net: {len(net_bytes)} bytes")
    print(gate_ladder_run.remote(net_bytes, gate_depth, gate_plies, move_time_ms,
                                 gate_shard_size, chunk_openings, max_openings))
```

- [ ] **Step 5: Docstrings + py_compile + commit.** Update the stale "vs the hand-crafted baseline" language in `gate_shard`, `nnue_gate_run`, `train_sf`, `train_wdl`, `gate_net` docstrings to "vs the bundled champion net" (they all gate net-vs-champion now). **Also fix `train_sf`'s docstring example invocations** (they still pass `--gate-openings 64` / `--gate-openings 2048`, an invalid flag after the swap) to `--chunk-openings`/`--max-openings`, consistent with Task 4 Step 3. `uv run --python 3.12 python -m py_compile modal/app.py`. Commit `feat(nnue): sequential SPRT gate ladder vs the champion net`.

---

### Task 4: PR, merge, Modal validation

- [ ] **Step 1:** Verify the branch diff is `engine-bench/src/{lib.rs,main.rs}`, `modal/{app.py,train_nnue.py}`, spec, plan — no strays. Open the PR (superpowers:finishing-a-development-branch); body: net-vs-champion gate + free val-precheck + early-stop sequential SPRT.
- [ ] **Step 2: Merge on green** (`Rusty Fish Tests` covers the Rust; Python isn't CI-gated).
- [ ] **Step 3: Modal validation** (from the branch or after merge — Modal builds the local tree):
  - **Self-gate sanity:** `gate_ladder` on the stored champion net (candidate == champion) — confirm from `modal app logs` it trends to **AcceptH0 / ~0 Elo** and stops (the unbiased check that champion-baseline works). Note games played.
  - **Early-reject:** train a deliberately weak net (`train_sf --hidden 8 --epochs 1 --chunk-openings 128 --max-openings 1024`) — confirm the ladder **rejects early** (AcceptH0, games far below the cap), proving the early-stop saves cost. (This also exercises the val pre-check path — a hidden-8/1-epoch net trains but should be far worse than the champion.)
- [ ] **Step 4: Update `D:/Work-Tracking/work-tracker-personal.md`:** the gate ladder is live — gating is now net-vs-champion with a free val-precheck and early-stop SPRT (bad candidates reject in a few hundred games). Note the follow-ons (#3 experiment harness, #4 cold-start) and that this + the label store make experiment iteration cheap end-to-end.

---

## Out of scope

The experiment harness (#3), cold-start reduction (#4), external-gauntlet changes, `train_wdl_run`'s cleanup (still deferred), auto-adopting a winner (bundling a new champion stays the manual adoption flow), and any RFNN/label-format change.
