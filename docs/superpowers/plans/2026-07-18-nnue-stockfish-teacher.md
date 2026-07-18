# NNUE Stockfish-Eval Teacher Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label ~3M Lichess-sampled positions with a fixed-100k-node Stockfish eval (a cleaner per-position teacher than the game outcome) and train the NNUE on those cp labels, to try to beat the WDL net's −37 Elo and the tuned hand-crafted eval. Still opt-in; adopt only on SPRT AcceptH1.

**Architecture:** A new `engine-bench gen-eval-positions` emits `fen<TAB>own_csv<TAB>opp_csv` from the same sampling as `gen-wdl-data`. A new `engine-bench label-sf` drives one persistent Stockfish process (`go nodes 100000`) to replace each FEN with its side-to-move cp eval, producing the trainer's existing `target<TAB>own<TAB>opp` format. The trainer needs no change (cp mode = `wdl_target=False`). A Modal `train_sf` pipeline bakes Stockfish into the image, fans labeling into a `/vol/sf/` subdirectory, trains hidden-512 in cp mode, and gates via the existing `gate_net`.

**Tech Stack:** Rust 2024 (`engine-bench`, `engine-core`, `engine-search`, `pgn-reader`, `shakmaty`), Stockfish (UCI subprocess), Python/PyTorch + Modal.

**Spec:** `docs/superpowers/specs/2026-07-18-nnue-stockfish-teacher-design.md`

---

## Global constraints

- **Never run Cargo/Rust binaries locally.** Rust unit tests run in the `Rusty Fish Tests` GHA workflow (`cargo test --workspace`; `engine-bench/**` and `engine-core/**` are in its filters). Format four-space by hand.
- **No CI on Python/Modal.** Verify Modal code with `uv run --python 3.12 python -m py_compile`; the real end-to-end validation is the Modal run (Task 4).
- **Modal runs** launch via `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 infisical run --env dev -- uv run --with modal --python 3.12 -- modal run --detach modal/app.py::<entrypoint>`, retrieved from `modal app logs <app-id>` (see the `modal-self-play-gating` memory). Modal builds from the local tree, so a branch runs before merge. The `rusty-fish-wdl` Volume already holds the six `export-<month>.pgn.zst` (cached) and the WDL `samples-*-*.tsv` at top level — SF shards go in `/vol/sf/` to stay disjoint.
- **Verify `gh` account is `benzsevern` before every remote op** (`gh auth switch --user benzsevern`); keyring PAT as `GH_TOKEN` for writes/PRs; push via tokenized URL if blocked; `git fetch origin --prune` before branching; **stage paths explicitly — never `git add -A`**.
- Conventional Commits. **Branch:** `feat/nnue-stockfish-teacher` (already created off latest main, spec committed).

## Background an engineer needs

**The WDL sampler** (`engine-bench/src/lib.rs`), which `gen-eval-positions` mirrors:
- `WdlSampleConfig { min_ply, end_trim, per_game, shard: (i, n) }` (lib.rs:1426).
- `WdlBuilder` is a `pgn_reader::Visitor`: `begin_movetext` computes `valid = in_shard && accepts_tags` and increments `stream_index` for every game (shard counts pre-filter); `san` (lib.rs:1529) tracks both `game.chess` and `game.board`, and for eligible plies (`ply >= min_ply`) captures `(active_features(&board, stm), active_features(&board, opposite(stm)))` **before** making the move, pushing `(own, opp, stm, ply)` to `game.positions` only after the move applies (`valid=false` on failure); `end_game` (lib.rs:1561) early-returns unless the `Result` tag is `1-0`/`0-1`/`1/2-1/2`, trims the last `end_trim` plies, and `evenly_spaced(&eligible, per_game)` subsamples.
- `gen_wdl_data_samples_from_reader<R: Read>(reader, config)` runs `Reader::new(reader).visit_all_games(&mut builder)`. `active_features`, `opposite`, `evenly_spaced` are reusable free items.

**The UCI helper** (`engine-bench/src/lib.rs:799`), which `label-sf` reuses:
- `UciProcess { child, stdin, stdout: Receiver<Result<String,String>>, response_timeout }`; `start(path, timeout)` spawns, runs `uci`/`uciok` + `setoption Threads 1` + `setoption Hash 16` + `isready`/`readyok`; `send(cmd)` (`&mut self`), `wait_for(expected)` (`&mut self`), `next_line()` (`&self`) with timeout; `Drop` sends `quit`+kill. `best_move` (lib.rs:848) issues `go movetime` and reads only `bestmove` via `classify_bestmove_token` — **it discards all `score` lines**; there is no `go nodes`/`score cp`/`score mate` parsing anywhere yet.

**Board→FEN** (`engine-core/src/lib.rs:404` `to_fen`, `:310` `from_fen`) already exists and already round-trips in a test (lib.rs:1593). `best_move` already feeds Stockfish `position fen {board.to_fen()}`.

**The trainer** (`modal/train_nnue.py`): `train(data_path, hidden, epochs, batch_size, lr, device, wdl_target=False)`. With `wdl_target=False` (the default) it fits `sigmoid(pred_cp/400)` to `sigmoid(target/400)` — exactly cp-teacher training. Reads `target<TAB>own_csv<TAB>opp_csv`. **No change needed.**

**The Modal WDL pipeline** (`modal/app.py`): `prepare_export`, `label_wdl_shard`, `train_wdl_run` (globs `/vol/samples-*-*.tsv`, deletes non-expected `/vol/samples-*.tsv`, trains `wdl_target=True`), `nnue_gate_run`/`gate_net`, `_load_wdl_corpus`, the `rust_image`/`torch_image`/`wdl_volume` objects.

## File structure

| Path | Change | Responsibility |
|------|--------|----------------|
| `engine-bench/src/lib.rs` | Modify | `EvalPositionBuilder` (FEN-capturing sampler), `gen_eval_positions_from_reader`, `UciProcess::score_position` + cp/mate parsing. |
| `engine-bench/src/main.rs` | Modify | `gen-eval-positions` and `label-sf` commands. |
| `modal/app.py` | Modify | Stockfish in `rust_image`, `label_sf_shard`, `train_sf_run`, `train_sf` entrypoint (all `/vol/sf/`). |

No `engine-core`, `engine-search`, or `train_nnue.py` change (board→FEN and cp-mode both already exist).

---

### Task 1: `gen-eval-positions` — FEN-labelled position sampler

**Files:** Modify `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`

- [ ] **Step 1: Write the failing test (fixture PGN → fen+features lines)**

Add to the engine-bench test module (reuse the existing `WDL_FIXTURE`-style constant if present, else a 2400-rated game long enough to sample past `min_ply`):

```rust
#[test]
fn gen_eval_positions_emits_fen_and_valid_features() {
    let samples = gen_eval_positions(WDL_FIXTURE, WdlSampleConfig {
        min_ply: 8, end_trim: 5, per_game: 6, shard: (0, 1),
    });
    assert!(!samples.is_empty());
    for s in &samples {
        // FEN parses back to a legal position.
        assert!(engine_core::Board::from_fen(&s.fen).is_ok(), "fen parses: {}", s.fen);
        assert!(!s.own.is_empty() && !s.opp.is_empty());
        assert!(s.own.iter().all(|&i| i < 768) && s.opp.iter().all(|&i| i < 768));
    }
    // Same sampled positions as the WDL sampler for the same config (same games,
    // same plies) — only the payload differs.
    let wdl = gen_wdl_data_samples(WDL_FIXTURE, WdlSampleConfig {
        min_ply: 8, end_trim: 5, per_game: 6, shard: (0, 1),
    });
    assert_eq!(samples.len(), wdl.len());
    assert!(samples.iter().zip(&wdl).all(|(e, w)| e.own == w.own && e.opp == w.opp && e.ply == w.ply));
}
```

Expose `gen_eval_positions(pgn: &str, config: WdlSampleConfig) -> Vec<EvalPositionSample>` and `gen_eval_positions_from_reader<R: std::io::Read>(reader, config) -> Vec<EvalPositionSample>`, where `EvalPositionSample { pub fen: String, pub own: Vec<usize>, pub opp: Vec<usize>, pub ply: u32 }`.

- [ ] **Step 2: Push, confirm it fails to compile (types missing)**

- [ ] **Step 3: Implement `EvalPositionBuilder`**

A parallel `Visitor` mirroring `WdlBuilder`, reusing `WdlSampleConfig`, `accepts_tags` (make it a free fn or duplicate the 4-line filter), `active_features`, `opposite`, `evenly_spaced`. It captures the **FEN at the same pre-move point** as the features, and **keeps the same `Result`-tag filter in `end_game`** (so the sampled game set matches `gen-wdl-data`), but attaches no outcome target:

```rust
struct EvalPositionGame {
    chess: Chess,
    board: Board,
    // (own, opp, fen, ply) for every eligible ply
    positions: Vec<(Vec<usize>, Vec<usize>, String, u32)>,
    ply: u32,
    valid: bool,
    result: String,
}

struct EvalPositionBuilder { config: WdlSampleConfig, stream_index: usize, out: Vec<EvalPositionSample> }
```

`san` mirrors `WdlBuilder::san` exactly, except the eligible-ply capture is
`(active_features(&board, stm), active_features(&board, opposite(stm)), game.board.to_fen())`,
pushed as `(own, opp, fen, ply)` after the move applies. `end_game` mirrors `WdlBuilder::end_game` (same `1-0|0-1|1/2-1/2` guard, same `end_trim` filter, same `evenly_spaced`) but pushes `EvalPositionSample { fen, own, opp, ply }` with no target. `gen_eval_positions_from_reader` runs `Reader::new(reader).visit_all_games(&mut builder).expect(...)`; `gen_eval_positions(&str, ...)` delegates via `pgn.as_bytes()`.

(If the duplication of the `san`/`end_game` body reads poorly to the code-quality reviewer, an acceptable refactor is to extract the shared board-tracking + trim/subsample into helpers both builders call; do not restructure `WdlBuilder`'s output — `gen-wdl-data` must stay byte-identical.)

- [ ] **Step 4: Add the `gen-eval-positions` command**

In `engine-bench/src/main.rs`, mirror the `gen-wdl-data` command: `gen-eval-positions <pgn_or_-> [--shard i/n] [--per-game N]` reads a path or `-` (stdin) via `gen_eval_positions_from_reader`, prints each as `{fen}\t{own_csv}\t{opp_csv}` (indices joined by `,`). Defaults: `min_ply 8, end_trim 5, per_game 12, shard 0/1`. Reject `--per-game 0` / `--shard n==0` as the WDL command does.

- [ ] **Step 5: More tests + green**

Determinism (same input → same output), `--shard` disjoint partition matches `gen-wdl-data`'s partition, and per-game cap. Push, confirm green in `Rusty Fish Tests`.

---

### Task 2: `label-sf` — Stockfish cp labeller

**Files:** Modify `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`

- [ ] **Step 1: Write failing tests (score parsing, transcript-based)**

The score extraction and mate-clamp are pure and must be unit-tested without spawning Stockfish. Factor them into a free fn and test it:

```rust
#[test]
fn parse_uci_score_reads_last_cp_and_clamps_mate() {
    // cp from the last info line before bestmove
    assert_eq!(parse_uci_score_cp("info depth 20 score cp 37 nodes 1 pv e2e4"), Some(37));
    assert_eq!(parse_uci_score_cp("info depth 20 score cp -145 pv"), Some(-145));
    // mate -> clamped +/- MATE_CP
    assert_eq!(parse_uci_score_cp("info depth 30 score mate 3 pv"), Some(MATE_CP));
    assert_eq!(parse_uci_score_cp("info depth 30 score mate -2 pv"), Some(-MATE_CP));
    // non-score info line -> None (caller keeps the previous score)
    assert_eq!(parse_uci_score_cp("info depth 1 nodes 20"), None);
}
```

with `pub const MATE_CP: i32 = 30000;` (well past the trainer's `EVAL_CLAMP`/sigmoid saturation) and `fn parse_uci_score_cp(line: &str) -> Option<i32>` scanning tokens for `score cp N` (→ `N`) or `score mate N` (→ `MATE_CP * sign(N)`), ignoring `lowerbound`/`upperbound` qualifiers (keep the value).

- [ ] **Step 2: Push, confirm it fails (fn/const missing)**

- [ ] **Step 3: Implement `UciProcess::score_position` + the command**

Add to `impl UciProcess`:

```rust
/// Fixed-node Stockfish evaluation of `fen`, side-to-move-relative centipawns.
/// Sends ucinewgame first to clear the TT so a fixed node count is reproducible
/// regardless of scan order.
fn score_position(&mut self, fen: &str, nodes: u64) -> Result<i32, String> {
    self.send("ucinewgame")?;
    self.send("isready")?;
    self.wait_for("readyok")?;
    self.send(&format!("position fen {fen}"))?;
    self.send(&format!("go nodes {nodes}"))?;
    let mut last = None;
    loop {
        let line = self.next_line()?;
        if let Some(cp) = parse_uci_score_cp(&line) {
            last = Some(cp);
        }
        if line.starts_with("bestmove") {
            return last.ok_or_else(|| format!("no score for fen {fen}"));
        }
    }
}
```

Add a `label-sf <positions_or_-> <nodes> [--engine PATH]` command in `main.rs` (default engine path `/usr/games/stockfish`, the Debian package location used on Modal): start one `UciProcess` with a generous `response_timeout` (e.g. 60s), read `fen<TAB>own<TAB>opp` lines from a path or `-`, and for each print `{cp}\t{own}\t{opp}`. On a `score_position` error for a line (or a malformed line), increment a skip counter and `eprintln!` it at the end rather than aborting — one bad position must not kill a shard of tens of thousands.

- [ ] **Step 4: Green**

Push, confirm the transcript tests pass in `Rusty Fish Tests`. (No Stockfish on the CI runner is required — the parsing tests are transcript-based; `score_position` itself is exercised on Modal in Task 4.)

---

### Task 3: Modal `train_sf` pipeline

**Files:** Modify `modal/app.py`

- [ ] **Step 1: Stockfish in `rust_image`**

Add `"stockfish"` to `rust_image`'s `.apt_install(...)` (Debian ships a baseline x86-64 build at `/usr/games/stockfish`, so no AVX/SIGILL risk and no tar extraction; this is the pragmatic reproducible-enough choice for a first attempt — a pinned release build is a later refinement). This rebuilds the image on first run (expected).

- [ ] **Step 2: `label_sf_shard`**

```python
@app.function(image=rust_image, volumes={"/vol": wdl_volume}, timeout=60 * 90)
def label_sf_shard(name: str, i: int, n: int, per_game: int, nodes: int) -> int:
    import os
    os.makedirs("/vol/sf", exist_ok=True)
    out = f"/vol/sf/samples-{name}-{i}.tsv"
    cmd = (
        f"set -euo pipefail; zstdcat /vol/export-{name}.pgn.zst | "
        f"{BIN} gen-eval-positions - --shard {i}/{n} --per-game {per_game} | "
        f"{BIN} label-sf - {nodes} > {out}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)
    wdl_volume.commit()
    with open(out, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
```

- [ ] **Step 3: `train_sf_run` (GPU, cp mode, `/vol/sf/` shards)**

Copy `train_wdl_run` but: glob/select `/vol/sf/samples-*-*.tsv`; delete stale shards only within `/vol/sf/` (the same explicit-expected-set cleanup, scoped so it never touches top-level WDL shards); write `data.tsv` to `/vol/sf/data.tsv`; call `train_nnue.train(..., wdl_target=False)` (cp mode). Keep `memory=32768`, `timeout=60*60*3`, `reload()`. Signature: `train_sf_run(shard_names: list[str], hidden: int, epochs: int) -> bytes` where `shard_names` are `sf/samples-<month>-<i>.tsv` basenames under `/vol/`.

- [ ] **Step 4: `train_sf` entrypoint**

Mirror `train_wdl`: params `shards_per_month=16, per_game=4, nodes=100000, hidden=512, epochs=60, gate_openings=2048, gate_plies=8, gate_depth=64, gate_shard_size=16, move_time_ms=50, months=""`. Load corpus, optional `months` subset + non-empty/unknown asserts, `list(prepare_export.starmap(...))`, `label_sf_shard.starmap([(m["name"], i, shards_per_month, per_game, nodes) ...])`, build the `/vol/sf/` `shard_names`, `train_sf_run.remote(shard_names, hidden, epochs)`, then `nnue_gate_run.remote(net_bytes, gate_depth, gate_openings, gate_plies, gate_shard_size, move_time_ms)`. Docstring with short-validation + real-run invocations.

- [ ] **Step 5: `py_compile` + commit.** `uv run --python 3.12 python -m py_compile modal/app.py`.

---

### Task 4: PR, merge, run on Modal, assess

- [ ] **Step 1:** Verify the branch diff is exactly `engine-bench/src/{lib.rs,main.rs}`, `modal/app.py`, the spec, and this plan — no strays. Open the PR (superpowers:finishing-a-development-branch); body: ships nothing by default (NNUE opt-in), the powered gate is the acceptance test.
- [ ] **Step 2:** Merge on green (`Rusty Fish Tests`; Python isn't CI-gated).
- [ ] **Step 3: Short Modal validation** — one month, tiny config, low nodes, to confirm the Stockfish binary runs, the `gen-eval-positions | label-sf` pipe works, cp-mode training runs, and the gate returns a verdict:

```
… modal run --detach modal/app.py::train_sf --months 2017-01 --shards-per-month 2 --per-game 2 --nodes 5000 --hidden 64 --epochs 2 --gate-openings 64 --gate-shard-size 16
```

Retrieve via `modal app logs`: a labeled-count, a trained-net size, an `NNUE_GATE_RESULT`. **Also confirm timing** — note seconds/position at 5000 nodes to sanity-check the real run's 100k-node fan-out fits `label_sf_shard`'s 90-min timeout (raise `shards_per_month` if a shard would exceed it).

- [ ] **Step 4: Real run** — six months, ~3M positions, 100k nodes, hidden 512, powered gate:

```
… modal run --detach modal/app.py::train_sf --shards-per-month 16 --per-game 4 --nodes 100000 --hidden 512 --epochs 60 --gate-openings 2048 --gate-shard-size 16 --move-time-ms 50
```

Watch the training log (`train_wdl_loss`/`val_wdl_loss` — now a cp-sigmoid loss, not directly comparable to the WDL loss). Retrieve the `NNUE_GATE_RESULT`.

- [ ] **Step 5: Assess** vs the WDL net (−37.4 Elo) and the tuned eval:
  - **AcceptH1 (beats the eval):** follow-up commits the `.rfnn` as an asset, wires it as the default eval, re-gated by CI; update `D:/Work-Tracking/work-tracker-personal.md`.
  - **Beats −37 but not the eval:** the cleaner teacher helped; next is the λ-blend or more positions / higher nodes. Update the tracker.
  - **Flat/worse than −37:** SF-eval at this scale isn't the lever; reconsider (HalfKA, far more data). Update the tracker.

---

## Out of scope

The λ·outcome + SF blend teacher, HalfKA, a pinned-release (vs apt) Stockfish, multi-PV/WDL Stockfish output, and any change to the trainer, RFNN format, or inference path.
