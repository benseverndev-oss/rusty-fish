# NNUE on Lichess Game Outcomes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an NNUE that beats the tuned hand-crafted eval by learning from Lichess game outcomes (a teacher stronger than itself), gated powered on Modal; adopt only if it wins.

**Architecture:** A new `engine-bench gen-wdl-data` command samples middlegame positions from the pinned Lichess export and labels each with the game outcome (side-to-move-relative). A `--wdl-target` trainer mode uses the outcome directly as the win-probability. A Modal pipeline shards labeling into a Volume, trains a hidden-256 net on GPU, and gates it against the tuned hand-crafted eval with the powered self-play SPRT.

**Tech Stack:** Rust 2024 (`engine-bench`, `pgn-reader`, `engine-search`, `engine-core`), Python/PyTorch (Modal), Modal (via infisical + uv).

**Spec:** `docs/superpowers/specs/2026-07-18-nnue-lichess-wdl-design.md`

---

## Global constraints

- **Never run Cargo or Rust binaries locally.** Rust unit tests run in the `Rusty Fish Tests` GHA workflow (`cargo test --workspace`; `engine-bench/**` is in its filters). Modal runs build in-cloud from the local tree.
- **Modal runs** launch via `infisical run --env dev -- uv run --with modal -- modal run modal/app.py::<entrypoint>` with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`; use `--detach` for long runs and retrieve results from `modal app logs` (see the `modal-self-play-gating` memory). Modal builds from the local tree, so a branch can run before merge.
- **Verify `gh` account is `benzsevern` before every remote op**; keyring PAT as `GH_TOKEN` for writes/PRs; push via tokenized URL; `git fetch origin --prune` before branching; **stage paths explicitly — never `git add -A`**.
- `cargo fmt` style by hand; Conventional Commits.
- **Branch:** `feat/nnue-lichess-wdl`, on latest main with the spec committed.

## Background an engineer needs

**The NNUE** (`engine-search/src/nnue.rs`): a `768 → hidden → 1` side-to-move-relative perspective net. `pub fn active_features(board, perspective) -> Vec<usize>` returns the active feature indices (one per piece), and is re-exported and already used by `engine-bench`. The RFNN format is quantized i16.

**The sample format** (`engine-bench` `gen-data` command → `train.rs`): one line per sample, `target<TAB>own_feature_csv<TAB>opp_feature_csv`. The GPU trainer (`modal/train_nnue.py`) reads it, computes `target_wp = sigmoid(target/400)`, and fits `sigmoid(pred_cp/400)` to it with squared error.

**book-tool's PGN filter** (which we mirror, not reuse — book-tool is binary-only and its visitor tracks the board only to ply 16): a game is accepted iff `white_elo >= 2200 && black_elo >= 2200 && Event contains "rated" && (Variant empty or "standard")`. Moves are applied to an `engine_core::Board` via `UciMove::from_standard(san.san.to_move(&chess)).to_string()` → `board.parse_uci_move(uci)` → `board.make_move`. Result is the `Result` tag (`1-0`/`0-1`/`1/2-1/2`).

**The Modal NNUE loop** (`modal/app.py`): `label_shard` (self-play labels), `train_net` (GPU, calls the Python `train()` directly), `gate_shard`/`sprt_verdict` (NNUE candidate vs hand-crafted baseline), the `run()` train→gate entrypoint, and the `eval_gate_run` remote-aggregation pattern for detached retrieval.

## File structure

| Path | Change | Responsibility |
|------|--------|----------------|
| `engine-bench/Cargo.toml` | Modify | Add `pgn-reader = "0.29"`. |
| `engine-bench/src/lib.rs` | Modify | The WDL PGN visitor, sampling, `--shard`, `gen_wdl_data`. |
| `engine-bench/src/main.rs` | Modify | The `gen-wdl-data` command. |
| `modal/train_nnue.py` | Modify | `wdl_target` mode (function param + argparse), extract `target_win_prob`. |
| `modal/app.py` | Modify | `prepare_export`, `label_wdl_shard`, `train_wdl` pipeline (Volume). |

---

### Task 1: `gen-wdl-data` — the WDL sample generator

**Files:** Modify `engine-bench/Cargo.toml`, `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`

- [ ] **Step 1: Add the dependency**

In `engine-bench/Cargo.toml`, add to `[dependencies]`: `pgn-reader = "0.29"` (match the version `book-tool/Cargo.toml` uses).

- [ ] **Step 2: Write the failing test (fixture PGN → correct WDL samples)**

Add to the engine-bench test module. Fixture: a White-win game and a draw, both with 2200+ tags, long enough to sample past ply 8.

```rust
const WDL_FIXTURE: &str = concat!(
    "[Event \"Rated Blitz\"]\n[WhiteElo \"2400\"]\n[BlackElo \"2400\"]\n[Result \"1-0\"]\n\n",
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0\n",
);

#[test]
fn gen_wdl_data_labels_positions_by_side_to_move_outcome() {
    // A 1-0 game: White-to-move sampled positions score 1.0, Black-to-move 0.0.
    let samples = gen_wdl_data_samples(WDL_FIXTURE, WdlSampleConfig {
        min_ply: 8, end_trim: 5, per_game: 6, shard: (0, 1),
    });
    assert!(!samples.is_empty());
    for s in &samples {
        assert!(s.target == 1.0 || s.target == 0.0, "1-0 game targets are 1.0/0.0");
        // white-to-move (even ply index) -> won -> 1.0; black-to-move -> 0.0
        assert_eq!(s.target, if s.ply % 2 == 0 { 1.0 } else { 0.0 });
        assert!(!s.own.is_empty() && !s.opp.is_empty());
        assert!(s.own.iter().all(|&i| i < 768) && s.opp.iter().all(|&i| i < 768));
    }
}
```

Expose a testable `gen_wdl_data_samples(pgn: &str, config: WdlSampleConfig) -> Vec<WdlSample>` where `WdlSample { target: f32, own: Vec<usize>, opp: Vec<usize>, ply: u32 }`.

- [ ] **Step 3: Push and confirm it fails (compile error — functions missing)**

- [ ] **Step 4: Implement the WDL visitor and sampler**

In `engine-bench/src/lib.rs`. The visitor mirrors book-tool's tag filter but keeps the board current for the **whole game** and buffers per-ply features:

```rust
use pgn_reader::shakmaty::{Chess, Position, uci::UciMove};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};

#[derive(Clone, Copy)]
pub struct WdlSampleConfig {
    pub min_ply: u32,     // skip the opening (e.g. 8)
    pub end_trim: u32,    // skip the last N plies (e.g. 5)
    pub per_game: usize,  // max sampled positions per game (e.g. 12)
    pub shard: (usize, usize), // (i, n): keep games where stream_index % n == i
}

#[derive(Clone)]
pub struct WdlSample { pub target: f32, pub own: Vec<usize>, pub opp: Vec<usize>, pub ply: u32 }
// `target` is f32 (one of 0.0/0.5/1.0) so the value-assertion tests read
// naturally. The disjoint-partition test (Step 6) needs SET comparison, so it
// compares the *printed TSV lines* (Strings) of the shards, not the structs —
// avoid deriving Eq/Hash on an f32-bearing struct.

#[derive(Default)]
struct WdlTags { event: String, variant: String, white_elo: u32, black_elo: u32, result: String }

struct WdlGame {
    chess: Chess,
    board: engine_core::Board,
    // (features_own, features_opp, side_to_move, ply) for every eligible ply
    positions: Vec<(Vec<usize>, Vec<usize>, engine_core::Color, u32)>,
    ply: u32,
    valid: bool,
    result: String,
}

struct WdlBuilder { config: WdlSampleConfig, stream_index: usize, out: Vec<WdlSample> }
```

The visitor: `begin_tags` increments nothing yet; `tag` parses Event/Variant/WhiteElo/BlackElo/Result (same tag bytes as book-tool); `begin_movetext` computes `valid` from the same rating/standard filter **and** from `stream_index % n == i`, then does `stream_index += 1` (so the shard index counts games in stream order *before* the filter — increment it for every game, disjoint by construction), and records `result`; `san` converts SAN → move and keeps both `chess` and `board` current for **every** ply (not just 16). Three correctness details the test will otherwise fight:

- **Opponent perspective:** use `opposite(stm)`, **not** `!stm` — `engine_core::Color` has no `Not` impl. Mirror `engine-bench/src/train.rs`, which uses a free `opposite(color)` helper for exactly this; reuse or copy it.
- **Move-failure guard:** if `san.san.to_move` or `board.make_move` fails, set `game.valid = false` (exactly as book-tool does), or the `board` desyncs from `chess` mid-game and later features are garbage.
- **Ply counter = half-moves already applied (0-based):** push the pre-move features **then** `ply += 1` *after*. This makes even `ply` ⇒ White to move (from startpos), which the Step-2 test relies on.

When `ply >= min_ply` (and the move applied cleanly), push `(active_features(&board, stm), active_features(&board, opposite(stm)), stm, ply)` to `game.positions` **before** making the move (own = side-to-move's pieces, opp = the other side). `end_game`:

```rust
    fn end_game(&mut self, game: WdlGame) -> Self::Output {
        if !game.valid || !matches!(game.result.as_str(), "1-0" | "0-1" | "1/2-1/2") {
            return;
        }
        // Trim the last `end_trim` plies (mechanical), then evenly subsample to
        // `per_game` positions (deterministic — no RNG).
        let last_ply = game.ply;
        let eligible: Vec<_> = game.positions.into_iter()
            .filter(|(_, _, _, ply)| *ply + self.config.end_trim < last_ply)
            .collect();
        let picked = evenly_spaced(&eligible, self.config.per_game);
        for (own, opp, stm, ply) in picked {
            let target = wdl_target_for(&game.result, stm);
            self.out.push(WdlSample { target, own, opp, ply });
        }
    }
```

with helpers:

```rust
fn wdl_target_for(result: &str, stm: engine_core::Color) -> f32 {
    use engine_core::Color::*;
    match (result, stm) {
        ("1-0", White) | ("0-1", Black) => 1.0,
        ("1/2-1/2", _) => 0.5,
        _ => 0.0, // the side to move lost
    }
}

fn evenly_spaced<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    if items.len() <= n { return items.to_vec(); }
    (0..n).map(|k| items[k * items.len() / n].clone()).collect()
}
```

`gen_wdl_data_samples(pgn, config)` builds a `WdlBuilder`, runs `Reader::new(pgn.as_bytes()).visit_all_games(&mut builder)` (`.expect(...)` the io `Result`, as book-tool `.map_err`s it), returns `builder.out`. `&[u8]` implements `io::Read`, so no `Cursor` is needed. Note the shard filter must count games in **stream order before** the rating filter (increment `stream_index` for every `begin_movetext`, and gate on it there), so shards partition disjointly regardless of filtering.

- [ ] **Step 5: Add the `gen-wdl-data` command**

In `engine-bench/src/main.rs`: `gen-wdl-data <pgn_or_-> [--shard i/n] [--per-game N]` reads the PGN from a path or `-` (stdin, streaming like the book refresh), calls `gen_wdl_data_samples`, and prints each as `{target}\t{own_csv}\t{opp_csv}` (join indices with `,`) — the exact format `train_nnue.py` reads. Defaults: min_ply 8, end_trim 5, per_game 12, shard 0/1.

- [ ] **Step 6: More tests + green**

Add tests: determinism (same input → same output), the opening/endgame skips (no sampled ply `< min_ply` or `>= last_ply - end_trim`), the per-game cap (`<= per_game` per game), and **`--shard` disjoint partition** (union of shards `0/3,1/3,2/3` over a multi-game fixture equals shard `0/1`, no duplicates). Push, confirm green.

---

### Task 2: `--wdl-target` trainer mode

**Files:** Modify `modal/train_nnue.py`

No Rust CI covers Python; validate the target logic with a torch-free unit and the rest on Modal (Task 4).

- [ ] **Step 1: Extract a torch-free target helper and use it**

```python
def target_win_prob(target: float, wdl_target: bool):
    """WDL mode: the target IS the win-probability (game outcome 0/0.5/1).
    cp mode: squash the centipawn target through the WDL sigmoid."""
    if wdl_target:
        return min(1.0, max(0.0, target))
    import math
    return 1.0 / (1.0 + math.exp(-target / WDL_SCALE))
```

In `train(...)`, add a `wdl_target: bool = False` parameter; replace `target_wp = torch.sigmoid(target / WDL_SCALE)` with a branch: if `wdl_target`, `target_wp = torch.clamp(target, 0.0, 1.0)` else the existing sigmoid. Add `--wdl-target` to argparse and pass `args.wdl_target` into `train(...)`. (The prediction still squashes through `sigmoid(pred_cp/400)`; both `pred_wp` and `target_wp` live in `[0,1]`, so the MSE is unchanged.)

- [ ] **Step 2: Torch-free unit check (run via uv)**

Not a CI test (Python), but a runnable check: `uv run --with pytest python -c "import train_nnue as t; assert t.target_win_prob(1.0, True)==1.0; assert t.target_win_prob(0.0, True)==0.0; assert t.target_win_prob(0.5, True)==0.5; assert abs(t.target_win_prob(1.0, False)-0.5006)<1e-3; print('ok')"` (run from `modal/`). Record the `ok` in the task notes. This guards the silent-collapse footgun (omitting the flag collapses outcomes to ~0.5).

---

### Task 3: The Modal `train_wdl` pipeline

**Files:** Modify `modal/app.py`

Mirror the existing train→gate loop, but label from Lichess into a Volume.

- [ ] **Step 0: Add `zstd` to `rust_image`**

`rust_image` (modal/app.py) only `apt_install`s `curl`, `build-essential`, `pkg-config` — it has no `zstd`, so `zstdcat` fails. Add `"zstd"` to its `apt_install(...)`. (This changes the image, so the first run rebuilds it — expected.)

- [ ] **Step 1: A Volume + export preparation**

Add `wdl_volume = modal.Volume.from_name("rusty-fish-wdl", create_if_missing=True)`. A `prepare_export` function (mounts the Volume) downloads the pinned URL from `assets/opening-book/manifest.toml`, verifies its SHA-256 (`sha256sum --check`), and writes `export.pgn.zst` into the Volume (skip if already present), then **`wdl_volume.commit()`** so label shards started afterward see the file. This downloads once, not per shard.

- [ ] **Step 2: `label_wdl_shard`**

An `@app.function(image=rust_image, volumes={"/vol": wdl_volume})` that runs the pipeline. `subprocess.run([...])` (no shell) cannot express a `|`, so use a shell string:

```python
subprocess.run(
    f"zstdcat /vol/export.pgn.zst | {BIN} gen-wdl-data - --shard {i}/{n} --per-game {p} "
    f"> /vol/samples-{i}.tsv",
    shell=True, check=True,
)
wdl_volume.commit()
```

Set `-o pipefail` semantics by running under `bash -c` with `set -euo pipefail`, or accept that `check=True` catches only the last stage — prefer the `bash -c 'set -euo pipefail; ...'` form so a `zstdcat` decode failure fails the shard. Returns the shard's line count.

- [ ] **Step 3: `train_wdl_run` (GPU) reads the Volume**

An `@app.function(image=torch_image, gpu="A10G", timeout=..., volumes={"/vol": wdl_volume})` that concatenates `/vol/samples-*.tsv`, calls `train_nnue.train(concatenated_path, hidden=256, epochs=..., wdl_target=True, ...)` and `quantize_and_write`, returns the RFNN bytes. (torch_image must also mount the volume and include train_nnue.py — it already adds the file.)

- [ ] **Step 4: `train_wdl` entrypoint + gate**

A `train_wdl` local_entrypoint: `prepare_export.remote()` → fan `label_wdl_shard` over `i in range(n)` → `net_bytes = train_wdl_run.remote()` → then the powered gate: reuse `gate_shard` (NNUE candidate vs hand-crafted baseline) over `make_openings` shards, sum, `sprt_verdict`. Move the gate aggregation into a remote `nnue_gate_run` (mirroring `eval_gate_run`) so a detached run's verdict is retrievable from logs. Print `NNUE_GATE_RESULT_BEGIN/END` markers.

- [ ] **Step 5: No Rust CI (Python).** Validated by the Modal run in Task 4.

---

### Task 4: Open the PR, merge, run on Modal, assess

- [ ] **Step 1:** Verify the branch diff (spec, plan, the 5 files, no strays). Open the PR (superpowers:finishing-a-development-branch); body states this ships nothing by default (NNUE is opt-in) and the gate is the acceptance test.
- [ ] **Step 2:** Merge on green (Rust tests; the Python isn't CI-gated).
- [ ] **Step 3:** **Short Modal validation:** run `train_wdl` with a tiny config (a few label shards, `--per-game 2`, few epochs, small gate) to confirm the whole pipeline works end-to-end (label→Volume→train→gate) and produces a net + a gate verdict.
- [ ] **Step 4:** **Real run:** `train_wdl` with hidden 256, a real per-game sample count, enough epochs, and a powered gate (2048 openings). Retrieve the SPRT verdict from `modal app logs` (`NNUE_GATE_RESULT`).
- [ ] **Step 5:** **Assess.** If the net beats the tuned hand-crafted eval under SPRT (positive Elo / AcceptH1): a follow-up commits the `.rfnn` as an asset and wires it as the default eval, re-gated by CI, and updates `D:/Work-Tracking/work-tracker-personal.md`. If flat/negative: the net stays opt-in; record that Lichess-WDL at 768/256 isn't enough and the next lever is a stronger teacher (Stockfish labels) or HalfKA. Update the tracker either way.

---

## Out of scope

HalfKA / king-buckets, Stockfish-eval or hybrid teachers, data augmentation/dedup beyond the per-game cap, multi-month datasets, training-loop refinements beyond the existing Adam, and any RFNN-format or inference change.
