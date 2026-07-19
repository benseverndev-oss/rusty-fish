# NNUE Adoption (Make the Bundled Net the Default Eval) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Stockfish-taught NNUE (+8.0 Elo, AcceptH1) the engine's default evaluation, bundled into the binary, with the hand-crafted eval kept as a selectable fallback.

**Architecture:** Commit the `.rfnn` as a binary asset; `include_bytes!` it into `engine-search` behind a `LazyLock<Arc<Nnue>>`; flip `Searcher::default()` to install the bundled net; wire the UCI `EngineState` to install it at startup with a `UseNNUE` toggle and `EvalFile` override; and explicitly disable NNUE at the engine-bench `Searcher::default()` sites that must stay hand-crafted.

**Tech Stack:** Rust 2024 (`engine-search`, `engine-uci`, `engine-bench`), `include_bytes!`, `LazyLock`.

**Spec:** `docs/superpowers/specs/2026-07-19-nnue-adopt-default-design.md`

---

## Global constraints

- **Never run Cargo/Rust locally.** All Rust validation is GitHub Actions: `Rusty Fish Tests` (`cargo test --workspace`), plus `Tactical Suite`, `Throughput Benchmark`, `Fixed Opponent Gauntlet` (these run the `engine-bench` binary and are affected by the flip). Format four-space by hand.
- **Verify `gh` account is `benzsevern`** (`gh auth switch --user benzsevern`); keyring PAT as `GH_TOKEN`; push via tokenized URL if blocked; `git fetch origin --prune` before branching; **stage paths explicitly — never `git add -A`**.
- Conventional Commits. **Branch:** `feat/nnue-adopt-default` (already created off latest main, spec committed).
- **The trained net bytes are already on this machine** at the scratchpad path `C:\Users\bsevern\AppData\Local\Temp\claude\D--show-case-rusty-fish\796b1bc1-fd5e-4f25-900c-0c2c080f894d\scratchpad\rusty-fish-sf-net.rfnn` (789,520 bytes, retrieved from the Modal Volume). Task 1 copies it into the repo.

## Background: the exact sites

- **`engine-search/src/lib.rs`:** `impl Default for Searcher` (lib.rs:521-545) sets `nnue: None` (line 538). The only other `Searcher` literal, `Searcher::helper`, takes `nnue` as a parameter and is fed `self.nnue.clone()` at spawn — so helper threads inherit the default automatically. `Searcher::evaluate` (1436) already prefers `self.nnue`. `set_nnue`/`has_nnue` at 782-788.
- **`engine-search/src/nnue.rs`:** `Nnue::from_bytes`, `to_bytes`, `from_file`, `hidden()` exist. `INPUT_DIMENSION = 768`. RFNN = magic/version(u32)/hidden(u32)/feature_weights(i16)/feature_bias(i16)/output_weights(i16)/output_bias(i32).
- **`engine-uci/src/main.rs`:** `EngineState` (155-163) `#[derive(Default)]` → `nnue: None`. `start_search` (104-132) builds `Searcher::default()` then **unconditionally** `searcher.set_nnue(nnue)` where `nnue` = `state.nnue`. `apply_option` (228-321): `EvalFile` (290-298) sets `state.nnue`. `write_uci_header` (165-199) lists options. `apply_option` arity guard `tokens.len() < 4` (233) is fine for a check option (always 4 tokens).
- **`engine-bench` `Searcher::default()` sites that must be forced hand-crafted:** `gate_searcher` (lib.rs:606, covers eval+mobility gates via `play_mobility_game` 630-631), `play_parameter_game` candidate+baseline (lib.rs:565,568, the search-param SPSA), `play_nnue_game` baseline (lib.rs:439), and `generate_training_samples` labeler (`train.rs:66`).

## File structure

| Path | Change | Responsibility |
|------|--------|----------------|
| `assets/nnue/rusty-fish-net.rfnn` | Create | The committed net (789,520 bytes). |
| `assets/nnue/README.md` | Create | Provenance. |
| `.gitattributes` | Modify | Mark `*.rfnn` binary. |
| `engine-search/src/nnue.rs` | Modify | `bundled_network()` loader + round-trip test. |
| `engine-search/src/lib.rs` | Modify | Flip `Searcher::default` nnue; re-export `bundled_network`. |
| `engine-uci/src/main.rs` | Modify | `EngineState` composition, `UseNNUE` option, header. |
| `engine-bench/src/lib.rs` | Modify | `set_nnue(None)` at 3 sites. |
| `engine-bench/src/train.rs` | Modify | `set_nnue(None)` on the labeler. |

---

### Task 1: Commit the net asset (build prerequisite)

**Files:** Create `assets/nnue/rusty-fish-net.rfnn`, `assets/nnue/README.md`; Modify `.gitattributes`

- [ ] **Step 1: Copy the net + mark it binary.** Copy the scratchpad `.rfnn` (path in Global constraints) to `assets/nnue/rusty-fish-net.rfnn`. Append to the repo-root `.gitattributes` (which already pins `assets/opening-book/*`):

```
assets/nnue/*.rfnn binary
```

`binary` (= `-text -diff`) prevents any CRLF/LF rewrite that would corrupt the bytes. **Verify the committed size is exactly 789520 bytes** (`git cat-file -s :assets/nnue/rusty-fish-net.rfnn` after staging, or check the working-tree size) — a mangled file is the top failure mode and would fail `from_bytes` at compile-adjacent test time.

- [ ] **Step 2: Provenance README.** `assets/nnue/README.md`: the net is `rusty-fish-net.rfnn` (768→512→1 RFNN, hidden 512), trained on ~3M Lichess positions labelled by Stockfish at 100k nodes (the `train_sf` Modal pipeline), gated at **+8.0 Elo, SPRT AcceptH1 over 16384 games** vs the tuned hand-crafted eval. It is the engine's default evaluation.

- [ ] **Step 3: Commit + push.** Stage the three paths explicitly. Conventional commit `feat(nnue): commit the Stockfish-taught default network asset`. (No code references it yet, so CI stays green.)

---

### Task 2: `bundled_network()` loader

**Files:** Modify `engine-search/src/nnue.rs`, `engine-search/src/lib.rs`

- [ ] **Step 1: Write the failing round-trip test** (in `nnue.rs` tests):

```rust
#[test]
fn bundled_network_round_trips() {
    let net = bundled_network();
    assert_eq!(net.hidden(), 512);
    // Re-serialising the parsed net reproduces the committed asset bytes exactly.
    assert_eq!(net.to_bytes(), include_bytes!("../../assets/nnue/rusty-fish-net.rfnn").to_vec());
}
```

- [ ] **Step 2: Push, confirm it fails to compile (`bundled_network` missing).**

- [ ] **Step 3: Implement the loader** in `nnue.rs`:

```rust
use std::sync::{Arc, LazyLock};

/// The engine's default NNUE, compiled into the binary. Parsed once and shared.
static BUNDLED_NETWORK: LazyLock<Arc<Nnue>> = LazyLock::new(|| {
    Arc::new(
        Nnue::from_bytes(include_bytes!("../../assets/nnue/rusty-fish-net.rfnn"))
            .expect("bundled NNUE asset is a valid RFNN network"),
    )
});

/// A shared handle to the bundled default network.
pub fn bundled_network() -> Arc<Nnue> {
    BUNDLED_NETWORK.clone()
}
```

(The `include_bytes!` path is relative to `nnue.rs`, i.e. `engine-search/src/` → repo root: `../../assets/...`. `Arc` is required — every install site takes `Arc<Nnue>`.) In `lib.rs`, add `bundled_network` to the `pub use nnue::{...}` re-export (line 14).

- [ ] **Step 4: Push, confirm green.** The default is NOT flipped yet — this task only adds the loader (no behaviour change), so all suites stay green.

---

### Task 3: Force hand-crafted at the engine-bench sites that need it

**Files:** Modify `engine-bench/src/lib.rs`, `engine-bench/src/train.rs`

These are **no-ops before the Task 4 flip** (the default is still `None`), so this task is green on its own; landing it first makes the flip safe. Add `set_nnue(None)` (with a one-line comment: "compare/label with the hand-crafted eval, not the now-default NNUE") at:

- [ ] **`gate_searcher`** (`lib.rs:606`): after `let mut searcher = Searcher::default();` add `searcher.set_nnue(None);`. (Covers `run_eval_gate_fens`, `run_mobility_gate_fens`, `run_eval_spsa_campaign` via `play_mobility_game`.)
- [ ] **`play_parameter_game`** (`lib.rs:565` and `:568`): `candidate.set_nnue(None);` and `baseline.set_nnue(None);` (the search-param SPSA keeps tuning the frozen hand-crafted engine).
- [ ] **`play_nnue_game` baseline** (`lib.rs:439`): after `let mut baseline = Searcher::default();` add `baseline.set_nnue(None);` (candidate keeps its installed net; baseline is the hand-crafted opponent).
- [ ] **`generate_training_samples` labeler** (`train.rs:66`): after `let mut labeler = Searcher::default();` add `labeler.set_nnue(None);` (leaves stay the hand-crafted eval, as the doc comment promises).

- [ ] **Push, confirm green** (behaviour unchanged; the SPSA/gauntlet/train tests still pass).

---

### Task 4: Flip the default + UCI wiring

**Files:** Modify `engine-search/src/lib.rs`, `engine-uci/src/main.rs`

- [ ] **Step 1: Flip `Searcher::default`.** In `engine-search/src/lib.rs:538`, change `nnue: None,` to `nnue: Some(bundled_network()),` (call the re-exported/`nnue::bundled_network()`). Helper threads inherit it via the existing `self.nnue.clone()` parameter — no other change.

- [ ] **Step 2: Wire `EngineState`.** In `engine-uci/src/main.rs`, replace the `#[derive(Default)]` on `EngineState` with a manual `Default`, and model the eval as a composition. Add fields `use_nnue: bool` and `eval_file: Option<Arc<Nnue>>`; keep `nnue: Option<Arc<Nnue>>` as the computed effective net:

```rust
struct EngineState {
    board: Board,
    searcher: Searcher,
    options: SearchOptions,
    syzygy_path: Option<String>,
    use_nnue: bool,
    eval_file: Option<Arc<Nnue>>,
    nnue: Option<Arc<Nnue>>, // effective network installed for the next search
    book: Option<OpeningBook>,
}

impl Default for EngineState {
    fn default() -> Self {
        let mut state = Self {
            board: Board::default(),
            searcher: Searcher::default(),
            options: SearchOptions::default(),
            syzygy_path: None,
            use_nnue: true,
            eval_file: None,
            nnue: None,
            book: None,
        };
        state.recompute_nnue();
        state
    }
}

impl EngineState {
    /// Effective net: none if NNUE is toggled off; else a custom EvalFile net if
    /// loaded, else the bundled default.
    fn recompute_nnue(&mut self) {
        self.nnue = if !self.use_nnue {
            None
        } else {
            self.eval_file
                .clone()
                .or_else(|| Some(engine_search::bundled_network()))
        };
    }
}
```

- [ ] **Step 3: `EvalFile` + `UseNNUE` in `apply_option`.** Change the `EvalFile` arm to set `eval_file` and recompute; add a `UseNNUE` arm:

```rust
"EvalFile" => {
    state.eval_file = match value {
        None => None,
        Some(path) => Some(Arc::new(Nnue::from_file(&path)?)),
    };
    state.recompute_nnue();
}
"UseNNUE" => {
    let value = value.ok_or_else(|| "missing option value".to_string())?;
    state.use_nnue = value
        .parse::<bool>()
        .map_err(|_| format!("invalid UseNNUE value: {value}"))?;
    state.recompute_nnue();
}
```

(On a `from_file` error the `?` returns before `eval_file` is reassigned, so a bad `EvalFile` keeps the previous net — preserving the existing "keeps it on error" behaviour.)

- [ ] **Step 4: Advertise the option.** In `write_uci_header`, after the `EvalFile` line add:

```rust
writeln!(stdout, "option name UseNNUE type check default true")?;
```

- [ ] **Step 5: Update tests + fix flip fallout.** Push and read the `Rusty Fish Tests` run. **Two engine-search tests will break — fix these specifically:**
  - **`nnue_evaluation_overrides_the_handcrafted_score` (`engine-search/src/lib.rs` ~2871):** it captures `let handcrafted = searcher.evaluate(&board);` from a default searcher, which now returns the *bundled NNUE* score (not hand-crafted), then compares it against the post-`set_nnue(None)` hand-crafted eval — they won't match. Fix: capture the `handcrafted` baseline from a searcher that has already had `set_nnue(None)` (so the test compares hand-crafted vs installed-net as intended).
  - **`finds_hanging_queen_tactic` (`engine-search/src/lib.rs` ~3071):** `assert!(result.score_cp > 700)` is an eval-scale-dependent magnitude now produced by NNUE. The best-move assertion still holds; the `> 700` threshold may not. This test is about *finding the tactic*, not the hand-crafted eval — rebake the threshold to the NNUE-scale value the run reports (keep it a meaningful "clearly winning" bar), or, if you'd rather keep it a hand-crafted-eval test, build the searcher with `set_nnue(None)`.
  - **`eval_file_loads_a_network_and_keeps_it_on_error` (`engine-uci/src/main.rs` ~453) does NOT break** — all its assertions are `state.nnue.is_some()`, which stays true under the composition (bundled default = Some, EvalFile = Some, keeps-previous-on-error = Some). Confirm it's green; no change needed.
  - **Add coverage:** default `EngineState` has `nnue.is_some()`; `UseNNUE false` → `nnue.is_none()`; `EvalFile <path>` then `UseNNUE true` keeps the custom net. And an `engine-search` test that `Searcher::default().has_nnue()` is `true` with a short start-position search returning a legal best move + finite score.
  - **Any other broken test:** mate-finding / best-move tests should still pass (NNUE + `evaluate_terminal` detects mates). For a hand-crafted-score-magnitude assertion, decide per test whether it means to test the hand-crafted eval (→ `set_nnue(None)`) or search behaviour (→ rebake). Do NOT blindly delete assertions.
  - Note: `app-desktop/src/main.rs` (~438) also builds `Searcher::default()` and thus inherits NNUE — this is desirable (the desktop engine gets the stronger eval) and no test there should break; just be aware it's a production site the flip reaches.
  - Iterate to green on `cargo test --workspace`.

---

### Task 5: PR, reassess the search-based suites, merge

- [ ] **Step 1:** Verify the branch diff is exactly the asset + README + `.gitattributes` + the four source files + spec + plan — no strays. Open the PR (superpowers:finishing-a-development-branch); body: makes the +8.0-Elo net the default eval, hand-crafted kept behind `UseNNUE false`.

- [ ] **Step 2: Reassess the search-based workflows on the PR** (they change because the engine now evaluates with NNUE):
  - **`Tactical Suite`:** may solve a different set. If it has a pass threshold, read the new result; adjust the expected count/threshold deliberately (NNUE is expected to solve at least as many — investigate any regression before lowering a bar).
  - **`Throughput Benchmark`:** nps drops (NNUE per-node cost). If it has a hard floor that now fails, relax the floor to a realistic NNUE value and note the before/after in the PR.
  - **`Fixed Opponent Gauntlet`:** now measures the NNUE engine (expected to improve); informational unless it gates.
  - `perft` and eval-snapshot tests must stay green untouched — if either moved, something is wrong (investigate, don't rebake).

- [ ] **Step 3: Merge on green** (per the repo standing rule) once `Rusty Fish Tests` and the reassessed suites pass.

- [ ] **Step 4: Update `D:/Work-Tracking/work-tracker-personal.md`:** NNUE is now the shipped default eval (+8.0 Elo over the prior hand-crafted default); the hand-crafted eval is a `UseNNUE false` fallback; the eval program's arc (mobility → SPSA +10.5 → WDL −37 → SF-teacher +8.0 → adopted) is complete for now. Note the follow-up: future "candidate vs current default" gates now need an NNUE baseline.

---

## Out of scope

Retraining/improving the net, HalfKA, incremental-accumulator changes, re-tuning the hand-crafted eval (now a frozen fallback), and a new powered gate (the +8.0 AcceptH1 result already stands).
