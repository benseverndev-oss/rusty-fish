# Syzygy Root Probing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure Syzygy directories through UCI and return an exact DTZ root move when a configured tablebase covers the position.

**Architecture:** `engine-search` converts Pyrrhic's root result to a legal Rusty Fish move and returns it before book or normal search. UCI stores only a validated path, then gives each single search worker sole ownership of its `SyzygyTablebases` handle; cancellation joins a replaced worker before another one loads Pyrrhic.

**Tech Stack:** Rust 2024, `pyrrhic-rs` 0.2, Rusty Fish UCI, GitHub Actions.

## Global Constraints

- Never run Cargo, benchmarks, or Rust processes locally; validation runs only in GitHub Actions.
- Do not commit Syzygy data files; a later workflow will download and checksum a small corpus for end-to-end DTZ verification.
- Preserve existing interior WDL probing and normal-search fallback for absent tables, uncovered positions, and all Pyrrhic probe errors.
- `probe_root` must run with exactly one Pyrrhic handle; configuration stores a path, not a cloned handle.

---

## File structure

- `engine-search/src/lib.rs`: tablebase conversion, root probing, and early root result.
- `engine-uci/src/main.rs`: `SyzygyPath`, worker lifecycle, and UCI protocol handling.
- `engine-uci/tests/protocol_stress.rs`: binary-level UCI advertisement assertion.
- `docs/superpowers/specs/2026-07-13-syzygy-root-probing-design.md`: Pyrrhic singleton decision.

### Task 1: Convert Pyrrhic DTZ root results

**Files:**
- Modify: `engine-search/src/lib.rs:1-205,496-550,1860-1870`

**Interfaces:**
- Produces: `pub struct SyzygyRootProbe { pub best_move: ChessMove, pub wdl: SyzygyWdl, pub dtz: u16 }`.
- Produces: `SyzygyTablebases::probe_root(&self, board: &Board) -> Option<SyzygyRootProbe>`.

- [ ] **Step 1: Write the failing conversion tests**

```rust
#[test]
fn tablebase_promotion_conversion_matches_uci_piece_kinds() {
    assert_eq!(promotion_from_tablebase(TbPiece::Queen), Some(PieceKind::Queen));
    assert_eq!(promotion_from_tablebase(TbPiece::Pawn), None);
}

#[test]
fn tablebase_wdl_categories_keep_cursed_results_on_the_winning_side() {
    assert_eq!(syzygy_wdl(WdlProbeResult::CursedWin), SyzygyWdl::Win);
    assert_eq!(syzygy_wdl(WdlProbeResult::BlessedLoss), SyzygyWdl::Loss);
}
```

- [ ] **Step 2: Verify red remotely**

Push the test-only commit and open a draft PR. Confirm `Rusty Fish Tests / workspace` fails because the two helpers are unresolved. Do not run Cargo locally.

- [ ] **Step 3: Add the root-probe API**

Import `DtzProbeValue`, `Piece as TbPiece`, and `Square`; factor the existing WDL match into `syzygy_wdl`. Reuse the existing bitboard arguments, adding `board.halfmove_clock()` for `rule50`. Reject positions above `max_pieces`, accept only `DtzProbeValue::DtzResult`, turn its squares/promotion into `ChessMove`, then validate it through `board.parse_uci_move(&candidate.to_uci())`.

```rust
Some(SyzygyRootProbe {
    best_move: board.parse_uci_move(&candidate.to_uci()).ok()?,
    wdl: syzygy_wdl(root.wdl),
    dtz: root.dtz,
})
```

- [ ] **Step 4: Verify green remotely**

Push and confirm `Rusty Fish Tests / workspace` passes on GitHub Actions.

- [ ] **Step 5: Commit**

```text
git add engine-search/src/lib.rs
git commit -m "feat: expose Syzygy root probes"
```

### Task 2: Return the DTZ move before book or normal search

**Files:**
- Modify: `engine-search/src/lib.rs:496-550`

**Interfaces:**
- Consumes: `SyzygyTablebases::probe_root` and `syzygy_score(wdl, 0)`.
- Produces: an immediate `SearchResult` with the tablebase move, zero nodes, and a one-move PV.

- [ ] **Step 1: Write the failing helper test**

```rust
#[test]
fn root_tablebase_result_uses_the_exact_move_and_existing_score_scale() {
    let board = Board::startpos();
    let root = SyzygyRootProbe {
        best_move: board.parse_uci_move("e2e4").unwrap(),
        wdl: SyzygyWdl::Win,
        dtz: 1,
    };
    let result = root_tablebase_search_result(root);
    assert_eq!(result.best_move, Some(root.best_move));
    assert_eq!(result.score_cp, syzygy_score(SyzygyWdl::Win, 0));
    assert_eq!(result.nodes, 0);
}
```

- [ ] **Step 2: Verify red remotely**

Push the test-only commit and confirm the workspace job fails because `root_tablebase_search_result` is absent.

- [ ] **Step 3: Implement the minimal early return**

At the beginning of `search_with_callback_and_stop_signal`, before opening-book selection:

```rust
if let Some(root) = self.syzygy.as_ref().and_then(|tables| tables.probe_root(board)) {
    return root_tablebase_search_result(root);
}
```

The helper must set `best_move: Some(root.best_move)`, `depth: 0`, `score_cp: syzygy_score(root.wdl, 0)`, `nodes: 0`, `elapsed: Duration::ZERO`, and `pv: vec![root.best_move]`.

- [ ] **Step 4: Verify green remotely**

Confirm the workspace job passes and inspect the fixed-opponent gauntlet artifact for a non-regression.

- [ ] **Step 5: Commit**

```text
git add engine-search/src/lib.rs
git commit -m "feat: use Syzygy DTZ moves at the root"
```

### Task 3: Configure tablebases safely through UCI

**Files:**
- Modify: `engine-uci/src/main.rs:1-210,300-390`
- Modify: `engine-uci/tests/protocol_stress.rs:1-150`

**Interfaces:**
- Produces: `EngineState { syzygy_path: Option<String>, .. }`.
- Produces: `start_search(board, options, syzygy_path, limits)`, whose worker is the only Pyrrhic handle owner.
- Produces: `stop_and_join_active_search(&mut Option<ActiveSearch>)`.

- [ ] **Step 1: Write failing UCI tests**

```rust
#[test]
fn syzygy_path_keeps_the_previous_path_on_error() {
    let mut state = EngineState::default();
    apply_option(&mut state, "setoption name SyzygyPath value .").unwrap();
    assert_eq!(state.syzygy_path.as_deref(), Some("."));
    assert!(apply_option(&mut state, "setoption name SyzygyPath value missing-tables").is_err());
    assert_eq!(state.syzygy_path.as_deref(), Some("."));
}
```

Also extend the binary UCI handshake assertion to require `option name SyzygyPath type string default` before `uciok`.

- [ ] **Step 2: Verify red remotely**

Push the test-only commit and confirm the GitHub workspace job fails because the field and option are absent.

- [ ] **Step 3: Implement configuration and lifecycle**

Advertise `SyzygyPath`. Parse it before generic nonempty-value validation so `setoption name SyzygyPath value` clears the path. For nonempty values, require every configured directory to exist, and assign only after validation.

Pass `state.syzygy_path.clone()` to `start_search`. In the worker, load with `SyzygyTablebases::load(&path).ok()` and set it on the worker-local searcher. Store the spawned `JoinHandle<()>` in `ActiveSearch`. For `ucinewgame`, `position`, `setoption`, replacement `go`, and `quit`, signal then join before discarding an active worker. On natural completion, receive, join, emit exactly one `bestmove`, then clear active state. `stop` only signals.

- [ ] **Step 4: Verify green remotely**

Confirm workspace, tactical-suite, and fixed-opponent-gauntlet checks pass. Download artifacts only to `%TEMP%` if comparison requires it.

- [ ] **Step 5: Commit**

```text
git add engine-uci/src/main.rs engine-uci/tests/protocol_stress.rs
git commit -m "feat: configure Syzygy probing through UCI"
```

### Task 4: Review and merge through GitHub

**Files:**
- Modify: `D:/Work-Tracking/work-tracker-personal.md` (only after merge)

- [ ] **Step 1: Open the pull request**

Push `feat/syzygy-root-probing`, create a PR that states root DTZ selection, UCI configuration, sole-handle ownership, and fallback behavior. Do not claim corpus-level DTZ validation.

- [ ] **Step 2: Review remote evidence**

Wait for all required checks. Resolve test, CodeQL, tactical, or gauntlet regressions before merging; never replace remote evidence with a local Cargo run.

- [ ] **Step 3: Merge and update tracking**

Merge on GitHub when checks are green. Then record the PR/date, delivered root probing behavior, and the remaining checksummed tablebase corpus workflow in the tracker. Switch to `main` and pull without invoking any Rust command.

## Self-review

Tasks 1–2 cover DTZ decoding, legal move validation, fallback, and precedence over book/search. Task 3 covers UCI configuration, empty-path disablement, error retention, and Pyrrhic singleton safety. Task 4 leaves tablebase-file download and exact corpus tests as an explicit later workflow.
