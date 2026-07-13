# Licensed Opening Book Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and consume a reproducible, weighted CC0 Lichess opening book with UCI-configurable variety.

**Architecture:** A new `book-tool` workspace binary converts a pinned, filtered PGN fixture into a deterministic v2 book and metrics TSV. `engine-search` loads v1/v2 books and selects weighted legal moves deterministically; `engine-uci` owns `BookPath` and `Book Variety` configuration and passes the loaded book to each worker.

**Tech Stack:** Rust 2024, existing engine-core/search/UCI, `pgn-reader`, GitHub Actions.

## Global Constraints

- Never run Cargo or Rust binaries locally; validate only in GitHub Actions.
- Source manifest must state CC0 license, URL, SHA-256, selection rules, and fixture provenance.
- Ordinary PR CI regenerates only the committed compact fixture; full database refresh is manual.
- Preserve safe normal-search fallback for missing, malformed, or disabled books.

---

### Task 1: Add reproducible source assets and generator

**Files:**
- Create: `book-tool/Cargo.toml`
- Create: `book-tool/src/main.rs`
- Create: `assets/opening-book/manifest.toml`
- Create: `assets/opening-book/lichess-cc0-fixture.pgn`
- Create: `assets/opening-book/rusty-fish-book-v2.txt`
- Modify: `Cargo.toml`

**Interfaces:**
- Produces `book-tool generate <pgn> <book> <metrics>`.
- Produces v2 records: `<fen>\t<uci>:<weight> ...`.

- [ ] **Step 1: Write failing generator tests**

```rust
#[test]
fn aggregates_transpositions_and_side_relative_results() {
    let report = build_book(FIXTURE_PGN, BookFilter { min_rating: 2200, max_plies: 16 }).unwrap();
    assert_eq!(report.accepted_games, 2);
    assert!(report.book.contains("e2e4:"));
}
```

- [ ] **Step 2: Verify red remotely**

Push only the test and require the workspace job to fail because `build_book` is absent.

- [ ] **Step 3: Implement parser and serializer**

Use `pgn-reader` to accept rated standard games with both ratings at least 2200, replay legal mainline moves through `Board`, aggregate by `position_signature`, score each move from the moving side, discard counts below three, and serialize sorted alternatives with integer weights.

- [ ] **Step 4: Verify green remotely and commit**

```text
git add Cargo.toml book-tool assets/opening-book
git commit -m "feat: generate licensed weighted opening book"
```

### Task 2: Load weighted v2 books and select deterministic variety

**Files:**
- Modify: `engine-search/src/lib.rs`

**Interfaces:**
- Produces `OpeningBook::from_text(&str) -> Result<OpeningBook, String>` for v1 and v2.
- Produces `OpeningBook::select(&self, &Board, variety: u8) -> Option<ChessMove>`.

- [ ] **Step 1: Write failing compatibility and variety tests**

```rust
#[test]
fn v2_book_uses_weighted_moves_but_zero_variety_is_best_move() {
    let book = OpeningBook::from_text("rusty-fish-book v2\n...\te2e4:9 d2d4:1\n").unwrap();
    assert_eq!(book.select(&Board::startpos(), 0).unwrap().to_uci(), "e2e4");
    assert!(matches!(book.select(&Board::startpos(), 100).unwrap().to_uci().as_str(), "e2e4" | "d2d4"));
}
```

- [ ] **Step 2: Verify red remotely**

Push the test-only change and require workspace failure from the changed `select` contract.

- [ ] **Step 3: Implement v2 parsing and deterministic selection**

Store `BookMove { mv, weight }`; parse `uci:weight`, reject zero weights and illegal moves, retain v1 moves at weight one, and use `position_hash % cumulative_weight` only when variety is nonzero.

- [ ] **Step 4: Verify green remotely and commit**

```text
git add engine-search/src/lib.rs
git commit -m "feat: support weighted opening book variety"
```

### Task 3: Configure books through UCI and publish metrics

**Files:**
- Modify: `engine-uci/src/main.rs`
- Modify: `engine-uci/tests/protocol_stress.rs`
- Create: `.github/workflows/opening-book.yml`

**Interfaces:**
- Produces UCI options `BookPath` (string) and `Book Variety` (spin 0–100).
- Worker loads configured text book and calls `Searcher::set_opening_book`.

- [ ] **Step 1: Write failing UCI configuration tests**

```rust
#[test]
fn book_options_advertise_and_keep_prior_path_on_invalid_input() {
    let mut state = EngineState::default();
    assert!(apply_option(&mut state, "setoption name Book Variety value 100").is_ok());
    assert_eq!(state.book_variety, 100);
}
```

- [ ] **Step 2: Verify red remotely**

Push only tests and require workspace failure for missing book fields/options.

- [ ] **Step 3: Implement UCI lifecycle and CI report**

Advertise options, validate `BookPath`, retain valid prior state on error, load only in worker ownership, and add a workflow that regenerates the committed fixture book then uploads book and TSV metrics artifacts.

- [ ] **Step 4: Verify all remote gates and commit**

Require workspace, tactical-suite, throughput, gauntlet, opening-book, and CodeQL checks.

```text
git add engine-uci .github/workflows/opening-book.yml
git commit -m "feat: configure licensed opening book through UCI"
```

### Task 4: Merge and record

- [ ] Open a PR with `benzsevern`, merge after every required remote check is green, and update `D:/Work-Tracking/work-tracker-personal.md` only after merge with the manifest, metrics, and book-hit evidence.
