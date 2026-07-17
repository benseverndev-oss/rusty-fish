# Opening Book Corpus Refresh Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the committed opening book from the pinned CC0 Lichess export through a manual, reviewable workflow, bounded to a size the repository can carry.

**Architecture:** Split the committed fixture outputs from the committed production book so ordinary CI keeps byte-comparing only the tiny synthetic fixture. Correct the generator's minimum-three filter to count observations rather than summed result points, teach it to stream PGN from stdin, and bound its output with `--max-positions`. A dispatch-only workflow downloads the pinned export, verifies its SHA-256, pipes it through `zstdcat`, and opens a PR carrying the regenerated production book.

**Tech Stack:** Rust 2024 (`book-tool`, `pgn-reader` 0.29, `shakmaty`), GitHub Actions, `zstd`.

**Spec:** `docs/superpowers/specs/2026-07-17-opening-book-corpus-refresh-design.md`

---

## Global constraints

Read these before starting. They are not negotiable and they change how every step below is verified.

- **Never run Cargo or Rust binaries locally.** Inherited from `docs/superpowers/plans/2026-07-13-licensed-opening-book.md`. Every red and every green is observed in GitHub Actions, never in a local terminal. In practice this means each "run the test" step is: commit, push, watch the run, read the log.
- **Verify the `gh` account before any remote operation.** `CLAUDE.md` and `AGENTS.md` both require the `benzsevern` account. Run `gh auth status` and `gh auth switch --user benzsevern` if it is anything else. Do this before the first push and re-check if a session gap occurs.
- **`cargo fmt` formatting, four-space indent, Conventional Commit subjects** (`feat:` / `fix:` / `test:` / `docs:`). Since Cargo cannot run locally, match the surrounding file's formatting by hand and let CI's fmt check catch drift.
- **Avoid unrelated refactors.** If you spot an adjacent flaw, note it; do not fix it here.
- **Branch:** `feat/opening-book-corpus-refresh`, already checked out with four spec commits ahead of `origin/main` and not yet pushed.

## Background an engineer with no context needs

**What the book is.** `assets/opening-book/rusty-fish-book-v2.txt` is a text file. Line one is the literal header `rusty-fish-book v2`. Every later line is `<fen>\t<uci>:<weight> <uci>:<weight> ...`, where `<fen>` is a *position signature* (a FEN with the halfmove and fullmove counters stripped, so transpositions collapse) and alternatives are ordered by descending weight then ascending UCI.

**How `book-tool` builds it.** `book-tool/src/main.rs` walks a PGN with `pgn-reader`'s `Visitor` trait. `Builder::counts` is a `BTreeMap<String, BTreeMap<String, u32>>` mapping signature to move to accumulated weight. `end_game` (main.rs:128-143) adds mover-relative points per move: 3 for a win, 2 for a draw, 1 for a loss. `build_book` (main.rs:146-192) then filters, sorts, and renders.

**Why `BTreeMap` matters.** Iteration order is sorted, which is where the book's determinism comes from. Any bound you add must preserve a total order or the byte-identical CI check becomes flaky. Do not switch to `HashMap`.

**The bug being fixed.** main.rs:165 reads `.filter(|(_, weight)| *weight >= 3)`. That tests *summed points*, not *observations*. A move seen exactly once in a won game scores 3 points and survives as though three different games had played it. The documented rule in `docs/superpowers/specs/2026-07-13-licensed-opening-book-design.md` is a minimum of three observations. Both the design doc and the work tracker have been claiming the documented behavior, not the real one.

**Why the fixture book does not change.** Every move in today's six-game fixture appears in exactly three games, so it has three observations under either rule. Fixing the filter alone would be provable by unit test only, with no committed evidence. So the fixture gains one single-occurrence decisive game whose moves the corrected rule must exclude. The book staying byte-identical *is* the regression evidence: revert the fix and CI fails.

## File structure

| Path | Change | Responsibility after this plan |
|------|--------|-------------------------------|
| `assets/opening-book/rusty-fish-book-v2.txt` | Renamed to `fixture-book-v2.txt` | Fixture output, byte-compared by CI. The production name holds no file until the first refresh PR lands. |
| `assets/opening-book/metrics.tsv` | Renamed to `fixture-metrics.tsv` | Fixture metrics, byte-compared by CI. |
| `assets/opening-book/lichess-cc0-fixture.pgn` | Modify (add one game) | Synthetic corpus that exercises both filter rules. |
| `assets/opening-book/manifest.toml` | Modify | Provenance for the production book, now including the bound. |
| `book-tool/src/main.rs` | Modify | Observation tracking, `--max-positions`, stdin streaming. |
| `book-tool/tests/generate.rs` | Modify | Repointed `include_str!` targets; new flag and filter coverage. |
| `.github/workflows/opening-book.yml` | Modify | Repointed at the fixture pair. Otherwise unchanged. |
| `.github/workflows/opening-book-refresh.yml` | Create | Dispatch-only production refresh that opens a PR. |

**Ordering rationale.** Task 1 (the rename) must land first: it is the only change that touches a compile-time `include_str!` dependency, and every later task's CI run needs a green baseline. Task 2 changes committed artifacts. Tasks 3 and 4 are additive and could be reordered, but the refresh workflow in Task 6 depends on both.

---

### Task 1: Make `book-tool`'s tests run in CI, then split the fixture assets from the production names

Two things happen here, and the first must come first.

**`book-tool`'s tests do not currently run in CI at all.** `cargo test --workspace` lives only in `.github/workflows/engine-core-perft.yml` ("Rusty Fish Tests"), whose push and pull_request path filters list `engine-core/**`, `engine-search/**`, `engine-bench/**`, `engine-uci/**`, `app-desktop/**`, `Cargo.toml`, `Cargo.lock`, and its own path. **`book-tool/**` and `assets/opening-book/**` are not among them.** Every commit in this plan touches only those two paths (plus `docs/` and the opening-book workflows), so the only workflow that fires is `opening-book.yml`, and it runs `cargo run --release -p book-tool` — never `cargo test`.

That matters more than it first looks. `cargo run -p book-tool` builds the **binary** target only; `book-tool/tests/generate.rs` is a separate **test** target that is never compiled. So without this fix:

- Every "verify it fails" step in Tasks 2, 3, and 4 is unobservable. No test runs, and `opening-book.yml` goes green because the fixture book is untouched. An implementer would stall, or wrongly conclude the bug is not real.
- The `include_str!` rename below would have no safety net whatsoever. A missed rename would go green and the break would surface much later.

This is a pre-existing repo gap, not one this plan introduces — `book-tool`'s test presumably ran once when `Cargo.toml` gained the member and never since. But this plan's entire verification strategy sits on top of it, so it is fixed here, before the first TDD red. Adding a `cargo test -p book-tool` step to `opening-book.yml` is the smallest fix that stays inside this plan's boundaries: that workflow already triggers on both relevant paths.

The second half is the rename. The refresh workflow will write `rusty-fish-book-v2.txt`. That is the exact file `opening-book.yml` currently diffs fixture output against. Landing the refresh without this split would make `opening-book.yml` compare a 3-position fixture book to a 5000-position production book on every future run, and it would fail forever.

**Files:**
- Modify: `.github/workflows/opening-book.yml` (new test step, and lines 31-32)
- Rename: `assets/opening-book/rusty-fish-book-v2.txt` → `assets/opening-book/fixture-book-v2.txt`
- Rename: `assets/opening-book/metrics.tsv` → `assets/opening-book/fixture-metrics.tsv`
- Modify: `book-tool/tests/generate.rs:6-7`

Both assets have exactly two consumers and both must move in this same commit. `book-tool/tests/generate.rs` pulls them in with `include_str!`, which is a **compile-time** dependency: renaming without repointing does not fail a test, it fails to build the `book-tool` test target. `.gitattributes` already covers the new names via its `assets/opening-book/*.txt` and `*.tsv` globs, so it needs no change.

- [ ] **Step 0: Run `book-tool`'s tests in CI**

In `.github/workflows/opening-book.yml`, add a step immediately after the `dtolnay/rust-toolchain@stable` step (before "Regenerate the book from the committed fixture"):

```yaml
      - name: Run the book generator tests
        run: cargo test -p book-tool
```

This compiles the test target, which is what makes the `include_str!` rename below verifiable and every later red observable.

- [ ] **Step 1: Rename both assets with git**

```bash
git mv assets/opening-book/rusty-fish-book-v2.txt assets/opening-book/fixture-book-v2.txt
git mv assets/opening-book/metrics.tsv assets/opening-book/fixture-metrics.tsv
```

- [ ] **Step 2: Repoint the test's compile-time includes**

In `book-tool/tests/generate.rs`, replace lines 6-7:

```rust
const EXPECTED_BOOK: &str = include_str!("../../assets/opening-book/fixture-book-v2.txt");
const EXPECTED_METRICS: &str = include_str!("../../assets/opening-book/fixture-metrics.tsv");
```

- [ ] **Step 3: Repoint the workflow's diff**

In `.github/workflows/opening-book.yml`, replace the two `diff --unified` lines under "Verify the committed book and metrics are byte-identical" (lines 31-32 before Step 0 shifted them down):

```yaml
          diff --unified assets/opening-book/fixture-book-v2.txt regenerated-book.txt
          diff --unified assets/opening-book/fixture-metrics.tsv regenerated-metrics.tsv
```

- [ ] **Step 4: Confirm no other consumer exists**

```bash
grep -rn "rusty-fish-book-v2\|opening-book/metrics.tsv" --exclude-dir=.git .
```

Expected: matches only inside `docs/` (spec and plan prose, which intentionally discuss the production names) and nothing under `book-tool/`, `engine-*/`, `app-desktop/`, or `.github/workflows/`. If a code or workflow match appears, repoint it here rather than in a later task.

- [ ] **Step 5: Commit and push**

```bash
gh auth status
git add assets/opening-book book-tool/tests/generate.rs .github/workflows/opening-book.yml
git commit -m "ci: run the book-tool tests and name its assets as fixture output"
git push -u origin feat/opening-book-corpus-refresh
```

- [ ] **Step 6: Verify green in GitHub Actions**

```bash
gh run list --branch feat/opening-book-corpus-refresh --limit 5
gh run watch <run-id> --exit-status
```

Expected: the `Opening Book` workflow passes, and its log now shows `cargo test -p book-tool` running `generate_writes_deterministic_book_and_metrics`. It regenerates from the untouched fixture, so both diffs are empty.

A compile error in the test step means an `include_str!` was missed — that step is the only thing in the repository that builds the test target, which is exactly why Step 0 came first. Confirm the test genuinely ran rather than assuming: if the log shows `running 1 test`, the safety net is live for every task that follows.

---

### Task 2: Filter on observations, not summed points

**Files:**
- Modify: `book-tool/src/main.rs` (`Builder::counts` type, `end_game`, `build_book`)
- Modify: `book-tool/tests/generate.rs` (new unit test)
- Modify: `assets/opening-book/lichess-cc0-fixture.pgn`
- Modify: `assets/opening-book/fixture-metrics.tsv`

`weight` keeps the existing integral 3/2/1 mover-relative sum and keeps ordering alternatives. `observations` is a new parallel counter and is what the minimum-three filter tests. `observations` is generator-internal: the v2 record format stays `<fen>\t<uci>:<weight> ...` and is unchanged, so no loader or format-version work follows.

- [ ] **Step 1: Write the failing test**

Append to `book-tool/tests/generate.rs`. This test runs the binary end to end because `build_book` is private to the binary crate.

```rust
const SINGLE_WIN: &str = "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n";

#[test]
fn a_single_decisive_game_does_not_satisfy_the_minimum_three_observations() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let input = root.join("single.pgn");
    let book = root.join("book.txt");
    let metrics = root.join("metrics.tsv");
    fs::write(&input, SINGLE_WIN).expect("write fixture");

    let status = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "generate",
            input.to_str().unwrap(),
            book.to_str().unwrap(),
            metrics.to_str().unwrap(),
        ])
        .status()
        .expect("run book generator");

    assert!(status.success());
    // White's e2e4 scores three points in a won game but has one observation,
    // so no move survives the minimum-three filter and the book is header-only.
    assert_eq!(
        fs::read_to_string(&book).expect("generated book"),
        "rusty-fish-book v2\n"
    );

    fs::remove_dir_all(root).expect("remove temporary directory");
}
```

- [ ] **Step 2: Push and verify it fails in CI**

```bash
git add book-tool/tests/generate.rs
git commit -m "test: pin that one decisive game does not satisfy the book filter"
git push
gh run list --branch feat/opening-book-corpus-refresh --limit 3
gh run watch <run-id> --exit-status
```

Expected: FAIL in the `Run the book generator tests` step. The assertion reports a book containing `e2e4:3` (`e7e5` scores 1 point in a lost game and is already filtered), not a bare header. That failure is the bug, reproduced.

If this step passes, stop. Either Task 1 Step 0 did not land or the test is not asserting what it claims. Do not proceed to the fix without seeing the red.

- [ ] **Step 3: Track observations alongside weight**

In `book-tool/src/main.rs`, change `Builder::counts` (main.rs:55) to carry both counters:

```rust
    counts: BTreeMap<String, BTreeMap<String, MoveStats>>,
```

Add above `Builder` (near the other small structs):

```rust
#[derive(Clone, Copy, Default)]
struct MoveStats {
    weight: u32,
    observations: u32,
}
```

In `end_game` (main.rs:134-142), accumulate both:

```rust
        for (fen, mv, side) in game.moves {
            // Twice the specified 0.5 + score_fraction weight, kept integral.
            let points = match (result, side) {
                ("1-0", Color::White) | ("0-1", Color::Black) => 3,
                ("1/2-1/2", _) => 2,
                _ => 1,
            };
            let stats = self.counts.entry(fen).or_default().entry(mv).or_default();
            stats.weight += points;
            stats.observations += 1;
        }
```

- [ ] **Step 4: Filter on observations and order on weight**

In `build_book` (main.rs:162-183), the filter tests `observations` while the sort and the rendered value still use `weight`:

```rust
    for (fen, moves) in builder.counts {
        let mut moves: Vec<_> = moves
            .into_iter()
            .filter(|(_, stats)| stats.observations >= 3)
            .collect();
        moves.sort_unstable_by(|(left_move, left), (right_move, right)| {
            right
                .weight
                .cmp(&left.weight)
                .then_with(|| left_move.cmp(right_move))
        });
        if moves.is_empty() {
            continue;
        }
        entries += 1;
        alternatives += moves.len();
        let alternatives = moves
            .into_iter()
            .map(|(mv, stats)| format!("{mv}:{}", stats.weight))
            .collect::<Vec<_>>()
            .join(" ");
        book.push_str(&format!("{fen}\t{alternatives}\n"));
    }
```

- [ ] **Step 5: Add the single-occurrence game to the fixture**

`assets/opening-book/lichess-cc0-fixture.pgn` currently ends `1. d4 d5 0-1\n` with no trailing blank line. **Append a blank line first, then the game.** PGN requires a blank line between one game's movetext and the next tag section, and every existing game boundary in the file has one. Without it `pgn-reader` merges the last two games, `source_games` comes back 6 instead of 7, and Step 7 fails with a metrics mismatch that reads like a filter bug.

So the file gains exactly this, blank line included:

```

[Event "Rated fixture"]
[WhiteElo "2300"]
[BlackElo "2300"]
[Result "1-0"]

1. c4 c5 1-0
```

This is the committed evidence. Under today's code `c2c4` scores three points and enters the startpos record, tie-breaking ahead of `d2d4:3` on ascending UCI. Under the corrected rule it has one observation and is excluded, so `fixture-book-v2.txt` stays byte-identical.

- [ ] **Step 6: Update the fixture metrics by hand**

`assets/opening-book/fixture-metrics.tsv` becomes:

```
metric	value
source_games	7
accepted_games	7
positions	4
entries	3
alternatives	4
```

Separators are literal tabs, not spaces. Three counters move. `source_games` and `accepted_games` rise to seven because the new game is parsed and accepted. `positions` rises from three to four because it is `builder.counts.len()` (main.rs:158), captured *before* the filter, so the one fresh signature the game reaches (the position after `c4`) counts even though none of its moves reach the book. `entries` and `alternatives` are post-filter and stay at three and four.

- [ ] **Step 7: Push and verify green in CI**

```bash
git add book-tool/src/main.rs assets/opening-book
git commit -m "fix: filter book moves on observations rather than summed weight"
git push
gh run watch <run-id> --exit-status
```

Expected: PASS. Both the new unit test and `generate_writes_deterministic_book_and_metrics` pass, and the `Opening Book` workflow's two diffs are empty. `fixture-book-v2.txt` is untouched in this commit's diff. If the book diff is non-empty, the filter fix is wrong.

---

### Task 3: Bound the book with `--max-positions N`

**Files:**
- Modify: `book-tool/src/main.rs` (`build_book`, `generate`, `run`)
- Modify: `book-tool/tests/generate.rs` (new unit test)

The pinned export yields roughly a million games. The rating filter admits a small fraction, but the result is still far larger than the repository should carry. The flag keeps the N most-observed positions and then re-sorts by FEN, so output stays deterministic and the byte-identical checks stay meaningful.

The bound applies **after** the minimum-three filter, to the positions that would otherwise be written. A position's rank is the sum of the observations of the alternatives it *retains*. Ties break on ascending FEN so the retained set is stable across runs. The default is unlimited, so the flag alone does not alter fixture output.

`entries` and `alternatives` are counted **after** the cap, so the production `metrics.tsv` describes the book that was actually emitted and its `entries` matches the committed book's record count. `positions` stays the pre-filter signature count, which measures corpus reach rather than book size.

- [ ] **Step 1: Write the failing test**

Append to `book-tool/tests/generate.rs`:

```rust
const THREE_POSITION_CORPUS: &str = concat!(
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"0-1\"]\n\n1. d4 d5 0-1\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"0-1\"]\n\n1. d4 d5 0-1\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"0-1\"]\n\n1. d4 d5 0-1\n",
);

#[test]
fn max_positions_keeps_the_most_observed_positions() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let input = root.join("corpus.pgn");
    let book = root.join("book.txt");
    let metrics = root.join("metrics.tsv");
    fs::write(&input, THREE_POSITION_CORPUS).expect("write fixture");

    let status = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "generate",
            input.to_str().unwrap(),
            book.to_str().unwrap(),
            metrics.to_str().unwrap(),
            "--max-positions",
            "1",
        ])
        .status()
        .expect("run book generator");

    assert!(status.success());
    // The start position retains six observations across its two alternatives;
    // the positions after e4 and after d4 retain three each, so the bound of
    // one keeps only the start position.
    assert_eq!(
        fs::read_to_string(&book).expect("generated book"),
        "rusty-fish-book v2\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9 d2d4:3\n"
    );
    // entries and alternatives are counted after the cap; positions is the
    // pre-filter signature count and is unaffected by it.
    assert_eq!(
        fs::read_to_string(&metrics).expect("generated metrics"),
        "metric\tvalue\nsource_games\t6\naccepted_games\t6\npositions\t3\nentries\t1\nalternatives\t2\n"
    );

    fs::remove_dir_all(root).expect("remove temporary directory");
}
```

- [ ] **Step 2: Push and verify it fails in CI**

```bash
git add book-tool/tests/generate.rs
git commit -m "test: pin --max-positions retention and its post-cap metrics"
git push
gh run watch <run-id> --exit-status
```

Expected: FAIL in the `Run the book generator tests` step. `run` (main.rs:227-231) rejects any fourth argument, so the process exits non-zero with the usage string and `assert!(status.success())` trips.

- [ ] **Step 3: Restructure `build_book` to rank, cap, then render**

In `book-tool/src/main.rs`, give `build_book` a bound and split rendering into a second pass. Replace the body from `let positions = builder.counts.len();` (main.rs:158) through the end of the function:

```rust
    let positions = builder.counts.len();
    let mut retained: Vec<(String, Vec<(String, MoveStats)>, u32)> = Vec::new();
    for (fen, moves) in builder.counts {
        let mut moves: Vec<_> = moves
            .into_iter()
            .filter(|(_, stats)| stats.observations >= 3)
            .collect();
        if moves.is_empty() {
            continue;
        }
        moves.sort_unstable_by(|(left_move, left), (right_move, right)| {
            right
                .weight
                .cmp(&left.weight)
                .then_with(|| left_move.cmp(right_move))
        });
        let observations = moves.iter().map(|(_, stats)| stats.observations).sum();
        retained.push((fen, moves, observations));
    }

    if let Some(max_positions) = max_positions
        && retained.len() > max_positions
    {
        // Rank by observations descending, breaking ties on ascending FEN so the
        // retained set is stable across runs, then restore FEN order for output.
        retained.sort_unstable_by(|(left_fen, _, left), (right_fen, _, right)| {
            right.cmp(left).then_with(|| left_fen.cmp(right_fen))
        });
        retained.truncate(max_positions);
        retained.sort_unstable_by(|(left_fen, _, _), (right_fen, _, _)| left_fen.cmp(right_fen));
    }

    let mut entries = 0;
    let mut alternatives = 0;
    let mut book = String::from("rusty-fish-book v2\n");
    for (fen, moves, _) in retained {
        entries += 1;
        alternatives += moves.len();
        let alternatives = moves
            .into_iter()
            .map(|(mv, stats)| format!("{mv}:{}", stats.weight))
            .collect::<Vec<_>>()
            .join(" ");
        book.push_str(&format!("{fen}\t{alternatives}\n"));
    }
    Ok(BookReport {
        source_games: builder.source_games,
        accepted_games: builder.accepted_games,
        positions,
        entries,
        alternatives,
        book,
    })
```

`builder.counts` is a `BTreeMap`, so `retained` is built in ascending FEN order and needs no initial sort. Change the signature at main.rs:146:

```rust
fn build_book(
    pgn: &str,
    filter: BookFilter,
    max_positions: Option<usize>,
) -> Result<BookReport, String> {
```

- [ ] **Step 4: Thread the bound through `generate`**

```rust
fn generate(
    input: &Path,
    book: &Path,
    metrics: &Path,
    max_positions: Option<usize>,
) -> Result<(), String> {
    let pgn = std::fs::read_to_string(input)
        .map_err(|error| format!("could not read {}: {error}", input.display()))?;
    let report = build_book(
        &pgn,
        BookFilter {
            min_rating: 2200,
            max_plies: 16,
        },
        max_positions,
    )?;
```

The rest of `generate` is unchanged.

- [ ] **Step 5: Parse the flag in `run`**

Replace `run` (main.rs:212-233). The flag is accepted in any position; positional arguments keep their existing order.

```rust
fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let program = args.next().unwrap_or_else(|| "book-tool".to_string());
    let usage = format!(
        "usage: {program} generate <input.pgn> <book.txt> <metrics.tsv> [--max-positions N]"
    );
    let Some(command) = args.next() else {
        return Err(usage);
    };
    if command != "generate" {
        return Err(format!("unknown command: {command}"));
    }

    let mut positional = Vec::new();
    let mut max_positions = None;
    while let Some(arg) = args.next() {
        if arg == "--max-positions" {
            let value = args.next().ok_or_else(|| usage.clone())?;
            let parsed = value
                .parse::<usize>()
                .map_err(|_| format!("invalid --max-positions value: {value}"))?;
            max_positions = Some(parsed);
        } else {
            positional.push(arg);
        }
    }

    let [input, book, metrics] = positional.as_slice() else {
        return Err(usage);
    };
    generate(
        Path::new(input),
        Path::new(book),
        Path::new(metrics),
        max_positions,
    )
}
```

- [ ] **Step 6: Push and verify green in CI**

```bash
git add book-tool/src/main.rs
git commit -m "feat: bound the generated opening book with --max-positions"
git push
gh run watch <run-id> --exit-status
```

Expected: PASS, including the unchanged `generate_writes_deterministic_book_and_metrics` — the default is unlimited, so fixture output is untouched.

---

### Task 4: Stream the PGN from stdin

**Files:**
- Modify: `book-tool/src/main.rs` (`build_book`, `generate`)
- Modify: `book-tool/tests/generate.rs` (new unit test)

`generate` currently calls `std::fs::read_to_string(input)` (main.rs:195), which would materialize the ~2 GB decompressed export in memory. `pgn-reader`'s `Reader` is already generic over `io::Read`, so switching to a reader keeps memory constant regardless of export size. No new Rust dependency is added; the runner's `zstd` performs decompression.

- [ ] **Step 1: Write the failing test**

Append to `book-tool/tests/generate.rs`. Add `use std::io::Write;` and `use std::process::Stdio;` to the imports at the top.

```rust
#[test]
fn a_dash_input_path_reads_the_pgn_from_stdin() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let book = root.join("book.txt");
    let metrics = root.join("metrics.tsv");

    let mut child = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "generate",
            "-",
            book.to_str().unwrap(),
            metrics.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .spawn()
        .expect("start book generator");
    child
        .stdin
        .take()
        .expect("generator stdin")
        .write_all(FIXTURE.as_bytes())
        .expect("stream fixture to stdin");
    let status = child.wait().expect("run book generator");

    assert!(status.success());
    assert_eq!(
        fs::read_to_string(&book).expect("generated book"),
        EXPECTED_BOOK
    );

    fs::remove_dir_all(root).expect("remove temporary directory");
}
```

Streaming the committed fixture through stdin must produce exactly the book the file path produces, which is what ties this test to the same `EXPECTED_BOOK` constant.

- [ ] **Step 2: Push and verify it fails in CI**

```bash
git add book-tool/tests/generate.rs
git commit -m "test: pin that a dash input path streams the PGN from stdin"
git push
gh run watch <run-id> --exit-status
```

Expected: FAIL in the `Run the book generator tests` step. `generate` treats `-` as a filename, `read_to_string` reports no such file, and the process exits non-zero.

- [ ] **Step 3: Take an `io::Read` in `build_book`**

In `book-tool/src/main.rs`, generalize the reader. Change the signature and the first line of the body (main.rs:146-147):

```rust
fn build_book<R: std::io::Read>(
    pgn: R,
    filter: BookFilter,
    max_positions: Option<usize>,
) -> Result<BookReport, String> {
    let mut reader = Reader::new(pgn);
```

Delete `use std::io::Cursor;` (main.rs:2). It had exactly one use, the `Cursor::new(pgn.as_bytes())` you just replaced, and `main.rs` has no `#[cfg(test)]` module. (The spec says "the `&str` form is retained for unit tests"; that is vestigial — there are no in-file unit tests to retain it for, and `book-tool/tests/generate.rs` drives the binary through `Command`, not the function.) Leaving it is an unused-import warning.

- [ ] **Step 4: Select stdin on `-` in `generate`**

```rust
fn generate(
    input: &Path,
    book: &Path,
    metrics: &Path,
    max_positions: Option<usize>,
) -> Result<(), String> {
    let filter = BookFilter {
        min_rating: 2200,
        max_plies: 16,
    };
    let report = if input == Path::new("-") {
        build_book(std::io::stdin().lock(), filter, max_positions)?
    } else {
        let file = std::fs::File::open(input)
            .map_err(|error| format!("could not read {}: {error}", input.display()))?;
        build_book(std::io::BufReader::new(file), filter, max_positions)?
    };
    let metrics_tsv = report.metrics_tsv();
    std::fs::write(book, report.book)
        .map_err(|error| format!("could not write {}: {error}", book.display()))?;
    std::fs::write(metrics, metrics_tsv)
        .map_err(|error| format!("could not write {}: {error}", metrics.display()))?;
    Ok(())
}
```

The `BufReader` matters: `Reader` issues many small reads and an unbuffered `File` would syscall on each one.

- [ ] **Step 5: Push and verify green in CI**

```bash
git add book-tool/src/main.rs
git commit -m "feat: stream the opening book source PGN from stdin"
git push
gh run watch <run-id> --exit-status
```

Expected: PASS, all four `book-tool` tests plus an empty pair of diffs in the `Opening Book` workflow.

---

### Task 5: Record the bound in the manifest

**Files:**
- Modify: `assets/opening-book/manifest.toml`

The manifest is the production book's provenance. Once the book is bounded, a manifest that omits the bound no longer describes how the committed artifact was selected.

- [ ] **Step 1: Add the bound and fold it into the selection rules**

In `assets/opening-book/manifest.toml`, replace the `selection` line and add `max_positions` after it:

```toml
selection = "standard rated games; both players >= 2200; first 16 plies; minimum three observations per move; the 5000 most-observed positions"
max_positions = 5000
```

Leave `source_name`, `license`, `license_url`, `source_url`, `source_sha256`, and `fixture_provenance` untouched.

- [ ] **Step 2: Commit and push**

```bash
git add assets/opening-book/manifest.toml
git commit -m "docs: record the opening book position bound in the manifest"
git push
```

No code reads the manifest, so this task has no test. The `Opening Book` workflow still runs (its `assets/opening-book/**` path filter matches) and must stay green.

---

### Task 6: The dispatch-only refresh workflow

**Files:**
- Create: `.github/workflows/opening-book-refresh.yml`

This is the only thing that ever writes the production pair. It is dispatch-only and never runs on push or pull request, honoring the constraint that ordinary PR CI never downloads the full database.

- [ ] **Step 1: Write the workflow**

```yaml
name: Opening Book Refresh

on:
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

jobs:
  refresh:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Install zstd
        run: sudo apt-get update && sudo apt-get install --yes zstd
      - name: Download the pinned CC0 export
        run: |
          set -euo pipefail
          curl --location --fail --retry 5 --retry-delay 10 --retry-all-errors \
            --output export.pgn.zst \
            https://database.lichess.org/standard/lichess_db_standard_rated_2014-12.pgn.zst
      - name: Verify the pinned SHA-256
        run: |
          set -euo pipefail
          echo "4589a1af622a893d196bc8eaede657652ce65dc79d2f289ff65fadd6a7076af4  export.pgn.zst" \
            | sha256sum --check --strict
      - name: Build the generator
        run: cargo build --release -p book-tool
      - name: Regenerate the production book
        run: |
          set -euo pipefail
          zstdcat export.pgn.zst \
            | ./target/release/book-tool generate - \
                assets/opening-book/rusty-fish-book-v2.txt \
                assets/opening-book/metrics.tsv \
                --max-positions 5000
      - name: Assert the book is not silently under-filled
        run: |
          set -euo pipefail
          entries="$(awk -F'\t' '$1 == "entries" { print $2 }' assets/opening-book/metrics.tsv)"
          echo "entries=$entries"
          if [ "${entries:-0}" -lt 4000 ]; then
            echo "::error::book has $entries entries, expected close to the 5000 cap" >&2
            exit 1
          fi
      - name: Report book metrics
        run: |
          {
            echo '### Refreshed opening book metrics'
            echo
            echo '```'
            cat assets/opening-book/metrics.tsv
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"
      - name: Open a pull request
        uses: peter-evans/create-pull-request@v6
        with:
          branch: chore/opening-book-refresh
          base: main
          add-paths: |
            assets/opening-book/rusty-fish-book-v2.txt
            assets/opening-book/metrics.tsv
          commit-message: "chore: refresh the opening book from the pinned CC0 export"
          title: "chore: refresh the opening book from the pinned CC0 export"
          body: |
            Regenerated by the Opening Book Refresh workflow from the CC0 Lichess
            export pinned in `assets/opening-book/manifest.toml`, bounded to the
            5000 most-observed positions. Metrics are in the run summary.
```

Points worth understanding rather than copying blindly:

- **`set -euo pipefail` on the generate step.** Without `pipefail`, a `zstdcat` decode failure mid-stream is swallowed: the pipeline's status is `book-tool`'s, which would happily exit zero having written a truncated book. This is the difference between a loud failure and a silently corrupt committed artifact.
- **`sha256sum --check --strict`** fails the job on mismatch, which is the point of pinning. Do not soften it to a warning.
- **The build step is separate from the generate step** so `zstdcat` is not holding a ~2 GB stream open while Cargo compiles.
- **`--retry-all-errors`** covers connection resets on a 259 MB download that plain `--retry` will not.
- **120 minutes** is generous for parsing roughly a million games. Tune down only with a real timing from a completed run.
- **The `entries` floor of 4000** catches a truncated or over-filtered corpus before a human reviews the PR. `entries` is counted after the cap, so a healthy run should sit at exactly 5000; anything far below means the rating filter or the decode admitted far less than expected. It is a smoke alarm, not a tuned threshold.
- **The refresh PR does not touch the fixture pair**, so `opening-book.yml` (whose `assets/opening-book/**` trigger still fires on it) regenerates the fixture, finds both diffs empty, and passes. That is the whole reason Task 1 came first.
- **`add-paths` lists only the production pair.** The workflow never modifies `manifest.toml`; Task 5 commits that by hand. Adding it here would be dead config.
- **Note that this workflow's own commit fires no CI at all** — `.github/workflows/opening-book-refresh.yml` matches no path filter in the repository. That is expected, not a gap. Step 3 below is the only check this task gets.

- [ ] **Step 2: Commit and push**

```bash
gh auth status
git add .github/workflows/opening-book-refresh.yml
git commit -m "feat: add a dispatch-only opening book refresh workflow"
git push
```

- [ ] **Step 3: Confirm it does not run on push**

```bash
gh run list --branch feat/opening-book-corpus-refresh --limit 10
```

Expected: no `Opening Book Refresh` run appears. It is `workflow_dispatch` only. Seeing one here means the triggers are wrong.

---

### Task 7: Open the pull request

- [ ] **Step 1: Verify the full branch diff**

```bash
gh auth status
git fetch origin
git diff --stat origin/main...HEAD
```

Expected: the spec and this plan under `docs/`, the two renamed assets, the modified fixture PGN and `fixture-metrics.tsv`, `manifest.toml`, `book-tool/src/main.rs`, `book-tool/tests/generate.rs`, and the two workflows. **`assets/opening-book/fixture-book-v2.txt` must appear as a pure rename with no content change.** If git reports content churn on it, the filter fix is wrong and the "book invariance is the evidence" argument no longer holds. Stop and investigate rather than accepting the diff.

- [ ] **Step 2: Open the PR**

Use the superpowers:finishing-a-development-branch skill. The body should lead with the two things a reviewer cannot infer from the diff: that the fixture book staying byte-identical is the deliberate regression evidence for the filter fix, and that the production names hold no committed file until the first refresh PR lands.

- [ ] **Step 3: Watch every required check to completion**

Poll the specific run ID rather than the branch's latest — a stale poll will report the previous push's result. Tolerate transient `error connecting to api.github.com` responses; those are not check failures.

- [ ] **Step 4: After merge, dispatch the first refresh**

```bash
gh auth status
gh workflow run "Opening Book Refresh" --ref main
```

This produces the first real production book as its own reviewable PR. Read the run summary's metrics before approving it: `entries` should be at or near 5000 (it is post-cap), and `positions` should be far larger, since it measures corpus reach.

- [ ] **Step 5: Update the work tracker**

Only after merge, update the epic entry in `D:/Work-Tracking/work-tracker-personal.md` (line ~374) and prepend the rolling log at line ~18. That file is outside the git repository. Two claims there need correcting rather than extending: the epic entry and the earlier design doc both describe the minimum-three rule as though it were already implemented on observations. Say plainly that it was implemented on summed points and that this change corrects it.

---

## Out of scope

Hit-rate metrics over a fixed opening-position suite. `metrics.tsv` emits no hit-rate column, and the measurement is only meaningful once a real corpus is committed, so it follows as its own spec. `EXTERNAL_SPRT_POSITIONS` (16 FENs) and `GAUNTLET_POSITIONS` in `engine-bench/src/main.rs` are reusable corpora for it.
