# Opening Book Hit-Rate Metric Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `book-tool hitrate` subcommand that replays a committed suite of canonical opening lines against a book and reports a coverage TSV, wired into CI as a report-only metric.

**Architecture:** A new `hitrate <book.txt> <suite.pgn>` command in `book-tool` parses the book's position signatures into a set, replays each suite line ply-by-ply through the same shakmaty + `engine_core::Board` path the generator uses, and checks each of the first sixteen positions per line against the set. It prints a deterministic `metric\tvalue` TSV to stdout. A committed SAN suite and a report-only CI step surface the number against the production book.

**Tech Stack:** Rust 2024 (`book-tool`, `pgn-reader` 0.29, `shakmaty`, `engine-core`), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-17-opening-book-hit-rate-design.md`

---

## Global constraints

- **Never run Cargo or Rust binaries locally.** Every red and green is observed in GitHub Actions. Each "run the test" step is: commit, push, watch the run, read the log.
- **Verify the `gh` account is `benzsevern` before every remote op.** It silently reverts to `benzsevern-mjh`. Run `gh auth status`; `gh auth switch --user benzsevern` if needed. For writes/PRs, the active OAuth context is pull-only — pass the keyring PAT as `GH_TOKEN` (`TOK=$(gh auth token --user benzsevern); GH_TOKEN="$TOK" gh ...`). Push via tokenized URL: `git push "https://benzsevern:$TOK@github.com/benseverndev-oss/rusty-fish.git" HEAD:<branch>`. **Stage paths explicitly — never `git add -A`** (it sweeps untracked files like `.infisical.json`, which `AGENTS.md` bans).
- **`cargo fmt` formatting, four-space indent, Conventional Commit subjects.** Match the file by hand; CI's fmt gate catches drift.
- **Avoid unrelated refactors.** The one refactor here (extracting `BOOK_MAX_PLIES`) is load-bearing: it is the shared bound the new code must not let drift from the generator.
- **Branch:** `feat/opening-book-hitrate`, already checked out with the spec committed, off `origin/main`.

## Background an engineer needs

**The book format.** `assets/opening-book/rusty-fish-book-v2.txt` (the committed production book, ~5000 entries): line 1 is the literal header `rusty-fish-book v2`; every later line is `<signature>\t<uci>:<weight> ...`. `<signature>` is a *position signature* — a FEN with the halfmove/fullmove counters stripped (so `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -` is the start position). The en-passant field is present whenever a pawn just double-pushed, even with no capture available — e.g. the committed book has a `...RNBQKBNR b KQkq a3` line after `1. a4`. So after `1. e4` the signature ends `... b KQkq e3`.

**How the generator replays PGN** (`book-tool/src/main.rs`, `Builder::san`, lines 111-131): it converts each SAN to a move with shakmaty (`san.san.to_move(&game.chess)`), computes `game.board.position_signature()` on an `engine_core::Board` *before* making the move, converts the move to UCI (`UciMove::from_standard`), and applies it to both the `Board` (`parse_uci_move` then `make_move`) and the shakmaty `Chess` (`play_unchecked`). It records a signature only while `game.moves.len() < max_plies` where `max_plies` is `16`. The hit-rate replay mirrors this exactly, so its "does the book contain this position" check uses the identical signature code — no drift.

**Why the 16-ply bound.** The generator records the position before each of the first sixteen plies (ply indices 0..15). A position at ply index sixteen or beyond can never be in the book by construction, so checking it would be a guaranteed miss that deflates the rate. Each line therefore checks at most its first sixteen positions.

**Generator leniency vs. hit-rate strictness.** In `Builder::san`, a SAN that will not parse or a move that will not apply sets `game.valid = false`, and `end_game` (lines 133-137) silently skips invalid games — the bulk run tolerates dirty data. `hitrate` does the opposite: the suite is small and hand-curated, so a move that will not replay is a curation bug to surface loudly, not skip.

## File structure

| Path | Change | Responsibility |
|------|--------|----------------|
| `book-tool/src/main.rs` | Modify | Add `BOOK_MAX_PLIES` const, `load_book_signatures`, a `HitRate` visitor, a `hitrate` fn, and a `hitrate` command in `run`. |
| `book-tool/tests/generate.rs` | Modify | Add hit-rate tests (stdout-capturing, byte-asserted). |
| `assets/opening-book/hitrate-suite.pgn` | Create | ~16 canonical opening mainlines in SAN. |
| `.github/workflows/opening-book.yml` | Modify | Report-only hit-rate step against the committed production book. |

---

### Task 1: The `hitrate` command

**Files:**
- Modify: `book-tool/src/main.rs`
- Modify: `book-tool/tests/generate.rs`

Test-driven, one red and one green in CI.

- [ ] **Step 1: Write the failing test**

Append to `book-tool/tests/generate.rs`. This drives the compiled binary and captures **stdout** with `.output()` (the existing tests use `.status()` and read files; `hitrate` prints to stdout). Add `use std::process::Stdio;` only if not already present — it is not needed here, so add nothing new; `Command`, `fs`, and `test_directory` already exist.

```rust
const HITRATE_BOOK: &str = concat!(
    "rusty-fish-book v2\n",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9\n",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3\te7e5:5\n",
);

// Line one checks three positions (start, after e4, after e4 e5); the book has
// the first two, so 2/3 with an in-book depth of 2 and no full coverage. Line
// two checks one position (start), a hit, fully covered. Totals: 2 lines, 4
// checked, 3 in book, hit_rate 3/4, mean_book_depth (2+1)/2, 1 fully covered.
const HITRATE_SUITE: &str = concat!(
    "[Event \"line one\"]\n\n1. e4 e5 2. Nf3 *\n\n",
    "[Event \"line two\"]\n\n1. e4 *\n",
);

const HITRATE_EXPECTED: &str = concat!(
    "metric\tvalue\n",
    "lines\t2\n",
    "plies_checked\t4\n",
    "plies_in_book\t3\n",
    "hit_rate\t0.750000\n",
    "mean_book_depth\t1.500000\n",
    "fully_covered_lines\t1\n",
);

#[test]
fn hitrate_reports_coverage_over_a_suite() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let book = root.join("book.txt");
    let suite = root.join("suite.pgn");
    fs::write(&book, HITRATE_BOOK).expect("write book");
    fs::write(&suite, HITRATE_SUITE).expect("write suite");

    let output = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args(["hitrate", book.to_str().unwrap(), suite.to_str().unwrap()])
        .output()
        .expect("run book generator");

    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8(output.stdout).expect("utf8 stdout"), HITRATE_EXPECTED);

    fs::remove_dir_all(root).expect("remove temporary directory");
}
```

- [ ] **Step 2: Push and verify it fails in CI**

```bash
git add book-tool/tests/generate.rs
git commit -m "test: pin book hitrate coverage output"
git push
gh run watch <run-id> --exit-status
```

Expected: FAIL in the `Run the book generator tests` step. `run` (main.rs:257) rejects any command other than `generate` with `unknown command: hitrate`, so the process exits non-zero and `assert!(output.status.success())` trips. Confirm via `--log-failed` it is that message, not a compile error.

- [ ] **Step 3: Extract the shared depth bound**

In `book-tool/src/main.rs`, add near the top (after the imports):

```rust
/// The generator records the position before each of the first this-many plies,
/// so `hitrate` checks at most this many positions per line. Both uses must stay
/// equal or the metric misreports.
const BOOK_MAX_PLIES: u32 = 16;
```

In `generate`, replace the literal `max_plies: 16` in the `BookFilter { min_rating: 2200, max_plies: 16 }` literal (main.rs:232) with `max_plies: BOOK_MAX_PLIES`. It is the only `max_plies: 16` in the file. No behavior change.

- [ ] **Step 4: Add `HashSet` to the imports and the book-signature loader**

Change the collections import (main.rs:1) from `use std::collections::BTreeMap;` to:

```rust
use std::collections::{BTreeMap, HashSet};
```

Add this function (near `build_book`):

```rust
fn load_book_signatures(book: &str) -> Result<HashSet<String>, String> {
    let mut lines = book.lines();
    match lines.next() {
        Some("rusty-fish-book v2") => {}
        _ => return Err("book is missing the rusty-fish-book v2 header".to_string()),
    }
    let mut signatures = HashSet::new();
    for line in lines {
        let Some((signature, _)) = line.split_once('\t') else {
            return Err(format!("malformed book line: {line}"));
        };
        signatures.insert(signature.to_string());
    }
    Ok(signatures)
}
```

`str::lines` drops the trailing newline, so a book ending in `\n` yields no spurious empty final line. An empty book (header only) yields an empty set, which is valid (every check misses).

- [ ] **Step 5: Add the replay visitor**

The visitor mirrors `Builder`/`Game` but accumulates coverage instead of weights. Add:

```rust
struct HitRate {
    signatures: HashSet<String>,
    lines: u32,
    plies_checked: u64,
    plies_in_book: u64,
    depth_sum: u64,
    fully_covered_lines: u32,
    error: Option<String>,
}

struct HitLine {
    chess: Chess,
    board: Board,
    checked: u32,
    in_book: u32,
    depth: u32,
    consecutive: bool,
    valid: bool,
}

impl Visitor for HitRate {
    type Tags = ();
    type Movetext = HitLine;
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, _tags: ()) -> ControlFlow<Self::Output, Self::Movetext> {
        ControlFlow::Continue(HitLine {
            chess: Chess::default(),
            board: Board::startpos(),
            checked: 0,
            in_book: 0,
            depth: 0,
            consecutive: true,
            valid: true,
        })
    }

    fn san(&mut self, line: &mut HitLine, san: SanPlus) -> ControlFlow<Self::Output> {
        if !line.valid || line.checked >= BOOK_MAX_PLIES {
            return ControlFlow::Continue(());
        }
        let Ok(mv) = san.san.to_move(&line.chess) else {
            self.error = Some(format!("illegal or unparsable suite move: {}", san.san));
            line.valid = false;
            return ControlFlow::Continue(());
        };
        let signature = line.board.position_signature();
        let hit = self.signatures.contains(&signature);
        line.checked += 1;
        if hit {
            line.in_book += 1;
            if line.consecutive {
                line.depth += 1;
            }
        } else {
            line.consecutive = false;
        }
        let uci = UciMove::from_standard(mv.clone()).to_string();
        match line
            .board
            .parse_uci_move(&uci)
            .and_then(|parsed| line.board.make_move(parsed))
        {
            Ok(_) => line.chess.play_unchecked(mv),
            Err(_) => {
                self.error = Some(format!("illegal suite move: {uci}"));
                line.valid = false;
            }
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, line: HitLine) -> Self::Output {
        if !line.valid {
            return;
        }
        self.lines += 1;
        self.plies_checked += u64::from(line.checked);
        self.plies_in_book += u64::from(line.in_book);
        self.depth_sum += u64::from(line.depth);
        if line.checked > 0 && line.in_book == line.checked {
            self.fully_covered_lines += 1;
        }
    }
}
```

Note the check happens **before** the move is applied (signature taken from `line.board` pre-move), exactly as the generator records it. A line already at `BOOK_MAX_PLIES` checks no further and stops advancing — the suite is curated short, so plies past sixteen are irrelevant.

- [ ] **Step 6: Add the `hitrate` function**

`&[u8]` implements `io::Read`, so no `Cursor` import is needed.

```rust
fn hitrate(book_path: &Path, suite_path: &Path) -> Result<(), String> {
    let book = std::fs::read_to_string(book_path)
        .map_err(|error| format!("could not read {}: {error}", book_path.display()))?;
    let signatures = load_book_signatures(&book)?;

    let suite = std::fs::read_to_string(suite_path)
        .map_err(|error| format!("could not read {}: {error}", suite_path.display()))?;
    let mut visitor = HitRate {
        signatures,
        lines: 0,
        plies_checked: 0,
        plies_in_book: 0,
        depth_sum: 0,
        fully_covered_lines: 0,
        error: None,
    };
    let mut reader = Reader::new(suite.as_bytes());
    reader
        .visit_all_games(&mut visitor)
        .map_err(|error| error.to_string())?;
    if let Some(error) = visitor.error {
        return Err(error);
    }
    if visitor.plies_checked == 0 {
        return Err("suite checked no positions".to_string());
    }

    let hit_rate = visitor.plies_in_book as f64 / visitor.plies_checked as f64;
    let mean_book_depth = visitor.depth_sum as f64 / f64::from(visitor.lines);
    print!(
        "metric\tvalue\nlines\t{}\nplies_checked\t{}\nplies_in_book\t{}\nhit_rate\t{hit_rate:.6}\nmean_book_depth\t{mean_book_depth:.6}\nfully_covered_lines\t{}\n",
        visitor.lines, visitor.plies_checked, visitor.plies_in_book, visitor.fully_covered_lines
    );
    Ok(())
}
```

`plies_checked == 0` holds exactly when there are no lines or every line is empty, so the `mean_book_depth` division by `visitor.lines` is only reached when `lines > 0`.

- [ ] **Step 7: Route the `hitrate` command in `run`**

Replace `run` (main.rs:249-284). The `generate` arm keeps its exact current parsing; a `hitrate` arm takes two positional args.

```rust
fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let program = args.next().unwrap_or_else(|| "book-tool".to_string());
    let usage = format!(
        "usage: {program} generate <input.pgn> <book.txt> <metrics.tsv> [--max-positions N]\n       {program} hitrate <book.txt> <suite.pgn>"
    );
    let Some(command) = args.next() else {
        return Err(usage);
    };
    match command.as_str() {
        "generate" => {
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
        "hitrate" => {
            let positional: Vec<String> = args.collect();
            let [book, suite] = positional.as_slice() else {
                return Err(usage);
            };
            hitrate(Path::new(book), Path::new(suite))
        }
        other => Err(format!("unknown command: {other}")),
    }
}
```

- [ ] **Step 8: Push and verify green in CI**

```bash
git add book-tool/src/main.rs
git commit -m "feat: add book-tool hitrate command"
git push
gh run watch <run-id> --exit-status
```

Expected: PASS. `hitrate_reports_coverage_over_a_suite` passes with the exact TSV, and the existing four `generate` tests plus the byte-identical fixture diffs stay green (the `BOOK_MAX_PLIES` extraction changed no behavior).

---

### Task 2: The suite asset and the CI report step

**Files:**
- Create: `assets/opening-book/hitrate-suite.pgn`
- Modify: `.github/workflows/opening-book.yml`

No unit test — the CI step running `hitrate` against the committed production book **is** the validation. Because `hitrate` fails loudly on any illegal or unparsable suite move, a green run proves every line in the suite replays legally; a red run names the offending move.

- [ ] **Step 1: Create the suite**

Write `assets/opening-book/hitrate-suite.pgn`. Each game is one canonical mainline; the `[Event]` tag names it (tags are ignored by `hitrate`, they document the line), a blank line separates tags from movetext, and `*` ends each game. These are standard, legal opening sequences of 12-16 plies:

```
[Event "Ruy Lopez, Closed"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O *

[Event "Italian Game"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 7. Re1 a6 *

[Event "Scotch Game"]

1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4 Nf6 5. Nxc6 bxc6 6. e5 Qe7 7. Qe2 Nd5 *

[Event "Petroff Defence"]

1. e4 e5 2. Nf3 Nf6 3. Nxe5 d6 4. Nf3 Nxe4 5. d4 d5 6. Bd3 Nc6 7. O-O Be7 *

[Event "Sicilian Najdorf"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be2 e5 7. Nb3 Be7 *

[Event "Sicilian Sveshnikov"]

1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 e5 6. Ndb5 d6 7. Bg5 a6 *

[Event "Sicilian Dragon"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 g6 6. Be3 Bg7 7. f3 O-O *

[Event "French Defence, Winawer"]

1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4 O-O *

[Event "Caro-Kann, Classical"]

1. e4 c6 2. d4 d5 3. Nc3 dxe4 4. Nxe4 Bf5 5. Ng3 Bg6 6. h4 h6 7. Nf3 Nd7 *

[Event "Queen's Gambit Declined"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 7. Bh4 b6 *

[Event "Slav Defence"]

1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4 5. a4 Bf5 6. e3 e6 7. Bxc4 Bb4 *

[Event "Nimzo-Indian"]

1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 *

[Event "King's Indian Defence"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 *

[Event "Grunfeld Defence"]

1. d4 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. e4 Nxc3 6. bxc3 Bg7 7. Nf3 c5 *

[Event "English Opening"]

1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5 6. Bg2 Nb6 7. O-O Be7 *

[Event "Catalan Opening"]

1. d4 Nf6 2. c4 e6 3. g3 d5 4. Bg2 Be7 5. Nf3 O-O 6. O-O dxc4 7. Qc2 a6 *
```

- [ ] **Step 2: Add the report-only CI step**

In `.github/workflows/opening-book.yml`, add after the existing "Report book metrics" step and before (or after) the upload step:

```yaml
      - name: Report opening book hit rate
        run: |
          cargo run --release -p book-tool -- hitrate \
            assets/opening-book/rusty-fish-book-v2.txt \
            assets/opening-book/hitrate-suite.pgn > hitrate.tsv
          {
            echo '### Opening book hit rate'
            echo
            echo '```'
            cat hitrate.tsv
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"
```

Then add `hitrate.tsv` to the paths of the existing `Upload book and metrics` step so the report is retained as an artifact:

```yaml
          path: |
            regenerated-book.txt
            regenerated-metrics.tsv
            hitrate.tsv
```

`cargo run --release` build chatter goes to stderr, so `> hitrate.tsv` captures only the TSV. The production book is committed (~459 KB), so this reads a local file and downloads nothing.

- [ ] **Step 3: Push and verify green in CI**

```bash
git add assets/opening-book/hitrate-suite.pgn .github/workflows/opening-book.yml
git commit -m "feat: add opening book hit-rate suite and CI report"
git push
gh run watch <run-id> --exit-status
```

Expected: PASS. The `Report opening book hit rate` step runs `hitrate` over the production book and the suite and prints a TSV to the step summary. **Read the step summary's numbers**: `hit_rate` should be high (the suite is mainlines the book was trained toward) and `mean_book_depth` should be several plies. If the step fails, the log names the illegal/unparsable suite move — fix that line and re-push. If `hit_rate` is surprisingly low (e.g. near zero), the production book may not be loading; investigate before proceeding.

---

### Task 3: Open the pull request

- [ ] **Step 1: Verify the full branch diff**

```bash
gh auth status
git fetch origin
git diff --stat origin/main...HEAD
```

Expected exactly: the spec and this plan under `docs/`, `book-tool/src/main.rs`, `book-tool/tests/generate.rs`, `assets/opening-book/hitrate-suite.pgn`, and `.github/workflows/opening-book.yml`. **No other files** — in particular no `.infisical.json` or `AGENTS.md` (a `git add -A` slip). If any appear, remove them from the commits before opening the PR.

- [ ] **Step 2: Open the PR**

Use the superpowers:finishing-a-development-branch skill. The body should state that hit-rate is report-only (no floor gate in v1), that it measures coverage not move-agreement, and quote the first observed `hit_rate` / `mean_book_depth` from the CI step summary so a reviewer sees the baseline.

- [ ] **Step 3: Merge on green**

Per the repo's standing rule, merge once all checks pass. Watch the specific run ID; tolerate transient `api.github.com` errors. Use `GH_TOKEN="$TOK" gh pr merge <n> --merge --delete-branch`.

- [ ] **Step 4: Update the work tracker**

After merge, update the opening-book epic entry in `D:/Work-Tracking/work-tracker-personal.md` (the hit-rate item was the last "Remaining" bullet) and prepend the rolling log. That file is outside the git repo.

---

## Out of scope

A regression floor/gate on hit-rate, move-agreement metrics, and measuring hit-rate against the synthetic fixture — all deferred per the spec.
