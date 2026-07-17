# Mobility Evaluation Term Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a piece-mobility term to the hand-crafted evaluation, shipped behind a `SearchParams.mobility_scale` knob defaulting to off, plus a self-play SPRT gate to prove it before enabling.

**Architecture:** `engine_core` exposes a public per-piece attack-set accessor. `evaluate_position` gains a tapered mobility term for knights/bishops/rooks/queens, scaled by a new `SearchParams.mobility_scale` (0–100, default 0) that the searcher threads in. An `engine-bench` self-play command pits `mobility_scale=100` against `=0` over generated openings and emits the existing SPRT TSV; a dispatch-only workflow runs it.

**Tech Stack:** Rust 2024 (`engine-core`, `engine-search`, `engine-bench`), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-17-eval-mobility-design.md`

---

## Global constraints

- **Never run Cargo or Rust binaries locally.** Every red and green is observed in GitHub Actions. A change under `engine-core/**`, `engine-search/**`, or `engine-bench/**` triggers the `Rusty Fish Tests` workflow (`cargo test --workspace`) — that is where unit tests run. Each "run the test" step is: commit, push, watch the run, read the log.
- **Verify the `gh` account is `benzsevern` before every remote op** (`gh auth status`; `gh auth switch --user benzsevern` if it reverted). For writes/PRs pass the keyring PAT as `GH_TOKEN`. Push via tokenized URL: `git push "https://benzsevern:$TOK@github.com/benseverndev-oss/rusty-fish.git" HEAD:feat/eval-mobility`. **Before branching off main, run a real `git fetch origin --prune`** — tokenized-URL pushes do not update `origin/*` tracking refs, so a naive `checkout -b origin/main` can land on a stale base.
- **Stage paths explicitly — never `git add -A`** (it sweeps banned untracked files like `.infisical.json`).
- **`cargo fmt` formatting, four-space indent, Conventional Commit subjects.** Match the file by hand; CI's fmt gate catches drift.
- **Branch:** `feat/eval-mobility`, already checked out on true main (`2f90e08`, includes #48/#49/#50) with the spec committed.

## Background an engineer needs

**The evaluation** (`engine-search/src/lib.rs`). `evaluate_position(board) -> i32` sums per-piece terms into `EvalFeatures` using `TaperedScore { middlegame, endgame }` interpolated by `endgame_phase(board)`. Terms: material, a procedural piece-square bonus, activity, bishop pair, pawn structure, rook file, king safety, threats. The `Searcher::evaluate` method (lib.rs:1421) returns NNUE eval when a net is loaded, else `evaluate_position(board)`. `hand_crafted_evaluation(board)` (lib.rs:1984) is a public wrapper that calls `evaluate_position(board)`.

**`SearchParams`** (lib.rs:97) is the per-side tunable struct: `Copy + PartialEq + Eq`, with a "Default reproduces the hand-set constants exactly" invariant guarded by `default_search_params_match_the_original_constants` (lib.rs:2768). It is also the SPSA tuning surface: `engine-bench`'s `search_params_to_vector` / `vector_to_search_params` project it onto `[f64; SPSA_DIMENSIONS]` (currently 8), and `vector_to_search_params` builds a **full struct literal with no `..default`** (engine-bench lib.rs:800).

**`engine_core` attacks** are private free functions (`knight_attacks(sq)`, `bishop_attacks(sq, occupied)`, `rook_attacks`, `queen_attacks`, `king_attacks`); `Board::occupancy(color) -> Bitboard` and `Bitboard = u64` are public.

**Self-play is deterministic**, so 16 fixed positions would yield only 32 identical games — far too few for SPRT. The gate therefore draws game variety from `random_opening_fens(count, plies, seed)` (engine-bench lib.rs:385), and both sides search at the **same** depth so the only difference is the mobility scale. `play_parameter_game(fen, candidate_color, candidate_params, baseline_params, config)` (lib.rs:545) plays one such game; `summarize` and `sprt_tsv_report` produce the report.

**Default-off is also zero-cost.** The mobility branch is guarded by `mobility_scale != 0`, so when off it does no attack computation — `main` stays byte-identical *and* the same NPS. The cost (slider attacks per piece) is paid only when the term is enabled.

## File structure

| Path | Change | Responsibility |
|------|--------|----------------|
| `engine-core/src/lib.rs` | Modify | Public `Board::attacks(square, piece) -> Bitboard`. |
| `engine-search/src/lib.rs` | Modify | `SearchParams.mobility_scale`; `mobility_score`; the mobility branch in `evaluate_position`; thread the scale. |
| `engine-bench/src/lib.rs` | Modify | `run_mobility_gate`; keep `vector_to_search_params` consistent. |
| `engine-bench/src/main.rs` | Modify | `mobility-gate` command. |
| `.github/workflows/mobility-gate.yml` | Create | Dispatch-only self-play SPRT. |

---

### Task 1: Expose attacks from `engine_core`

**Files:**
- Modify: `engine-core/src/lib.rs`

- [ ] **Step 1: Write the failing test**

Add to `engine-core`'s test module. `Square(idx)` uses `idx = rank*8 + file`, so d4 is `Square(27)` and a1 is `Square(0)`.

```rust
#[test]
fn attacks_reports_pseudo_legal_targets() {
    use crate::{Color, Piece, PieceKind, Square};
    let knight = Piece { color: Color::White, kind: PieceKind::Knight };
    // Central knight on d4 reaches 8 squares; a cornered knight on a1 reaches 2.
    let central = Board::from_fen("8/8/8/8/3N4/8/8/8 w - - 0 1").unwrap();
    assert_eq!(central.attacks(Square(27), knight).count_ones(), 8);
    let corner = Board::from_fen("8/8/8/8/8/8/8/N7 w - - 0 1").unwrap();
    assert_eq!(corner.attacks(Square(0), knight).count_ones(), 2);
    // A rook's slide stops at the first blocker (own pawn on d6 blocks past it).
    let rook = Piece { color: Color::White, kind: PieceKind::Rook };
    let blocked = Board::from_fen("8/8/3P4/8/8/8/8/3R4 w - - 0 1").unwrap();
    let up_file = blocked.attacks(Square(3), rook); // rook on d1
    assert!(up_file & (1 << 19) != 0, "attacks d3");   // d3 reachable
    assert!(up_file & (1 << 43) != 0, "attacks d6");   // d6 (the blocker) reachable
    assert!(up_file & (1 << 51) == 0, "stops at d6");  // d7 not reachable
}
```

- [ ] **Step 2: Push and verify it fails in CI**

```bash
git add engine-core/src/lib.rs
git commit -m "test: pin engine_core piece attack accessor"
git push
gh run watch <run-id> --exit-status
```

Expected: FAIL in the `workspace` job — `no method named attacks`. Confirm via `--log-failed` it is a compile error on `attacks`, not something else. If it compiles, stop and report.

- [ ] **Step 3: Add the accessor**

In `engine-core/src/lib.rs`, inside `impl Board` (near `occupancy` / `is_square_attacked`):

```rust
    /// The pseudo-legal squares `piece` attacks from `square`, given the current
    /// all-piece occupancy. Sliders stop at the first blocker (the blocker's
    /// square is included). Defined for knight, bishop, rook, queen, and king;
    /// pawns are never passed here (mobility does not use pawn attacks) and yield
    /// an empty set.
    pub fn attacks(&self, square: Square, piece: Piece) -> Bitboard {
        let occupied = self.occupancy(Color::White) | self.occupancy(Color::Black);
        match piece.kind {
            PieceKind::Knight => knight_attacks(square),
            PieceKind::Bishop => bishop_attacks(square, occupied),
            PieceKind::Rook => rook_attacks(square, occupied),
            PieceKind::Queen => queen_attacks(square, occupied),
            PieceKind::King => king_attacks(square),
            PieceKind::Pawn => 0,
        }
    }
```

- [ ] **Step 4: Push and verify green**

```bash
git add engine-core/src/lib.rs
git commit -m "feat: expose per-piece attack sets from engine_core"
git push
gh run watch <run-id> --exit-status
```

Expected: PASS — the new test and the whole workspace suite green.

---

### Task 2: The mobility term and its scale

**Files:**
- Modify: `engine-search/src/lib.rs`
- Modify: `engine-bench/src/lib.rs` (`vector_to_search_params`)

- [ ] **Step 1: Write the failing test**

Add to `engine-search`'s test module (it already imports `evaluate_position`). The position gives White a central knight (d4, 8 mobility) and Black a cornered knight (a8, 2 mobility), kings present so the FEN is legal; with mobility on, White's eval rises relative to off.

```rust
#[test]
fn mobility_scale_rewards_the_more_active_side() {
    // White knight on d4 (8 targets), black knight on a8 (2 targets).
    let board = Board::from_fen("n6k/8/8/8/3N4/8/8/7K w - - 0 1").unwrap();
    let off = evaluate_position(&board, 0);
    let on = evaluate_position(&board, 100);
    // White is to move, so a positive mobility difference raises the score.
    assert!(on > off, "mobility should favor the side with the more active knight: on={on} off={off}");
}
```

- [ ] **Step 2: Push and verify it fails in CI**

```bash
git add engine-search/src/lib.rs
git commit -m "test: pin mobility scale changes evaluation"
git push
gh run watch <run-id> --exit-status
```

Expected: FAIL — `evaluate_position` takes one argument today, so the test does not compile (`this function takes 1 argument but 2 were supplied`). Confirm it is that compile error.

- [ ] **Step 3: Add `mobility_scale` to `SearchParams`**

In `engine-search/src/lib.rs`, add the field to the `SearchParams` struct (lib.rs:97):

```rust
    pub null_move_reduction: u8,
    /// Scales the mobility evaluation term, 0–100. 0 disables it (and skips its
    /// cost). Excluded from the SPSA vector; tuned in a later sub-project.
    pub mobility_scale: i32,
```

Add to its `Default` impl (lib.rs:108), keeping every other field:

```rust
            null_move_reduction: 3,
            mobility_scale: 0,
```

- [ ] **Step 4: Keep the SPSA vector round-trip consistent**

`mobility_scale` is **not** an SPSA dimension. Leave `SPSA_DIMENSIONS` and `search_params_to_vector` unchanged. In `engine-bench/src/lib.rs`, `vector_to_search_params` builds a full literal — add the field set to the default so a round-trip preserves it (the vector never carries it, and SPSA runs with mobility off):

```rust
        null_move_reduction: clamped[7].round() as u8,
        mobility_scale: 0,
```

- [ ] **Step 5: Add the mobility term and thread the scale**

In `engine-search/src/lib.rs`, add a helper near the other eval helpers:

```rust
fn mobility_score(kind: PieceKind, mobility: i32) -> TaperedScore {
    // Centered by a per-piece offset so an average-mobility piece scores near
    // zero (material already accounts for having the piece). Hand-set starting
    // weights; tuned later via SPSA.
    let (mg, eg, offset) = match kind {
        PieceKind::Knight => (4, 4, 4),
        PieceKind::Bishop => (3, 3, 6),
        PieceKind::Rook => (2, 4, 7),
        PieceKind::Queen => (1, 2, 13),
        _ => (0, 0, 0),
    };
    let centered = mobility - offset;
    TaperedScore::new(mg * centered, eg * centered)
}
```

Change `evaluate_position`'s signature and add the mobility branch inside the first `for idx in 0..64` loop, right after the material/piece-square `features.add(...)` call. The `mobility_scale != 0` guard makes the term zero-cost when off:

```rust
fn evaluate_position(board: &Board, mobility_scale: i32) -> i32 {
```

```rust
        if mobility_scale != 0
            && matches!(
                piece.kind,
                PieceKind::Knight | PieceKind::Bishop | PieceKind::Rook | PieceKind::Queen
            )
        {
            let mobility =
                (board.attacks(square, piece) & !board.occupancy(piece.color)).count_ones() as i32;
            let raw = mobility_score(piece.kind, mobility);
            let scaled = TaperedScore::new(
                raw.middlegame * mobility_scale / 100,
                raw.endgame * mobility_scale / 100,
            );
            features.add(piece.color, scaled, endgame_phase);
        }
```

Thread the scale from the searcher — `Searcher::evaluate` (lib.rs:1436):

```rust
            None => evaluate_position(board, self.params.mobility_scale),
```

And the public wrapper `hand_crafted_evaluation` (lib.rs:1984) evaluates at the stable baseline:

```rust
pub fn hand_crafted_evaluation(board: &Board) -> i32 {
    evaluate_position(board, 0)
}
```

- [ ] **Step 6: Fix the other `evaluate_position` call sites**

Search the whole `engine-search` crate for `evaluate_position(` and give every remaining caller (all in the test module — e.g. `evaluation_prefers_passed_pawn_and_bishop_pair`) a `0` second argument, so their expected values are unchanged. These tests passing at scale 0 is the proof that mobility is inert when off — no separate byte-identical test is needed.

```bash
grep -n "evaluate_position(" engine-search/src/lib.rs
```

Also add one line to `default_search_params_match_the_original_constants` (lib.rs:2768) to pin the new default: `assert_eq!(params.mobility_scale, 0);`.

- [ ] **Step 7: Push and verify green**

```bash
git add engine-search/src/lib.rs engine-bench/src/lib.rs
git commit -m "feat: add mobility evaluation term behind a default-off scale"
git push
gh run watch <run-id> --exit-status
```

Expected: PASS — `mobility_scale_rewards_the_more_active_side`, the SPSA round-trip test, the default-constants test, and every existing eval test all green. The Throughput Benchmark (push-only) should show no NPS regression, because mobility is off by default and its branch is skipped.

---

### Task 3: The self-play gate

**Files:**
- Modify: `engine-bench/src/lib.rs`
- Modify: `engine-bench/src/main.rs`
- Create: `.github/workflows/mobility-gate.yml`

- [ ] **Step 1: Write the failing test**

Add to `engine-bench`'s test module. A tiny, fast configuration (few openings, shallow depth) that only checks the harness runs and reports.

```rust
#[test]
fn mobility_gate_plays_games_and_reports() {
    let config = MatchConfig { candidate_depth: 2, baseline_depth: 2, max_plies: 20 };
    let records = run_mobility_gate(2, 0xC0FFEE, config).expect("gate runs");
    assert_eq!(records.len(), 4); // 2 openings x 2 colors
    let report = sprt_tsv_report(summarize(&records), SprtConfig::default());
    assert!(report.contains("decision"));
}
```

The engine-bench test module uses an explicit (non-glob) `use super::{...}` list. It already imports `summarize`, `SprtConfig`, and `MatchConfig`, but **not** `run_mobility_gate` or `sprt_tsv_report` — add both to that list, or this test fails to compile at Step 5 with `cannot find function sprt_tsv_report`. (So the Step 2 red will name two missing symbols, `run_mobility_gate` and `sprt_tsv_report`; that is expected.)

- [ ] **Step 2: Push and verify it fails in CI**

```bash
git add engine-bench/src/lib.rs
git commit -m "test: pin the mobility self-play gate harness"
git push
gh run watch <run-id> --exit-status
```

Expected: FAIL — `cannot find function run_mobility_gate`. Confirm it is that compile error.

- [ ] **Step 3: Implement the gate function**

In `engine-bench/src/lib.rs`:

```rust
/// Plays mobility-on (`mobility_scale = 100`) against mobility-off (`= 0`) over
/// `openings` generated openings, color-swapped, at equal depth for both sides.
/// Everything but the mobility scale is identical, so the SPRT isolates the term.
pub fn run_mobility_gate(
    openings: usize,
    seed: u64,
    config: MatchConfig,
) -> Result<Vec<GameRecord>, String> {
    let fens = random_opening_fens(openings, 8, seed);
    let candidate = SearchParams { mobility_scale: 100, ..SearchParams::default() };
    let baseline = SearchParams::default(); // mobility_scale == 0
    let mut records = Vec::with_capacity(fens.len() * 2);
    for fen in &fens {
        for candidate_color in [Color::White, Color::Black] {
            records.push(play_parameter_game(fen, candidate_color, candidate, baseline, config)?);
        }
    }
    Ok(records)
}
```

- [ ] **Step 4: Add the `mobility-gate` command**

In `engine-bench/src/main.rs`, following the existing `if std::env::args().nth(1).as_deref() == Some("...")` pattern, add before `main`'s fallthrough. Optional positional args `[openings] [depth]` default to 200 and 6:

```rust
    if std::env::args().nth(1).as_deref() == Some("mobility-gate") {
        let openings = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(200);
        let depth = std::env::args().nth(3).and_then(|a| a.parse().ok()).unwrap_or(6);
        let config = MatchConfig { candidate_depth: depth, baseline_depth: depth, max_plies: 160 };
        let records = run_mobility_gate(openings, 0xC0FFEE, config)?;
        eprint!("{}", tsv_report(&records, config));
        print!("{}", sprt_tsv_report(summarize(&records), SprtConfig::default()));
        return Ok(());
    }
```

Import whatever names this needs (`run_mobility_gate`, `MatchConfig`, `summarize`, `sprt_tsv_report`, `SprtConfig`, `tsv_report`) — match how the neighboring commands import from the library crate.

- [ ] **Step 5: Push and verify green**

```bash
git add engine-bench/src/lib.rs engine-bench/src/main.rs
git commit -m "feat: add mobility self-play gate command"
git push
gh run watch <run-id> --exit-status
```

Expected: PASS — `mobility_gate_plays_games_and_reports` and the workspace suite green.

- [ ] **Step 6: Add the dispatch-only workflow**

Create `.github/workflows/mobility-gate.yml`:

```yaml
name: Mobility Gate

on:
  workflow_dispatch:
    inputs:
      openings:
        description: Number of generated openings (games = openings x 2)
        required: false
        default: '200'
      depth:
        description: Fixed search depth for both sides
        required: false
        default: '6'

jobs:
  mobility-gate:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Run the mobility self-play SPRT
        run: |
          set -euo pipefail
          mkdir -p artifacts
          cargo run --release -p engine-bench -- mobility-gate \
            "${{ inputs.openings }}" "${{ inputs.depth }}" \
            > artifacts/mobility-sprt.tsv \
            2> artifacts/mobility-games.tsv
          echo '=== mobility-sprt.tsv ==='
          cat artifacts/mobility-sprt.tsv
          {
            echo '### Mobility self-play SPRT'
            echo
            echo '```'
            cat artifacts/mobility-sprt.tsv
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"
      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: mobility-gate-tsv
          path: |
            artifacts/mobility-sprt.tsv
            artifacts/mobility-games.tsv
```

- [ ] **Step 7: Commit and confirm no push-triggered run**

```bash
git add .github/workflows/mobility-gate.yml
git commit -m "ci: add dispatch-only mobility gate workflow"
git push
gh run list --branch feat/eval-mobility --limit 5
```

Expected: no `Mobility Gate` run appears (it is `workflow_dispatch`-only). The workflow file matches no push path filter, so its commit triggers nothing.

---

### Task 4: Open the PR, merge, and dispatch the gate

- [ ] **Step 1: Verify the full branch diff**

```bash
gh auth status
git fetch origin
git diff --stat origin/main...HEAD
```

Expected exactly: the spec and this plan under `docs/`, `engine-core/src/lib.rs`, `engine-search/src/lib.rs`, `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`, `.github/workflows/mobility-gate.yml`. No other files — in particular no `.infisical.json` or `AGENTS.md`.

- [ ] **Step 2: Open the PR**

Use the superpowers:finishing-a-development-branch skill. The body must make clear: mobility ships **off** (`mobility_scale` default 0), so `main` is byte-identical and same-NPS; the gate is a separate dispatch that proves the term; and enabling it is a one-line follow-up PR after the gate result.

- [ ] **Step 3: Merge on green**

Per the repo's standing rule, merge once all checks pass. Watch the specific run ID; tolerate transient `api.github.com` errors. Use `GH_TOKEN="$TOK" gh pr merge <n> --merge --delete-branch`.

- [ ] **Step 4: Dispatch the gate and assess**

```bash
gh auth status
gh workflow run "Mobility Gate" --ref main
```

Read the run's step summary: wins/draws/losses (mobility-on vs off), Elo estimate, LLR, decision. Interpretation, per the spec: accept if the SPRT accepts H1, or if it is inconclusive (`Continue`) but the Elo estimate is clearly positive — in which case a larger rerun (more openings) can confirm. If the estimate is negative, the term does not help as weighted; leave it off and revisit weights (the deferred SPSA sub-project) rather than reverting.

- [ ] **Step 5: If it gains, flip the default (separate one-line PR)**

Only if the gate shows a gain: a follow-up PR sets `SearchParams::default().mobility_scale` to `100` (and updates the `default_search_params_match_the_original_constants` assertion and any eval test whose expected value now includes mobility). That PR is where the Throughput Benchmark's NPS cost of active mobility shows up and is judged against the Elo gain. Update `D:/Work-Tracking/work-tracker-personal.md` after it lands.

---

## Out of scope

SPSA-tuning the mobility weights, safe/king/pawn mobility, and UCI exposure of the scale — all deferred per the spec. Enabling mobility by default is Task 5's follow-up PR, gated on a positive self-play result.
