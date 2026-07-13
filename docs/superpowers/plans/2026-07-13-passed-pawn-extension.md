# Passed-Pawn Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a genuinely passed pawn one bounded extra search ply when it advances near promotion.

**Architecture:** A pure `is_passed_pawn` predicate becomes the sole definition of passed-pawn geometry for both evaluation and search. `passed_pawn_extension` uses that predicate before a move is made and returns either zero or one, which the existing negamax loop combines with its check extension via `max`.

**Tech Stack:** Rust 2024 workspace, engine-core board API, GitHub Actions.

## Global Constraints

- Do not run Cargo locally; GitHub Actions is the sole Rust validation environment.
- Extension depth is capped at one ply even if the pawn move also gives check.
- Promotions, non-pawns, non-advanced pawns, and pawns opposed ahead on an adjacent file never extend.

---

### Task 1: Specify the extension with a failing regression test

**Files:**
- Modify: `engine-search/src/lib.rs:1580-1840`

**Interfaces:**
- Consumes: `Board`, `ChessMove`, `Color`, and `PieceKind`.
- Produces: `passed_pawn_extension(&Board, ChessMove) -> u8`.

- [ ] **Step 1: Add the failing test and import**

```rust
use super::{passed_pawn_extension, /* existing imports */};

#[test]
fn passed_pawn_extension_requires_an_advanced_unblocked_pawn_push() {
    let white = Board::from_fen("4k3/8/3P4/8/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(passed_pawn_extension(&white, white.parse_uci_move("d6d7").unwrap()), 1);

    let black = Board::from_fen("4k3/8/8/8/8/3p4/8/4K3 b - - 0 1").unwrap();
    assert_eq!(passed_pawn_extension(&black, black.parse_uci_move("d3d2").unwrap()), 1);

    let blocked = Board::from_fen("4k3/2p5/3P4/8/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(passed_pawn_extension(&blocked, blocked.parse_uci_move("d6d7").unwrap()), 0);

    let promotion = Board::from_fen("4k3/3P4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(passed_pawn_extension(&promotion, promotion.parse_uci_move("d7d8q").unwrap()), 0);
}
```

- [ ] **Step 2: Push the test-only commit and verify the GitHub red run**

Run: push the branch, then inspect `Rusty Fish Tests`.

Expected: compilation fails because `passed_pawn_extension` is not defined.

- [ ] **Step 3: Commit the red test**

```powershell
git add engine-search/src/lib.rs
git commit -m "test: cover passed-pawn extension policy"
```

### Task 2: Share passed-pawn geometry and apply the extension

**Files:**
- Modify: `engine-search/src/lib.rs:768-774,1430-1505`

**Interfaces:**
- Consumes: `is_passed_pawn(board, mv.from, color)` before `board.make_move(mv)`.
- Produces: a `u8` depth extension in `{0, 1}`.

- [ ] **Step 1: Add shared helpers**

```rust
fn is_passed_pawn(board: &Board, square: engine_core::Square, color: Color) -> bool {
    let ranks = match color {
        Color::White => square.rank() + 1..8,
        Color::Black => 0..square.rank(),
    };
    for file in square.file().saturating_sub(1)..=((square.file() + 1).min(7)) {
        for rank in ranks.clone() {
            if board.piece_at(engine_core::Square::from_file_rank(file, rank).expect("in bounds"))
                == Some(Piece { color: color.opposite(), kind: PieceKind::Pawn })
            {
                return false;
            }
        }
    }
    true
}

fn passed_pawn_extension(board: &Board, mv: ChessMove) -> u8 {
    let Some(piece) = board.piece_at(mv.from) else { return 0; };
    let advanced = matches!(piece.color, Color::White) && mv.to.rank() == 6
        || matches!(piece.color, Color::Black) && mv.to.rank() == 1;
    u8::from(piece.kind == PieceKind::Pawn && mv.promotion.is_none() && advanced
        && is_passed_pawn(board, mv.from, piece.color))
}
```

- [ ] **Step 2: Replace evaluation's inline passed-pawn loops**

```rust
let is_passed = is_passed_pawn(board, square, color);
```

- [ ] **Step 3: Use the helper before making each negamax move**

```rust
let pawn_extension = passed_pawn_extension(board, mv);
let undo = board.make_move(mv).expect("generated move must be legal");
let extension = u8::from(board.in_check(board.side_to_move)).max(pawn_extension);
```

- [ ] **Step 4: Push and verify GitHub green checks**

Require green `Rusty Fish Tests`, tactical suite, fixed-opponent gauntlet,
throughput benchmark, and CodeQL checks.

- [ ] **Step 5: Commit the implementation**

```powershell
git add engine-search/src/lib.rs
git commit -m "feat: extend advanced passed-pawn searches"
```

### Task 3: Merge and track

**Files:**
- Modify: `D:/Work-Tracking/work-tracker-personal.md:332`

- [ ] **Step 1: Open a pull request with red and green GitHub run evidence**
- [ ] **Step 2: Squash merge only after all fresh checks succeed**
- [ ] **Step 3: Record the completed passed-pawn extension in the search-quality tracker row**

## Self-review

- The test covers each allowed color and key excluded category.
- The shared predicate removes duplicated geometry rather than adding a second definition.
- Check and passed-pawn extensions are capped at one ply.
