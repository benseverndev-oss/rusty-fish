# Tapered Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate positions with tunable middlegame/endgame components interpolated by material phase.

**Architecture:** `TaperedScore` represents a pair of centipawn values and is the only value accumulated by `EvalFeatures`. `EvalParams` maps evaluation-only piece values and piece-square bonuses to pairs; existing scalar features explicitly become equal, middlegame-only, or endgame-only pairs. Search still consumes the same signed `i32` from `evaluate_position`.

**Tech Stack:** Rust 2024, engine-core board API, GitHub Actions.

## Global Constraints

- Do not run Cargo locally; GitHub Actions is the only Rust validation environment.
- Retain the 0..24 material-phase scale from `endgame_phase`.
- Keep SEE and move-ordering material values scalar and unchanged.
- Preserve the side-to-move score convention at the evaluator boundary.

---

### Task 1: Define taper interpolation with red tests

**Files:**
- Modify: `engine-search/src/lib.rs:1200-1300,1580-1860`

**Interfaces:**
- Produces: `TaperedScore::new(middlegame, endgame)`, `TaperedScore::equal(value)`, and `TaperedScore::interpolate(phase)`.

- [ ] **Step 1: Add the missing-type test**

```rust
use super::{TaperedScore, /* existing imports */};

#[test]
fn tapered_scores_interpolate_between_phase_endpoints() {
    let score = TaperedScore::new(120, 40);
    assert_eq!(score.interpolate(0), 120);
    assert_eq!(score.interpolate(24), 40);
    assert_eq!(score.interpolate(12), 80);
    assert_eq!(TaperedScore::equal(17).interpolate(9), 17);
}
```

- [ ] **Step 2: Push the test-only branch and verify GitHub red**

Expected: `Rusty Fish Tests` fails with unresolved import `TaperedScore`.

- [ ] **Step 3: Commit the red test**

```powershell
git add engine-search/src/lib.rs
git commit -m "test: specify tapered evaluation interpolation"
```

### Task 2: Convert the evaluator to score pairs

**Files:**
- Modify: `engine-search/src/lib.rs:1169-1320`

**Interfaces:**
- Consumes: `endgame_phase(board) -> i32` in the inclusive range 0..24.
- Produces: `evaluate_position(board) -> i32` with unchanged side-to-move sign.

- [ ] **Step 1: Add the score and parameter types**

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TaperedScore { middlegame: i32, endgame: i32 }

impl TaperedScore {
    const fn new(middlegame: i32, endgame: i32) -> Self { Self { middlegame, endgame } }
    const fn equal(value: i32) -> Self { Self::new(value, value) }
    fn interpolate(self, phase: i32) -> i32 {
        (self.middlegame * (24 - phase) + self.endgame * phase) / 24
    }
}
```

- [ ] **Step 2: Make `EvalFeatures` accumulate score pairs**

```rust
struct EvalFeatures { white_score: TaperedScore, black_score: TaperedScore }

fn net(self, side_to_move: Color, phase: i32) -> i32 {
    let score = (self.white_score - self.black_score).interpolate(phase);
    if side_to_move == Color::White { score } else { -score }
}
```

Implement `Add`, `Sub`, and `AddAssign` for `TaperedScore` or perform both
component additions explicitly; use one approach consistently.

- [ ] **Step 3: Parameterize evaluation-only material and piece-square scores**

```rust
fn tapered_piece_value(piece: Piece) -> TaperedScore {
    match piece.kind {
        PieceKind::Pawn => TaperedScore::new(100, 120),
        PieceKind::Knight => TaperedScore::new(320, 300),
        PieceKind::Bishop => TaperedScore::new(330, 340),
        PieceKind::Rook => TaperedScore::new(500, 520),
        PieceKind::Queen => TaperedScore::new(900, 900),
        PieceKind::King => TaperedScore::equal(0),
    }
}
```

Return a pair from the piece-square helper: pawn advancement and knight
centralization are middlegame-weighted; king centrality is endgame-weighted.

- [ ] **Step 4: Assign existing positional features to explicit phases**

Wrap bishop pair, pawn structure, rook files, activity, and threats with
`TaperedScore::equal`. Wrap `king_safety_bonus` with
`TaperedScore::new(value, 0)` and `king_endgame_activity` with
`TaperedScore::new(0, value)`.

- [ ] **Step 5: Finish at the single interpolation boundary**

```rust
features.net(board.side_to_move, endgame_phase)
```

- [ ] **Step 6: Push and verify GitHub green**

Require green workspace tests, tactical suite, fixed-opponent gauntlet,
throughput benchmark, and CodeQL analysis.

- [ ] **Step 7: Commit implementation**

```powershell
git add engine-search/src/lib.rs
git commit -m "feat: add tapered evaluation components"
```

### Task 3: Merge and track

**Files:**
- Modify: `D:/Work-Tracking/work-tracker-personal.md:333`

- [ ] **Step 1: Create the PR with GitHub red/green evidence**
- [ ] **Step 2: Squash merge only after fresh PR checks are green**
- [ ] **Step 3: Update the evaluation tracker row with the tapered foundation and remaining model work**

## Self-review

- The plan preserves evaluator API and scalar SEE/move-order material.
- It has a concrete red condition and one final interpolation point.
- It intentionally does not claim NNUE, pawn cache, or trained tuning complete.
