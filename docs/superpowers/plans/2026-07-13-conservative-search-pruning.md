# Conservative Search Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add guarded razoring, reverse futility, and late-move pruning to reduce shallow quiet-node work without weakening tactical safety.

**Architecture:** Private pure helpers define conservative depth-scaled margins and quiet-move limits. `Searcher::negamax` uses those helpers only in non-checking, non-mate, non-pawn-endgame interior nodes; root and quiescence behavior remain unchanged.

**Tech Stack:** Rust 2024, existing Rusty Fish alpha-beta/PVS search, GitHub Actions.

## Global Constraints

- Never run Cargo, benchmarks, or Rust processes locally; validation runs only in GitHub Actions.
- Do not prune root, check, mate-adjacent, or pawn-only positions.
- Preserve PVS re-search, null move, LMR, SEE ordering, extensions, and tablebase fallbacks.
- Tactical-suite, fixed-opponent gauntlet, workspace tests, and CodeQL are required remote evidence.

---

### Task 1: Specify pruning policy helpers

**Files:**
- Modify: `engine-search/src/lib.rs`

**Interfaces:**
- Produces: `razor_margin(depth: u8) -> i32`.
- Produces: `reverse_futility_margin(depth: u8) -> i32`.
- Produces: `late_move_pruning_limit(depth: u8) -> usize`.

- [ ] **Step 1: Write failing helper tests**

```rust
#[test]
fn conservative_pruning_margins_increase_with_depth() {
    assert!(razor_margin(2) > razor_margin(1));
    assert!(reverse_futility_margin(3) > reverse_futility_margin(2));
    assert!(late_move_pruning_limit(3) > late_move_pruning_limit(2));
}
```

- [ ] **Step 2: Verify red remotely**

Push the test-only commit and confirm `Rusty Fish Tests / workspace` fails because the helpers are unresolved.

- [ ] **Step 3: Add minimal fixed policy**

```rust
fn razor_margin(depth: u8) -> i32 { 120 + 80 * i32::from(depth) }
fn reverse_futility_margin(depth: u8) -> i32 { 100 + 90 * i32::from(depth) }
fn late_move_pruning_limit(depth: u8) -> usize { 3 + usize::from(depth) * 2 }
```

- [ ] **Step 4: Verify green remotely and commit**

Push and require the workspace job to pass before committing:

```text
git add engine-search/src/lib.rs
git commit -m "feat: define conservative pruning policy"
```

### Task 2: Apply guarded interior pruning

**Files:**
- Modify: `engine-search/src/lib.rs:775-910`

**Interfaces:**
- Consumes: the policy helpers from Task 1.
- Produces: bounded shallow-node pruning that returns only alpha-beta bounds.

- [ ] **Step 1: Write a failing policy regression test**

```rust
#[test]
fn pruning_policy_excludes_check_and_mate_windows() {
    assert!(!can_apply_static_pruning(2, true, 0, 50, true));
    assert!(!can_apply_static_pruning(2, false, MATE_SCORE - 512, MATE_SCORE, true));
    assert!(can_apply_static_pruning(2, false, 0, 50, true));
}
```

- [ ] **Step 2: Verify red remotely**

Push the test-only commit; `Rusty Fish Tests / workspace` must fail because `can_apply_static_pruning` is missing.

- [ ] **Step 3: Implement guarded static and late-move pruning**

Add `can_apply_static_pruning(depth, in_check, alpha, beta, has_non_pawn_material)`. In `negamax`, after draw/tablebase checks and before null-move/move generation, use one static evaluation to:

```rust
if depth == 1 && static_eval + razor_margin(depth) <= alpha {
    return (self.quiescence(board, alpha, beta), Vec::new());
}
if depth <= 3 && static_eval - reverse_futility_margin(depth) >= beta {
    return (static_eval, Vec::new());
}
```

In the move loop, only after the first ordered move, break when the move index reaches `late_move_pruning_limit(depth)` and the move is quiet with no extension, TT move, killer, or counter-move priority.

- [ ] **Step 4: Verify green remotely and commit**

Require workspace, tactical-suite, fixed-opponent-gauntlet, and CodeQL success. Commit:

```text
git add engine-search/src/lib.rs
git commit -m "feat: prune guarded quiet search nodes"
```

### Task 3: Review and merge

**Files:**
- Modify: `D:/Work-Tracking/work-tracker-personal.md` after merge only.

- [ ] **Step 1: Open a PR and await all checks**

Describe the three heuristics, all safety exclusions, and the remote tactical/gauntlet gates.

- [ ] **Step 2: Merge with the `benzsevern` GitHub account**

Only merge if every required check passes.

- [ ] **Step 3: Record result and remaining search-quality work**

Update the tracker with PR/date, the delivered pruning, and remaining singular extensions plus the external established-engine SPRT campaign.

## Self-review

The plan covers all three requested pruning heuristics, makes every guard explicit, and requires focused unit tests plus the existing tactical, gauntlet, workspace, and CodeQL evidence before merge.
