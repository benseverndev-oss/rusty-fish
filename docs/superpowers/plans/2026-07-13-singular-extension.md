# Singular Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend only deeply verified, TT-backed singular moves by one ply.

**Architecture:** Add an optional excluded move to interior negamax calls. A private eligibility helper protects the verification search; its failure to find an alternative grants the ordinary TT move one additional ply.

**Tech Stack:** Rust 2024, existing Rusty Fish alpha-beta/PVS search, GitHub Actions.

## Global Constraints

- Never run Cargo, benchmarks, or Rust processes locally; validate in GitHub Actions.
- Do not apply singular extensions at root, quiescence, checks, pawn-only positions, shallow nodes, or mate-adjacent scores.
- Preserve existing PVS, null-move, LMR, pruning, extensions, and tablebase behavior.

---

### Task 1: Specify singular-candidate safety

**Files:**
- Modify: `engine-search/src/lib.rs`

- [ ] **Step 1: Write failing eligibility tests**

```rust
#[test]
fn singular_extension_requires_a_deep_exact_non_mate_tt_entry() {
    let entry = TranspositionEntry { depth: 8, score: 40, bound: Bound::Exact, best_move: Some(ChessMove::from_uci("e2e4").unwrap()) };
    assert!(can_try_singular_extension(6, false, true, entry));
    assert!(!can_try_singular_extension(5, false, true, entry));
    assert!(!can_try_singular_extension(6, true, true, entry));
}
```

- [ ] **Step 2: Verify red remotely**

Push the test-only commit and require `Rusty Fish Tests / workspace` to fail because the helper is unresolved.

- [ ] **Step 3: Implement helper and excluded-move search plumbing**

Add `can_try_singular_extension(depth, in_check, has_non_pawn_material, entry) -> bool` and an `excluded_move: Option<ChessMove>` parameter to interior negamax. Skip an excluded move in its ordered loop and avoid TT exact storage for such a search.

- [ ] **Step 4: Verify green remotely and commit**

```text
git add engine-search/src/lib.rs
git commit -m "feat: support singular verification searches"
```

### Task 2: Apply the one-ply singular extension

**Files:**
- Modify: `engine-search/src/lib.rs`

- [ ] **Step 1: Write failing policy test**

```rust
#[test]
fn singular_extension_uses_a_fixed_verification_margin() {
    assert_eq!(singular_verification_beta(80), 48);
}
```

- [ ] **Step 2: Verify red remotely**

Push and confirm the workspace job fails because the margin helper is unresolved.

- [ ] **Step 3: Add verification search before the TT move**

For an eligible exact TT entry, search at `depth / 2` with the TT move excluded and window `[-beta, -beta + 1]`, where `beta = entry.score - 32`. If it fails below that bound, add one ply only to the matching TT move's normal search.

- [ ] **Step 4: Verify all remote gates and commit**

Require workspace, tactical-suite, fixed-opponent gauntlet, throughput, and CodeQL success.

```text
git add engine-search/src/lib.rs
git commit -m "feat: extend verified singular moves"
```

### Task 3: Review and merge

- [ ] Open a PR, wait for all required checks, merge with `benzsevern`, and update `D:/Work-Tracking/work-tracker-personal.md` only after merge.
