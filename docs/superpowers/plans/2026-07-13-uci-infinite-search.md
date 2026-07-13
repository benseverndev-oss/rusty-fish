# UCI Infinite Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make UCI `go infinite` run until cancellation and prove stop/replacement behavior with a real-process stress test.

**Architecture:** `Searcher` owns the interpretation of `SearchLimits::infinite`, so all callers share the same cancellation semantics. `engine-uci` integration tests spawn the compiled executable and use bounded stdout waits, exercising stdin, the async command loop, worker search, and output as a UCI client would.

**Tech Stack:** Rust 2024 workspace, `std::process`, `std::sync::mpsc`, GitHub Actions.

## Global Constraints

- Do not run Cargo locally; GitHub Actions is the sole Rust validation environment.
- No unbounded child-process or channel waits in tests.
- Tests exercise the real `engine-uci` binary via `CARGO_BIN_EXE_engine-uci`.

---

### Task 1: Reproduce infinite-search protocol behavior

**Files:**
- Create: `engine-uci/tests/protocol_stress.rs`

**Interfaces:**
- Consumes: executable path from `env!("CARGO_BIN_EXE_engine-uci")`.
- Produces: bounded helpers `UciProcess::send`, `UciProcess::expect_line`, and `UciProcess::expect_no_line`.

- [ ] **Step 1: Write the failing real-binary test**

```rust
#[test]
fn infinite_search_waits_for_stop_even_when_max_depth_is_one() {
    let mut uci = UciProcess::spawn();
    uci.send("uci");
    uci.expect_line("uciok", Duration::from_secs(1));
    uci.send("setoption name Max Depth value 1");
    uci.send("position startpos");
    uci.send("go infinite");
    uci.expect_no_line(Duration::from_millis(150));
    uci.send("stop");
    assert!(uci.expect_line_starting_with("bestmove ", Duration::from_secs(2)).is_some());
}
```

- [ ] **Step 2: Run it remotely to verify it fails**

Push the test-only commit and inspect the `Rusty Fish Tests` GitHub Actions log. Expected: the test receives `bestmove` before `stop` because `go infinite` currently honors `Max Depth`.

- [ ] **Step 3: Add stale-result replacement coverage**

```rust
#[test]
fn replacing_an_infinite_search_emits_only_the_replacement_bestmove() {
    let mut uci = UciProcess::spawn();
    uci.handshake();
    uci.send("position startpos");
    uci.send("go infinite");
    uci.send("position startpos moves e2e4");
    uci.send("go depth 1");
    uci.expect_line_starting_with("bestmove ", Duration::from_secs(2));
    uci.expect_no_line(Duration::from_millis(150));
}
```

- [ ] **Step 4: Commit the red test**

```powershell
git add engine-uci/tests/protocol_stress.rs
git commit -m "test: stress UCI infinite search control"
```

### Task 2: Honor `SearchLimits::infinite`

**Files:**
- Modify: `engine-search/src/lib.rs:535-540`

**Interfaces:**
- Consumes: `SearchLimits { infinite: bool, depth: Option<u8> }` and `SearchOptions::max_depth`.
- Produces: a maximum iterative-deepening depth of `u8::MAX` for infinite searches; unchanged caps for all other limits.

- [ ] **Step 1: Make the minimal search policy change**

```rust
let max_depth = if limits.infinite {
    u8::MAX
} else {
    limits
        .depth
        .unwrap_or(self.options.max_depth)
        .max(1)
        .min(self.options.max_depth)
};
```

- [ ] **Step 2: Push and verify the remote green run**

Push the implementation commit and require the `Rusty Fish Tests` GitHub Actions workspace job to pass. Expected: the protocol test observes no output before `stop`, then one `bestmove`.

- [ ] **Step 3: Commit the implementation**

```powershell
git add engine-search/src/lib.rs
git commit -m "fix: keep infinite UCI searches running until stop"
```

### Task 3: Review and merge

**Files:**
- Modify: `D:/Work-Tracking/work-tracker-personal.md:338`

- [ ] **Step 1: Create a pull request with remote validation links**

- [ ] **Step 2: Require green workspace and CodeQL checks, then squash merge**

- [ ] **Step 3: Update the engine-operations tracker entry to mark UCI stress testing complete**

## Self-review

- Scope covers the exact missing UCI-stress and infinite-search behavior, not multithreading or GUI work.
- All waits in the plan are bounded and every code change has a named test.
- The test's expected red condition is specific: premature `bestmove` with maximum depth one.
