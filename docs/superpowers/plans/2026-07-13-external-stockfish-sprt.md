# External Stockfish SPRT Campaign Implementation Plan

**Goal:** Run a reproducible, 32-game color-balanced external Stockfish 18
campaign and publish raw-game and SPRT TSV results.

**Architecture:** Add a small, standard-library UCI subprocess adapter to
`engine-bench`. Reuse the engine-core board as the game authority and the
existing score/SPRT reporting. A dispatch-only GitHub workflow installs the
pinned official release after checksum verification.

**Tech Stack:** Rust 2024 standard library, existing engine-core/search,
GitHub Actions, official Stockfish 18 Linux release.

## Global Constraints

- Never run Cargo, benchmarks, or Rust processes locally; validate remotely.
- Do not change the existing self-play fixed-opponent gauntlet.
- Require an explicit external UCI path; never download or execute an
  unverified binary from benchmark code.

### Task 1: Test and implement the UCI game adapter

**Files:**
- Modify: `engine-bench/src/lib.rs`

- [ ] Add failing policy tests for external configuration validation and
  color-balanced game scheduling, then verify the remote workspace is red.
- [ ] Add `ExternalMatchConfig`, UCI command parsing/handshake/response
  handling, and an external-opponent game loop that uses full FEN positions.
- [ ] Emit a raw-game TSV that records the opponent identifier, movetime, and
  game limits alongside every existing game field.
- [ ] Verify the remote workspace is green.

### Task 2: Expose a fixed corpus and dispatch-only campaign

**Files:**
- Modify: `engine-bench/src/main.rs`
- Create: `.github/workflows/external-stockfish-sprt.yml`

- [ ] Add the 16-position corpus and `external-sprt` CLI mode. Its output
  contract is summary TSV on stdout and raw-game TSV on stderr.
- [ ] Add the pinned Stockfish 18 download, SHA-256 verification, extraction,
  run, and two artifact uploads to a `workflow_dispatch` workflow.
- [ ] Verify the workflow through a GitHub Actions dispatch and retain its
  artifact evidence.

### Task 3: Review and record

- [ ] Open and merge a PR with `benzsevern` after all required checks pass.
- [ ] Update the Rusty Fish tracker entries only with the actual campaign
  result and its run URL; do not mark the SPRT gate complete unless it reaches
  a valid decision without a regression.
