# Task 2 report: bounded Stockfish labeler and calibration

## Commit

- Pending at report creation: `feat: add bounded Stockfish NNUE labels`

## Files

- `engine-bench/src/stockfish.rs` — bounded UCI labeler, binary digest verification, score parsing, calibration, and fake-UCI tests.
- `engine-bench/src/lib.rs` — exports the Stockfish module.
- `engine-bench/src/main.rs` — `stockfish-calibrate` and `stockfish-label` commands plus immutable config I/O and manifest/split verification.

## TDD evidence

1. Added parser and calibration tests before implementations.
2. Ran `cargo test -p engine-bench stockfish::tests`; it failed as expected with unresolved `choose_budget`, `parse_info_score`, and `MATE_LABEL_CP` imports.
3. Implemented the smallest parser/calibration surface and reran: 2 tests passed.
4. Added fake transport tests before the helper existed.
5. Reran the focused suite; it failed as expected because `evaluate_one_with_transport` was absent.
6. Added the test-only fake transport helper; the focused suite then passed with 4 tests.

## Commands and results

- `cargo test -p engine-bench stockfish::tests` — passed: 4 unit tests.
- `cargo test -p engine-bench` — passed: 32 tests total (30 library, 2 binary tests).
- `git diff --check` — passed.
- `rustfmt engine-bench/src/stockfish.rs engine-bench/src/main.rs engine-bench/src/lib.rs` — could not format `main.rs` because an existing Rust-2024 let-chain elsewhere in that file is incompatible with the installed formatter edition. It did not modify unrelated files; compilation/tests passed afterward.

## Self-review

- The binary SHA-256 is checked before every process launch; no internal-evaluation fallback exists.
- The process is initialized once for a label invocation with UCI, one thread, configured hash, and readiness confirmation; one CLI label invocation processes one verified dataset shard.
- Each label enforces a timeout, parseable score, non-`0000` best move, reported node count at least the requested budget, and detects a non-success exit observed before returning a label.
- Mate labels use the documented signed `MATE_LABEL_CP = 10000` clamp.
- Labels contain exactly `score_cp`, own/opp active-feature CSV, canonical FEN from the manifest shard, and reported nodes. Existing external-match code was not changed.
- Calibration compares standard candidate budgets against a 400k-node reference and chooses the first budget with p95 absolute error no greater than 20 cp.

## Concerns

- Live Stockfish was not available in this workspace, so process interaction is covered through portable fake-UCI tests rather than an executable integration test.
- The project has an existing `engine-search` unused-mut warning; this task introduces no new warnings in `engine-bench`.
