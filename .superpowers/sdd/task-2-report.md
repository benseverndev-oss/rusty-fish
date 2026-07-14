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

## Review-fix addendum

### Root cause

- Calibration compared every lower budget to a 400k reference, included an unrequested 250k candidate, and its selector returned the final candidate when no candidate met the tolerance.
- The public parser represented both an absent `score` field and malformed score values as `None`; the UCI loop treated both as ignorable.
- The CLI feature extraction parsed the label FEN but wrote the original text instead of the Task 1 canonical form.

### Corrections

- Calibration now evaluates exactly 25,000, 100,000, and 400,000 nodes. It compares 25k to 100k, then 100k to 400k; it selects the first P95 error at or below 20 cp and otherwise returns 400k. A timeout/error in any run propagates, so no budget is selected after a timed-out evaluation.
- Added an internal checked score parser. Absent scores remain ignorable info lines, while malformed `score cp`/`score mate` records produce an immediate labeling error, including when followed by a valid score.
- `stockfish-label` now calls Task 1 `canonical_fen` before deriving features and writing each TSV row, rejecting invalid FENs and emitting the canonical representation.

### Review-fix TDD and verification

1. Changed the no-qualifying-budget expectation to 400k; focused test failed with `left: None`, `right: Some(400000)` before the calibration implementation changed.
2. Added regression coverage for malformed score then valid score, and canonicalization/validation of label FENs. Before the implementation, the focused test build failed because the checked parser and canonical label helper were absent; after implementation, all passed.
3. `cargo test -p engine-bench stockfish::tests` — passed: 6 tests.
4. `cargo test -p engine-bench --bin engine-bench label_fens_are_canonicalized_and_validated` — passed: 1 test.
5. `cargo test -p engine-bench` — passed: 35 tests total (32 library, 3 binary tests).
6. `git diff --check` — passed.

## Resolved calibration-reference policy

The committed plan/spec at `597c209` resolves the prior conflict: 400k is the mandatory calibration reference. Task 2 now evaluates exactly 25k, 100k, and 400k nodes, calculates P95 absolute score deltas for 25k-vs-400k and 100k-vs-400k, then selects 25k first, 100k second, and 400k as the fallback. Any labeling timeout/error still propagates and prevents a selection.

TDD evidence: added `calibration_compares_each_lower_budget_to_the_400k_reference` before `calibration_candidates` existed; `cargo test -p engine-bench stockfish::tests` failed with the expected unresolved-import error. After extracting the reference-based candidate calculation, focused tests passed (7 tests), `cargo test -p engine-bench` passed (36 tests), and `git diff --check` passed.
