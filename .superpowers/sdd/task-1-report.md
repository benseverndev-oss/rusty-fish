# Task 1 report: Deterministic corpus and manifest primitives

## Commit

- `723378697bdcaa447515a856bdc175681c8d5bbb feat: add deterministic NNUE dataset manifests`

## Files changed

- `engine-bench/src/dataset.rs` — deterministic FEN canonicalization, stable FNV-1a split assignment, canonical deduplication, stable TSV manifest read/write, and in-repository SHA-256.
- `engine-bench/src/lib.rs` — exports the dataset module.
- `engine-bench/src/main.rs` — adds `dataset-build` validation, deterministic shard construction, manifests, and terminal-position regression coverage.

## TDD evidence

1. Added the two specified dataset tests and exposed the module declaration; `cargo test -p engine-bench dataset::tests -- --nocapture` failed to compile because the required interfaces did not exist.
2. Added the minimal interfaces and implementation; the same dataset tests passed.
3. The requested smoke command initially failed with `terminal position`. The root cause was the existing random opening generator recording terminal boards after a game ended. Added `generated_dataset_records_exclude_terminal_positions`, watched it fail with the exact smoke input, then filtered and replenished candidate records deterministically. The regression test passed afterward.

## Verification

- `cargo test -p engine-bench generated_dataset_records_exclude_terminal_positions -- --nocapture` — passed (1 test).
- `cargo test -p engine-bench dataset::tests` — passed (2 dataset tests).
- `cargo run -p engine-bench -- dataset-build smoke artifacts/smoke 400 400 200 1 --smoke` — passed.
- Smoke manifest inspection verified exactly three `split_count` entries and at least four 64-character SHA-256 values. `Get-FileHash` independently matched each generated shard hash in `manifest.tsv`.
- `git diff --check -- engine-bench/src/dataset.rs engine-bench/src/lib.rs engine-bench/src/main.rs` — passed before commit.

## Self-review

- Canonical FENs are sorted bytewise through `BTreeMap` before shard emission, and all three split files are emitted in fixed train/validation/test order.
- Deduplication selects a lexical-minimum source for the same canonical FEN, making source attribution independent of input ordering.
- The manifest format is explicit versioned, line-oriented TSV and rejects tabs/newlines in serialized fields.
- SHA-256 is implemented locally and the smoke output was independently checked with the platform SHA-256 tool.
- The CLI rejects non-production counts unless `--smoke` is given, and smoke totals over 1,000 are rejected.

## Concerns

- Existing unrelated `engine-search` warnings (`unused_mut` at `engine-search/src/lib.rs:786`) appear in all cargo verification output.
- The verification command generates untracked `artifacts/smoke/` files. They are intentional runtime output and were not included in the Task 1 commit.
- `Cargo.lock` and `.superpowers/sdd/progress.md` were already/unrelated working-tree changes and were not staged.

## Review-fix follow-up

- Added legal-position validation for king cardinality, adjacent kings, back-rank pawns, and an already-in-check prior mover.
- Added manifest immutability, SHA-256 syntax validation, and duplicate singleton/map-field rejection.
- Corpus construction now keeps a cross-source canonical-FEN set while replenishing each requested source, then asserts the post-dedup source and total counts before writing shards.
- Focused verification: `cargo test -p engine-bench dataset::tests` (5 passed) and the smoke `dataset-build` command (passed).
