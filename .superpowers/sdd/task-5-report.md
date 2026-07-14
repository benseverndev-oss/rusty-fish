# Task 5 — HalfKA GPU training and full promotion gate

## Implemented

- Added schema-aware PyTorch construction/training and an RFNN v2 exporter.
  HalfKA exports version 2, schema tag 1, 64 buckets, explicit dimension
  40,960, then the existing parameter payload; v1 keeps its old layout.
- Added the deterministic HalfKA width mapping and stop-on-first-failed-
  promotion behavior. HalfKA candidates use the same-manifest v1-128 control.
- Added a Rust one-opening load check before screening and an eligible-only full
  gate of 12 × 96 openings, depth 4, `gate-file ... 100`, asserting 2,304
  games.
- Converted the GitHub workflow into a verified promoted-network gate: it takes
  a network URL/checksum and explicit manifest checksum, then prints aggregate
  W/D/L and the SPRT output.
- Updated Modal and handoff documentation. No campaign was launched and no
  adoption was proposed.

## TDD evidence

1. Added the RFNN v2 export-header test and ran pytest: it failed because
   `tiny_model` did not exist.
2. Implemented the model/exporter and reran pytest successfully.
3. Added a one-epoch schema-aware training regression; it failed with the
   missing minibatch-tensor initialization, then passed after the minimal fix.

## Verification

- `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m pytest modal -q` — 5 passed.
- `cargo test --workspace` — passed (with the pre-existing `unused_mut` warning
  in `engine-search/src/lib.rs`).
- `git diff --check` — clean.

## Limitation

Modal credentials/GPU capacity and campaign artifacts were not available in
this environment, so neither the smoke Modal run nor a 2,304-game promotion
campaign was launched. The code only dispatches a full gate after an eligible
candidate passes the offline and 384-game screen criteria.

## Review follow-up (P1 fixes)

- HalfKA now executes each selected width as one ordered train → parity →
  screen → full-gate decision. The next width cannot start unless the preceding
  width reaches `AcceptH1`; offline, screen, `AcceptH0`, and `Continue` failures
  stop the ladder.
- Both the 384-game screen and 2,304-game full gate now persist immutable,
  input-addressed outcome `report.json` artifacts. They record run ID; network,
  candidate/control-report, manifest, and config checksums; W/D/L; Elo; raw and
  parsed SPRT output; and the non-adopting/adoption-design decision.
- Added focused Python tests for the stop-before-next-training behavior and
  immutable gate evidence behavior. No Modal run or campaign was launched.

### Follow-up verification

- `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m pytest modal -q` — 8 passed.
- `cargo test --workspace` — passed (only the existing `unused_mut` warning in
  `engine-search/src/lib.rs`).
- `git diff --check` — clean.
