# Task 4 — RFNN v2 schema and HalfKA feature extraction

## Scope

- Implemented only Task 4 files: `engine-search/src/nnue.rs`,
  `engine-search/src/lib.rs`, and `engine-bench/src/train.rs`.
- No Task 5 Modal, workflow, or documentation changes were made.

## TDD evidence

1. Added RFNN v2 round-trip / wrong-dimension tests before the v2 API existed.
2. Ran `cargo test -p engine-search nnue -- --nocapture` and observed the
   expected compile failures for missing `FeatureSchema` and
   `Nnue::from_seed_with_schema`.
3. Implemented the smallest schema-owned parser/writer and HalfKA feature path.
4. Added the search-level special-move regression covering castling, en
   passant, promotion, and ordinary capture. It exercises the existing
   make/unmake refresh-equivalence assertion before and after unmake.

## Implementation evidence

- RFNN v1 still writes precisely its former `RFNN`, version, hidden-width,
  weights/biases layout; its schema is implied by version 1.
- RFNN v2 writes version 2, HalfKA schema tag, bucket count, and explicit input
  dimension before the existing parameter payload. Loading rejects unsupported
  tags, zero bucket counts, dimensions inconsistent with the declared schema,
  arithmetic overflow, truncation, and trailing bytes.
- HalfKA dimensions are `buckets * 2 * 5 * 64`; the 64-bucket production
  schema is 40,960 inputs. Kings are anchors, not transformed features.
- HalfKA features are perspective-relative; a king move (including castling)
  refreshes the accumulator. Non-king changes retain square-delta incremental
  add/remove behavior, including captures, promotions, and en passant.
- `TrainingSample` now carries its schema and training rejects mixed-schema
  batches; quantized output is constructed with the sample schema.

## Verification

- `git diff --check -- engine-search/src/nnue.rs engine-search/src/lib.rs engine-bench/src/train.rs` — clean.
- `cargo test -p engine-search` — 46 passed.
- `cargo test -p engine-bench train::tests` — 4 passed.

## Self-review

- Confirmed v1 serializer branch adds no new header fields and v1 round-trip
  remains covered.
- Confirmed v2 header input dimension is checked before allocating its weight
  vector.
- Confirmed the permanent debug refresh equality assertion is exercised for
  all special move classes under HalfKA, including unmake.
- The pre-existing `unused_mut` warning in `engine-search/src/lib.rs:787` is
  unrelated to this task; the touched code introduces no warnings.
