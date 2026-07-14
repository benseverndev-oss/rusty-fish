# Task 3 report: manifest-backed Modal corpus and v1 capacity ladder

## Delivered

- Replaced the legacy seed-shard Modal scaffold with explicit `build_corpus`,
  `label_manifest`, `train_net`, and `run_screen` stages.
- Corpus and derived outputs are immutable Modal Volume artifacts below
  `runs/<run-id>/<stage>-<input-sha256>/`; an existing artifact can only be
  reused when its bytes match.
- Labels carry the `v1` schema prefix and the trainer rejects mismatched schema
  and feature indices outside the declared input dimension before PyTorch use.
- Training uses `EmbeddingBag(input_dimension, hidden, mode="sum")`, retains WDL
  loss, evaluates validation/test splits, records a checksum and quantization
  error, and emits a `report.json` for every requested width.
- The local entrypoint runs the 128/256/512 ladder from the same manifest and
  calls the 12-shard, 16-opening deterministic screen only for offline-eligible
  candidates. Every `gate-file` invocation explicitly passes `100` ms.
- Updated Modal documentation and added the schema-loader pytest coverage.

## Verification

- `uv run --with pytest python -m pytest modal -q` — passed (1 test).
- `uv run python -m py_compile modal/app.py modal/train_nnue.py modal/test_train_nnue.py` — passed.
- `cargo test -p engine-bench` — passed (36 tests); one pre-existing
  `unused_mut` warning in `engine-search/src/lib.rs` remains.
- `git diff --check` — passed (only CRLF conversion warnings from the Windows
  worktree).

## Modal limitation

The prescribed `modal run modal/app.py --run-id smoke-v1 --smoke --schema v1
--widths 128,256,512` was not started. This environment has no `modal` CLI
installed or authenticated Modal credentials, and it also lacks the required
Stockfish configuration/binary and GPU. The limitation affects remote execution
only; local Python and Rust verification completed.

## Scope

Only Task 3 files are intended for the Task 3 commit: `modal/app.py`,
`modal/train_nnue.py`, `modal/README.md`, and `modal/test_train_nnue.py`, plus
this report. Existing Task 1/2 worktree modifications were left untouched.
