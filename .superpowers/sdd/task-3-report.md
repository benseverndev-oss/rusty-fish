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

## Review-fix addendum

- Capacity candidates now receive the explicit shared `seed` argument before
  PyTorch initialization and minibatch shuffling. The seed participates in the
  artifact hash and is included in every report.
- Reports now include `train_wdl_loss` as well as validation and sealed-test
  losses. `quantization_max_error_cp` is the maximum test-sample prediction
  delta from a cloned model with exactly the RFNN-export rounding/clamping
  applied, including the output bias; it is no longer a parameter-rounding
  fraction.
- The entrypoint performs the offline predicates, runs a 12×16 / 384-game,
  explicit-100-ms screen only for eligible candidates, and then calls
  `promotes` to print the final decision.
- The corpus manifest address includes `run_id`, smoke mode, counts, and seed.
  The Modal label image installs pinned `stockfish=15.1-4`; the label stage
  verifies the supplied config's digest against `/usr/games/stockfish` before
  replacing the caller-local binary path.
- Added feature-dimension and sealed-test prediction-delta tests. Focused
  verification after this change: `uv run --with pytest --with torch python -m
  pytest modal -q` passed (3 tests).
