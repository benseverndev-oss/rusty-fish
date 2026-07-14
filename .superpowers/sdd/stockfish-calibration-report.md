# Stockfish calibration follow-up

## Result

`stockfish-calibrate` now consumes exactly the first 1,000 records of the
manifest's deterministic training split and errors when that split contains
fewer than 1,000 positions. The Modal entrypoint below creates the corpus for
the named run, calibrates inside the pinned Linux Stockfish 18 image, and
writes the resulting config to the requested local path:

```text
modal run modal/app.py::calibrate --run-id stockfish18-v1 --output stockfish-config.tsv
```

The config is bound to `REMOTE_STOCKFISH` using the SHA-256 computed in the
image. No credential values are accepted, logged, or written by this path.

## Red / green evidence

- RED (Rust): `cargo test -p engine-bench --bin engine-bench calibration_sample`
  failed with unresolved import `calibration_sample`, before implementation.
- GREEN (Rust): the same command passed: 2 passed, 0 failed.
- RED (Python): `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m pytest modal/test_train_nnue.py -q`
  failed with missing `_calibrate_remote_stockfish_config`; the initial test
  invocation first exposed a missing `json` import, which was corrected before
  the feature-level RED run.
- RED (entrypoint): the same test command then failed with missing
  `app.calibrate`.
- GREEN: final focused verification passed:
  - `cargo fmt --check -p engine-bench`
  - `cargo test -p engine-bench` — 41 passed, 0 failed
  - `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m pytest modal/test_train_nnue.py -q` — 15 passed, 1 pre-existing Modal local-volume warning
  - `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m modal run modal/app.py::calibrate --help` — exposes `--run-id`, `--smoke`, and `--output`

## Changed files

- `engine-bench/src/main.rs`
- `modal/app.py`
- `modal/test_train_nnue.py`
- `modal/README.md`

## Implementation commit

`1e78189aaed56571a447c6b1bb0baf65d4bc9d75` (`fix: calibrate Stockfish in Modal image`)

## Concerns

- A smoke corpus has only 400 training positions, so calibration intentionally
  fails with the new clear minimum-size error. Use the default non-smoke corpus
  for a usable config.
- The Modal calibration itself was not launched during this change; it would
  consume remote compute. The entrypoint's CLI shape and local behavior are
  verified, while the remote subprocess is covered with a focused unit test.

## Reproducibility remediation

The Rust/Stockfish Modal image now starts from the immutable official Debian
bookworm-slim reference
`debian@sha256:60eac759739651111db372c07be67863818726f754804b8707c90979bda511df`
via `modal.Image.from_registry(..., add_python="3.12")`, rather than a moving
`debian_slim()` tag. The existing package install, Stockfish archive digest
check, Rust build, and application behavior are unchanged.

- RED: `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m pytest modal/test_train_nnue.py -q`
  failed because `RUST_IMAGE_BASE` did not exist.
- GREEN: the same command passed: 16 passed, 1 existing Modal local-volume warning.
- CLI regression: `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m modal run modal/app.py::calibrate --help`
  succeeded and still exposes the calibration entrypoint options.
