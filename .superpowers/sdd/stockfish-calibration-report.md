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

## Direct remote calibration remediation

The Windows local-entrypoint invocation created Modal objects but did not invoke
any function after `Created objects`, so no calibration container appeared in
the app logs. `calibrate_run` is now a direct `@app.function` that builds (or
reuses the immutable corpus artifacts) and calibrates within one remote
execution; it does not rely on a local entrypoint dispatching nested remote
calls.

Use the Modal CLI's global result writer (before the function reference):

```text
modal run --write-result stockfish-config.tsv modal/app.py::calibrate_run --run-id stockfish18-v1
```

- RED: the new direct-function regression failed because
  `_build_corpus_artifacts` did not exist.
- GREEN: `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m pytest modal/test_train_nnue.py -q`
  passed: 18 passed, 2 expected Modal local-execution warnings.
- CLI verification: `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m modal run --help`
  confirms the global `--write-result` option; function help for
  `modal/app.py::calibrate_run` confirms `--run-id` and `--smoke`.

## Early Stockfish completion remediation

Real Stockfish accepted `go nodes 25000` and returned a valid score and
`bestmove` after 6,624 nodes, which is legitimate when search finishes early.
The labeler incorrectly rejected that response for being below the requested
limit. The requested `go nodes` commands (25k/100k/400k during calibration)
are unchanged; the recorded label now retains Stockfish's actual reported node
count without treating an early valid completion as an error.

- RED: `cargo test -p engine-bench stockfish::tests::fake_transport_accepts_valid_early_completion_and_records_reported_nodes`
  failed because the shared reported-node parser and label finalizer did not exist.
- GREEN: the same test passed, accepting a score/bestmove at 6,624 nodes after
  `go nodes 25000` and recording `nodes == 6624`.
- Focused/full verification: `cargo fmt --check -p engine-bench`; `cargo test -p engine-bench`
  (42 passed); and `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m pytest modal/test_train_nnue.py -q`
  (18 passed, 2 existing local-execution warnings).

## Modal image-path remediation

Real Modal image construction exposed a path collision: the Stockfish archive
extracts `/opt/stockfish/stockfish` as a directory, so linking the executable
to that same path caused `ln` to resolve it as a directory and fail while
creating an already-existing nested name. This was an image-layout issue, not
credentials or calibration.

`REMOTE_STOCKFISH` and the installation command now use the non-colliding
stable path `/opt/stockfish/stockfish-bin`. All calibration and labeling code
continues to read the binary through that one constant.

- RED: `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m pytest modal/test_train_nnue.py -q`
  failed because the configured executable path was `/opt/stockfish/stockfish`
  instead of `/opt/stockfish/stockfish-bin`.
- GREEN: the same command passed: 17 passed, 1 existing Modal local-volume warning.
- CLI regression: `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe -m modal run modal/app.py::calibrate --help`
  succeeded and still exposes the calibration entrypoint options.
