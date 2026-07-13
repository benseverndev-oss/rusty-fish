# Syzygy Root Probing Design

## Goal

Expose configured Syzygy tablebases through UCI and return an exact DTZ root
move when a loaded tablebase covers the root position, while safely falling
back to ordinary search when tablebases are absent or cannot probe a position.

## Design

`SyzygyTablebases` will convert Pyrrhic's `probe_root` result into a Rusty Fish
legal `ChessMove`, WDL category, and DTZ value. `Searcher` probes that result
before book/search work and returns the DTZ move with the existing tablebase
score convention. Existing interior WDL probing remains unchanged.

The UCI `EngineState` owns an optional configured tablebase *path* set with
`setoption name SyzygyPath value <path>`. A `go` worker loads that path and is
the sole owner of the Pyrrhic handle for its entire search. This is required:
Pyrrhic rejects `probe_root` while a `TableBases` handle is cloned, so keeping
one in UCI state would make every root probe fail. An empty path disables
tablebases. Invalid path configuration reports an `info string` and retains
the previous valid path; a load/probe failure in a worker simply uses ordinary
search.

## Constraints

- `probe_root` is serialized by current single-worker UCI search; before a
  replacement search or configuration command, UCI signals and joins its old
  worker so a new worker can initialize Pyrrhic safely. Later multithreading
  must coordinate root probes because Pyrrhic documents them as non-thread-safe.
- Missing DTZ files, uncovered positions, malformed paths, and probe errors
  return `None` and preserve normal search.
- The engine does not commit Syzygy files. A later asset workflow will download
  a minimal, checksummed corpus in GitHub Actions for exact WDL/DTZ validation.
