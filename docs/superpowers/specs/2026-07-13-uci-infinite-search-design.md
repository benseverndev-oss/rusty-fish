# UCI Infinite Search Design

## Goal

Make `go infinite` conform to UCI by continuing iterative deepening until a
`stop`, replacement command, or process exit cancels it, and prove the behavior
through an end-to-end process test that runs only in GitHub Actions.

## Scope

`engine-search` will treat `SearchLimits::infinite` as an instruction to ignore
the configured maximum depth. It will retain the existing `u8` depth type and
use `u8::MAX` as the practical ceiling; normal depth, movetime, and clock
searches retain their current configured cap and time controls.

`engine-uci` will gain integration tests that spawn the built UCI executable,
exchange lines over stdin/stdout, and bound every wait. The central regression
sets `Max Depth` to one, starts `go infinite`, verifies it emits no `bestmove`
before `stop`, then requires exactly one valid `bestmove` promptly after stop.
A second case replaces an infinite search with a finite one and requires only
the replacement result, preventing stale output.

## Alternatives considered

1. Keep the depth cap for `go infinite`. This is simpler but violates UCI and
   makes stop testing weak because the search can finish independently.
2. Make only the UCI frontend override `Max Depth`. This duplicates search
   policy and leaves the public `SearchLimits::infinite` API inconsistent.
3. Recommended: honor `infinite` in `Searcher` and test through the UCI
   process. This gives one cancellation contract to every caller.

## Constraints and validation

- Do not run Cargo locally; all Rust validation is GitHub Actions only.
- Every child-process wait has a short timeout and cleanup sends `quit` before
  killing a hung child.
- The existing `Rusty Fish Tests` workflow must pass for the branch and PR.
- The protocol test must use the real binary, not a mocked command loop.
