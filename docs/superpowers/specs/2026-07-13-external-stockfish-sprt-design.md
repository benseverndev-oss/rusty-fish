# External Stockfish SPRT Campaign Design

## Goal

Provide a reproducible, external fixed-opponent campaign against a pinned
official Stockfish release, producing both per-game and sequential-test TSV
reports in GitHub Actions.

## Contract

`engine-bench external-sprt` will require an executable path in
`RUSTY_FISH_EXTERNAL_UCI`. It will start one UCI subprocess per game, perform
the `uci`/`isready` handshake, send a full FEN before each opponent turn, and
search the opponent at a fixed 100 ms `movetime`. Rusty Fish searches at its
existing deterministic depth limit. The benchmark plays every supplied FEN
twice, swapping Rusty Fish between White and Black, and reports the result
from Rusty Fish's perspective.

The first campaign corpus contains sixteen legal, fixed, middlegame FENs;
therefore the run always consists of thirty-two color-balanced games. Each
game is capped at 160 plies. The command emits a per-game TSV to stderr and
the existing SPRT summary TSV to stdout. This preserves a machine-readable
result record while allowing the workflow to capture each report separately.

The workflow downloads Stockfish 18's `stockfish-ubuntu-x86-64.tar` release
asset from the official Stockfish GitHub release. It verifies the published
SHA-256 `5c6f38b02a4da5f3ffe763f27da6c3e743eebefd92b50cb3661623b96696adff`
before extracting the executable. It runs only on `workflow_dispatch`, so an
intentional long campaign is never silently added to ordinary pull-request CI.

## Safety and reproducibility

- The benchmark refuses a missing or non-executable opponent path.
- A bounded UCI response timeout makes a hung opponent fail the campaign with
  a diagnostic instead of blocking the runner indefinitely.
- Each command is flushed immediately and unexpected EOF, malformed
  `bestmove`, and illegal opponent moves are reported as errors.
- Game state remains Rusty Fish's `Board`; the external process receives FEN
  snapshots and cannot desynchronize the authoritative move list.
- Artifact names, release URL, checksum, game corpus, side swapping, movetime,
  and SPRT settings are all fixed in repository-controlled files.

## Verification

Remote GitHub Actions validation covers the workspace after test-first UCI
adapter tests, followed by a manually dispatched checksum-verified campaign.
The run uploads both the raw game TSV and the SPRT TSV as artifacts.
