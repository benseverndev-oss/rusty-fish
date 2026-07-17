# External Gauntlet Time-Parity Design

## Goal

Make the external Stockfish gauntlet a fair strength read. Today the candidate
(rusty-fish) searches a fixed depth 5 while Stockfish gets 100 ms per move, so
the resulting Elo (-338 on the first completed run) is a combined depth+eval
gap, not a measure of the engine's quality. Give the candidate the same time
budget as the opponent so the number reflects playing strength at equal time and
can be used to judge future evaluation work.

## Scope

Only the external match is affected. `ExternalMatchConfig` and `play_external_game`
drive the external Stockfish campaign; the SPSA tuner and self-play gauntlet use
a separate depth-based `MatchConfig` and are untouched.

## Changes

### `ExternalMatchConfig`

- Remove `candidate_depth: u8`.
- Add `candidate_movetime: Duration`, default 100 ms (parity with the existing
  `opponent_movetime`, also 100 ms).
- Add `candidate_move_overhead: Duration`, default 10 ms.

The default therefore pits both engines at 100 ms per move.

### `play_external_game`

The candidate searches under a time budget instead of a fixed depth:

- Configure the candidate `Searcher` with a small move overhead
  (`candidate_move_overhead`) via `set_options`, so it actually spends close to
  its full budget. The searcher's default `move_overhead` is 25 ms, which on a
  100 ms budget would let the candidate use only ~75 ms while Stockfish uses the
  full 100 ms; 10 ms restores real parity (this is an automated harness with no
  GUI latency to reserve for).
- Search with `SearchLimits { movetime: Some(config.candidate_movetime), .. }`
  instead of `depth: Some(config.candidate_depth)`.

The searcher already honors `SearchLimits.movetime` through its internal
deadline, so no engine-search change is needed.

### `external_tsv_report`

The per-game report's `candidate_depth` column becomes `candidate_movetime_ms`,
emitting `config.candidate_movetime.as_millis()`. This is the stderr games TSV;
the stdout SPRT report is unchanged.

### Workflow

`.github/workflows/external-stockfish-sprt.yml` raises `timeout-minutes` from 15
to 30. Both sides now spend real time per move, so a fixed-depth-5 assumption of
near-instant candidate moves no longer holds; long games (up to the 160-ply cap)
plus build and download need headroom.

## Determinism

A movetime search is machine-dependent, so the campaign's result varies run to
run. This is inherent to time parity and acceptable: the workflow is
dispatch-only and report-only, and SPRT is designed to absorb per-game variance.
No output is byte-asserted.

## Verification

- A unit test pins `ExternalMatchConfig::default()`: `candidate_movetime` is
  100 ms and `candidate_move_overhead` is 10 ms.
- A unit test pins that `external_tsv_report` emits a `candidate_movetime_ms`
  header column and no `candidate_depth` column.
- The end-to-end gauntlet is validated by re-dispatching the campaign (it needs
  a real Stockfish binary, so it cannot run in ordinary CI); a completed run
  reporting a fresh Elo is the acceptance signal.
- All in-repo validation runs in GitHub Actions; Cargo is never run locally.

## Out of scope

- The SPRT hypothesis (`elo0 = 0`, `elo1 = 5`) stays as-is; it is an A/B
  regression gate, and making it an absolute-strength test is a separate change.
- Draw adjudication of games that hit the 160-ply cap.
- Multiple time controls or opening-suite sweeps.
- Any change to `MatchConfig`, the SPSA campaign, or the self-play gauntlet.
