# Opening Book Corpus Refresh Design

## Goal

Regenerate the committed opening book from the pinned CC0 Lichess export
through a manual, reviewable workflow, bounded to a size the repository can
carry. The book currently committed is a 3-position synthetic fixture; this
replaces it with real data while keeping ordinary PR CI free of any multi-
hundred-megabyte download.

## Source and provenance

`assets/opening-book/manifest.toml` already pins the CC0 source: Lichess
`lichess_db_standard_rated_2014-12.pgn.zst` (259,216,467 bytes, SHA-256
`4589a1af622a893d196bc8eaede657652ce65dc79d2f289ff65fadd6a7076af4`), standard
rated games, both players rated at least 2200, first 16 plies, minimum three
observations per move. The manifest gains `max_positions` and folds the bound
into its selection rules so the committed book's provenance stays complete.

## Observation counting and the min-three filter

The generator aggregates mover-relative result points (3 win, 2 draw, 1 loss)
per move and currently excludes alternatives whose summed points fall below
three. That is not the documented rule: a move seen exactly once in a won game
scores three points and survives as though it had three observations.

The generator therefore tracks `observations` alongside `weight`. `weight`
keeps the existing integral 3/2/1 mover-relative sum and continues to order
alternatives; the minimum-three filter applies to `observations`, matching
`2026-07-13-licensed-opening-book-design.md`. This changes the committed
fixture book, whose regenerated assets are part of the change.

## Bounding the committed book

The pinned export yields roughly a million games; the rating filter admits a
small fraction, but the result is still far larger than the current fixture.
The generator gains `--max-positions N`, which keeps the N most-observed
positions and then re-sorts by FEN, so output stays deterministic and the
byte-identical checks stay meaningful. Ties in observation count are broken by
FEN so the retained set is stable. The default is unlimited, so the flag alone
does not alter fixture output. The refresh uses `N = 5000`.

## Streaming

`generate` accepts `-` as the input path, meaning read the PGN from stdin.
`build_book` takes an `io::Read` instead of a `&str`; the `&str` form is
retained for unit tests. `pgn-reader`'s `Reader` is already generic over
`io::Read`, so memory stays constant regardless of export size and the ~2 GB
decompressed export is never materialized. No new Rust dependency is added;
the runner's `zstd` performs decompression.

## Refresh workflow

`.github/workflows/opening-book-refresh.yml` is dispatch-only and never runs
on push or pull request. It downloads the manifest's pinned URL with retries,
verifies the pinned SHA-256 and fails loudly on mismatch, then runs
`zstdcat export.zst | book-tool generate - book.txt metrics.tsv
--max-positions 5000` with `pipefail` so a decode failure fails the job. It
opens a pull request containing the regenerated book, metrics, and manifest,
so every book change is reviewed and gated by the normal required checks. The
job needs `contents: write` and `pull-requests: write` and a timeout generous
enough to parse the full export.

## Ordinary CI is unchanged

`.github/workflows/opening-book.yml` continues to regenerate only the
committed synthetic fixture and assert it is byte-identical, honoring the
constraint that ordinary PR CI never downloads the full database. The
production book is byte-verified inside the refresh workflow, the only place
that has the export.

## Verification

- Unit tests cover reader-based generation, `--max-positions` retention and
  its deterministic FEN tie-break, and stdin selection via `-`.
- A regression pins that a single decisive game no longer satisfies the
  minimum-three filter.
- The committed fixture book and metrics are regenerated and reviewed as part
  of this change; the opening-book workflow proves they are reproducible.
- All validation runs in GitHub Actions; Cargo is never run locally.

## Out of scope

Hit-rate metrics over a fixed opening-position suite. `metrics.tsv` emits no
hit-rate column and the measurement is only meaningful once a real corpus is
committed, so it follows as its own spec.
