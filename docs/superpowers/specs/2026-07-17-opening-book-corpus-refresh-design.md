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

## File layout

The committed fixture outputs and the committed production book are distinct
files. Ordinary CI regenerates and byte-compares only the fixture pair; the
production pair is only ever written by the refresh workflow.

| Path | Produced by | Verified by |
|------|-------------|-------------|
| `assets/opening-book/lichess-cc0-fixture.pgn` | committed by hand | - |
| `assets/opening-book/fixture-book-v2.txt` | fixture generation | `opening-book.yml`, byte-identical |
| `assets/opening-book/fixture-metrics.tsv` | fixture generation | `opening-book.yml`, byte-identical |
| `assets/opening-book/rusty-fish-book-v2.txt` | refresh workflow | refresh workflow only |
| `assets/opening-book/metrics.tsv` | refresh workflow | refresh workflow only |

The existing committed `rusty-fish-book-v2.txt` and `metrics.tsv` are fixture
output and are renamed to `fixture-book-v2.txt` and `fixture-metrics.tsv`;
`opening-book.yml` is repointed at those names. The production names hold no
committed file until the first refresh PR lands. Nothing depends on a book
existing: `BookPath` is opt-in and has no default, so the engine's behavior is
unchanged until a book is configured.

Without this split, the refresh PR would overwrite the file the fixture check
diffs against, and `opening-book.yml` would compare 3-position fixture output
against a 5000-position production book and fail permanently.

## Observation counting and the min-three filter

The generator aggregates mover-relative result points (3 win, 2 draw, 1 loss)
per move and currently excludes alternatives whose summed points fall below
three. That is not the documented rule: a move seen exactly once in a won game
scores three points and survives as though it had three observations.

The generator therefore tracks `observations` alongside `weight`. `weight`
keeps the existing integral 3/2/1 mover-relative sum and continues to order
alternatives; the minimum-three filter applies to `observations`, matching
`2026-07-13-licensed-opening-book-design.md`. `observations` is generator-
internal: the v2 record format stays `<fen>\t<uci>:<weight> ...` and is
unchanged, so no loader or format version work follows from this.

Every move in the current fixture has exactly three observations, so the
filter change alone leaves the fixture book byte-identical and would be
proven by unit test only. To make the rule observable in the committed,
CI-verified artifacts, the fixture gains one single-occurrence decisive game.
Under today's code that game contributes a move at three points and it enters
the book; under the corrected rule its moves have one observation and are
excluded. The committed `fixture-book-v2.txt` therefore stays as it is today
and that invariance is the regression evidence, while `fixture-metrics.tsv`
records the higher source and accepted game counts.

Note for readers of the earlier spec: `2026-07-13-licensed-opening-book-design.md`
describes the weight as `frequency * (0.5 + score_fraction)`. The generator
doubles that to stay integral, which is the same ordering; the 3/2/1 sum
described here is that same quantity and is not a change.

## Bounding the committed book

The pinned export yields roughly a million games; the rating filter admits a
small fraction, but the result is still far larger than the current fixture.
The generator gains `--max-positions N`, which keeps the N most-observed
positions and then re-sorts by FEN, so output stays deterministic and the
byte-identical checks stay meaningful.

A position's observation count is the sum of the observations of the
alternatives it retains after the minimum-three filter, so the bound keeps the
positions the engine most often reaches. Ties in that count are broken by
ascending FEN, so the retained set is stable across runs. The default is
unlimited, so the flag alone does not alter fixture output. The refresh uses
`N = 5000`.

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
`zstdcat export.zst | book-tool generate - assets/opening-book/rusty-fish-book-v2.txt
assets/opening-book/metrics.tsv --max-positions 5000` with `pipefail` so a
decode failure fails the job. It opens a pull request containing the
regenerated production book, its metrics, and the manifest, so every book
change is reviewed and gated by the normal required checks. That PR does not
touch the fixture pair, so the fixture byte-identity check passes on it
unchanged. The job needs `contents: write` and `pull-requests: write` and a
timeout generous enough to parse the full export.

## Ordinary CI is unchanged

`.github/workflows/opening-book.yml` continues to regenerate only the
committed synthetic fixture and assert it is byte-identical, honoring the
constraint that ordinary PR CI never downloads the full database. It is
repointed at `fixture-book-v2.txt` and `fixture-metrics.tsv` and never reads
or verifies the production pair, which no runner can reproduce without the
export. Its existing `assets/opening-book/**` triggers are unchanged, so a
refresh PR still runs it; the check passes because that PR leaves the fixture
pair untouched.

## Verification

- Unit tests cover reader-based generation, `--max-positions` retention and
  its deterministic FEN tie-break, and stdin selection via `-`.
- A regression pins that a single decisive game no longer satisfies the
  minimum-three filter.
- The fixture gains one single-occurrence decisive game. `fixture-book-v2.txt`
  is unchanged by it, which is the committed evidence for the corrected
  filter; `fixture-metrics.tsv` is regenerated with the higher game counts.
  The opening-book workflow proves both remain reproducible.
- All validation runs in GitHub Actions; Cargo is never run locally.

## Out of scope

Hit-rate metrics over a fixed opening-position suite. `metrics.tsv` emits no
hit-rate column and the measurement is only meaningful once a real corpus is
committed, so it follows as its own spec.
