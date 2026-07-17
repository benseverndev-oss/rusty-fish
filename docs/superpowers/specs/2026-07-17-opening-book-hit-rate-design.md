# Opening Book Hit-Rate Metric Design

## Goal

Quantify how well the committed opening book covers the openings people
actually play, as a reportable metric. Replay a fixed suite of canonical
opening lines against a book and measure, per position reached, whether the
book has an entry. This makes book coverage visible and catches regressions: a
future corpus refresh or filter change that guts coverage of mainlines shows up
as a dropped number.

## Why a separate metric, not a column in `metrics.tsv`

`assets/opening-book/metrics.tsv` (and the fixture's `fixture-metrics.tsv`) is
the generator's byte-reproduced description of the book it built from the
*training* corpus; CI asserts it byte-for-byte. Hit-rate is a different
measurement entirely: coverage of an *independent* curated suite. Folding it
into `metrics.tsv` would couple book generation to a suite and churn a
byte-identical artifact. Hit-rate is therefore emitted as its own TSV, written
to stdout by a new command, and never mixed into generation output.

## Home: a `book-tool hitrate` subcommand

The command lives in `book-tool`, not `engine-bench`. `book-tool` already parses
PGN (via `pgn-reader`/`shakmaty`) and owns the `position_signature()` logic the
book is *built* with, so the "does the book contain this position" check reuses
the exact same signature code with no drift and no dependency on
`engine-search`'s runtime loader. `engine-bench` is about search strength (SPRT,
tactical, NPS) and is the wrong home.

Invocation:

```
book-tool hitrate <book.txt> <suite.pgn>
```

It prints a TSV report to stdout and writes nothing to disk. `run`'s argument
parsing gains a `hitrate` command alongside `generate`; it takes exactly two
positional arguments and rejects any others with a usage string.

## How a book is read

The book file is the v2 text format: line 1 is the literal header
`rusty-fish-book v2`; every later line is `<signature>\t<uci>:<weight> ...`,
where `<signature>` is a position signature (a FEN with the halfmove and
fullmove counters stripped). `hitrate` skips the header line and collects the
substring before the first tab on each remaining line into a set of signatures.
It reads only signatures; move weights are irrelevant to coverage. A line
without a tab is malformed and fails the command loudly rather than being
silently skipped.

## How the suite is replayed

`<suite.pgn>` is a PGN whose games are short opening lines in SAN. For each
game, replay from the standard start position move by move using the same dual
path `build_book` uses: convert each SAN to a move with `shakmaty`, and mirror
it on an `engine_core::Board` so `position_signature()` is computed by the same
code that built the book. Before making each move, compute the signature of the
current position and test membership in the book's signature set.

Positions are checked only up to the book's build depth. The generator records
signatures for the positions before each of the first sixteen plies
(`max_plies = 16`), so a position at ply index sixteen or beyond can never be in
the book by construction. Checking those would deflate the rate with guaranteed
misses, so each line checks at most its first sixteen positions (ply indices
zero through fifteen). This bound is the same sixteen the generator uses; the
spec and code note the coupling so the two stay aligned. A line of `L` moves
contributes `min(L, 16)` checked positions (the positions before moves one
through `min(L, 16)`).

A malformed or illegal suite move (SAN that does not parse or is not legal in
the reached position) fails the command loudly, matching how the generator
treats a corrupt input.

## Metrics emitted

A `metric\tvalue` TSV, byte-deterministic so it can be asserted exactly:

| metric | meaning |
|--------|---------|
| `lines` | number of games in the suite |
| `plies_checked` | total positions checked = sum of `min(L, 16)` over lines |
| `plies_in_book` | count of checked positions whose signature is in the book |
| `hit_rate` | `plies_in_book / plies_checked`, fixed six decimals |
| `mean_book_depth` | mean over lines of the count of consecutive in-book positions from ply zero before the first miss, fixed six decimals |
| `fully_covered_lines` | count of lines whose every checked position is in the book |

`hit_rate` is coverage, not agreement: a position is a hit when the book has any
entry for it, regardless of whether the book's move equals the suite's move.
Agreement is a stricter, separate metric and is out of scope. Floats use a fixed
six-decimal format (matching the existing tactical-suite solve-rate style, e.g.
`1.000000`) so output is reproducible. `mean_book_depth` and `hit_rate` are
computed over `plies_checked`; when a suite is empty (`plies_checked == 0`) the
command fails loudly rather than dividing by zero, since an empty suite is a
mistake.

## The corpus: `assets/opening-book/hitrate-suite.pgn`

A committed, hand-curated PGN of roughly twenty canonical opening mainlines in
SAN (for example Ruy Lopez, Italian, Scotch, Sicilian Najdorf and Sveshnikov,
French, Caro-Kann, Queen's Gambit Declined, Slav, Nimzo-Indian, King's Indian,
Grünfeld, English, Catalan, London), each about twelve to sixteen plies. SAN so
a reviewer can read the openings directly; parsed through `book-tool`'s existing
PGN path. The suite is chosen to represent openings a broad population plays
rather than being derived from the training data, so the number is informative
and a coverage regression is visible.

## CI: report-only

`.github/workflows/opening-book.yml` gains a step that runs `hitrate` against the
committed *production* book and the committed suite, then writes the TSV to
`$GITHUB_STEP_SUMMARY` and uploads it as an artifact:

```
cargo run --release -p book-tool -- hitrate \
  assets/opening-book/rusty-fish-book-v2.txt \
  assets/opening-book/hitrate-suite.pgn
```

The production book is committed (about 459 KB), so this reads a local file and
downloads nothing; ordinary PR CI stays light, honoring the constraint that it
never fetches the full database. The step is report-only: no byte-assertion and
no floor gate in v1. Hit-rate is a visibility metric, and an untuned floor would
be noise; a regression floor can follow once real numbers are observed. The
workflow already triggers on `book-tool/**` and `assets/opening-book/**`, so a
change to the command, the book, or the suite re-reports the number.

The existing fixture regeneration and byte-identity checks are unchanged. The
hit-rate step runs against the production book, not the three-position fixture,
against which coverage would be near zero and uninformative.

## Verification

- A unit test drives the compiled `book-tool` binary over a tiny fixture book
  and a tiny suite PGN with a hand-computed expected TSV, asserting the exact
  bytes. The fixture exercises a partial hit (a line the book covers for some
  plies then misses) so `hit_rate`, `mean_book_depth`, and `fully_covered_lines`
  are all non-trivial.
- A test pins that a suite move outside the book still counts as a checked miss
  rather than an error, and that a genuinely malformed book or illegal suite
  move fails loudly.
- All validation runs in GitHub Actions; Cargo is never run locally.

## Out of scope

- A regression floor / gate on hit-rate (report-only in v1).
- Move-agreement (does the book's move equal the suite's move) as a metric.
- Measuring hit-rate against the synthetic fixture book.
- Any change to the v2 book format, the generator, or the refresh workflow.
