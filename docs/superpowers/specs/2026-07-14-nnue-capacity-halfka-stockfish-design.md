# NNUE Capacity, HalfKA, and Stockfish-Label Design

## Decision

Use a controlled experimental ladder with Modal as the primary execution path.
Build one reproducible Stockfish 18 label corpus, keep its train/validation/test
splits fixed, and change one model variable at a time. A candidate advances from
offline metrics to a small deterministic screen and only then to the existing
2,304-game SPRT gate.

This replaces speculative all-at-once redesigns with attributable experiments.
The current WDL NNUE lost the decisive campaign `29342974072` by 23W/435D/1846L
(-373.40 Elo; SPRT `AcceptH0`), so it is not an adoption candidate.

## Goals

- Train static NNUE candidates from reproducible, stronger Stockfish 18 labels.
- Measure whether hidden-layer capacity helps before changing feature semantics.
- Add HalfKA king-bucketed features without weakening the existing incremental
  accumulator guarantees.
- Preserve `RFNN` v1 loading while defining a versioned format for HalfKA nets.
- Make every result reproducible and comparable: dataset provenance, training
  configuration, quantization checks, screening result, and full SPRT result.
- Run GPU training and parallel CPU labeling/gating through Modal; retain the
  local/GitHub path for small smoke tests and regression tests.

## Non-goals

- Do not embed an NNUE as the default evaluation in this project phase.
- Do not replace UCI `EvalFile`, the hand-crafted fallback, or the current
  `RFNN` v1 inference path.
- Do not chase opening-book, search, or desktop changes as part of this work.
- Do not use an unbounded search as a labeler or gate; all external engine
  processes and candidate gates require explicit resource limits.

## Data and labels

### Corpus

Generate exactly one million legal, nonterminal positions from this deterministic
mixture:

- 400,000 seeded legal random-walk positions from the existing FEN seeds;
- 400,000 opening-derived positions from the generated-opening utility; and
- 200,000 quiet positions sampled after legal moves from the same deterministic
  walks.

The implementation must define the exact source counts, seeds, maximum plies,
and filtering rules in a dataset manifest before a run starts. It must remove
duplicate canonical FENs, reject terminal positions, retain side-to-move, and
write samples in a stable canonical order. The manifest is immutable input to a
training run and includes the SHA-256 of every shard and of the concatenated
dataset.

### Stockfish labels

Use the pinned Stockfish 18 binary already used by the external-engine harness.
Each worker receives an explicit configuration: binary SHA-256, UCI options,
one engine process, one search thread, fixed hash allocation, `go nodes N`, and
a per-position wall-clock timeout. Fixed nodes, not depth, make the work budget
portable across Modal workers. Record the requested and reported nodes, score,
mate conversion/clamp rule, and any timeout/error.

Use a fixed node budget selected by a small calibration run before the corpus
run. The calibration labels the same 1,000 held-out FENs at 25,000, 100,000,
and 400,000 nodes. Treat the 400,000-node score as the reference. Select the
lowest of 25,000 or 100,000 nodes whose 95th-percentile absolute centipawn
difference from the 400,000-node score is at most 20 cp and whose shard has no
timeouts; otherwise use 400,000 nodes. Timed-out or malformed labels fail the
shard; they are never silently replaced by the hand-crafted evaluation.

Convert non-mate centipawn labels to the existing WDL target with
`sigmoid(cp / 400)`. Convert mate scores to a finite signed centipawn value
before that transform, using one documented clamp shared by the Rust and Python
paths.

### Splits and artifacts

Assign train/validation/test by a stable hash of canonical FEN rather than by
input order: 90%/5%/5%. Store each split's count and SHA-256 in the manifest.
The validation split drives early selection; the test split is read only after a
candidate is selected for screening. Modal stores the corpus, manifest, trainer
report, quantized net, and gate records under a run identifier. The run
identifier is included in every printed report and artifact name.

## Model ladder

All candidates use the same corpus and splits for a given experiment series.
Training uses the existing quantization-aware, WDL-sigmoid objective unless a
later, separately approved experiment changes the objective.

### Capacity phase

Keep the current 768 side-relative piece-square feature encoding and evaluate
hidden widths 128, 256, and 512. Width 128 is the control. Select the smallest
width whose validation WDL loss is at least 2% lower than the control and whose
quantization checks pass. Do not run a full SPRT for widths that do not clear
those offline conditions.

### HalfKA phase

Implement a HalfKA feature schema after the capacity phase identifies a width.
The feature key is, for each perspective, the bucket of that perspective's
king square plus the relative colour, piece kind, and relative square of every
non-king piece. Kings choose the bucket but are not encoded as ordinary pieces.
The exact bucket count and index formula are constants shared by inference,
feature extraction, data export, and Python training. The first implementation
uses 64 king buckets; bucket reduction is a later experiment, not part of this
scope.

For an ordinary piece move, capture, promotion, or en passant, update the two
perspective accumulators incrementally. When either king moves (including
castling), rebuild the affected perspective accumulator from the board because
the king bucket changes. Existing debug assertions must compare incremental
state against a full refresh at every search node for both v1 and HalfKA models.

Evaluate HalfKA at the winning capacity and, if it clears the same offline
conditions, one larger width. Do not combine a feature-schema change with a new
label corpus or loss-function change in the same comparison.

## RFNN compatibility

Keep `RFNN` v1 as the current 768-feature format. Define `RFNN` v2 with an
explicit feature-schema discriminator, input dimension, and HalfKA bucket count
in the header before the parameter arrays. A loader must reject inconsistent
schema/dimension combinations, truncated files, trailing bytes, zero width, and
unknown versions. It must continue loading valid v1 files without conversion.

`Nnue` owns its feature schema. Accumulator refresh, incremental update, and
the trainer/exporter use that schema rather than free-standing assumptions about
the input dimension. UCI `EvalFile` loads either valid version and leaves the
hand-crafted evaluation active on any load error.

## Modal orchestration

Extend `modal/app.py` into named stages:

1. build or reuse a corpus manifest;
2. fan out Stockfish label shards over CPU containers;
3. verify and merge shards into stable train/validation/test artifacts;
4. launch GPU training for one explicit candidate configuration;
5. run Rust-side load/quantization parity validation;
6. run a small deterministic screen; and
7. fan out the existing bounded-move-time full gate only for a promoted
   candidate.

Each stage is idempotent by run identifier and input hashes. Re-running a stage
may reuse a verified artifact but may not overwrite it. The command line exposes
the candidate schema, width, dataset manifest, label node budget, epochs,
learning rate, screen configuration, and `--full-gate` as explicit arguments;
there are no hidden production defaults.

The gate keeps the PR #38 100 ms candidate-search bound. Label workers use their
own fixed-node request and wall-clock process timeout, which are separate from
the candidate gate setting.

## Promotion rules

A candidate advances only when all of these hold:

1. its validation WDL loss improves over the width-128 v1 control trained on
   the same dataset by at least 2%;
2. the quantized `RFNN` passes byte-format, load, and float-versus-quantized
   prediction-tolerance tests (maximum absolute difference 32 cp over the
   sealed test split);
3. its sealed test WDL loss is at least 1% lower than the width-128 v1 control;
   and
4. it scores at least 50% against the control in a deterministic 384-game
   screen (12 shards × 16 generated openings × two colours).

A promoted candidate receives the existing 12-shard, 2,304-game depth-4 gate.
Only `AcceptH1` under the repository SPRT configuration qualifies a net for a
separate adoption proposal. `AcceptH0` ends that candidate branch and records
the negative result. `Continue` increases only the gate sample count; it does
not change architecture, data, or engine settings mid-test.

## Testing and verification

- Unit-test v1 and v2 format round trips, invalid headers, incompatible schema
  dimensions, and cross-version loading behavior.
- Unit-test HalfKA feature indices for both perspectives, vertical mirroring,
  every king bucket, captures, promotions, en passant, and castling.
- Add randomized legal make/unmake tests that assert each accumulator equals a
  full refresh after every move for v1 and HalfKA models.
- Test Stockfish worker protocol parsing with a fake UCI engine: fixed-node
  command, timeout, mate conversion, malformed response, and nonzero exit.
- Test manifest determinism, duplicate-FEN removal, split assignment, shard
  merge hash, and refusal to mix incompatible shard configurations.
- Keep a small local/GitHub smoke corpus that proves Modal-produced v1/v2 nets
  load in Rust and complete a bounded gate.
- Publish each full run's manifest, net checksum, W/D/L, Elo estimate, SPRT
  decision, and the exact campaign URL.

## Delivery sequence

1. Establish dataset-manifest and Stockfish-label primitives with fake-engine
   tests and a small end-to-end smoke run.
2. Add Modal artifact staging and GPU trainer support for manifest-backed v1
   experiments; train the 128/256/512 capacity ladder.
3. Add RFNN v2 schema dispatch and HalfKA incremental accumulators with
   correctness-first tests.
4. Extend the GPU trainer and Modal command to train HalfKA; run the selected
   width and optional larger-width comparison.
5. Apply the promotion rules, run full gates for viable candidates, and write a
   result record. Start a separate adoption design only after `AcceptH1`.

## Risks and controls

- **Expensive or nondeterministic labels:** fixed binary/hash/options/nodes,
  canonical ordering, manifest hashes, and shard-fail behavior make labels
  auditable.
- **Feature-index or accumulator desynchronization:** schema-owned indexing,
  full-refresh assertions, and move-class regression tests protect correctness.
- **Training success that vanishes after quantization:** select on quantized
  parity and require Rust-side load/evaluation before screening.
- **False strength claims:** only the preconfigured full SPRT changes adoption
  status; all other measurements are screening evidence.
