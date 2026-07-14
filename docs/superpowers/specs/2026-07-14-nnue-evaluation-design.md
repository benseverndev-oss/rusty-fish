# NNUE Evaluation Foundation Design

## Goal

Lay the foundation for an NNUE (efficiently updatable neural network)
evaluation — the single biggest strength lever on the roadmap — by building the
inference machinery, feature encoding, quantised forward pass, and a versioned
network file format, wired as an **opt-in** evaluator that defaults to the
existing hand-crafted evaluation so nothing regresses until a trained network is
adopted.

## Scope

This first slice delivers the NNUE *engineering*, which is fully testable
without a trained network:

- A `nnue` module in `engine-search` with:
  - A perspective feature set (768 inputs: own/their × 6 piece kinds × 64
    side-relative squares).
  - An accumulator with a from-scratch `refresh(board)` plus incremental
    `add_feature`/`remove_feature` primitives, proven equivalent by test.
  - A quantised forward pass (clipped-ReLU activation, integer output layer)
    producing a side-to-move-relative centipawn score, clamped away from mate
    scores.
  - A versioned little-endian network file format (`RFNN`) with a
    bytes/file loader and round-trip serialisation.
  - A deterministic seeded test-network generator so CI can exercise the whole
    pipeline.
- Integration seam: `Searcher` gains an optional `Arc<Nnue>`; `evaluate` uses it
  when present and the hand-crafted evaluation otherwise. Lazy SMP helper
  searchers share the same network so the shared table stays eval-consistent.
- A UCI `EvalFile` option that loads a network from a path.

Explicitly **out of scope** (documented follow-ups):

- Training a real network (an offline PyTorch pipeline over millions of
  positions). Until then the default evaluation is unchanged.
- Wiring the incremental accumulator into `make_move`/`unmake_move` for the
  in-tree speed win; this slice refreshes per evaluation but proves the
  incremental primitives correct so that hook is a drop-in follow-up.
- King-bucketed (HalfKP/HalfKA) feature sets; the 768-input perspective set is
  the simplest correct starting point.

## Alternatives considered

1. **Ship a full HalfKA net + incremental updates + trainer in one change.**
   The end state, but far too large and un-reviewable in a single slice, and it
   needs an offline training pipeline that cannot live in this repo.
2. **Only define a file format and loader.** Too thin to be useful and leaves
   the inference untested.
3. **Full inference machinery + accumulator (refresh now, incremental proven),
   opt-in and non-regressive (chosen).** Delivers the hard, testable
   engineering, keeps the default engine unchanged, and leaves clean seams for
   training and make/unmake integration.

## Architecture

- **Features.** For a perspective `p`, a piece on square `s` maps to
  `((own_or_their) * 6 + kind) * 64 + relative_square`, where the square is
  vertically flipped for the black perspective and colours are taken relative to
  `p`. Two accumulators (one per perspective) are maintained.
- **Network.** A feature transformer (`768 × H` int16 weights + `H` int16 bias)
  builds each perspective accumulator. The output layer concatenates the
  clipped-ReLU of the side-to-move accumulator and the opponent accumulator
  (`2H` inputs), applies `2H` int16 weights and an int32 bias, and scales the
  integer result into centipawns.
- **Quantisation.** Accumulators sum int16 columns into int32; activations clip
  to `[0, 127]`; the output is divided by a fixed scale and clamped to a safe
  non-mate range.
- **Format.** `RFNN` + `u32` version + `u32` hidden size + `W1`, `b1`, `W2`
  (int16 LE) + `b2` (int32 LE). The loader validates the magic, version, and
  exact byte length.
- **Integration.** `Nnue` holds only immutable weights, so it is shared via
  `Arc` across search threads; each `evaluate` builds a local accumulator, which
  keeps it trivially thread-safe.

## Safety rules

- With no network loaded, `evaluate` is byte-for-byte the current hand-crafted
  evaluation; the default engine is unchanged.
- The loader rejects a wrong magic, unknown version, or mismatched length rather
  than reading garbage weights.
- NNUE output is clamped to a non-mate range so it can never masquerade as a
  mate score.
- Lazy SMP helpers share the primary searcher's network, keeping shared-table
  scores consistent.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including the NNUE tests:
feature-index bounds, incremental-accumulator equals refresh, forward-pass
determinism, format round-trip, and loader rejection of malformed data. The
tactical suite and fixed-opponent gauntlet must not regress (they run with the
default hand-crafted evaluation). Adopting a trained network as the default is a
separate change gated on an external Stockfish SPRT.
