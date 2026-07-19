# NNUE Adoption: Make the Stockfish-Taught Net the Default Eval

## Goal

Make the Stockfish-eval-taught NNUE the engine's **default evaluation**, bundled
into the binary, with the hand-crafted evaluation kept as a selectable fallback.
The net was gated at **+8.0 Elo, SPRT AcceptH1 over 16384 games** vs the tuned
hand-crafted eval, so this is the first NNUE to beat it. No new strength work — the
net's strength is already established; this slice is the wiring plus keeping CI
honest about the fact that the engine now evaluates with NNUE.

## Background: how eval selection works today

- `engine_search::Searcher` holds `nnue: Option<Arc<Nnue>>` (lib.rs). `set_nnue`
  installs or clears it; `has_nnue()` reports it.
- `Searcher::evaluate` (lib.rs ~1436): if `self.nnue` is `Some`, it evaluates with
  the network (incremental accumulator); if `None`, it calls
  `evaluate_position(board, mobility_scale, &eval_params)` (the hand-crafted eval).
- Today the default is `None` (opt-in): NNUE is installed only when the UCI
  `EvalFile` option is set (`engine-uci/src/main.rs`: `Nnue::from_file(path)` →
  `set_nnue`).
- `Nnue::from_bytes(&[u8])` and `from_file(path)` already exist
  (`engine-search/src/nnue.rs`), and the RFNN format is the byte layout the Modal
  trainer exports (magic/version/hidden/feature_weights/feature_bias/output_weights/
  output_bias). So bundling a net via `include_bytes!` + `from_bytes` is a small
  addition.

## The net asset

The trained network is committed as `assets/nnue/rusty-fish-net.rfnn` (789,520
bytes — hidden 512). A `.gitattributes` entry marks it binary (`-text` /
`binary`) so no CRLF/LF mangling corrupts the bytes (the opening-book fixtures
needed the same discipline; a corrupted net would fail `from_bytes`). A short
`assets/nnue/README.md` (or a comment header where it's referenced) records
provenance: trained on ~3M Lichess positions labelled by Stockfish at 100k nodes,
hidden 512, and gated at +8.0 Elo / AcceptH1 over 16384 games vs the tuned
hand-crafted eval.

## Bundling and the default flip

- `engine-search` gains `Nnue::bundled() -> &'static Nnue` (or a
  `fn default_network() -> Arc<Nnue>`): `include_bytes!` the asset and parse it
  once via the existing `from_bytes`, behind a `OnceLock`/`LazyLock` so it is
  parsed a single time and shared. A parse failure is a programming/asset error
  (the bytes are compiled in), so it may `expect(...)` with a clear message.
- The default flips at the **library `Searcher` level**: the default construction
  path installs the bundled net (`nnue = Some(bundled)`), so every `Searcher`
  evaluates with NNUE unless a caller explicitly clears it. This is the honest
  adoption — the engine plays NNUE everywhere, not only through the UCI binary.
  `Searcher::evaluate` already prefers the net, so no evaluation-path change is
  needed beyond the default.

## Hand-crafted fallback

The hand-crafted eval stays first-class and reachable:

- **Library:** `set_nnue(None)` restores `evaluate_position` exactly as today.
- **UCI:** a new `UseNNUE` check option (default `true`). `true` keeps the bundled
  net (or a custom `EvalFile` net); `false` clears the net so the engine plays the
  hand-crafted eval. `EvalFile <path>` continues to override with a custom network.
  The option ordering must let `UseNNUE false` win over the bundled default (the
  engine installs the bundled net at startup; `UseNNUE false` calls
  `set_nnue(None)`).
- Rationale: debugging, a safety valve if a net regresses, and the gate baseline.

## engine-bench ripple (required by the default flip)

Because the library default is now NNUE, code that builds a `Searcher` and relied
on the hand-crafted default changes behaviour. Two deliberate, opposite
adjustments:

- **SPSA eval / mobility gate baselines must explicitly disable NNUE.**
  `run_eval_gate_fens`, `run_mobility_gate_fens`, and the SPSA campaign compare
  *hand-crafted eval parameter configs* against each other. With NNUE now the
  default, both sides would silently become the identical bundled net (zero
  signal). These functions must `set_nnue(None)` on both Searchers so they keep
  comparing hand-crafted eval configs. (This is the same "gate baselines are
  pinned, not 'current default'" caveat recorded when the tuned eval shipped —
  here it becomes a required code change, not just a note.)
- **The external SF gauntlet should run the bundled net.** The gauntlet measures
  "how strong is our engine vs Stockfish"; with NNUE as the engine's eval, the
  gauntlet's candidate should now use the bundled net (either via the new default,
  or by explicitly installing it) so the measurement reflects the shipped engine.
  If the gauntlet's `Searcher` uses the default construction path, it picks up
  NNUE automatically — confirm and keep it.

## CI impact

- **Unaffected (stay green):** `perft` (pure move-gen) and the eval-snapshot /
  `EvalParams` byte-identical tests (they call `evaluate_position` directly, not
  through `Searcher::evaluate`).
- **Rebaked / reassessed to reflect the NNUE engine:**
  - **Tactical suite:** search now uses NNUE, so solved-position sets/scores may
    change (NNUE may solve *more*). Reassess from the first CI run; update expected
    results deliberately, not blindly.
  - **Throughput benchmark:** NNUE costs more per node, so nps drops. If there is a
    hard floor, relax it to a realistic NNUE value; document the before/after.
  - **External gauntlet:** now measures the NNUE engine — expected to improve. Not
    a pass/fail gate, but note the new baseline.
- The approach is to run CI, read what actually moves, and adjust each suite
  deliberately, rather than predicting the deltas.

## Verification

- **Bundled-net round-trip:** a unit test that `Nnue::bundled()` (from the
  compiled-in bytes) parses to a hidden-512 network and that re-serialising it
  (`to_bytes`) equals the committed asset bytes — pins that the asset is intact and
  the loader matches the format.
- **Default is NNUE:** `Searcher::default().has_nnue()` is `true`.
- **Toggle works:** the UCI `UseNNUE false` path yields a searcher with
  `has_nnue() == false` (hand-crafted), and `true`/`EvalFile` install a net; a
  characterisation test in `engine-uci`.
- **Search smoke:** a short search on the start position with the default
  (NNUE) searcher returns a legal best move and a finite score.
- All Rust validation runs in GitHub Actions (`cargo test --workspace` + the
  suites); Cargo is never run locally.
- **Strength is not re-measured here** — the +8.0 Elo / AcceptH1 gate already
  established it. CI green is the acceptance bar for the wiring.

## Out of scope

- Retraining or improving the net; HalfKA; incremental-accumulator changes.
- Re-tuning the hand-crafted eval (it is now a fallback, frozen).
- A new powered gate (already done); future "candidate vs current default" gating
  will need NNUE-aware baselines, tracked separately.
