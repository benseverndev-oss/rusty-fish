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
bytes — hidden 512). **This is step zero and a hard build prerequisite:**
`include_bytes!` is compile-time, so the workspace will not build until the real
trained net bytes are committed at that path (the file does not exist yet;
`assets/nnue/` currently holds only `wdl-corpus.toml`). The bytes are already
retrieved off the Modal Volume to the local scratchpad — the plan copies them in.

The repo's `.gitattributes` already exists (it pins `assets/opening-book/*` to
`text eol=lf`). **Append** an entry for the net that marks it binary —
`assets/nnue/*.rfnn binary` (equivalently `-text`), NOT `text eol=lf` — so no
CRLF/LF mangling corrupts the bytes (a corrupted net fails `from_bytes`). A short
`assets/nnue/README.md` records provenance: trained on ~3M Lichess positions
labelled by Stockfish at 100k nodes, hidden 512, gated at +8.0 Elo / AcceptH1 over
16384 games vs the tuned hand-crafted eval.

## Bundling and the default flip

- `engine-search` gains `fn bundled_network() -> Arc<Nnue>` returning a cloned
  `Arc` from a `static BUNDLED: LazyLock<Arc<Nnue>>` that `include_bytes!`es the
  asset (`include_bytes!("../../assets/nnue/rusty-fish-net.rfnn")` from
  `engine-search/src/`) and parses it once via the existing `from_bytes`. The
  `Arc<Nnue>` shape (not `&'static Nnue`) is required because every install site is
  `set_nnue(Some(Arc<Nnue>))`. A parse failure is a compiled-in-asset programming
  error, so `expect(...)` with a clear message is fine.
- The default flips at the **library `Searcher` level**: `impl Default for
  Searcher` (the single origination point — `lib.rs` ~521) installs the bundled net
  (`nnue: Some(bundled_network())`) instead of `None`. This is a **one-place**
  change: the only other `Searcher` struct literal, `Searcher::helper`, takes
  `nnue` as a parameter and is fed `self.nnue.clone()` at spawn, so Lazy SMP helper
  threads inherit the net automatically with no extra edit. `Searcher::evaluate`
  already prefers the net, so no evaluation-path change is needed beyond the
  default.

## UCI startup wiring (the library flip alone is NOT enough)

**Critical:** flipping `Searcher::default()` does not make the shipped UCI binary
play NNUE. `start_search` builds a fresh `Searcher::default()` and then
*unconditionally* calls `searcher.set_nnue(state.nnue)` (`engine-uci/src/main.rs`
~116-122), and `EngineState` derives `Default` → `nnue: None`. So the flipped
library default is immediately overwritten by `None` on every `go`. The binary
plays NNUE only if **`state.nnue` is initialised to the bundled net** — a separate,
required change from the library flip.

Model `EngineState`'s eval selection as a composed state, not a single `Option`, so
the toggle and a custom `EvalFile` net compose correctly:

- Store `use_nnue: bool` (default `true`) and `eval_file: Option<Arc<Nnue>>`
  (default `None`) on `EngineState`, and recompute the effective net whenever
  either changes: `state.nnue = if !use_nnue { None } else {
  eval_file.clone().or_else(|| Some(bundled_network())) }`. A custom `EngineState`
  default (not `#[derive(Default)]`) sets `use_nnue = true` so the bundled net is
  installed at startup.
- **`EvalFile <path>`** sets `eval_file = Some(from_file(path))` and recomputes;
  it continues to override with a custom network.
- A new **`UseNNUE` check option** (default `true`) sets `use_nnue` and recomputes.
  `false` → hand-crafted (`state.nnue = None`); `true` → the custom `EvalFile` net
  if one is loaded, else the bundled net. This composition means `UseNNUE true`
  after an `EvalFile` load does NOT clobber the custom net (the bug a naive
  `true → Some(bundled)` would cause).

Advertise it in `write_uci_header` as a static line `option name UseNNUE type check
default true` (no signature change). The `apply_option` arity guard
(`tokens.len() < 4`) is not a problem: a check option always carries `true`/`false`
(4 tokens), so it never hits the empty-value gotcha.

## Hand-crafted fallback

The hand-crafted eval stays first-class and reachable: at the **library** level
`set_nnue(None)` restores `evaluate_position` exactly as today, and at the **UCI**
level `UseNNUE false` reaches it (above). Rationale: debugging, a safety valve if a
net regresses, and the gate baseline.

## engine-bench ripple (required by the default flip)

Because the library default is now NNUE, **every `Searcher::default()` site in
engine-bench that relied on the hand-crafted default changes behaviour.** The plan
must handle each one deliberately. Sites that must **explicitly disable NNUE**
(`set_nnue(None)`) to keep their meaning:

- **`gate_searcher`** (`engine-bench/src/lib.rs` ~605) — the shared builder behind
  `run_eval_gate_fens`, `run_mobility_gate_fens`, and `run_eval_spsa_campaign`.
  These compare *hand-crafted eval-parameter configs*; under the flip both sides
  become the identical bundled net (zero signal). `set_nnue(None)` here covers all
  three at once.
- **`play_game` / `play_parameter_game`** (~565) — the search-parameter SPSA
  (`run_spsa_campaign`). This is not zero-signal (both sides differ in search
  params), but under the flip it silently starts tuning search params *on top of
  NNUE*. To keep it tuning the frozen hand-crafted engine (so its frozen snapshot
  stays meaningful), disable NNUE here too. (Re-tuning search on NNUE is future
  work, out of scope.)
- **`play_nnue_game`'s baseline** (`lib.rs` ~439) — the NNUE gauntlet builds
  `candidate` (installs a net) vs `baseline` (`Searcher::default()`, commented
  "hand-crafted evaluation"). Under the flip the baseline silently becomes NNUE,
  making it NNUE-vs-NNUE. `baseline.set_nnue(None)`.
- **`generate_training_samples`'s labeler** (`engine-bench/src/train.rs` ~66) — the
  local self-play data-gen labels positions with a depth-N search whose leaves are
  documented as "the hand-crafted evaluation." Under the flip the leaves become
  NNUE, silently changing the distilled teacher. `labeler.set_nnue(None)`. (Its
  test only asserts labels *differ* from the static eval, so this would rot
  silently otherwise.)

Sites that should **keep NNUE** (they measure the shipped engine, so the flip is
correct — reassess their CI output, do not disable):

- **The external SF gauntlet** (`play_external_game`, `lib.rs` ~756) builds
  `Searcher::default()` and never calls `set_nnue`, so it inherits NNUE
  automatically — the honest "how strong is our engine vs Stockfish" measurement.
- **The tactical suite** and **throughput benchmark** — see CI impact below.

None of these mis-flips fail a test loudly (the gauntlet/labeler tests only check
counts / that labels differ), so each must be handled by inspection, not by relying
on CI to catch it.

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
