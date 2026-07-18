# SPSA Eval Tuning Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SPSA-tune a bounded set of hand-crafted eval weights (mobility + existing terms) via self-play on Modal, gated against today's eval; ship the weights only if a powered gate shows a gain.

**Architecture:** Generalize the SPSA primitives to be size-agnostic and specs-parameterized so they serve both the existing search-param vector and a new eval vector. Extend `EvalParams` into a settable, threaded struct (default byte-identical). Run the eval-only SPSA campaign as a Rust `run_spsa_campaign` inside one long Modal container. Gate the tuned eval against the default with the powered parallel Modal gate.

**Tech Stack:** Rust 2024 (`engine-search`, `engine-bench`), Modal (Python, via infisical + uv), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-18-eval-spsa-tuning-design.md`

---

## Global constraints

- **Never run Cargo or Rust binaries locally.** Rust unit tests run in the `Rusty Fish Tests` GHA workflow (`cargo test --workspace`); Modal runs build the engine in-cloud. Each "run the test" step is: commit, push, watch the run.
- **Modal runs** launch from this machine via `infisical run --env dev -- uv run --with modal -- modal run modal/app.py::<entrypoint>` with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` set (Infisical injects `MODAL_TOKEN_ID`/`SECRET`; Modal builds from the local tree, so it can run a branch before merge). See the `modal-self-play-gating` memory.
- **Verify the `gh` account is `benzsevern` before every remote op**; pass the keyring PAT as `GH_TOKEN` for writes/PRs; push via tokenized URL; **`git fetch origin --prune` before branching** (tokenized pushes don't update tracking refs); **stage paths explicitly — never `git add -A`**.
- `cargo fmt` style by hand; Conventional Commits.
- **Branch:** `feat/eval-spsa-tuning`, already on latest main with the spec committed.

## Background an engineer needs

**The SPSA tuner** (`engine-bench/src/lib.rs`): `SPSA_DIMENSIONS = 8`; `SPSA_SPECS: [SpsaSpec; 8]` (name/min/max/step per search param); `search_params_to_vector`/`vector_to_search_params` project `SearchParams`; `SpsaRng` (xorshift64* → Rademacher `direction()`); `perturb`, `clamp_vector`, `spsa_update` all hardcode `[f64; SPSA_DIMENSIONS]` and read the global `SPSA_SPECS`; `run_spsa_campaign(positions, initial, config)` loops iterations in-process: perturb ±, play `theta+` vs `theta-` via `play_parameter_game`, `spsa_update`. `mobility_scale` was added to `SearchParams` but excluded from the vector (`vector_to_search_params` sets it to 0).

**The eval** (`engine-search/src/lib.rs`): `evaluate_position(board, mobility_scale: i32)` sums terms. A private `struct EvalParams` (with `const EVAL_PARAMS`) holds only the five piece values as `TaperedScore::equal(...)` (pawn 100, knight 320, bishop 330, rook 500, queen 900). `tapered_piece_value` reads `EVAL_PARAMS`. `mobility_score(kind, mob)` has hardcoded weight/offset match arms (knight `(4,4,4)`, bishop `(3,3,6)`, rook `(2,4,7)`, queen `(1,2,13)`). Bishop pair is `TaperedScore::equal(35)` inline. Passed-pawn base is `score += 20 + advancement * 10` inside `pawn_structure_bonus` (a separate `20` in `rook_file_bonus` is unrelated — do not touch it). `Searcher::evaluate` calls `evaluate_position(board, self.params.mobility_scale)`.

**Modal** (`modal/app.py`): `train_net` runs a long single-container job (`timeout=60*60`); `mobility_gate` fans `mobility-gate-file` shards via `starmap`, `sprt_verdict` runs `engine-bench sprt`.

## File structure

| Path | Change | Responsibility |
|------|--------|----------------|
| `engine-bench/src/lib.rs` | Modify | Generalize SPSA primitives; eval vector + `EVAL_SPSA_SPECS`; generic campaign; per-side eval in match/gate; `spsa-eval`/`eval-gate-file` support. |
| `engine-bench/src/main.rs` | Modify | `spsa-eval` and `eval-gate-file` commands. |
| `engine-search/src/lib.rs` | Modify | Extend `EvalParams`; thread into `evaluate_position`; `Searcher::set_eval_params`. |
| `modal/app.py` | Modify | `spsa_tune` and `eval_gate` entrypoints. |

---

### Task 1: Generalize the SPSA primitives

**Files:** Modify `engine-bench/src/lib.rs`

Make `spsa_update`, `clamp_vector`, `perturb`, and `SpsaRng::direction` size-agnostic and specs-parameterized, so the same code drives the 8-dim search vector and the new eval vector. This changes no numeric behavior for the search campaign — a regression test pins that.

The regression guard must capture the tuned params the **current (pre-refactor)**
campaign produces, then assert the generalized code still produces them. A
same-binary determinism check would only prove determinism, not invariance across
the refactor — so we bake a pre-refactor snapshot.

- [ ] **Step 1: Write the regression test with the tuned params printed**

```rust
#[test]
fn search_param_spsa_matches_the_frozen_tuned_params() {
    let cfg = SpsaConfig { iterations: 3,
        match_config: MatchConfig { candidate_depth: 2, baseline_depth: 2, max_plies: 16 },
        ..SpsaConfig::default() };
    let pos = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"];
    let report = run_spsa_campaign(&pos, SearchParams::default(), cfg).unwrap();
    eprintln!("FROZEN_TUNED = {:?}", report.tuned); // TEMP: read from CI, then bake below
    // assert_eq!(report.tuned, SearchParams { ...frozen from CI... });
}
```

- [ ] **Step 2: Push, read the pre-refactor tuned from the CI log, bake it, confirm green**

Push; the test passes (no assertion yet) and the `workspace` log prints
`FROZEN_TUNED = SearchParams { ... }`. Copy those exact values into the
`assert_eq!`, delete the `eprintln!`, push again, confirm green. This snapshot is
now the pre-refactor behavior the generalization must preserve. (The
`SpsaConfig::default()` seed is fixed, so the campaign is deterministic and the
frozen value is stable.)

- [ ] **Step 3: Generalize the four primitives**

Change signatures to slices + a specs argument (returning `Vec<f64>`), reading the passed specs rather than the global:

```rust
fn clamp_vector(vector: &[f64], specs: &[SpsaSpec]) -> Vec<f64> {
    vector.iter().zip(specs).map(|(v, s)| v.clamp(s.min, s.max)).collect()
}

fn perturb(theta: &[f64], direction: &[f64], sign: f64, specs: &[SpsaSpec]) -> Vec<f64> {
    let stepped: Vec<f64> = theta
        .iter()
        .zip(direction)
        .zip(specs)
        .map(|((t, d), s)| t + sign * d * s.step)
        .collect();
    clamp_vector(&stepped, specs)
}

pub fn spsa_update(theta: &[f64], direction: &[f64], candidate_score: f64, learning_rate: f64, specs: &[SpsaSpec]) -> Vec<f64> {
    let gradient = 2.0 * candidate_score - 1.0;
    let next: Vec<f64> = theta
        .iter()
        .zip(direction)
        .zip(specs)
        .map(|((t, d), s)| t + learning_rate * gradient * d * s.step)
        .collect();
    clamp_vector(&next, specs)
}
```

And `SpsaRng::direction` takes a length:

```rust
    pub fn direction(&mut self, dimensions: usize) -> Vec<f64> {
        (0..dimensions).map(|_| if self.next_u64() & 1 == 0 { -1.0 } else { 1.0 }).collect()
    }
```

- [ ] **Step 4: Update the search-param call sites**

In `run_spsa_campaign`, thread `&SPSA_SPECS` and `SPSA_DIMENSIONS`:

```rust
    let mut theta = search_params_to_vector(&initial).to_vec();
    ...
    let direction = rng.direction(SPSA_DIMENSIONS);
    let plus = vector_to_search_params(&to_array(&perturb(&theta, &direction, 1.0, &SPSA_SPECS)));
    let minus = vector_to_search_params(&to_array(&perturb(&theta, &direction, -1.0, &SPSA_SPECS)));
    ...
    theta = spsa_update(&theta, &direction, fraction, config.learning_rate, &SPSA_SPECS);
```

`search_params_to_vector`/`vector_to_search_params` keep their `[f64; 8]` types; add a small `fn to_array(v: &[f64]) -> [f64; SPSA_DIMENSIONS] { v.as_slice().try_into().expect("len") }` helper. `run_spsa_campaign` has **four** call sites to fix once `theta: Vec<f64>`: the two `perturb(...)` (wrap in `to_array` before `vector_to_search_params`), the per-iteration `params: vector_to_search_params(&to_array(&theta))` record push, and the final `tuned: vector_to_search_params(&to_array(&theta))`. Update the existing SPSA unit tests (`spsa_update_moves_toward_the_winning_side`, `spsa_update_clamps_to_bounds`, `spsa_vector_round_trips_default_params`, `spsa_rng_is_reproducible_and_seed_sensitive`) to pass `&SPSA_SPECS` / a length and use `Vec`/slices; `assert_eq!(Vec, [f64; 8])` holds via `PartialEq`.

- [ ] **Step 5: Push and confirm green**

The deterministic-tune test and all updated SPSA unit tests pass; the search-param campaign behavior is unchanged.

---

### Task 2: Extend `EvalParams` and thread it into evaluation (byte-identical default)

**Files:** Modify `engine-search/src/lib.rs`

Extend `EvalParams` to hold all tunable weights, defaulting to today's values, and thread it into `evaluate_position` replacing the hardcoded reads. Default eval stays byte-identical.

The byte-identical guard must capture the scores today's **two-arg**
`evaluate_position(&board, 0)` produces, *before* the signature changes, then
assert the threaded version still produces them. Baking values from a post-change
run would be circular (it would freeze the new code's output and catch nothing).

- [ ] **Step 1: At baseline, freeze today's eval scores for a corpus**

Add a corpus of ~6 varied FENs (startpos, an open middlegame, a closed one, a
pawn endgame, a position with a bishop pair, one with a passed pawn) and a test
that prints today's scores:

```rust
const EVAL_CORPUS: [&str; 6] = [ /* startpos + 5 varied FENs */ ];

#[test]
fn default_eval_is_byte_identical() {
    for fen in EVAL_CORPUS {
        let board = Board::from_fen(fen).unwrap();
        eprintln!("EVAL {fen} = {}", evaluate_position(&board, 0)); // TEMP: two-arg, today's code
    }
    // After baking: assert_eq!(evaluate_position(&board, 0, &EvalParams::default()), FROZEN[i]);
}
```

- [ ] **Step 2: Push, read the frozen scores, bake them (two-arg, pre-change)**

Push; read `EVAL <fen> = <score>` from the `workspace` log. Bake a `FROZEN: [i32; 6]`
and rewrite the test to assert `evaluate_position(&board, 0) == FROZEN[i]` (still
two-arg). Push, confirm green. These are today's exact scores; Step 5 changes the
call to three-arg and they must still match.

- [ ] **Step 3: Make `TaperedScore` public and extend `EvalParams`**

`TaperedScore` and its fields/constructors are **private** today, but the SPSA
vector projection in engine-bench (Task 3) must read and rebuild them across the
crate boundary. Make it public: `pub struct TaperedScore`, `pub middlegame`/`pub
endgame`, and `pub const fn new`/`equal`. Then replace the current private
`EvalParams`/`EVAL_PARAMS` with a public struct carrying every tunable weight, and
a `Default` reproducing today's constants:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalParams {
    pub knight: TaperedScore,
    pub bishop: TaperedScore,
    pub rook: TaperedScore,
    pub queen: TaperedScore,
    pub knight_mobility: TaperedScore,
    pub bishop_mobility: TaperedScore,
    pub rook_mobility: TaperedScore,
    pub queen_mobility: TaperedScore,
    pub bishop_pair: i32,
    pub passed_pawn_base: i32,
}

impl Default for EvalParams {
    fn default() -> Self {
        Self {
            knight: TaperedScore::equal(320),
            bishop: TaperedScore::equal(330),
            rook: TaperedScore::equal(500),
            queen: TaperedScore::equal(900),
            knight_mobility: TaperedScore::new(4, 4),
            bishop_mobility: TaperedScore::new(3, 3),
            rook_mobility: TaperedScore::new(2, 4),
            queen_mobility: TaperedScore::new(1, 2),
            bishop_pair: 35,
            passed_pawn_base: 20,
        }
    }
}
```

Pawn value stays the fixed `TaperedScore::equal(100)` constant (the anchor, not in `EvalParams`); mobility offsets stay the fixed `(4,6,7,13)` constants.

- [ ] **Step 4: Thread `EvalParams` through evaluation**

Change `evaluate_position(board: &Board, mobility_scale: i32, params: &EvalParams)`. Rewire the reads:
- `tapered_piece_value(piece)` → take `params`, return `params.knight` etc. for N/B/R/Q, fixed pawn 100 / king 0.
- `mobility_score(kind, mob)` → take `params`, use `params.knight_mobility` etc. (weights) with the fixed offsets.
- Bishop-pair `TaperedScore::equal(35)` → `TaperedScore::equal(params.bishop_pair)`.
- `pawn_structure_bonus`'s passed-pawn `score += 20 + advancement * 10` → `params.passed_pawn_base + advancement * 10`. **Leave the `rook_file_bonus` `20` alone.**

Thread `params` into every helper that needs it (their signatures gain `&EvalParams`). `Searcher::evaluate` passes `&self.eval_params` (new field, `EvalParams::default()`); `hand_crafted_evaluation` passes `&EvalParams::default()`. Add `Searcher::set_eval_params`. **Update the frozen byte-identical test's call to the three-arg `evaluate_position(&board, 0, &EvalParams::default())`** — the `FROZEN` scores from Step 2 must still match, which is the byte-identical proof. Fix all other `evaluate_position(` test call sites (see the crate-wide grep from the mobility work) to pass `&EvalParams::default()`.

- [ ] **Step 5: Confirm green + pin the default constants**

Add a `default_eval_params_match_the_original_constants` test asserting each `EvalParams::default()` field equals its known constant (knight 320, `knight_mobility` (4,4), bishop_pair 35, passed_pawn_base 20, ...). Push; confirm the byte-identical corpus test (now three-arg, matching the Step-2 FROZEN), the default-constants test, and all existing eval tests pass. The FROZEN scores matching under the threaded code is the guarantee that no term was dropped or altered.

---

### Task 3: Eval vector, per-side `EvalParams` plumbing, and the eval gate

**Files:** Modify `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`

This lands the foundations the campaign (Task 4) builds on: the `EvalParams` ↔ vector projection, per-side eval in matches, and the `eval-gate-file` command. Per-side plumbing must precede the campaign, which plays `theta+` vs `theta-` as two `EvalParams`.

- [ ] **Step 1 (vector projection, TDD):** Add `EVAL_SPSA_SPECS: [SpsaSpec; N]` (one spec per weight — e.g. piece values default±120 step 12, mobility 0..12 step 1, bishop_pair 0..80 step 6, passed_pawn 0..60 step 6) and `eval_params_to_vector(&EvalParams) -> [f64; N]` / `vector_to_eval_params(&[f64; N]) -> EvalParams` (reading `TaperedScore.middlegame`/`.endgame`, now public). Test: round-trip `EvalParams::default()`, and that `vector_to_eval_params` clamps out-of-range. Push red (functions missing) → implement → green. **Order in the vector must be fixed and documented** — the `eval-gate-file`/`spsa-eval` TSV interchange depends on it.
- [ ] **Step 2 (per-side eval plumbing):** Give `play_parameter_game` and the gate's `gate_searcher`/`play_mobility_game` a per-side `EvalParams` (set via the `Searcher::set_eval_params` added in Task 2). This lets a match pit two eval configurations. Existing callers pass `EvalParams::default()` for both sides (behavior unchanged).
- [ ] **Step 3 (eval gate, TDD):** `run_eval_gate_fens(fens, candidate: EvalParams, baseline: EvalParams, move_time, max_plies)` plays candidate (mobility on, `mobility_scale = 100`) vs baseline (mobility off) over the FENs, color-swapped. An `eval-gate-file <openings_file> <tuned_eval_tsv_file> [movetime_ms]` command reads the tuned `EvalParams` from a **file** (a TSV of the vector, path passed as argv — mirror how `gate_shard` writes the net to a temp file), plays vs default, emits `W\tD\tL`. Test: default-vs-default is well-formed; a lopsided candidate (queen value 200) loses. Push red → implement → green.

---

### Task 4: Eval SPSA campaign and `spsa-eval` command

**Files:** Modify `engine-bench/src/lib.rs`, `engine-bench/src/main.rs`

- [ ] **Step 1 (TDD):** A short in-process `run_eval_spsa_campaign(positions, initial: EvalParams, config) -> EvalSpsaReport` mirroring `run_spsa_campaign` but over the eval vector + `EVAL_SPSA_SPECS`, playing `theta+` vs `theta-` via the per-side plumbing from Task 3, mobility enabled on both sides. Reuse the generalized `spsa_update`/`perturb`/`SpsaRng`. Smoke test (2 iterations, few positions, depth-2/short) returns in-bounds `EvalParams`. Push red → implement → green.
- [ ] **Step 2:** `spsa-eval [iterations] [openings] [movetime_ms]` command: generate openings, run the campaign, print the tuned `EvalParams` to stdout as the **same vector TSV** `eval-gate-file` parses (so the campaign's output feeds the gate directly), and a per-iteration trace to stderr. Push, confirm green.

---

### Task 5: Modal `spsa_tune` and `eval_gate` entrypoints

**Files:** Modify `modal/app.py`

- [ ] **Step 1:** `spsa_tune` local_entrypoint — an `@app.function(image=rust_image, timeout=...)` runs `engine-bench spsa-eval` in one container (long timeout, mirroring `train_net`), returns the tuned `EvalParams` TSV; the entrypoint prints it.
- [ ] **Step 2:** `eval_gate` local_entrypoint — takes the tuned `EvalParams` TSV, `make_openings` → `_chunks` → `eval_gate_shard.starmap` (each runs `eval-gate-file` with the tuned TSV) → sum → `sprt_verdict`. Mirrors `mobility_gate`.
- [ ] **Step 3:** No CI (Python). Validated by the Modal runs in Task 6.

---

### Task 6: Open the PR, merge, run on Modal, assess

- [ ] **Step 1:** Verify the branch diff (spec, plan, the four modified files, no strays). Open the PR (superpowers:finishing-a-development-branch); body states default eval is byte-identical (nothing ships until a gated flip), and the campaign proposes / the gate disposes.
- [ ] **Step 2:** Merge on green.
- [ ] **Step 3:** Run a **short** Modal campaign to validate the pipeline: `... modal run modal/app.py::spsa_tune --iterations 3 --openings 64`. Confirm it returns tuned `EvalParams`.
- [ ] **Step 4:** Run the **real** campaign (e.g. `--iterations 40`, a per-iteration match sized to fit the container timeout), capture the tuned `EvalParams` TSV.
- [ ] **Step 5:** Run the powered eval gate on the tuned params: `... modal run modal/app.py::eval_gate --tuned <tsv> --gate-openings 2048 --movetime-ms 50`. Read the SPRT: wins/draws/losses, Elo, decision.
- [ ] **Step 6:** **If it gains** (positive Elo / trends AcceptH1): a one-line follow-up PR sets `EvalParams::default()` (and the mobility default on) to the tuned values, updating the byte-identical/default-constants tests to the new numbers, re-gated by normal CI. **If flat/negative:** leave the default; record that the hand-crafted eval is near its ceiling and the next lever is NNUE or new terms. Update `D:/Work-Tracking/work-tracker-personal.md` either way.

---

## Out of scope

Parallelizing the SPSA campaign's per-iteration match across containers (a future optimization needing a Python driver + thin Rust step commands); tuning search params, king-safety internals, pawn-structure penalties, activity/threats, or PSTs; second-order SPSA; NNUE.
