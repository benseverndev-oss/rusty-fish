# Gate Ladder Design

## Goal

Make gating cheap enough to run on every experiment. A candidate net climbs a
ladder of increasingly expensive checks; a clearly-worse net is rejected after a
few hundred games (near-free), and only a real contender pays for a full
evaluation. This is sub-project #2 of the faster-iteration effort (after the
persistent label store).

Critically, the ladder gates a candidate against the **current champion net**, not
the hand-crafted eval. Now that the +8.0 NNUE is the shipped default, "beats the
hand-crafted eval" is a ~8-Elo-low bar every decent net clears, so it can't tell
whether a *new* net is an improvement worth adopting. The gate ladder resolves the
"future gates need an NNUE baseline" caveat recorded when NNUE was adopted.

## Baseline: the bundled champion net (net-vs-net)

The `engine-bench` gate binary is compiled from the tree, so it already has the
**champion net bundled** (`bundled_network()`, installed by `Searcher::default()`).
A candidate is the just-trained net loaded from a file. So a net-vs-net gate is:
candidate-net-from-file vs bundled-champion — no separate champion download needed,
and the baseline auto-tracks whatever net is currently shipped.

Today `play_nnue_game` (`engine-bench/src/lib.rs`) builds the candidate with
`set_nnue(Some(file_net))` and the baseline with `Searcher::default()` then
`set_nnue(None)` (forced hand-crafted, from the adoption ripple). The change: thread
a **baseline mode** through `run_nnue_gauntlet…` → `play_nnue_game`:

- **champion** (the new default): the baseline keeps `Searcher::default()`'s bundled
  net — do NOT call `set_nnue(None)`. This is candidate-vs-champion.
- **handcrafted** (opt-in, for the record / eval-development): the baseline calls
  `set_nnue(None)` as today.

Thread the mode as a small `BaselineMode { Champion, Handcrafted }` enum
(defaulting to `Champion`) added only to the inner
`run_nnue_gauntlet_with_optional_move_time` + `play_nnue_game`; keep the public
`run_nnue_gauntlet` / `run_nnue_gauntlet_with_move_time` signatures unchanged
(default `Champion`) so the `nnue-sprt` caller in `main.rs` is untouched.

The `gate-file` command gains an optional trailing arg selecting the mode
(`gate-file <net> <depth> <openings_file> [move_time_ms] [champion|handcrafted]`,
default `champion`). SPRT hypothesis is unchanged (`elo0=0`, `elo1=5`): AcceptH1 =
candidate is ≥5 Elo better than the champion = adopt. **Arg-parse care:** the mode
token sits after the optional `move_time_ms`, so a caller passing a mode must also
pass a movetime (else the mode lands in the u64 movetime slot and is silently
ignored) — the ladder always passes a movetime, so this is safe; the plan parses
the mode by matching the `champion`/`handcrafted` token rather than strict
position, and `gate_shard` forwards the mode only after the movetime.

**This flips the default baseline for the existing gates too, deliberately.**
`gate_shard` → `nnue_gate_run` is called by `train_wdl`, `train_sf`, and
`gate_net`; with `gate-file` now defaulting to `champion`, all three gate
candidate-vs-champion instead of candidate-vs-hand-crafted. That is the intended
direction (it is this spec's whole thesis — hand-crafted is no longer a meaningful
bar). The plan must **update the now-stale "vs the hand-crafted baseline"
docstrings** on `gate_shard`, `nnue_gate_run`, `train_sf`, `train_wdl`, and
`gate_net` so the docs match the new semantics.

**Sanity property:** gating a net against an identical champion (candidate file ==
bundled net) should trend to ~0 Elo / AcceptH0 — a useful smoke check that the gate
isn't biased.

## The rungs

### Rung 0 — validation-loss pre-check (free)

`train_from_store` trains a net whose per-epoch `val_wdl_loss` is currently only
**printed to stderr**, not returned — `train_nnue.py`'s `train(...)` returns just
the `model`. So this rung requires a small trainer change: **`train()` returns
`(model, final_val_loss)`**, and its three callers unpack the tuple (`train_net`,
`train_wdl_run`, and `train_from_store` — the plan must touch all three, not only
`train_from_store`). `train_from_store` then returns `(net_bytes, val_loss)`, and
the ladder **rejects before any game** if `val_loss` is `NaN` or above a generous
sanity ceiling (default ~0.1; a healthy cp-mode net is < 0.02, so this only kills
diverged/broken training, it does not rank close nets). Zero games spent on a
broken run.

### Rung 1 — sequential SPRT gate vs champion (the main rung)

Replace the fixed-N parallel gate with a **sequential SPRT** that stops as soon as
the result is decisive:

1. Play a **chunk** of `chunk_openings` openings (default 256; each opening → 2
   colour-swapped games), fanned across gate shards in parallel (net-vs-champion,
   movetime-bounded) exactly as the current gate does — but generated with a
   distinct per-chunk seed so each chunk is fresh positions.
2. Accumulate cumulative `W/D/L` across all chunks so far and call the existing
   `sprt` command → parse its `decision`. Parse the **TSV `decision` column** (the
   last field of the values line), which is the bare token `AcceptH0` / `AcceptH1`
   / `Continue` — not the stderr `decision = Some(...)` blob.
3. **Stop on a decision:** `AcceptH0` → **reject** (candidate not better; a clearly
   worse net dies in ~1–2 chunks / a few hundred games), `AcceptH1` → **adopt**
   (candidate ≥5 Elo better). `Continue` → play another chunk.
4. Stop at a **cap** of `max_openings` (default 8192); hitting the cap with
   `Continue` = **inconclusive** (too close to call within budget).

The verdict reports the final W/D/L, the Elo estimate, the decision, and the
**games actually played** (so the cost of each gate is visible — the whole point of
the ladder is that most gates cost far less than the cap).

Cost profile: a clearly-worse net rejects after ~1–2 chunks (~500–1000 games,
~2–4 min); a clear improvement accepts similarly fast; only a genuinely-close net
pays up to the cap. Versus today's always-4096-game gate, the common case (most
experiments produce nets that are clearly better or worse than the champion) is
several times cheaper.

### Rung 2 — external SF gauntlet (ground truth, separate)

Unchanged: the dispatch-only 32-game SF18 workflow. It is not a pass/fail rung in
the automated ladder — it is run manually to track the real gap trajectory after a
candidate is *adopted* as the new champion.

## Pipeline shape

- **Rust (`engine-bench`):** the baseline-mode thread + `gate-file` arg + a unit
  test (champion-mode baseline `has_nnue()` true; handcrafted-mode false).
- **Modal (`modal/app.py`):**
  - `train_from_store` returns `(net_bytes, val_loss)`.
  - a `gate_ladder_run(net_bytes, chunk_openings, max_openings, gate_plies,
    gate_depth, move_time_ms, gate_shard_size)` remote function implementing the
    sequential loop, emitting `NNUE_LADDER_RESULT` markers (verdict + games played),
    retrievable from `modal app logs` for a detached run.
  - the sequential loop reuses `make_openings` (per-chunk seed), `gate_shard`
    (now champion-baseline by default), and `sprt_verdict` (parse the decision).
  - `train_sf` wires it: `net_bytes, val_loss = train_from_store.remote(...)`; if
    the val-loss pre-check fails, print a rejected verdict and skip the gate; else
    `gate_ladder_run.remote(...)`.
  - a `gate_ladder` entrypoint (mirroring `gate_net`) re-runs the ladder on the
    stored net without retraining.
- Composed as functions the experiment harness (#3) will orchestrate.

## Verification

- **Rust (CI, `cargo test --workspace`):** the baseline-mode unit test above, and
  that `gate-file` parses the optional mode arg (defaulting to champion). The
  gate's game-play itself is exercised by the Modal run.
- **Modal:** run `gate_ladder` on the stored champion net **against itself**
  (candidate == champion) and confirm it trends to AcceptH0 / ~0 Elo and stops
  (the unbiased sanity check). Then run it on a deliberately-weak net (e.g. a
  few-epoch net from `train_from_store`) and confirm it **rejects early** (AcceptH0
  after ~1–2 chunks, games-played far below the cap) — proving the early-stop
  saves cost.
- All Rust validation is GitHub Actions; Cargo is never run locally.

## Out of scope

- The experiment harness (#3) and cold-start reduction (#4).
- Changing the external SF gauntlet, `train_wdl_run`, or the RFNN/label formats.
- Multi-baseline gauntlets or Elo-model changes beyond the existing SPRT.
- Auto-adopting a winning candidate (bundling + shipping a new champion stays the
  manual adoption flow already used).
