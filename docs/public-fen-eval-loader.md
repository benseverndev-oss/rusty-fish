# Public FEN+eval loader (`label-fens`) — spike

A spike to feed **pre-evaluated public datasets** (Kaggle / Hugging Face FEN+eval
dumps) straight into the NNUE trainer, skipping the self-labeling pipeline we run
on Modal.

## What it is

`engine-bench label-fens` converts `FEN<sep>eval` rows into the exact
`cp<TAB>own<TAB>opp` training rows the trainer reads — taking the centipawn label
**from the file** instead of running a Stockfish pass over every position.

```bash
# public dataset (CSV or TSV, White-relative evals) -> trainer-format shard
engine-bench label-fens chessData.csv > shard.tsv
zstdcat some_hf_dump.tsv.zst | engine-bench label-fens - --stm-relative > shard.tsv
```

- Input: `FEN<sep>eval`, tab- **or** comma-separated. Eval may be integer
  centipawns (`24`), a mate (`#3`, `#-2`, `mate 4` → ±`MATE_CP`), or decimal pawn
  units (`0.24` → `24`cp).
- Sign convention: public dumps are usually **White-relative**; the trainer's
  labels are **side-to-move-relative** (`own` == side to move), so the sign is
  flipped for black-to-move positions by default. Pass `--stm-relative` when the
  source is already side-to-move POV. **This is the #1 correctness knob** — a
  wrong convention trains the net against negated targets.
- Malformed rows (header lines, bad FENs) are counted and skipped, not fatal.

Feature extraction reuses the engine's own `active_features`, so the emitted
`own`/`opp` indices are **byte-identical to what the network sees at inference**.
A Python reimplementation of the 768-feature layout would risk silent divergence;
doing it in the engine binary makes that impossible.

## What it replaces (and what it doesn't)

The current data pipeline, per month, is:

```
export.pgn.zst → gen-eval-positions (self-play sampling)
              → label-sf  (fixed-node Stockfish over every FEN)   ← hours of Modal fan-out
              → cp<TAB>own<TAB>opp  → store → train_from_store → gate
```

`label-fens` collapses the first two stages into one instant pass:

```
public FEN+eval → label-fens (feature extraction only)
              → cp<TAB>own<TAB>opp  → store → train_from_store → gate
```

- **Replaced:** the entire `gen-eval-positions | label-sf` stage — i.e. the
  self-play position sampling **and** the fixed-node Stockfish labeling fan-out
  that is the expensive, multi-hour Modal workload (the 24-month run was ~19M
  positions × 100k-node Stockfish across ~100 containers).
- **Unchanged:** the store, `train_from_store`, `train_nnue.py`, and the SPRT
  gate ladder. The output format is identical, so the downstream is untouched.
- **Not addressed:** feature set (still 768) and net architecture. Public **deep**
  evals (SF binpacks, Leela) are higher quality than our 100k-node labels, but the
  official `.binpack`s are HalfKAv2 for `nnue-pytorch` and need a format converter
  — a separate, larger piece than this loader. `label-fens` targets the common
  `FEN,cp` CSV/TSV dumps, which need no converter.

## How to compare against the 19M self-labeled net

Once the 24-month self-labeled net gates, run the same procedure on a public
corpus and gate it head-to-head:

1. Download a public FEN+eval dataset (e.g. a `chessData`-style CSV).
2. `engine-bench label-fens data.csv > /store/sf/public/samples-public-0.tsv`
   (write it as its own store dataset dir so it never mixes with `n100000-pg4`).
3. `train_from_store(["public"], hidden=1024, epochs=240)` — identical trainer.
4. Gate the resulting net vs the champion with the existing ladder.

That isolates **data source** as the only variable: same features, same trainer,
same gate. If the public-data net matches or beats the 19M self-labeled net at a
fraction of the labeling cost, it validates pivoting off self-labeling.

## Status

Spike: `label-fens` subcommand + unit tests landed and validated locally
(build + tests green on rustc 1.97; feature indices verified equal to
`active_features`, sign-flip and mate handling covered). A thin Modal entrypoint
to fan a large public file into the store is the obvious next step once a
specific dataset is chosen.
