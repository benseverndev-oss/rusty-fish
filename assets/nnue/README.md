# NNUE networks

## `rusty-fish-net.rfnn` — the engine's default evaluation

A `768 → 512 → 1` side-to-move-relative perspective network in the quantised
`RFNN` format (magic / version / hidden / feature-weights i16 / feature-bias i16 /
output-weights i16 / output-bias i32), 789,520 bytes, hidden 512.

**Training:** distilled from a Stockfish teacher. ~3M middlegame positions were
sampled from six months of 2200+-rated Lichess games (2017-01 … 2017-06) and each
was labelled with Stockfish's centipawn evaluation from a fixed 100,000-node
search (the `train_sf` Modal pipeline: `engine-bench gen-eval-positions` →
`engine-bench label-sf` → cp-mode training).

**Strength:** gated at **+8.0 Elo, SPRT AcceptH1 over 16,384 games** (5520W / 5719D
/ 5145L, movetime-bounded self-play) against the tuned hand-crafted evaluation —
the first NNUE in the engine to beat it. It is bundled into the binary
(`include_bytes!`) and installed as the default eval; the hand-crafted evaluation
remains reachable via `UseNNUE false` (UCI) or `set_nnue(None)` (library).

Provenance for the teacher labels is fixed by `assets/nnue/wdl-corpus.toml` (the
pinned, SHA-256-verified Lichess months).
