# NNUE networks

## `rusty-fish-net.rfnn` — the engine's default evaluation

A `768 → 1024 → 1` side-to-move-relative perspective network in the quantised
`RFNN` format (magic / version / hidden / feature-weights i16 / feature-bias i16 /
output-weights i16 / output-bias i32), 1,579,024 bytes, hidden 1024.

**Training:** distilled from a Stockfish teacher. ~3M middlegame positions were
sampled from six months of 2200+-rated Lichess games (2017-01 … 2017-06) and each
was labelled with Stockfish's centipawn evaluation from a fixed 100,000-node
search (the `train_sf` / label-store pipeline: `gen-eval-positions` → `label-sf` →
cp-mode training), trained hidden 1024 for 120 epochs.

**Strength:** found by an experiment-harness sweep over the persistent SF labels
and gated at **+~50–58 Elo, SPRT AcceptH1** (movetime-bounded self-play) against
the previous champion — the hidden-512 / 60-epoch net (itself +8.0 vs the tuned
hand-crafted eval). The gain came mostly from *training longer* (60 → 120 epochs;
the prior net was under-trained) plus the capacity bump (512 → 1024). It is bundled
into the binary (`include_bytes!`) and installed as the default eval; the
hand-crafted evaluation remains reachable via `UseNNUE false` (UCI) or
`set_nnue(None)` (library). The engine reads `hidden` from the RFNN header, so the
width change needs no code change.

Provenance for the teacher labels is fixed by `assets/nnue/wdl-corpus.toml` (the
pinned, SHA-256-verified Lichess months).
