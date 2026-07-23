# NNUE networks

## `rusty-fish-net.rfnn` — the engine's default evaluation

A `768 → 1024 → 1` side-to-move-relative perspective network in the quantised
`RFNN` format (magic / version / hidden / feature-weights i16 / feature-bias i16 /
output-weights i16 / output-bias i32), 1,579,024 bytes, hidden 1024.

**Training:** distilled from a Stockfish teacher. **~16M** middlegame positions
were sampled from **~20 months** of 2200+-rated Lichess games (2017-01 … 2018-11
plus partial later months) and each was labelled with Stockfish's centipawn
evaluation from a fixed 100,000-node search (the `label_sf` / persistent
label-store pipeline: `gen-eval-positions` → `label-sf` → cp-mode training),
trained hidden 1024 for 160 epochs (converged; val-loss flat from ~epoch 145).

**Strength:** the **data-scale** lever. Scaling the SF-label corpus ~5× (the ~3M
six-month set → ~16M across ~20 months) and re-training 1024 produced a net gated
at **+45.3 Elo, SPRT AcceptH1** (689W 357D 490L over 1536 games, movetime-bounded
self-play) against the previous champion — the hidden-1024 / 240-epoch net on the
~3M corpus (itself +~48–54 over the 120-epoch net, … +8.0 over the tuned
hand-crafted eval). Width stays 1024 and epochs 160 (converged; the earlier
epoch-plateau sweep settled 240 as the ceiling for the 3M set, and 160 reaches the
same converged val-loss on the larger corpus). A head-to-head control confirmed the
gain is real and *data-targeted*: a 5M-position net trained the same way on the
public **lichess eval database** (deeper Stockfish evals, but an endgame/mate-heavy
position mix) gated at **−159.7 Elo, AcceptH0** — position distribution, not eval
depth, is what matters, and the self-labeled middlegame sampling is well-matched to
gameplay. It is bundled
into the binary (`include_bytes!`) and installed as the default eval; the
hand-crafted evaluation remains reachable via `UseNNUE false` (UCI) or
`set_nnue(None)` (library). The engine reads `hidden` from the RFNN header, so the
width change needs no code change.

Provenance for the teacher labels is fixed by `assets/nnue/wdl-corpus.toml` (the
pinned, SHA-256-verified Lichess months).
