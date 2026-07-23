# NNUE Trainer Performance — Handoff

This documents the performance work on the GPU NNUE trainer
(`modal/train_nnue.py` + its callers in `modal/app.py`), what changed, how each
change was validated, the current performance ceiling, and which directions were
explored and **rejected with data** so they are not re-tried blindly.

Scope: the *training* pipeline only (data load → GPU fit → quantised RFNN
export). The inference/search engine is a separate body of perf work.

## TL;DR — current state

The trainer is **at its bandwidth floor** for its architecture on an A10G. Every
behaviour-preserving win has been banked and measured on real hardware:

- Forward gather+sum fused (`EmbeddingBag`), optimizer fused (`fused=True`),
  loss/forward compiled (`torch.compile`), perspectives batched into one call.
- Corpus parses ~6× faster / ~7× lighter and is **cached** across sweep configs.
- Container memory request cut 128 GB → 32 GB.

End-to-end, the 19M-position × 60-epoch run went from **~32 min → ~18 min** of GPU
epochs (steady-state), plus a one-time ~7–10 s compile.

**Do not** spend more effort chasing training *speed* — it is measurably maxed
(see "Explored and rejected"). The only remaining speed frontier is **hardware**
(bigger GPU) or an **architecture change**, not code tuning.

## Reproducibility contract (important)

Two tiers, deliberately:

- **CPU / CLI path (`device="cpu"`): bit-for-bit reproducible.** Exporting the
  same net from the same seed+corpus is byte-identical to before all of this
  work. This is verified after every change (train two versions, diff the
  exported `.rfnn`).
- **CUDA path: identical within quantisation noise, NOT bit-exact.** `fused=True`
  Adam and `torch.compile` reorder float ops, so the CUDA-trained net can differ
  from an eager run by **≤1 int16 unit** across the whole 394k-weight net
  (≈0.00 Elo; the net is SPRT-gated regardless). Everything *before* the
  `ef7058f` commit is byte-identical on CUDA too.

If you need bit-exact CUDA training for debugging, set `fused=False` and skip the
`torch.compile` call in `train_arrays` (both are gated on `on_cuda`).

## What changed

| commit | change | speedup (A10G) | net effect |
|---|---|---|---|
| `7aa9dd4` | `Embedding`+`.sum` → `EmbeddingBag(mode="sum")` | fwd+bwd 1.89→0.94 ms (**2.0×**) | **byte-identical** |
| `afb4c64` | list-of-lists parse → preallocated numpy | parse ~6× faster, ~7× lighter | **array-equal** |
| `fe78933` | cache parsed corpus on store, keyed by shard manifest | parse once/sweep, not once/config | **byte-identical net** |
| `ef7058f` | fused Adam + `torch.compile` + single-bag forward | steady-state **~1.7×** over baseline | ≤1 i16 unit (CUDA) |
| `103121d` | iterate only whole batches on the compiled path | removes 9.8 s compile warmup | CPU byte-identical |
| `13f57c3` | `train_from_store` memory 128 GB → 32 GB | container cost | none (load now ~5 GB) |

Detail:

1. **EmbeddingBag** — the forward gathered every active feature row and reduced
   with `.sum(dim=1)`, materialising a `[B, 32, hidden]` activation. `EmbeddingBag`
   fuses gather+sum → `[B, hidden]` directly. Same math (padding rows sum to
   zero). CPU proxy showed 11.7×; on A10G the real win is 2.0× (the linear layer +
   Adam don't shrink).

2. **numpy loader** (`_load_padded` / `_load_padded_shards`) — parses the TSV
   straight into preallocated `[N,32]` int32 matrices with `np.fromstring`, no
   Python list-of-lists. On 1M rows: 29.9 s → 4.9 s, peak RSS 1987 → 274 MB. At
   19M rows the old loader peaked ~40 GB (the reason for the old 128 GB request);
   the new one ~5 GB.

3. **Corpus cache** (`load_corpus`) — memoises the parsed arrays + a shard
   manifest under `/store/sf/_cache/<datasets>`. A sweep spins a fresh container
   per config, each of which used to re-parse the whole corpus; now the first
   parses, the rest load in seconds. **Safe by construction:** used only when the
   stored manifest (full path + byte size; the store is append-only) exactly
   matches; any mismatch/missing/corrupt/read-error silently falls back to a fresh
   parse. Writes are `os.replace`-atomic; loads use `allow_pickle=False`.

4. **fused Adam + torch.compile + single-bag** — `fused=True` runs Adam's update
   in one CUDA kernel; `torch.compile` fuses forward+loss; own/opp go through one
   stacked `EmbeddingBag` call. All three gated on `on_cuda`; `torch.compile`
   falls back to eager on any error. Steady-state ~1.7× over the pre-round-4
   baseline (bench: 1.534 → 0.889 s per 800k-row epoch).

5. **Whole-batch iteration** — the ragged final batch made `torch.compile`
   compile a second time for the tail shape (9.76 s first-epoch spike). Iterating
   only full batches gives it one static shape (first epoch → 0.65 s). CUDA-only;
   the dropped <1024-row tail is a fresh random slice each epoch (~0.005% of 19M).
   CPU path keeps every sample (stays byte-identical).

## Validation methodology

- **Byte-identity / drift:** train two versions on the same seed+corpus, quantise
  to RFNN, diff the int16 weights (`np.array_equal` or max delta). Used for every
  change to classify it as byte-identical vs ≤N-unit drift.
- **Loader equivalence:** `array_equal` of the padded arrays vs the old path,
  including a 55%-dropped stress case; multi-shard parse == single-file parse of
  the concatenation.
- **Cache:** hit==fresh-parse; adding a shard rebuilds; corrupt file falls back;
  `train_arrays`(cached) == `train`(file) byte-for-byte.
- **A10G timing:** real Modal A10G, `torch.cuda.synchronize()` around each epoch,
  steady-state = median of post-warmup epochs. CPU numbers are only ever a proxy —
  always confirmed on-GPU before believing a speedup.

## Performance characterisation

The trainer is **memory-bandwidth-bound on the embedding gather**, ceiling
~**1.3 Msamp/s** on an A10. Evidence:

- **Batch-size scaling is flat** — 1024→32768 gives only ~5–6% and plateaus by
  batch 2048. If it were launch- or dispatch-bound, larger/fewer steps would help;
  they don't. Same bytes gathered regardless of batch → same time.

| batch | s/epoch (800k) | Msamp/s |
|---|---|---|
| 1024 | 0.654 | 1.22 |
| 4096 | 0.620 | 1.29 |
| 16384 | 0.617 | 1.30 |
| 32768 | 0.621 | 1.29 |

## Explored and rejected (with data)

Do not re-try these on the current net/GPU without new reason:

| idea | result on A10G | why it loses |
|---|---|---|
| `torch.compile(mode="reduce-overhead")` (CUDA graphs) | **1.9× slower** | not launch-bound; graph capture/management overhead exceeds any launch saving |
| `torch.compile(mode="max-autotune")` | **1.9× slower** | autotuned kernels don't beat default for this shape; big compile cost |
| bf16 autocast on the forward | **6% slower** | gather is bandwidth-bound on *indices*, not FLOP-bound; cast ops cost more |
| bigger batch + LR retune | ~5% throughput | bandwidth-bound (above); not worth an SPRT campaign + net-identity risk |
| parallel `gen-data` labeling | n/a | `label-fens` reads eval from file (no search) → labeling already cheap; only relevant for a search-based teacher, and it perturbs the corpus |
| rewrite in JAX | not benchmarked | bandwidth-bound → framework can't beat hardware bandwidth; JAX's strengths (fusion, `vmap`/`grad`, multi-device, TPU) don't apply on a single A10 with a tiny net. JAX would only help *on TPU*, where the win is the TPU's bandwidth, not JAX. |

## Open directions (not code tuning)

- **Bigger GPU (A100/H100):** ~2–3× the memory bandwidth → directly lifts the
  1.3 Msamp/s ceiling, no net-identity change. A cost/availability call on Modal.
  This is the highest-leverage remaining option and requires zero code changes
  beyond the `gpu=` string.
- **TPU + JAX:** only if moving off GPU entirely; a rewrite, justified solely by
  TPU bandwidth.
- **int8/quantised feature table:** could halve gather bytes and raise the
  ceiling, but is a net redesign + retrain + SPRT — a project, not a tweak.

## Where things live

- `modal/train_nnue.py` — trainer. Key functions: `_load_padded`,
  `_load_padded_shards`, `load_corpus` (+ `_shard_manifest`), `train`
  (parse → `train_arrays`), `train_arrays` (the fit; holds the `Nnue` module,
  `on_cuda` gating of fused Adam + `torch.compile`, the whole-batch loop),
  `quantize_and_write`.
- `modal/app.py` — callers: `train_from_store` (sweep/store path, wires
  `load_corpus` + cache dir), `train_wdl_run`, `train_net`; `run_experiment`
  trains a config and runs the **SPRT ladder vs champion** (the gate any
  identity-changing experiment must pass).
- Benchmarks were run as standalone `modal run` scripts (self-contained A10G
  functions that ship `train_nnue.py`'s source in and time it). They were
  session-scratch, not committed; recreate by wrapping `train_arrays` / the
  forward in an `@app.function(gpu="A10G")` and timing epochs with
  `torch.cuda.synchronize()`. Ask if you want them added under `modal/bench/`.
