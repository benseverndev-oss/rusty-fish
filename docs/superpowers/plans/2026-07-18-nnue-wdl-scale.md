# NNUE Lichess-WDL v2 (Scale + Capacity) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the outcome-trained NNUE its best shot at beating the tuned hand-crafted eval by scaling the corpus ~6x (2017-01..2017-06), rewriting the trainer for GPU throughput, bumping capacity to hidden 512, and gating movetime-bounded — still opt-in, adopted only on SPRT AcceptH1.

**Architecture:** A new committed `assets/nnue/wdl-corpus.toml` pins six Lichess months (URL + SHA-256). The Modal pipeline downloads/verifies each into the Volume and fans `gen-wdl-data` labeling over `(month, shard)` pairs. `train_nnue.py` is rewritten to pre-tokenize the whole dataset once into fixed-width `[N,32]` int32 index tensors (padding index 768, frozen), so minibatching is pure GPU work; it adds a cosine LR schedule and a validation-loss report. The gate switches to movetime (orchestration-only — `gate-file` already accepts a movetime arg). No Rust changes.

**Tech Stack:** Python/PyTorch (`modal/train_nnue.py`), Modal (via infisical + uv), TOML (`assets/nnue/wdl-corpus.toml`). The `engine-bench gen-wdl-data` labeller and the RFNN format are unchanged.

**Spec:** `docs/superpowers/specs/2026-07-18-nnue-wdl-scale-design.md`

---

## Global constraints

- **No Rust change in this slice.** Do not touch `engine-bench` / `engine-search`. The labeller (`gen-wdl-data`) already streams, shards, and caps; the gate binary (`gate-file <net> <depth> <openings> [move_time_ms]`) already accepts a movetime budget.
- **No CI gate on Python.** Verify trainer changes with a **torch parity check run via `uv run --with torch`** and `py_compile`; the real end-to-end validation is the Modal run (Task 4). Never run Cargo locally (not needed here anyway).
- **Modal runs** launch via `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 infisical run --env dev -- uv run --with modal -- modal run --detach modal/app.py::<entrypoint> ...`, retrieved from `modal app logs <app-id>` (see the `modal-self-play-gating` memory). Modal builds from the local tree, so a branch runs before merge. The `rusty-fish-wdl` Volume already holds `export.pgn.zst` from the v1 run (harmless; v2 writes `export-<month>.pgn.zst`).
- **Verify `gh` account is `benzsevern` before every remote op** (`gh auth switch --user benzsevern`); keyring PAT as `GH_TOKEN` for writes/PRs; push via tokenized URL `https://benzsevern:$(gh auth token --user benzsevern)@github.com/benseverndev-oss/rusty-fish.git HEAD:feat/nnue-wdl-scale` if the plain push blocks; `git fetch origin --prune` before branching; **stage paths explicitly — never `git add -A`**.
- `cargo fmt`-style is irrelevant (no Rust); match the existing 4-space Python style in the two files. Conventional Commits.
- **Branch:** `feat/nnue-wdl-scale` (already created off latest main, spec committed).

## Background an engineer needs

**The v1 pipeline** (merged in #56, all in `modal/app.py` + `modal/train_nnue.py`):
- `prepare_export()` downloads the single pinned export to `/vol/export.pgn.zst`, SHA-verifies, `wdl_volume.commit()`.
- `label_wdl_shard(i, n, per_game)` runs `bash -c 'set -euo pipefail; zstdcat /vol/export.pgn.zst | {BIN} gen-wdl-data - --shard {i}/{n} --per-game {p} > /vol/samples-{i}.tsv'`, commits, returns the line count.
- `train_wdl_run(hidden, epochs)` (GPU A10G, mounts the Volume, `reload()`s it) concatenates `sorted(glob.glob("/vol/samples-*.tsv"))` into `/vol/data.tsv`, calls `train_nnue.train(data_path, hidden, epochs, batch_size=1024, lr=1e-3, device="cuda", wdl_target=True)` + `quantize_and_write`, returns RFNN bytes.
- `gate_shard(net_bytes, depth, openings_text)` calls `[BIN, "gate-file", net_path, str(depth), openings_path]` — **note it never forwards a movetime**, so it uses the binary's default.
- `nnue_gate_run(net_bytes, gate_depth, gate_openings, gate_plies, gate_shard_size)` mirrors `eval_gate_run`: `make_openings` -> `_chunks` -> `gate_shard.starmap` -> sum -> `sprt_verdict`, printing `NNUE_GATE_RESULT_BEGIN/END`.
- `train_wdl(...)` entrypoint wires `prepare_export.remote()` -> `label_wdl_shard.starmap` -> `train_wdl_run.remote` -> `nnue_gate_run.remote`.

**The current trainer** (`modal/train_nnue.py`):
- `_load_samples(path)` -> `(owns, opps, targets)` (lists of index-lists + float targets).
- `_ragged_to_bag(rows)` -> `(values, offsets)` for `nn.EmbeddingBag`, **rebuilt per minibatch** (`train_nnue.py:131-132`) — the bottleneck.
- Model: `nn.EmbeddingBag(768, hidden, mode="sum")` + `feature_bias[hidden]` + `nn.Linear(2*hidden, 1)`; forward clamps to `[0,127]`, concatenates the two perspectives, and divides the linear output by 64.
- `train(data_path, hidden, epochs, batch_size, lr, device, wdl_target=False)`: builds `target_wp` (WDL mode: `clamp(target,0,1)`), loops epochs, per minibatch rebuilds ragged bags, MSE on win-probabilities.
- `quantize_and_write(model, hidden, out_path)`: reads `model.transformer.weight` `[768, hidden]`, `feature_bias`, `output.weight.squeeze(0)` `[2*hidden]`, `output.bias`, quantizes to i16 (output bias i32), writes `MAGIC | version(u32) | hidden(u32) | feature_weights | feature_bias | output_weights | output_bias`.
- `target_win_prob(target, wdl_target)` helper exists and is unchanged.

**RFNN invariant:** the feature-weights block is exactly `768 * hidden` i16 values in row-major `feature*hidden + i` order. The rewrite uses `nn.Embedding(769, hidden, padding_idx=768)`; **only rows `0..=767` are exported** (the padding row is dropped), so the block is byte-identical in format.

## File structure

| Path | Change | Responsibility |
|------|--------|----------------|
| `assets/nnue/wdl-corpus.toml` | Create | Pins the six months (name/url/sha256). Training input, separate from the opening-book manifest. |
| `modal/train_nnue.py` | Modify | Batched padded-tensor trainer, cosine LR, validation-loss report, `quantize_and_write` slice. |
| `modal/app.py` | Modify | Corpus loader, multi-month `prepare_export`, `(month, shard)` labeling, movetime gate, hidden-512 defaults, SHA probe. |

---

### Task 1: WDL corpus manifest + SHA probe

**Files:** Create `assets/nnue/wdl-corpus.toml`; Modify `modal/app.py` (probe entrypoint + loader)

- [ ] **Step 1: Create the corpus manifest with the known SHA and placeholders**

`assets/nnue/wdl-corpus.toml` — the Lichess standard-rated monthly exports for 2017-01..2017-06. Only `2017-01`'s SHA is known (from `assets/opening-book/manifest.toml`); the rest are filled by the probe in Step 3.

```toml
# Lichess standard-rated monthly exports used as NNUE WDL training input.
# SEPARATE from assets/opening-book/manifest.toml: that pins a shipped book asset;
# this is training data. SHAs verified in the Modal pipeline before use.
source_name = "Lichess Open Database standard rated games"
license = "CC0-1.0"
license_url = "https://database.lichess.org/"

[[month]]
name = "2017-01"
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-01.pgn.zst"
sha256 = "d1236dcd954089aee162c7b0d82f51162f7c912882343d47b77ba5c0e05512f6"

[[month]]
name = "2017-02"
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-02.pgn.zst"
sha256 = ""

[[month]]
name = "2017-03"
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-03.pgn.zst"
sha256 = ""

[[month]]
name = "2017-04"
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-04.pgn.zst"
sha256 = ""

[[month]]
name = "2017-05"
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-05.pgn.zst"
sha256 = ""

[[month]]
name = "2017-06"
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-06.pgn.zst"
sha256 = ""
```

- [ ] **Step 2: Add a corpus loader + a SHA probe to `modal/app.py`**

At the top of `app.py`, add `import tomllib`. Add a client-side loader (runs in the local entrypoint, which executes from the repo root):

```python
def _load_wdl_corpus() -> list[dict]:
    """Read the committed WDL corpus manifest (name/url/sha256 per month)."""
    manifest = pathlib.Path(REPO_ROOT) / "assets" / "nnue" / "wdl-corpus.toml"
    with open(manifest, "rb") as handle:
        return tomllib.load(handle)["month"]
```

Add a probe function + entrypoint (downloads each month, prints its sha256 so the manifest can be pinned). It does not write anything committed:

```python
@app.function(image=rust_image, timeout=60 * 60)
def sha_probe_one(name: str, url: str) -> str:
    path = f"/tmp/{name}.zst"
    subprocess.run(["curl", "-L", "-o", path, url], check=True)
    out = subprocess.run(["sha256sum", path], capture_output=True, text=True, check=True)
    digest = out.stdout.split()[0]
    print(f"SHA_PROBE {name} {digest}", flush=True)
    return f"{name} {digest}"


@app.local_entrypoint()
def sha_probe():
    """Print sha256 for EVERY corpus month (incl. the already-pinned 2017-01, so it
    serves as a cross-check anchor against the digest copied from the book manifest).

        modal run modal/app.py::sha_probe
    """
    months = _load_wdl_corpus()
    for line in sha_probe_one.starmap([(m["name"], m["url"]) for m in months]):
        print(line)
```

(Probing all six re-downloads 2017-01 once — cheap for a one-time step and worth it: 2017-01's probed digest must match the pinned `d1236dcd…12f6`, catching any typo in the pinned value.)

- [ ] **Step 3: Run the probe, pin the SHAs, commit**

Run (detached not needed — it is short-ish but downloads ~5 files; use `--detach` and read logs to be safe):

```
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 infisical run --env dev -- uv run --with modal --python 3.12 -- modal run modal/app.py::sha_probe
```

Copy each printed `SHA_PROBE <name> <digest>` into the matching `sha256 = ""` in `assets/nnue/wdl-corpus.toml`. Sanity check: `2017-01`'s probed digest MUST equal the already-pinned `d1236dcd…12f6` (same file) — if it does not, stop and investigate. Then:

```
git add assets/nnue/wdl-corpus.toml modal/app.py
git commit -m "feat(nnue): pin 6-month Lichess WDL training corpus + SHA probe"
```

(No CI runs on these paths; that is expected.)

---

### Task 2: Batched padded-tensor trainer

**Files:** Modify `modal/train_nnue.py`

The rewrite must (a) tokenize the whole dataset once into `[N,32]` int32 tensors, (b) use `nn.Embedding(769, hidden, padding_idx=768)`, (c) add a cosine LR schedule, (d) report a held-out validation loss, and (e) still export a byte-format-identical RFNN. `_load_samples`, `target_win_prob`, and `quantize_and_write`'s output layout stay.

- [ ] **Step 1: Add constants and a padded-tensor builder**

Near the other constants add:

```python
MAX_FEATURES = 32        # >= max active features per perspective (<=16 pieces/side)
PAD_INDEX = INPUT_DIMENSION  # 768: a dedicated padding row, frozen to zero
VAL_EVERY = 50           # 1-in-50 samples (~2%) held out for validation
```

Add a vectorized builder that turns ragged index-lists into a `[N,32]` int32 tensor padded with `PAD_INDEX` (one-time cost; uses the same flatten the old per-batch path used, but once):

```python
def _pad_rows(rows):
    """Ragged index lists -> a [N, MAX_FEATURES] int32 tensor padded with PAD_INDEX.

    Rows never exceed MAX_FEATURES (a perspective has <=16 pieces). Built once for
    the whole dataset via a single flatten + masked scatter (no per-row Python)."""
    import torch

    lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
    if (lengths > MAX_FEATURES).any():
        raise SystemExit("a sample exceeded MAX_FEATURES active features")
    flat = torch.tensor([x for r in rows for x in r], dtype=torch.int32)
    padded = torch.full((len(rows), MAX_FEATURES), PAD_INDEX, dtype=torch.int32)
    mask = torch.arange(MAX_FEATURES)[None, :] < lengths[:, None]
    padded[mask] = flat
    return padded
```

- [ ] **Step 2: Rewrite the model to `nn.Embedding` + masked sum**

Replace the `EmbeddingBag`-based `Nnue` module. The forward gathers rows and sums over the feature axis; `padding_idx=PAD_INDEX` keeps the padding row zero and its gradient zero, so the sum equals the old accumulator:

```python
    class Nnue(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            # 769 rows: 0..767 are real features, 768 is a frozen zero pad row.
            self.transformer = nn.Embedding(INPUT_DIMENSION + 1, hidden, padding_idx=PAD_INDEX)
            self.feature_bias = nn.Parameter(torch.zeros(hidden))
            self.output = nn.Linear(2 * hidden, 1)
            nn.init.uniform_(self.transformer.weight[:INPUT_DIMENSION], -0.1, 0.1)
            nn.init.uniform_(self.output.weight, -0.1, 0.1)
            nn.init.zeros_(self.output.bias)

        def forward(self, own_rows, opp_rows):
            # own_rows/opp_rows: [B, MAX_FEATURES] int; embed -> [B, F, hidden] -> sum F.
            acc_own = self.transformer(own_rows).sum(dim=1) + self.feature_bias
            acc_opp = self.transformer(opp_rows).sum(dim=1) + self.feature_bias
            a_own = torch.clamp(acc_own, 0.0, ACTIVATION_CLIP)
            a_opp = torch.clamp(acc_opp, 0.0, ACTIVATION_CLIP)
            features = torch.cat([a_own, a_opp], dim=1)
            return self.output(features).squeeze(1) / OUTPUT_SCALE
```

- [ ] **Step 3: Rewrite `train()` — GPU-resident tensors, cosine LR, validation split**

```python
def train(data_path, hidden, epochs, batch_size, lr, device, wdl_target=False):
    import torch
    from torch import nn

    owns, opps, targets = _load_samples(data_path)
    if not owns:
        raise SystemExit(f"no training samples in {data_path}")

    own_t = _pad_rows(owns).to(device)             # [N, 32] int32
    opp_t = _pad_rows(opps).to(device)
    target = torch.tensor(targets, dtype=torch.float32, device=device)
    target_wp = torch.clamp(target, 0.0, 1.0) if wdl_target else torch.sigmoid(target / WDL_SCALE)

    count = len(owns)
    all_idx = torch.arange(count, device=device)
    is_val = (all_idx % VAL_EVERY) == 0
    train_idx = all_idx[~is_val]
    val_idx = all_idx[is_val]

    model = Nnue(hidden).to(device)   # the module class from Step 2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def loss_on(idx):
        pred_cp = model(own_t[idx], opp_t[idx])
        pred_wp = torch.sigmoid(pred_cp / WDL_SCALE)
        return ((pred_wp - target_wp[idx]) ** 2).mean()

    for epoch in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        total = 0.0
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start:start + batch_size]
            loss = loss_on(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * idx.numel()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val = loss_on(val_idx).item() if val_idx.numel() else float("nan")
        print(
            f"epoch {epoch + 1}/{epochs}: train_wdl_loss {total / train_idx.numel():.6f} "
            f"val_wdl_loss {val:.6f} lr {scheduler.get_last_lr()[0]:.2e}",
            file=sys.stderr,
        )

    return model
```

(The model class stays defined inside `train` as today, or hoist it to module scope — either is fine; keep it inside to minimize churn.)

**int32-index note:** `own_t`/`opp_t` are int32. Modern torch (2.x) indexes `nn.Embedding` with int32 fine, and the Step 5 parity check exercises this exact path on CPU so a failure surfaces there. If a given torch build raises `Expected ... scalar type Long`, cast the minibatch at the call site — `model(own_t[idx].long(), opp_t[idx].long())` — rather than storing int64 (keeps the resident tensors at ~2.3 GB; the transient batch cast is negligible).

- [ ] **Step 4: Fix `quantize_and_write` to drop the padding row**

The only change: read `model.transformer.weight` sliced to the real rows.

```python
        w1 = model.transformer.weight.detach().cpu()[:INPUT_DIMENSION]   # [768, hidden]
```

Everything else (i16 rounding, the write order, `output.weight.squeeze(0)`, `output.bias`) is unchanged, so the RFNN block layout is identical.

- [ ] **Step 5: Torch parity + round-trip check (via uv, not CI)**

Runnable check (from `modal/`), proving the padded forward equals a direct row-sum and that a tiny train + quantize round-trips the RFNN header:

```
uv run --with torch python -c "
import torch, train_nnue as t
emb = torch.nn.Embedding(769, 4, padding_idx=768)
torch.nn.init.uniform_(emb.weight[:768], -0.1, 0.1)
rows = t._pad_rows([[1,5,9],[2]])            # ragged -> [2,32]
got = emb(rows).sum(dim=1)                    # masked sum
want = torch.stack([emb.weight[[1,5,9]].sum(0), emb.weight[[2]].sum(0)])
assert torch.allclose(got, want), 'padded sum != direct row sum'
assert torch.count_nonzero(emb.weight[768]) == 0, 'padding row must be zero'
print('parity ok')
"
```

Also add a couple of samples to a temp TSV and run a 2-epoch `train()` on CPU + `quantize_and_write`, asserting the output starts with `b'RFNN'` and is `8 + 4 + 768*hidden*2 + hidden*2 + 2*hidden*2 + 4` bytes. Record `parity ok` + the byte assertion in the task notes. (No commit gate; this is a developer check.)

- [ ] **Step 6: `py_compile` + commit**

```
uv run --python 3.12 python -m py_compile modal/train_nnue.py
git add modal/train_nnue.py
git commit -m "feat(nnue): batched padded-tensor trainer with cosine LR + val loss"
```

---

### Task 3: Modal pipeline — multi-month + movetime gate

**Files:** Modify `modal/app.py`

- [ ] **Step 1: Multi-month `prepare_export`**

Change `prepare_export` to take a month's `name`/`url`/`sha256` and write `export-{name}.pgn.zst` (idempotent per month):

```python
@app.function(image=rust_image, volumes={"/vol": wdl_volume}, timeout=60 * 60)
def prepare_export(name: str, url: str, sha256: str) -> None:
    export_path = f"/vol/export-{name}.pgn.zst"
    if pathlib.Path(export_path).exists():
        print(f"{export_path} present, skipping download", flush=True)
        return
    subprocess.run(["curl", "-L", "-o", export_path, url], check=True)
    subprocess.run(["sha256sum", "--check"], input=f"{sha256}  {export_path}\n", text=True, check=True)
    wdl_volume.commit()
    print(f"downloaded + verified {export_path}", flush=True)
```

- [ ] **Step 2: `(month, shard)` labeling**

`label_wdl_shard` gains a `name` argument and writes `samples-{name}-{i}.tsv`:

```python
@app.function(image=rust_image, volumes={"/vol": wdl_volume}, timeout=60 * 60)
def label_wdl_shard(name: str, i: int, n: int, per_game: int) -> int:
    out = f"/vol/samples-{name}-{i}.tsv"
    cmd = (
        f"set -euo pipefail; zstdcat /vol/export-{name}.pgn.zst | "
        f"{BIN} gen-wdl-data - --shard {i}/{n} --per-game {per_game} > {out}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)
    wdl_volume.commit()
    with open(out, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())
```

**`train_wdl_run` needs three changes** (do NOT leave its glob as-is):

1. **Narrow the glob** from `samples-*.tsv` to `samples-*-*.tsv`. The `rusty-fish-wdl` Volume is persistent and **still holds v1's `samples-0.tsv`…`samples-15.tsv`** (the single-month shards that produced the −325 baseline). The broad glob would silently concatenate ~1.5M stale single-month samples into the new corpus. The `samples-*-*.tsv` pattern matches the new `samples-<month>-<i>.tsv` names and excludes the v1 `samples-<i>.tsv` names. As a belt-and-braces cleanup, delete stale one-token shards first: `for p in glob.glob("/vol/samples-*.tsv"):` and remove any whose basename has fewer than two `-`-separated index segments (or simply `os.remove` every `/vol/samples-*.tsv` that does not match `/vol/samples-*-*.tsv`), plus remove a stale `/vol/data.tsv` before rewriting it.
2. **Set a host-memory floor:** add `memory=32768` to the `@app.function(...)` decorator. `_load_samples` holds ~9M own+opp Python index-lists and `_pad_rows` transiently builds a ~150M-int `flat` list per perspective — this is host RAM, not GPU RAM, and can OOM a default-sized container before anything reaches the GPU. 32 GB is a safe floor.
3. Receive `hidden`/`epochs` (already parameters) and raise `timeout` to `60 * 60 * 3` for the larger dataset.

- [ ] **Step 3: Movetime gate (orchestration-only)**

`gate_shard` forwards a movetime as the 5th `gate-file` arg; callers pass a high `depth` so movetime binds:

```python
@app.function(image=rust_image)
def gate_shard(net_bytes, depth, openings_text, move_time_ms=0):
    ...
        cmd = [BIN, "gate-file", net_path, str(depth), openings_path]
        if move_time_ms:
            cmd.append(str(move_time_ms))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    ...
```

`move_time_ms=0` (falsy -> arg omitted) preserves the existing `run()` entrypoint's behavior exactly. Thread `move_time_ms` through `nnue_gate_run(net_bytes, gate_depth, gate_openings, gate_plies, gate_shard_size, move_time_ms)` into its `gate_shard.starmap([(net_bytes, gate_depth, text, move_time_ms) for text in shard_texts])`.

- [ ] **Step 4: `train_wdl` entrypoint — corpus fan-out, hidden 512, movetime**

```python
@app.local_entrypoint()
def train_wdl(
    shards_per_month: int = 8,
    per_game: int = 12,
    hidden: int = 512,
    epochs: int = 60,
    gate_openings: int = 2048,
    gate_plies: int = 8,
    gate_depth: int = 64,        # high; movetime binds first
    gate_shard_size: int = 32,
    move_time_ms: int = 50,
    months: str = "",            # comma-sep subset (e.g. "2017-01,2017-02") for short runs; empty = all
):
    corpus = _load_wdl_corpus()
    if months:
        wanted = set(months.split(","))
        corpus = [m for m in corpus if m["name"] in wanted]
    for m in corpus:
        assert m.get("sha256"), f"month {m['name']} has no pinned sha256 — run sha_probe first"

    prepare_export.starmap([(m["name"], m["url"], m["sha256"]) for m in corpus])
    label_args = [(m["name"], i, shards_per_month, per_game)
                  for m in corpus for i in range(shards_per_month)]
    counts = label_wdl_shard.starmap(label_args)
    print(f"labeled {sum(counts)} WDL samples across {len(label_args)} shards "
          f"({len(corpus)} months x {shards_per_month})")

    net_bytes = train_wdl_run.remote(hidden, epochs)
    print(f"trained network: {len(net_bytes)} bytes")

    print(nnue_gate_run.remote(net_bytes, gate_depth, gate_openings, gate_plies,
                               gate_shard_size, move_time_ms))
```

Update `nnue_gate_run`'s signature/body per Step 3. Give its docstring short-validation + real-run example invocations.

- [ ] **Step 5: `py_compile` + commit**

```
uv run --python 3.12 python -m py_compile modal/app.py
git add modal/app.py
git commit -m "feat(nnue): multi-month WDL corpus fan-out + movetime gate, hidden 512"
```

---

### Task 4: PR, merge, run on Modal, assess

- [ ] **Step 1: Verify the diff + open the PR**

`git fetch origin --prune`; confirm the branch diff is exactly `assets/nnue/wdl-corpus.toml` (SHAs filled), `modal/train_nnue.py`, `modal/app.py`, the spec, and this plan — no strays. Open the PR (superpowers:finishing-a-development-branch); body: ships nothing by default (NNUE opt-in), the powered SPRT gate is the acceptance test, no Rust change.

- [ ] **Step 2: Merge on green** (Rust tests still pass trivially since no Rust changed; the Python is not CI-gated). Merge per the repo standing rule.

- [ ] **Step 3: Short Modal validation** — 2 months, tiny config, to confirm multi-month download, `(month,shard)` fan-out, the batched trainer, and the movetime gate all work end-to-end:

```
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 infisical run --env dev -- uv run --with modal --python 3.12 -- \
  modal run --detach modal/app.py::train_wdl \
  --months 2017-01,2017-02 --shards-per-month 2 --per-game 2 --hidden 64 --epochs 2 \
  --gate-openings 64 --gate-shard-size 16 --move-time-ms 50
```

Retrieve via `modal app logs <app-id>`: confirm a labeled-sample count, a trained-net byte size, and an `NNUE_GATE_RESULT` verdict. (The verdict itself is meaningless at this size — plumbing only.)

- [ ] **Step 4: Real run** — all six months, hidden 512, 60 epochs, powered gate:

```
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 infisical run --env dev -- uv run --with modal --python 3.12 -- \
  modal run --detach modal/app.py::train_wdl \
  --shards-per-month 8 --per-game 12 --hidden 512 --epochs 60 \
  --gate-openings 2048 --gate-shard-size 32 --move-time-ms 50
```

Watch the training log: `train_wdl_loss` / `val_wdl_loss` per epoch. Retrieve the `NNUE_GATE_RESULT` from `modal app logs`.

- [ ] **Step 5: Assess and record.** Compare against the v1 result (-325 Elo, val loss ~0.219):
  - **AcceptH1 (net wins):** follow-up commits the `.rfnn` as an asset, wires it as the default eval, re-gated by CI; update `D:/Work-Tracking/work-tracker-personal.md`.
  - **Material improvement but not a win** (gate well off -325 toward 0, `val_wdl_loss` below ~0.20): record that scaling helps; next is more months / longer training or the Stockfish teacher.
  - **Flat** (still ~-325, `val_wdl_loss` ~0.219): record that raw outcome labels are the ceiling at this architecture; the next lever is a stronger teacher (Stockfish-eval labels), not more data.
  - Update the tracker with the verdict and reasoning either way.

---

## Out of scope

HalfKA / king-buckets, Stockfish-eval or hybrid teachers, incremental accumulator updates, early stopping / dropout (v1 reports val loss but does not act on it), dedup beyond the per-game cap, multi-year datasets, and any RFNN-format or Rust inference change.
