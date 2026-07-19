"""Modal orchestration for the Rusty Fish NNUE train -> gate loop.

Runs the two embarrassingly-parallel halves of the pipeline across many cloud
containers and the training on a GPU:

  1. Labeling   - fan out `engine-bench gen-data` shards (each a different seed)
                  and concatenate the labelled samples.
  2. Training   - one GPU container runs the PyTorch trainer (`train_nnue.py`),
                  exporting a quantised RFNN network.
  3. Gating     - generate many opening positions, shard them, run
                  `engine-bench gate-file` per shard in parallel, sum the
                  win/draw/loss counts, and take the SPRT verdict.

There is also a WDL path (`train_wdl`): pull a multi-month Lichess corpus
(`assets/nnue/wdl-corpus.toml`), use `sha_probe` to pin each month's SHA, then
`train_wdl` labels the game outcomes into the shared Volume, GPU-trains with the
WDL loss, and runs a movetime-gated SPRT vs the hand-crafted baseline.

This is scaffolding: it needs your own Modal account/token and cannot be run
from the agent sandbox. See modal/README.md.

    modal run modal/app.py                 # defaults
    modal run modal/app.py --hidden 256 --epochs 60 --gate-openings 2048
"""

import glob
import pathlib
import subprocess
import tempfile
import tomllib

import modal

REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
BIN = "/repo/target/release/engine-bench"


def _load_wdl_corpus() -> list[dict]:
    """Read the committed WDL corpus manifest (name/url/sha256 per month)."""
    manifest = pathlib.Path(REPO_ROOT) / "assets" / "nnue" / "wdl-corpus.toml"
    with open(manifest, "rb") as handle:
        return tomllib.load(handle)["month"]

# Image A: builds the engine-bench release binary from the repo source.
rust_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "build-essential", "pkg-config", "zstd", "stockfish")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --default-toolchain stable"
    )
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
    .run_commands("cd /repo && $HOME/.cargo/bin/cargo build --release -p engine-bench")
)

# Image B: GPU training image with PyTorch and the trainer script.
torch_image = (
    modal.Image.debian_slim()
    .pip_install("torch")
    .add_local_file(str(pathlib.Path(__file__).parent / "train_nnue.py"), "/root/train_nnue.py")
)

app = modal.App("rusty-fish-nnue")

# Persistent volume holding the (large) Lichess export and the labelled shards, so
# the download happens once and the GPU trainer can read every shard by path.
wdl_volume = modal.Volume.from_name("rusty-fish-wdl", create_if_missing=True)
labels_volume = modal.Volume.from_name("rusty-fish-labels", create_if_missing=True)


def _sf_dataset(nodes: int, per_game: int) -> str:
    """The store dataset dir keyed on data identity (teacher budget + density)."""
    return f"n{nodes}-pg{per_game}"


@app.function(volumes={"/store": labels_volume}, timeout=60 * 30)
def migrate_flat_sf_labels() -> str:
    import os, glob, re, shutil
    labels_volume.reload()
    dataset = _sf_dataset(100000, 4)
    dst = f"/store/sf/{dataset}"
    os.makedirs(dst, exist_ok=True)
    moved, months = 0, set()
    for p in glob.glob("/store/sf/samples-*.tsv"):  # flat, top-level only (no recursion)
        base = os.path.basename(p)
        shutil.move(p, f"{dst}/{base}")
        moved += 1
        m = re.match(r"samples-(\d{4}-\d{2})-\d+\.tsv", base)
        if m:
            months.add(m.group(1))
    for month in sorted(months):
        with open(f"{dst}/{month}.complete", "w") as h:
            h.write("migrated\n")
    if os.path.exists("/store/sf/data.tsv"):
        os.remove("/store/sf/data.tsv")  # stale flat concat; datasets are re-concatenated on train
    labels_volume.commit()
    report = f"moved {moved} shards into sf/{dataset}, markers: {sorted(months)}"
    print(f"MIGRATE_DONE {report}", flush=True)
    return report


@app.local_entrypoint()
def migrate_labels():
    print(migrate_flat_sf_labels.remote())


@app.function(image=rust_image)
def label_shard(seed: int, plies: int, label_depth: int) -> str:
    """One labeling shard: emit gen-data TSV for a distinct random seed."""
    result = subprocess.run(
        [BIN, "gen-data", str(plies), str(label_depth), str(seed)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


@app.function(image=rust_image)
def make_openings(count: int, plies: int, seed: int) -> str:
    result = subprocess.run(
        [BIN, "gen-openings", str(count), str(plies), str(seed)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


# 30-minute timeout: a movetime gate shard plays gate_shard_size*2 games at
# move_time_ms per move; with a hidden-512 net that can exceed Modal's 300s
# default (which timed out the first hidden-512 gate).
@app.function(image=rust_image, timeout=60 * 30)
def gate_shard(net_bytes: bytes, depth: int, openings_text: str, move_time_ms: int = 0, baseline: str = "champion") -> tuple[int, int, int]:
    """Play the NNUE candidate vs the bundled champion net by default over one
    opening shard.

    `move_time_ms` is forwarded as the optional 5th `gate-file` arg only when
    truthy, so `move_time_ms=0` preserves the depth-only behavior the `run()`
    entrypoint relies on. `baseline` ("champion"|"handcrafted") is the optional
    6th arg and is appended only AFTER a truthy movetime (its positional slot),
    so a non-default baseline requires a movetime — which the ladder always passes.
    """
    with tempfile.TemporaryDirectory() as directory:
        net_path = f"{directory}/net.rfnn"
        openings_path = f"{directory}/openings.txt"
        with open(net_path, "wb") as handle:
            handle.write(net_bytes)
        with open(openings_path, "w", encoding="utf-8") as handle:
            handle.write(openings_text)
        cmd = [BIN, "gate-file", net_path, str(depth), openings_path]
        if move_time_ms:
            cmd.append(str(move_time_ms))
            cmd.append(baseline)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wins, draws, losses = (int(x) for x in result.stdout.strip().split("\t"))
    return wins, draws, losses


@app.function(image=rust_image)
def mobility_gate_shard(movetime_ms: int, openings_text: str) -> tuple[int, int, int]:
    """Play mobility-on (scale 100) vs mobility-off over one opening shard."""
    with tempfile.TemporaryDirectory() as directory:
        openings_path = f"{directory}/openings.txt"
        with open(openings_path, "w", encoding="utf-8") as handle:
            handle.write(openings_text)
        result = subprocess.run(
            [BIN, "mobility-gate-file", openings_path, str(movetime_ms)],
            capture_output=True, text=True, check=True,
        )
    wins, draws, losses = (int(x) for x in result.stdout.strip().split("\t"))
    return wins, draws, losses


@app.function(image=rust_image)
def eval_gate_shard(movetime_ms: int, tuned_tsv: str, openings_text: str) -> tuple[int, int, int]:
    """Play the tuned eval (mobility on) vs the default eval over one opening shard."""
    with tempfile.TemporaryDirectory() as directory:
        openings_path = f"{directory}/openings.txt"
        tuned_path = f"{directory}/tuned.tsv"
        with open(openings_path, "w", encoding="utf-8") as handle:
            handle.write(openings_text)
        with open(tuned_path, "w", encoding="utf-8") as handle:
            handle.write(tuned_tsv)
        result = subprocess.run(
            [BIN, "eval-gate-file", openings_path, tuned_path, str(movetime_ms)],
            capture_output=True, text=True, check=True,
        )
    wins, draws, losses = (int(x) for x in result.stdout.strip().split("\t"))
    return wins, draws, losses


# A generous timeout: the eval SPSA campaign runs its iterations sequentially in
# one container (the loop is Rust, only the matches are sequential here), so it
# needs hours, not minutes. Matches how train_net runs the trainer.
@app.function(image=rust_image, timeout=60 * 60 * 3)
def spsa_tune_run(iterations: int, openings: int, movetime_ms: int) -> str:
    """Run the eval SPSA campaign in one container; return the tuned EvalParams TSV."""
    result = subprocess.run(
        [BIN, "spsa-eval", str(iterations), str(openings), str(movetime_ms)],
        capture_output=True, text=True, check=True,
    )
    tuned = result.stdout.strip()
    # Emit to the (remote) function log so a detached run stays retrievable via
    # `modal app logs` even if the launching client disconnects.
    print(f"TUNED_TSV_BEGIN\n{tuned}\nTUNED_TSV_END", flush=True)
    return tuned


@app.function(image=rust_image)
def sprt_verdict(wins: int, draws: int, losses: int) -> str:
    result = subprocess.run(
        [BIN, "sprt", str(wins), str(draws), str(losses)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout + result.stderr


@app.function(image=torch_image, gpu="A10G", timeout=60 * 60)
def train_net(data_text: str, hidden: int, epochs: int) -> bytes:
    import train_nnue

    with tempfile.TemporaryDirectory() as directory:
        data_path = f"{directory}/data.tsv"
        out_path = f"{directory}/net.rfnn"
        with open(data_path, "w", encoding="utf-8") as handle:
            handle.write(data_text)
        model, _ = train_nnue.train(data_path, hidden, epochs, batch_size=1024, lr=1e-3, device="cuda")
        train_nnue.quantize_and_write(model, hidden, out_path)
        with open(out_path, "rb") as handle:
            return handle.read()


@app.function(image=rust_image, volumes={"/vol": wdl_volume}, timeout=60 * 60)
def prepare_export(name: str, url: str, sha256: str) -> None:
    """Download + verify one pinned Lichess month into the shared volume, once.

    Idempotent per month: if `/vol/export-{name}.pgn.zst` is already present this
    returns early and skips the multi-GB re-download. That early return is safe
    because a file at the FINAL path is always a fully-downloaded, SHA-verified
    one: we download to a `.tmp` sibling, verify it, and only then atomically
    rename it into place — so a truncated leftover from a crashed download can
    never occupy the final path and be used unverified.
    """
    export_path = f"/vol/export-{name}.pgn.zst"
    if pathlib.Path(export_path).exists():
        print(f"{export_path} present, skipping download", flush=True)
        return
    tmp_path = f"{export_path}.tmp"
    subprocess.run(["curl", "-L", "-o", tmp_path, url], check=True)
    # sha256sum --check reads "<hash>  <path>" and exits non-zero on mismatch,
    # which check=True turns into a raised CalledProcessError (function fails).
    # Verify the TEMP file, then atomically rename — the final path only ever
    # holds a verified export.
    subprocess.run(["sha256sum", "--check"], input=f"{sha256}  {tmp_path}\n", text=True, check=True)
    pathlib.Path(tmp_path).replace(export_path)
    wdl_volume.commit()
    print(f"downloaded + verified {export_path}", flush=True)


@app.function(image=rust_image, volumes={"/vol": wdl_volume}, timeout=60 * 60)
def label_wdl_shard(name: str, i: int, n: int, per_game: int) -> int:
    """One WDL labeling shard: stream a month's export through gen-wdl-data.

    Runs the pipe under bash with pipefail so a zstdcat failure (e.g. truncated
    export) fails the shard instead of silently yielding an empty TSV.
    """
    out = f"/vol/samples-{name}-{i}.tsv"
    cmd = (
        f"set -euo pipefail; zstdcat /vol/export-{name}.pgn.zst | "
        f"{BIN} gen-wdl-data - --shard {i}/{n} --per-game {per_game} > {out}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)
    wdl_volume.commit()
    with open(out, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


@app.function(volumes={"/store": labels_volume}, timeout=60 * 10)
def missing_sf_months(dataset: str, months: list[str]) -> list[str]:
    """Return the months in `months` that have no `<month>.complete` marker in the
    store dataset dir (i.e. still need labeling). Read-only on the store."""
    import os
    labels_volume.reload()
    return [m for m in months if not os.path.exists(f"/store/sf/{dataset}/{m}.complete")]


@app.function(volumes={"/store": labels_volume}, timeout=60 * 10)
def wipe_sf_month(dataset: str, month: str) -> int:
    """Remove any existing shards for one (dataset, month) before its labelers run,
    so a re-label can't leave stale shards from a different shard count. Run once
    per month by the orchestrator, before the parallel shards."""
    import os, glob
    labels_volume.reload()
    removed = 0
    for p in glob.glob(f"/store/sf/{dataset}/samples-{month}-*.tsv"):
        os.remove(p); removed += 1
    labels_volume.commit()
    return removed


@app.function(volumes={"/store": labels_volume}, timeout=60 * 10)
def mark_sf_month_complete(dataset: str, month: str) -> None:
    """Write the `<month>.complete` marker after all of a month's shards land, so a
    later run treats the month as a cache hit."""
    import os
    os.makedirs(f"/store/sf/{dataset}", exist_ok=True)
    with open(f"/store/sf/{dataset}/{month}.complete", "w") as h:
        h.write("done\n")
    labels_volume.commit()


@app.function(image=rust_image, volumes={"/vol": wdl_volume, "/store": labels_volume}, timeout=60 * 90)
def label_sf_shard(dataset: str, name: str, i: int, n: int, per_game: int, nodes: int) -> int:
    """One Stockfish-eval labeling shard: sample FEN positions, then replace each
    with its fixed-node Stockfish cp eval.

    Streams a month's export through `gen-eval-positions` (same sampling as
    `gen-wdl-data`) and pipes the FEN+features lines through `label-sf`, which
    drives one persistent Stockfish process at `go nodes {nodes}`. Reads the export
    from `/vol` and writes the labelled shard into the store dataset dir
    `/store/sf/{dataset}/`. It does NOT wipe the month (that's `wipe_sf_month`, run
    once per month by the orchestrator, so parallel shards can't clobber each other).
    """
    import os
    os.makedirs(f"/store/sf/{dataset}", exist_ok=True)
    out = f"/store/sf/{dataset}/samples-{name}-{i}.tsv"
    cmd = (
        f"set -euo pipefail; zstdcat /vol/export-{name}.pgn.zst | "
        f"{BIN} gen-eval-positions - --shard {i}/{n} --per-game {per_game} | "
        f"{BIN} label-sf - {nodes} > {out}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)
    labels_volume.commit()
    with open(out, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def ensure_sf_labels(corpus, per_game, nodes, shards_per_month) -> None:
    """Client-side orchestration: label only the missing (dataset, month) shards
    into the store, skipping months that already have a `<month>.complete` marker.

    Derives the dataset from the data identity, asks the store which months are
    missing, then for each missing month wipes it once, fans out its shards, and
    writes the completion marker. A fully-cached corpus does zero labeling."""
    dataset = _sf_dataset(nodes, per_game)
    names = [m["name"] for m in corpus]
    missing = missing_sf_months.remote(dataset, names)
    print(f"SF store {dataset}: {len(names) - len(missing)} months cached, labeling {missing}")
    for month in missing:
        wipe_sf_month.remote(dataset, month)  # once, before the shards
        list(label_sf_shard.starmap(
            [(dataset, month, i, shards_per_month, per_game, nodes) for i in range(shards_per_month)]
        ))
        mark_sf_month_complete.remote(dataset, month)


@app.function(
    image=torch_image, gpu="A10G", timeout=60 * 60 * 3, memory=32768, volumes={"/vol": wdl_volume}
)
def train_wdl_run(shard_names: list[str], hidden: int, epochs: int) -> bytes:
    """GPU train on EXACTLY this run's WDL shards -> quantised RFNN bytes.

    `shard_names` is the explicit set of basenames the current run's labelers
    produced (e.g. "samples-2017-01-0.tsv"). We train on those and only those.
    """
    import train_nnue

    wdl_volume.reload()
    # The persistent Volume can hold shards from prior runs with a DIFFERENT
    # month set or shard count, plus v1's one-token shards (samples-0.tsv..) and
    # a stale /vol/data.tsv. Any of those would silently contaminate training, so
    # delete every /vol/samples-*.tsv that is NOT one of this run's expected
    # shards (this removes both v1 one-token names and stale two-token shards).
    expected = {f"/vol/{n}" for n in shard_names}
    for stale in glob.glob("/vol/samples-*.tsv"):
        if stale not in expected:
            pathlib.Path(stale).unlink(missing_ok=True)
    stale_data = pathlib.Path("/vol/data.tsv")
    if stale_data.exists():
        stale_data.unlink()
    # Train on exactly the expected shards; a missing one means a labeler didn't
    # produce it, so fail loudly rather than train on a partial corpus.
    shard_paths = sorted(expected)
    for shard_path in shard_paths:
        assert pathlib.Path(shard_path).exists(), f"expected shard missing: {shard_path}"
    data_path = "/vol/data.tsv"
    with open(data_path, "w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
    out_path = "/vol/net.rfnn"
    model, _ = train_nnue.train(
        data_path, hidden, epochs, batch_size=1024, lr=1e-3, device="cuda", wdl_target=True
    )
    train_nnue.quantize_and_write(model, hidden, out_path)
    with open(out_path, "rb") as handle:
        return handle.read()


@app.function(
    image=torch_image, gpu="A10G", timeout=60 * 60 * 3, memory=32768,
    volumes={"/store": labels_volume},
)
def train_from_store(datasets: list[str], hidden: int, epochs: int, lr: float = 1e-3) -> tuple[bytes, float]:
    """Train (cp mode) on the concatenation of the given store datasets. Read-only
    on the store: it globs + concatenates, NEVER deletes."""
    import os, train_nnue

    labels_volume.reload()
    # Glob every shard across the requested datasets and concatenate them onto the
    # ephemeral container disk (NOT the store) — the store stays append-only.
    shard_paths = sorted(
        p for d in datasets for p in glob.glob(f"/store/sf/{d}/samples-*.tsv")
    )
    assert shard_paths, f"no shards found for datasets {datasets}"
    data_path = "/tmp/data.tsv"  # ephemeral container disk, NOT the store
    with open(data_path, "w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
    model, val_loss = train_nnue.train(
        data_path, hidden, epochs, batch_size=1024, lr=lr, device="cuda", wdl_target=False
    )
    os.makedirs("/store/nets", exist_ok=True)
    out_path = "/store/nets/latest.rfnn"
    train_nnue.quantize_and_write(model, hidden, out_path)
    labels_volume.commit()
    with open(out_path, "rb") as handle:
        return (handle.read(), val_loss)


@app.function(timeout=60 * 60 * 3)
def run_experiment(config: dict) -> dict:
    """Train one net from the store, val-precheck it, gate it vs the champion, and
    return a structured result. Catches its own errors so a bad config becomes an
    `error` row rather than sinking the whole sweep."""
    import math
    import re

    base = {
        "dataset": config["dataset"], "hidden": config["hidden"],
        "epochs": config["epochs"], "lr": config["lr"], "val_loss": "NA",
        "wins": "NA", "draws": "NA", "losses": "NA", "games": "NA",
        "elo": "NA", "decision": "error",
    }
    try:
        net_bytes, val_loss = train_from_store.remote(
            [config["dataset"]], config["hidden"], config["epochs"], config["lr"]
        )
        base["val_loss"] = val_loss
        if math.isnan(val_loss) or val_loss > 0.1:
            return {**base, "decision": "rejected"}
        verdict = gate_ladder_run.remote(
            net_bytes, config["gate_depth"], config["gate_plies"], config["move_time_ms"],
            config["gate_shard_size"], config["chunk_openings"], config["max_openings"],
        )
        m = re.search(r"gate ladder: (\d+)W (\d+)D (\d+)L over (\d+) games .* decision (\w+)", verdict)
        if not m:
            return base  # decision stays "error"
        w, d, l, games = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        total = w + d + l
        if total == 0:
            elo = "NA"
        else:
            score = (w + 0.5 * d) / total
            elo = -800.0 if score <= 0 else 800.0 if score >= 1 else round(-400 * math.log10(1 / score - 1), 1)
        return {**base, "wins": w, "draws": d, "losses": l, "games": games, "elo": elo, "decision": m[5]}
    except Exception as error:  # noqa: BLE001 — any failure becomes one error row
        return {**base, "decision": f"error: {error}"[:120]}


@app.function(volumes={"/store": labels_volume}, timeout=60 * 10)
def append_results(rows: list[str]) -> None:
    import os
    labels_volume.reload()
    os.makedirs("/store/experiments", exist_ok=True)
    path = "/store/experiments/results.tsv"
    header = ("sweep_id\ttimestamp\tdataset\thidden\tepochs\tlr\tval_loss\t"
              "wins\tdraws\tlosses\tgames\telo\tdecision")
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as handle:
        if not exists:
            handle.write(header + "\n")
        for row in rows:
            handle.write(row + "\n")
    labels_volume.commit()


@app.function(volumes={"/store": labels_volume})
def read_results() -> str:
    import os
    labels_volume.reload()
    path = "/store/experiments/results.tsv"
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _chunks(lines, size):
    for start in range(0, len(lines), size):
        yield "\n".join(lines[start:start + size])


@app.local_entrypoint()
def run(
    label_shards: int = 16,
    plies: int = 64,
    label_depth: int = 6,
    hidden: int = 128,
    epochs: int = 40,
    gate_openings: int = 512,
    gate_plies: int = 8,
    gate_depth: int = 5,
    gate_shard_size: int = 32,
):
    # 1. Parallel labeling across distinct seeds.
    shards = label_shard.starmap([(seed, plies, label_depth) for seed in range(1, label_shards + 1)])
    data_text = "".join(shards)
    print(f"labeled {data_text.count(chr(10))} samples across {label_shards} shards")

    # 2. GPU training -> RFNN bytes.
    net_bytes = train_net.remote(data_text, hidden, epochs)
    print(f"trained network: {len(net_bytes)} bytes")

    # 3. Parallel SPRT gate over many openings.
    openings = [line for line in make_openings.remote(gate_openings, gate_plies, 1).splitlines() if line]
    shard_texts = list(_chunks(openings, gate_shard_size))
    results = gate_shard.starmap([(net_bytes, gate_depth, text) for text in shard_texts])
    wins = draws = losses = 0
    for w, d, l in results:
        wins += w
        draws += d
        losses += l
    print(f"gate over {len(openings) * 2} games: {wins}W {draws}D {losses}L")
    print(sprt_verdict.remote(wins, draws, losses))


@app.local_entrypoint()
def mobility_gate(
    gate_openings: int = 2048,
    gate_plies: int = 8,
    movetime_ms: int = 50,
    gate_shard_size: int = 32,
):
    """Self-play SPRT for the mobility eval term, fanned out across containers.

    Plays mobility-on (mobility_scale=100) against mobility-off over many
    openings, color-swapped. Candidate wins => the term helps.

        modal run modal/app.py::mobility_gate
        modal run modal/app.py::mobility_gate --gate-openings 4096 --movetime-ms 50
    """
    openings = [line for line in make_openings.remote(gate_openings, gate_plies, 1).splitlines() if line]
    shard_texts = list(_chunks(openings, gate_shard_size))
    results = mobility_gate_shard.starmap([(movetime_ms, text) for text in shard_texts])
    wins = draws = losses = 0
    for w, d, l in results:
        wins += w
        draws += d
        losses += l
    print(f"mobility gate over {len(openings) * 2} games: {wins}W {draws}D {losses}L")
    print(sprt_verdict.remote(wins, draws, losses))


@app.local_entrypoint()
def spsa_tune(iterations: int = 40, openings: int = 24, movetime_ms: int = 20):
    """Tune the hand-crafted eval weights via SPSA self-play, in one container.

    Prints the tuned EvalParams as an 18-value TSV — feed it to `eval_gate` to
    verify it beats the current default.

        modal run modal/app.py::spsa_tune
        modal run modal/app.py::spsa_tune --iterations 60 --openings 32 --movetime-ms 20
    """
    tuned_tsv = spsa_tune_run.remote(iterations, openings, movetime_ms)
    print("TUNED_EVAL_PARAMS_TSV")
    print(tuned_tsv)


@app.local_entrypoint()
def eval_gate(
    tuned: str,
    gate_openings: int = 2048,
    gate_plies: int = 8,
    movetime_ms: int = 50,
    gate_shard_size: int = 32,
):
    """Powered SPRT of a tuned eval (mobility on) vs today's default, fanned out.

    `tuned` is the 18-value TSV printed by `spsa_tune`.

        modal run modal/app.py::eval_gate --tuned "$(cat tuned.tsv)"
    """
    print(eval_gate_run.remote(tuned, gate_openings, gate_plies, movetime_ms, gate_shard_size))


@app.function(image=rust_image, timeout=60 * 30)
def eval_gate_run(
    tuned: str, gate_openings: int, gate_plies: int, movetime_ms: int, gate_shard_size: int
) -> str:
    """Runs the whole eval gate (fan shards, sum, SPRT) inside one remote function
    and prints the verdict to its log, so a detached run stays retrievable via
    `modal app logs` even if the launching client disconnects."""
    openings = [line for line in make_openings.remote(gate_openings, gate_plies, 1).splitlines() if line]
    shard_texts = list(_chunks(openings, gate_shard_size))
    results = eval_gate_shard.starmap([(movetime_ms, tuned, text) for text in shard_texts])
    wins = draws = losses = 0
    for w, d, l in results:
        wins += w
        draws += d
        losses += l
    summary = f"eval gate over {len(openings) * 2} games: {wins}W {draws}D {losses}L"
    verdict = sprt_verdict.remote(wins, draws, losses)
    out = f"EVAL_GATE_RESULT_BEGIN\n{summary}\n{verdict}\nEVAL_GATE_RESULT_END"
    print(out, flush=True)
    return out


@app.function(image=rust_image, timeout=60 * 30)
def nnue_gate_run(
    net_bytes: bytes,
    gate_depth: int,
    gate_openings: int,
    gate_plies: int,
    gate_shard_size: int,
    move_time_ms: int = 0,
) -> str:
    """Runs the whole NNUE gate (fan shards, sum, SPRT) inside one remote function
    and prints the verdict to its log, so a detached run stays retrievable via
    `modal app logs` even if the launching client disconnects. Gates the candidate
    vs the bundled champion net.

    A truthy `move_time_ms` makes the gate movetime-bounded (callers pass a high
    `gate_depth` so the movetime budget binds first)."""
    openings = [line for line in make_openings.remote(gate_openings, gate_plies, 1).splitlines() if line]
    shard_texts = list(_chunks(openings, gate_shard_size))
    results = gate_shard.starmap([(net_bytes, gate_depth, text, move_time_ms) for text in shard_texts])
    wins = draws = losses = 0
    for w, d, l in results:
        wins += w
        draws += d
        losses += l
    summary = f"nnue gate over {len(openings) * 2} games: {wins}W {draws}D {losses}L"
    verdict = sprt_verdict.remote(wins, draws, losses)
    out = f"NNUE_GATE_RESULT_BEGIN\n{summary}\n{verdict}\nNNUE_GATE_RESULT_END"
    print(out, flush=True)
    return out


@app.function(image=rust_image, timeout=60 * 60)
def gate_ladder_run(
    net_bytes: bytes, gate_depth: int, gate_plies: int, move_time_ms: int,
    gate_shard_size: int, chunk_openings: int = 256, max_openings: int = 8192,
) -> str:
    """Sequential SPRT of a candidate net vs the bundled champion: play openings in
    chunks, check the SPRT on the cumulative W/D/L after each, stop on a decision."""
    wins = draws = losses = 0
    played = 0
    decision = "Continue"
    chunk = 0
    while played < max_openings:
        chunk += 1
        openings = [l for l in make_openings.remote(chunk_openings, gate_plies, chunk).splitlines() if l]
        if not openings:
            break  # no openings (e.g. chunk_openings=0) — never advances `played`; avoid an infinite loop
        shard_texts = list(_chunks(openings, gate_shard_size))
        for w, d, l in gate_shard.starmap(
            [(net_bytes, gate_depth, text, move_time_ms, "champion") for text in shard_texts]
        ):
            wins += w; draws += d; losses += l
        played += len(openings)
        verdict = sprt_verdict.remote(wins, draws, losses)
        # Defensively find the TSV values line: the last line whose final
        # tab-field is a decision token (avoids off-by-one if stderr is empty).
        tokens = {"AcceptH0", "AcceptH1", "Continue"}
        decision = next(
            (ln.split("\t")[-1] for ln in reversed(verdict.splitlines())
             if ln.split("\t")[-1] in tokens),
            "Continue",
        )
        if decision in ("AcceptH0", "AcceptH1"):
            break
    summary = (f"gate ladder: {wins}W {draws}D {losses}L over {played * 2} games "
               f"({played}/{max_openings} openings), decision {decision}")
    out = f"NNUE_LADDER_RESULT_BEGIN\n{summary}\nNNUE_LADDER_RESULT_END"
    print(out, flush=True)
    return out


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
    """Train an NNUE on real Lichess game outcomes (WDL) across many months, then
    gate it movetime-bounded via SPRT.

    Downloads + verifies each pinned month from `assets/nnue/wdl-corpus.toml` into
    a shared volume once, fans out `engine-bench gen-wdl-data` labeling over every
    (month, shard) pair, trains on a GPU with the WDL loss (hidden 512 by default),
    and runs the parallel movetime SPRT gate vs the bundled champion net.

        # short validation (2 months, tiny config — plumbing only):
        modal run modal/app.py::train_wdl --months 2017-01,2017-02 --shards-per-month 2 --per-game 2 --hidden 64 --epochs 2 --gate-openings 64 --gate-shard-size 16 --move-time-ms 50

        # real run (all six months, hidden 512, powered gate):
        modal run modal/app.py::train_wdl --shards-per-month 8 --per-game 12 --hidden 512 --epochs 60 --gate-openings 2048 --gate-shard-size 32 --move-time-ms 50
    """
    corpus = _load_wdl_corpus()
    if months:
        wanted = set(months.split(","))
        corpus = [m for m in corpus if m["name"] in wanted]
        missing = wanted - {m["name"] for m in corpus}
        assert not missing, f"unknown months: {sorted(missing)}"
        assert corpus, "no months selected"
    for m in corpus:
        assert m.get("sha256"), f"month {m['name']} has no pinned sha256 — run sha_probe first"

    # .starmap is lazy; wrap in list() so every export is downloaded + verified
    # before any labeler runs.
    list(prepare_export.starmap([(m["name"], m["url"], m["sha256"]) for m in corpus]))
    label_args = [(m["name"], i, shards_per_month, per_game)
                  for m in corpus for i in range(shards_per_month)]
    shard_names = [f"samples-{m['name']}-{i}.tsv"
                   for m in corpus for i in range(shards_per_month)]
    counts = label_wdl_shard.starmap(label_args)
    print(f"labeled {sum(counts)} WDL samples across {len(label_args)} shards "
          f"({len(corpus)} months x {shards_per_month})")

    net_bytes = train_wdl_run.remote(shard_names, hidden, epochs)
    print(f"trained network: {len(net_bytes)} bytes")

    print(nnue_gate_run.remote(net_bytes, gate_depth, gate_openings, gate_plies,
                               gate_shard_size, move_time_ms))


@app.local_entrypoint()
def train_sf(
    shards_per_month: int = 16,
    per_game: int = 4,
    nodes: int = 100000,
    hidden: int = 512,
    epochs: int = 60,
    chunk_openings: int = 256,
    max_openings: int = 8192,
    gate_plies: int = 8,
    gate_depth: int = 64,        # high; movetime binds first
    gate_shard_size: int = 16,
    move_time_ms: int = 50,
    months: str = "",            # comma-sep subset (e.g. "2017-01,2017-02") for short runs; empty = all
):
    """Train an NNUE on fixed-node Stockfish cp labels across many months, then
    gate it movetime-bounded via a sequential SPRT ladder.

    Downloads + verifies each pinned month from `assets/nnue/wdl-corpus.toml` into
    a shared volume once, fans out `gen-eval-positions | label-sf` labeling over
    every (month, shard) pair into `/vol/sf/`, trains on a GPU in cp mode
    (`wdl_target=False`, hidden 512 by default), then runs a free val-loss
    pre-check and the sequential SPRT gate ladder vs the bundled champion net.

        # short validation (1 month, tiny config, low nodes — plumbing only):
        modal run modal/app.py::train_sf --months 2017-01 --shards-per-month 2 --per-game 2 --nodes 5000 --hidden 64 --epochs 2 --chunk-openings 64 --max-openings 512 --gate-shard-size 16

        # real run (all six months, 100k nodes, hidden 512, powered gate):
        modal run modal/app.py::train_sf --shards-per-month 16 --per-game 4 --nodes 100000 --hidden 512 --epochs 60 --chunk-openings 256 --max-openings 8192 --gate-shard-size 16 --move-time-ms 50
    """
    corpus = _load_wdl_corpus()
    if months:
        wanted = set(months.split(","))
        corpus = [m for m in corpus if m["name"] in wanted]
        missing = wanted - {m["name"] for m in corpus}
        assert not missing, f"unknown months: {sorted(missing)}"
        assert corpus, "no months selected"
    for m in corpus:
        assert m.get("sha256"), f"month {m['name']} has no pinned sha256 — run sha_probe first"

    # .starmap is lazy; wrap in list() so every export is downloaded + verified
    # before any labeler runs.
    list(prepare_export.starmap([(m["name"], m["url"], m["sha256"]) for m in corpus]))
    # Label only the months not already in the persistent store (cache hit skips
    # them), then train read-only over the whole dataset. A fully-labeled corpus
    # goes straight to training with zero Stockfish cost.
    ensure_sf_labels(corpus, per_game, nodes, shards_per_month)
    net_bytes, val_loss = train_from_store.remote([_sf_dataset(nodes, per_game)], hidden, epochs)
    print(f"trained network: {len(net_bytes)} bytes, val_loss {val_loss:.6f}")
    import math
    if math.isnan(val_loss) or val_loss > 0.1:
        print(f"NNUE_LADDER_RESULT_BEGIN\nrejected: val_loss {val_loss} failed the pre-check\nNNUE_LADDER_RESULT_END")
    else:
        print(gate_ladder_run.remote(net_bytes, gate_depth, gate_plies, move_time_ms,
                                     gate_shard_size, chunk_openings, max_openings))


@app.function(volumes={"/store": labels_volume})
def read_net() -> bytes:
    """Read the last store-trained net (/store/nets/latest.rfnn) back off the store."""
    labels_volume.reload()
    with open("/store/nets/latest.rfnn", "rb") as handle:
        return handle.read()


@app.local_entrypoint()
def gate_net(
    gate_openings: int = 2048,
    gate_plies: int = 8,
    gate_depth: int = 64,
    gate_shard_size: int = 16,
    move_time_ms: int = 50,
):
    """Gate the store-trained net (/store/nets/latest.rfnn) vs the bundled champion net.

    Re-gates the already-trained store net without re-labeling or re-training — use
    to re-measure a saved net or after a gate that timed out. Smaller default
    gate_shard_size (16) keeps each movetime shard well under the per-container
    timeout for a slower hidden-512 net.

        modal run modal/app.py::gate_net
        modal run modal/app.py::gate_net --gate-openings 2048 --move-time-ms 50
    """
    net_bytes = read_net.remote()
    print(f"loaded net: {len(net_bytes)} bytes")
    print(nnue_gate_run.remote(net_bytes, gate_depth, gate_openings, gate_plies,
                               gate_shard_size, move_time_ms))


@app.local_entrypoint()
def gate_ladder(gate_depth: int = 64, gate_plies: int = 8, move_time_ms: int = 50,
                gate_shard_size: int = 16, chunk_openings: int = 256, max_openings: int = 8192):
    """Re-run the sequential SPRT gate ladder on the stored net vs the bundled
    champion, without retraining.

        modal run modal/app.py::gate_ladder
        modal run modal/app.py::gate_ladder --chunk-openings 128 --max-openings 1024
    """
    net_bytes = read_net.remote()
    print(f"loaded net: {len(net_bytes)} bytes")
    print(gate_ladder_run.remote(net_bytes, gate_depth, gate_plies, move_time_ms,
                                 gate_shard_size, chunk_openings, max_openings))


@app.function(image=rust_image, timeout=60 * 60)
def sha_probe_one(name: str, url: str) -> str:
    """Download one corpus month and print its sha256 (for pinning wdl-corpus.toml)."""
    path = f"/tmp/{name}.zst"
    subprocess.run(["curl", "-L", "-o", path, url], check=True)
    out = subprocess.run(["sha256sum", path], capture_output=True, text=True, check=True)
    digest = out.stdout.split()[0]
    print(f"SHA_PROBE {name} {digest}", flush=True)
    return f"{name} {digest}"


@app.local_entrypoint()
def sha_probe():
    """Print sha256 for EVERY corpus month so 2017-01 cross-checks its pinned SHA.

        modal run modal/app.py::sha_probe
    """
    months = _load_wdl_corpus()
    for line in sha_probe_one.starmap([(m["name"], m["url"]) for m in months]):
        print(line)
