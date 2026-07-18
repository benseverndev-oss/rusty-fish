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

This is scaffolding: it needs your own Modal account/token and cannot be run
from the agent sandbox. See modal/README.md.

    modal run modal/app.py                 # defaults
    modal run modal/app.py --hidden 256 --epochs 60 --gate-openings 2048
"""

import glob
import pathlib
import subprocess
import tempfile

import modal

REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
BIN = "/repo/target/release/engine-bench"

# Pinned Lichess standard-rated export (matches assets/opening-book/manifest.toml).
# The WDL pipeline labels game outcomes from this file.
WDL_EXPORT_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_2017-01.pgn.zst"
)
WDL_EXPORT_SHA256 = "d1236dcd954089aee162c7b0d82f51162f7c912882343d47b77ba5c0e05512f6"

# Image A: builds the engine-bench release binary from the repo source.
rust_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "build-essential", "pkg-config", "zstd")
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


@app.function(image=rust_image)
def gate_shard(net_bytes: bytes, depth: int, openings_text: str) -> tuple[int, int, int]:
    """Play the NNUE candidate vs the hand-crafted baseline over one opening shard."""
    with tempfile.TemporaryDirectory() as directory:
        net_path = f"{directory}/net.rfnn"
        openings_path = f"{directory}/openings.txt"
        with open(net_path, "wb") as handle:
            handle.write(net_bytes)
        with open(openings_path, "w", encoding="utf-8") as handle:
            handle.write(openings_text)
        result = subprocess.run(
            [BIN, "gate-file", net_path, str(depth), openings_path],
            capture_output=True, text=True, check=True,
        )
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
        model = train_nnue.train(data_path, hidden, epochs, batch_size=1024, lr=1e-3, device="cuda")
        train_nnue.quantize_and_write(model, hidden, out_path)
        with open(out_path, "rb") as handle:
            return handle.read()


@app.function(image=rust_image, volumes={"/vol": wdl_volume}, timeout=60 * 60)
def prepare_export() -> None:
    """Download + verify the pinned Lichess export into the shared volume, once.

    Idempotent: if the file is already present (a prior run committed it) this
    returns early and skips the multi-GB re-download. The SHA-256 check fails the
    function on any mismatch, so a corrupt/partial download never reaches labeling.
    """
    export_path = "/vol/export.pgn.zst"
    if pathlib.Path(export_path).exists():
        print(f"export already present at {export_path}, skipping download", flush=True)
        return
    subprocess.run(["curl", "-L", "-o", export_path, WDL_EXPORT_URL], check=True)
    # sha256sum --check reads "<hash>  <path>" and exits non-zero on mismatch,
    # which check=True turns into a raised CalledProcessError (function fails).
    checkfile = f"{WDL_EXPORT_SHA256}  {export_path}\n"
    subprocess.run(["sha256sum", "--check"], input=checkfile, text=True, check=True)
    wdl_volume.commit()
    print(f"downloaded + verified {export_path}", flush=True)


@app.function(image=rust_image, volumes={"/vol": wdl_volume}, timeout=60 * 60)
def label_wdl_shard(i: int, n: int, per_game: int) -> int:
    """One WDL labeling shard: stream the export through gen-wdl-data.

    Runs the pipe under bash with pipefail so a zstdcat failure (e.g. truncated
    export) fails the shard instead of silently yielding an empty TSV.
    """
    cmd = (
        f"set -euo pipefail; zstdcat /vol/export.pgn.zst | "
        f"{BIN} gen-wdl-data - --shard {i}/{n} --per-game {per_game} > /vol/samples-{i}.tsv"
    )
    subprocess.run(["bash", "-c", cmd], check=True)
    wdl_volume.commit()
    with open(f"/vol/samples-{i}.tsv", "r", encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    return count


@app.function(image=torch_image, gpu="A10G", timeout=60 * 60 * 2, volumes={"/vol": wdl_volume})
def train_wdl_run(hidden: int, epochs: int) -> bytes:
    """GPU train on the WDL outcome labels (all shards) -> quantised RFNN bytes."""
    import train_nnue

    wdl_volume.reload()
    shard_paths = sorted(glob.glob("/vol/samples-*.tsv"))
    data_path = "/vol/data.tsv"
    with open(data_path, "w", encoding="utf-8") as out:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
    out_path = "/vol/net.rfnn"
    model = train_nnue.train(
        data_path, hidden, epochs, batch_size=1024, lr=1e-3, device="cuda", wdl_target=True
    )
    train_nnue.quantize_and_write(model, hidden, out_path)
    with open(out_path, "rb") as handle:
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
) -> str:
    """Runs the whole NNUE gate (fan shards, sum, SPRT) inside one remote function
    and prints the verdict to its log, so a detached run stays retrievable via
    `modal app logs` even if the launching client disconnects."""
    openings = [line for line in make_openings.remote(gate_openings, gate_plies, 1).splitlines() if line]
    shard_texts = list(_chunks(openings, gate_shard_size))
    results = gate_shard.starmap([(net_bytes, gate_depth, text) for text in shard_texts])
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


@app.local_entrypoint()
def train_wdl(
    label_shards: int = 16,
    per_game: int = 12,
    hidden: int = 256,
    epochs: int = 40,
    gate_openings: int = 2048,
    gate_plies: int = 8,
    gate_depth: int = 5,
    gate_shard_size: int = 32,
):
    """Train an NNUE on real Lichess game outcomes (WDL), then gate it via SPRT.

    Downloads the pinned Lichess export once into a shared volume, fans out
    `engine-bench gen-wdl-data` shards to label game outcomes, trains on a GPU
    with the WDL loss, and runs the parallel SPRT gate vs the hand-crafted
    baseline.

        modal run modal/app.py::train_wdl --label-shards 2 --per-game 2 --epochs 2 --gate-openings 64
        modal run modal/app.py::train_wdl --label-shards 16 --per-game 12 --hidden 256 --epochs 40 --gate-openings 2048
    """
    prepare_export.remote()
    counts = label_wdl_shard.starmap([(i, label_shards, per_game) for i in range(label_shards)])
    total = sum(counts)
    print(f"labeled {total} WDL samples across {label_shards} shards")

    net_bytes = train_wdl_run.remote(hidden, epochs)
    print(f"trained network: {len(net_bytes)} bytes")

    print(nnue_gate_run.remote(net_bytes, gate_depth, gate_openings, gate_plies, gate_shard_size))
