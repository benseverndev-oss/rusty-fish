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

import pathlib
import subprocess
import tempfile

import modal

REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
BIN = "/repo/target/release/engine-bench"

# Image A: builds the engine-bench release binary from the repo source.
rust_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "build-essential", "pkg-config")
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
