"""Manifest-addressed Modal training and screening for v1 NNUE candidates."""

import hashlib
import json
import pathlib
import subprocess
import tempfile

import modal

REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
BIN = "/repo/target/release/engine-bench"
SCHEMA = "v1"
INPUT_DIMENSION = 768
LEARNING_RATE = 1e-3
STOCKFISH_PACKAGE = "stockfish=15.1-4"
REMOTE_STOCKFISH = "/usr/games/stockfish"

rust_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "build-essential", "pkg-config", STOCKFISH_PACKAGE)
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable")
    .add_local_dir(REPO_ROOT, remote_path="/repo", copy=True)
    .run_commands("cd /repo && $HOME/.cargo/bin/cargo build --release -p engine-bench")
)
torch_image = modal.Image.debian_slim().pip_install("torch").add_local_file(
    str(pathlib.Path(__file__).parent / "train_nnue.py"), "/root/train_nnue.py"
)
app = modal.App("rusty-fish-nnue")
artifacts = modal.Volume.from_name("rusty-fish-nnue-artifacts", create_if_missing=True)


def _sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def _artifact_path(run_id: str, stage: str, input_hash: str, name: str = "artifact") -> str:
    return f"/artifacts/runs/{run_id}/{stage}-{input_hash}/{name}"


def _write_artifact(run_id: str, stage: str, input_text: str, contents: str | bytes,
                    name: str = "artifact") -> str:
    """Atomically make an immutable, input-addressed stage artifact."""
    path = _artifact_path(run_id, stage, _sha256(input_text), name)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = contents.encode("utf-8") if isinstance(contents, str) else contents
    if target.exists():
        if target.read_bytes() != data:
            raise RuntimeError(f"refusing to overwrite {path} with different input result")
        return path
    target.write_bytes(data)
    artifacts.commit()
    return path


def _corpus_payload(directory: str) -> str:
    return json.dumps({split: pathlib.Path(directory, f"{split}.tsv").read_text(encoding="utf-8")
                       for split in ("train", "validation", "test")}, sort_keys=True)


def _remote_stockfish_config(stockfish_config_text: str) -> str:
    """Bind a calibrated config to the pinned engine installed in this image."""
    fields = {}
    for line in stockfish_config_text.splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            fields[key] = value
    actual_hash = _sha256(pathlib.Path(REMOTE_STOCKFISH).read_bytes())
    if fields.get("binary_sha256") != actual_hash:
        raise ValueError(
            "Stockfish config binary_sha256 does not match the pinned remote Stockfish binary"
        )
    lines = []
    for line in stockfish_config_text.splitlines():
        lines.append(f"binary\t{REMOTE_STOCKFISH}" if line.startswith("binary\t") else line)
    return "\n".join(lines) + "\n"


@app.function(image=rust_image, volumes={"/artifacts": artifacts}, timeout=60 * 60)
def build_corpus(run_id: str, smoke: bool) -> tuple[str, str]:
    args = [BIN, "dataset-build", run_id, "/tmp/corpus", "400000", "400000", "200000", "1"]
    if smoke:
        # dataset-build deliberately caps smoke totals; preserve the same composition ratio.
        args[4:7] = ["400", "400", "200"]
        args.append("--smoke")
    subprocess.run(args, check=True)
    manifest = pathlib.Path("/tmp/corpus/manifest.tsv").read_text(encoding="utf-8")
    positions = _corpus_payload("/tmp/corpus")
    corpus_input = json.dumps({"run_id": run_id, "smoke": smoke, "counts": args[4:7], "seed": args[7]}, sort_keys=True)
    _write_artifact(run_id, "corpus-manifest", corpus_input, manifest)
    _write_artifact(run_id, "corpus-positions", manifest, positions)
    return manifest, positions


@app.function(image=rust_image, volumes={"/artifacts": artifacts}, timeout=60 * 60)
def label_manifest(run_id: str, manifest_text: str, stockfish_config_text: str) -> dict[str, str]:
    """Label every immutable manifest split; return schema-tagged training rows."""
    # The corpus is recovered by its manifest digest, never by hidden process state.
    artifacts.reload()
    positions_path = _artifact_path(run_id, "corpus-positions", _sha256(manifest_text))
    payload = json.loads(pathlib.Path(positions_path).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        manifest_path = root / "manifest.tsv"
        manifest_path.write_text(manifest_text, encoding="utf-8")
        for split, content in payload.items():
            (root / f"{split}.tsv").write_text(content, encoding="utf-8")
        config = root / "stockfish-config.tsv"
        config.write_text(_remote_stockfish_config(stockfish_config_text), encoding="utf-8")
        result = {}
        for split in ("train", "validation", "test"):
            output = root / f"{split}-labels.tsv"
            subprocess.run([BIN, "stockfish-label", str(manifest_path), split, str(config), str(output)], check=True)
            result[split] = "".join(f"{SCHEMA}\t{line}\n" for line in output.read_text(encoding="utf-8").splitlines())
    _write_artifact(run_id, "labels", manifest_text + stockfish_config_text, json.dumps(result, sort_keys=True))
    return result


@app.function(image=torch_image, gpu="A10G", volumes={"/artifacts": artifacts}, timeout=60 * 60)
def train_net(data_text: str, schema: str, input_dimension: int, hidden: int, epochs: int,
              run_id: str, seed: int = 1) -> bytes:
    import train_nnue

    splits = json.loads(data_text)
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        paths = {name: root / f"{name}.tsv" for name in ("train", "validation", "test")}
        for name, path in paths.items():
            path.write_text(splits[name], encoding="utf-8")
        model = train_nnue.train(
            str(paths["train"]), hidden, epochs, 1024, LEARNING_RATE, "cuda",
            schema, input_dimension, seed,
        )
        train_loss = model.train_wdl_loss
        validation_loss = train_nnue.wdl_loss(model, str(paths["validation"]), "cuda", schema, input_dimension)
        test_loss = train_nnue.wdl_loss(model, str(paths["test"]), "cuda", schema, input_dimension)
        quantization_error = train_nnue.quantization_max_error_cp(
            model, str(paths["test"]), "cuda", schema, input_dimension,
        )
        net = root / "net.rfnn"
        train_nnue.quantize_and_write(model, hidden, str(net))
        net_bytes = net.read_bytes()
    report = {
        "train_wdl_loss": train_loss, "validation_wdl_loss": validation_loss,
        "test_wdl_loss": test_loss,
        "model_sha256": _sha256(net_bytes), "input_dimension": input_dimension,
        "schema": schema, "epochs": epochs, "learning_rate": LEARNING_RATE,
        "quantization_max_error_cp": quantization_error, "seed": seed,
    }
    input_hash = _sha256(data_text + f"{schema}:{input_dimension}:{hidden}:{epochs}:{seed}")
    _write_artifact(run_id, f"net-{hidden}", input_hash, net_bytes, "net.rfnn")
    _write_artifact(run_id, f"report-{hidden}", input_hash, json.dumps(report, sort_keys=True, indent=2), "report.json")
    return net_bytes


@app.function(image=rust_image, timeout=60 * 60)
def run_screen(net_bytes: bytes, openings_per_shard: int = 16) -> tuple[int, int, int]:
    """Run exactly 12 deterministic, bounded opening shards (384 games)."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        net = root / "net.rfnn"
        net.write_bytes(net_bytes)
        openings = subprocess.run([BIN, "gen-openings", str(12 * openings_per_shard), "8", "1"], capture_output=True, text=True, check=True).stdout.splitlines()
        totals = [0, 0, 0]
        for shard in range(12):
            opening_file = root / f"openings-{shard}.txt"
            opening_file.write_text("\n".join(openings[shard * openings_per_shard:(shard + 1) * openings_per_shard]), encoding="utf-8")
            result = subprocess.run([BIN, "gate-file", str(net), "5", str(opening_file), "100"], capture_output=True, text=True, check=True)
            for index, value in enumerate(result.stdout.strip().split("\t")):
                totals[index] += int(value)
    return tuple(totals)


def promotes(report: dict, control: dict, screen: tuple[int, int, int]) -> bool:
    return (
        report["validation_wdl_loss"] <= control["validation_wdl_loss"] * 0.98
        and report["test_wdl_loss"] <= control["test_wdl_loss"] * 0.99
        and report["quantization_max_error_cp"] <= 32
        and screen[0] + 0.5 * screen[1] >= 192
    )


@app.function(volumes={"/artifacts": artifacts})
def read_report(run_id: str, width: int, input_hash: str) -> dict:
    artifacts.reload()
    path = _artifact_path(run_id, f"report-{width}", input_hash, "report.json")
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


@app.local_entrypoint()
def run(run_id: str = "smoke-v1", smoke: bool = False, schema: str = SCHEMA,
        widths: str = "128,256,512", epochs: int = 40, stockfish_config: str = "stockfish-config.tsv",
        seed: int = 1):
    if schema != SCHEMA:
        raise ValueError(f"unsupported schema: {schema}")
    config_text = pathlib.Path(stockfish_config).read_text(encoding="utf-8")
    manifest, _ = build_corpus.remote(run_id, smoke)
    labels = label_manifest.remote(run_id, manifest, config_text)
    data_text = json.dumps(labels, sort_keys=True)
    reports = []
    for width in (int(value) for value in widths.split(",") if value):
        net = train_net.remote(data_text, schema, INPUT_DIMENSION, width, epochs, run_id, seed)
        input_hash = _sha256(data_text + f"{schema}:{INPUT_DIMENSION}:{width}:{epochs}:{seed}")
        report = read_report.remote(run_id, width, input_hash)
        reports.append((width, net, report))
        print(f"manifest SHA {_sha256(manifest)}; seed {seed}; width {width}: {json.dumps(report, sort_keys=True)}")
    control = reports[0][2]
    for width, net, report in reports:
        offline_ok = (
            report["validation_wdl_loss"] <= control["validation_wdl_loss"] * 0.98
            and report["test_wdl_loss"] <= control["test_wdl_loss"] * 0.99
            and report["quantization_max_error_cp"] <= 32
        )
        if offline_ok:
            screen = run_screen.remote(net)
            print(f"width {width} screen (384 games): {screen[0]}W {screen[1]}D {screen[2]}L")
            print(f"width {width} promotion: {'promoted' if promotes(report, control, screen) else 'rejected'}")
