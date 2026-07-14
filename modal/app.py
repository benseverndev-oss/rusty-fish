"""Manifest-addressed Modal training, HalfKA screening, and promotion gates."""

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
HALFKA_SCHEMA = "halfka-v2-64"
HALFKA_INPUT_DIMENSION = 64 * 2 * 5 * 64
LEARNING_RATE = 1e-3
STOCKFISH_18_URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar"
STOCKFISH_18_ARCHIVE_SHA256 = "5c6f38b02a4da5f3ffe763f27da6c3e743eebefd92b50cb3661623b96696adff"
REMOTE_STOCKFISH = "/opt/stockfish/stockfish"

rust_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands(
        f"curl --fail --location --retry 3 --output /tmp/stockfish-18.tar {STOCKFISH_18_URL}",
        f"echo '{STOCKFISH_18_ARCHIVE_SHA256}  /tmp/stockfish-18.tar' | sha256sum --check --status",
        "mkdir -p /opt/stockfish && tar --extract --file /tmp/stockfish-18.tar --directory /opt/stockfish",
        "stockfish=$(find /opt/stockfish -type f -name stockfish-ubuntu-x86-64 -print -quit) && test -n \"$stockfish\" && chmod +x \"$stockfish\" && ln -s \"$stockfish\" /opt/stockfish/stockfish",
    )
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


def _sprt_fields(sprt_text: str) -> dict[str, str | None]:
    """Decode the engine's one-row SPRT TSV without losing its evidence fields."""
    lines = [line for line in sprt_text.splitlines() if line]
    if len(lines) != 2:
        return {"raw": sprt_text, "elo_estimate": None, "llr": None, "decision": None}
    names, values = lines
    fields = dict(zip(names.split("\t"), values.split("\t"), strict=True))
    return {
        "raw": sprt_text,
        "elo_estimate": fields.get("elo_estimate") or None,
        "llr": fields.get("llr") or None,
        "decision": fields.get("decision") or None,
    }


def _gate_outcome(*, stage: str, run_id: str, net_bytes: bytes, candidate_report: dict,
                  control_report: dict, manifest_sha256: str, config_sha256: str,
                  wins: int, draws: int, losses: int, sprt_text: str,
                  promotion_decision: str) -> tuple[dict, dict]:
    """Build the complete, content-addressed evidence record for a gate result."""
    inputs = {
        "stage": stage,
        "run_id": run_id,
        "network_sha256": _sha256(net_bytes),
        "candidate_report_sha256": _sha256(json.dumps(candidate_report, sort_keys=True)),
        "control_report_sha256": _sha256(json.dumps(control_report, sort_keys=True)),
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
    }
    sprt = _sprt_fields(sprt_text)
    outcome = {
        "run_id": run_id,
        "stage": stage,
        "inputs": inputs,
        "wdl": {"wins": wins, "draws": draws, "losses": losses,
                "games": wins + draws + losses},
        "elo_estimate": sprt["elo_estimate"],
        "sprt": sprt,
        "promotion_decision": promotion_decision,
    }
    return inputs, outcome


def _write_gate_outcome(**kwargs) -> dict:
    """Persist one immutable candidate outcome report and return its contents."""
    inputs, outcome = _gate_outcome(**kwargs)
    _write_artifact(
        kwargs["run_id"], f"{kwargs['stage']}-outcome",
        json.dumps(inputs, sort_keys=True), json.dumps(outcome, sort_keys=True, indent=2),
        "report.json",
    )
    return outcome


def _parse_screen_shard_wdl(output: str) -> tuple[int, int, int]:
    """Require a complete nonnegative W/D/L result from one screen shard."""
    fields = output.strip().split("\t")
    if len(fields) != 3 or any(not field.isascii() or not field.isdigit() for field in fields):
        raise ValueError("screen shard must output exactly three nonnegative integer W/D/L fields")
    return tuple(int(field) for field in fields)


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
def label_manifest(run_id: str, manifest_text: str, stockfish_config_text: str,
                   schema: str = SCHEMA) -> dict[str, str]:
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
            subprocess.run([BIN, "stockfish-label", str(manifest_path), split, str(config), str(output), schema], check=True)
            result[split] = output.read_text(encoding="utf-8")
    _write_artifact(run_id, f"labels-{schema}", manifest_text + stockfish_config_text + schema,
                    json.dumps(result, sort_keys=True))
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
        buckets = 64 if schema == HALFKA_SCHEMA else 0
        model = train_nnue.train(str(paths["train"]), schema, input_dimension, hidden,
                                 epochs, 1024, LEARNING_RATE, "cuda", seed)
        train_loss = model.train_wdl_loss
        validation_loss = train_nnue.wdl_loss(model, str(paths["validation"]), "cuda", schema, input_dimension)
        test_loss = train_nnue.wdl_loss(model, str(paths["test"]), "cuda", schema, input_dimension)
        quantization_error = train_nnue.quantization_max_error_cp(
            model, str(paths["test"]), "cuda", schema, input_dimension,
        )
        net = root / "net.rfnn"
        train_nnue.quantize_and_write(model, schema, input_dimension, buckets, hidden, str(net))
        net_bytes = net.read_bytes()
    report = {
        "train_wdl_loss": train_loss, "validation_wdl_loss": validation_loss,
        "test_wdl_loss": test_loss,
        "model_sha256": _sha256(net_bytes), "input_dimension": input_dimension,
        "schema": schema, "buckets": buckets, "hidden": hidden,
        "epochs": epochs, "learning_rate": LEARNING_RATE,
        "quantization_max_error_cp": quantization_error, "seed": seed,
    }
    input_hash = _sha256(data_text + f"{schema}:{input_dimension}:{hidden}:{epochs}:{seed}")
    _write_artifact(run_id, f"net-{hidden}", input_hash, net_bytes, "net.rfnn")
    _write_artifact(run_id, f"report-{hidden}", input_hash, json.dumps(report, sort_keys=True, indent=2), "report.json")
    return net_bytes


@app.function(image=rust_image, volumes={"/artifacts": artifacts}, timeout=60 * 60)
def run_screen(net_bytes: bytes, run_id: str, candidate_report: dict, control_report: dict,
               manifest_sha256: str, config_sha256: str,
               openings_per_shard: int = 16) -> dict:
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
            for index, value in enumerate(_parse_screen_shard_wdl(result.stdout)):
                totals[index] += value
    score = tuple(totals)
    if sum(score) != 384:
        raise RuntimeError(f"{run_id}: screen returned {sum(score)} games, not 384")
    sprt_text = subprocess.run(
        [BIN, "sprt", *(str(value) for value in score)], capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    return _write_gate_outcome(
        stage="screen", run_id=run_id, net_bytes=net_bytes,
        candidate_report=candidate_report, control_report=control_report,
        manifest_sha256=manifest_sha256, config_sha256=config_sha256,
        wins=score[0], draws=score[1], losses=score[2], sprt_text=sprt_text,
        promotion_decision=("screen-passed" if score[0] + 0.5 * score[1] >= 192
                            else "screen-rejected"),
    )


@app.function(image=rust_image, timeout=60 * 60)
def rust_parity_check(net_bytes: bytes) -> None:
    """Load an exported candidate in Rust before spending screen resources."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        net = root / "net.rfnn"
        net.write_bytes(net_bytes)
        openings = root / "opening.txt"
        openings.write_text(subprocess.run(
            [BIN, "gen-openings", "1", "8", "1"], capture_output=True,
            text=True, check=True,
        ).stdout, encoding="utf-8")
        subprocess.run([BIN, "gate-file", str(net), "4", str(openings), "100"], check=True)


@app.function(image=rust_image, volumes={"/artifacts": artifacts}, timeout=6 * 60 * 60)
def run_full_gate(net_bytes: bytes, run_id: str, candidate_report: dict, control_report: dict,
                  manifest_sha256: str, config_sha256: str) -> dict:
    """Run the 2,304-game, depth-4 bounded promotion gate for one candidate."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        net = root / "net.rfnn"
        net.write_bytes(net_bytes)
        openings = subprocess.run([BIN, "gen-openings", "1152", "8", "1"], capture_output=True,
                                  text=True, check=True).stdout.splitlines()
        totals = [0, 0, 0]
        for shard in range(12):
            opening_file = root / f"openings-{shard}.txt"
            opening_file.write_text("\n".join(openings[shard * 96:(shard + 1) * 96]), encoding="utf-8")
            result = subprocess.run([BIN, "gate-file", str(net), "4", str(opening_file), "100"],
                                    capture_output=True, text=True, check=True)
            for index, value in enumerate(result.stdout.strip().split("\t")):
                totals[index] += int(value)
        verdict = subprocess.run([BIN, "sprt", *(str(value) for value in totals)],
                                 capture_output=True, text=True, check=True).stdout.strip()
    if sum(totals) != 2304:
        raise RuntimeError(f"{run_id}: full gate returned {sum(totals)} games, not 2304")
    sprt = _sprt_fields(verdict)
    return _write_gate_outcome(
        stage="full-gate", run_id=run_id, net_bytes=net_bytes,
        candidate_report=candidate_report, control_report=control_report,
        manifest_sha256=manifest_sha256, config_sha256=config_sha256,
        wins=totals[0], draws=totals[1], losses=totals[2], sprt_text=verdict,
        promotion_decision=("adoption-design-required" if sprt["decision"] == "AcceptH1"
                            else "not-promoted"),
    )


def promotes(report: dict, control: dict, screen: tuple[int, int, int]) -> bool:
    return (
        report["validation_wdl_loss"] <= control["validation_wdl_loss"] * 0.98
        and report["test_wdl_loss"] <= control["test_wdl_loss"] * 0.99
        and report["quantization_max_error_cp"] <= 32
        and screen[0] + 0.5 * screen[1] >= 192
    )


def selected_halfka_widths(capacity_selection_report: dict[str, int]) -> list[int]:
    """Select the deterministic HalfKA ladder from the promoted v1 width."""
    selected = capacity_selection_report.get("selected_width", capacity_selection_report.get("width"))
    try:
        return {128: [128, 256], 256: [256, 512], 512: [512]}[int(selected)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("capacity selection report must name selected_width 128, 256, or 512") from error


def _run_ordered_ladder(widths, train_candidate, evaluate_candidate, *, stop_on_failure: bool):
    """Train and decide one width at a time; HalfKA cannot pre-start later widths."""
    results = []
    for width in widths:
        candidate = train_candidate(width)
        promoted = evaluate_candidate(width, candidate)
        results.append((width, promoted))
        if stop_on_failure and not promoted:
            break
    return results


@app.function(volumes={"/artifacts": artifacts})
def read_report(run_id: str, width: int, input_hash: str) -> dict:
    artifacts.reload()
    path = _artifact_path(run_id, f"report-{width}", input_hash, "report.json")
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


@app.local_entrypoint()
def run(run_id: str = "smoke-v1", smoke: bool = False, schema: str = SCHEMA,
        widths: str = "128,256,512", epochs: int = 40, stockfish_config: str = "stockfish-config.tsv",
        seed: int = 1, capacity_selection: str | None = None):
    if schema not in (SCHEMA, HALFKA_SCHEMA):
        raise ValueError(f"unsupported schema: {schema}")
    config_text = pathlib.Path(stockfish_config).read_text(encoding="utf-8")
    manifest, _ = build_corpus.remote(run_id, smoke)
    control_labels = label_manifest.remote(run_id, manifest, config_text, SCHEMA)
    control_data = json.dumps(control_labels, sort_keys=True)
    labels = control_labels if schema == SCHEMA else label_manifest.remote(run_id, manifest, config_text, schema)
    data_text = json.dumps(labels, sort_keys=True)
    input_dimension = INPUT_DIMENSION if schema == SCHEMA else HALFKA_INPUT_DIMENSION
    if schema == HALFKA_SCHEMA:
        if not capacity_selection:
            raise ValueError("--capacity-selection is required for HalfKA")
        selected = json.loads(pathlib.Path(capacity_selection).read_text(encoding="utf-8"))
        candidate_widths = selected_halfka_widths(selected)
    else:
        candidate_widths = [int(value) for value in widths.split(",") if value]
    control_net = train_net.remote(control_data, SCHEMA, INPUT_DIMENSION, 128, epochs, run_id, seed)
    control_hash = _sha256(control_data + f"{SCHEMA}:{INPUT_DIMENSION}:128:{epochs}:{seed}")
    control = read_report.remote(run_id, 128, control_hash)
    manifest_sha256 = _sha256(manifest)
    config_sha256 = _sha256(config_text)
    def train_candidate(width: int) -> tuple[bytes, dict]:
        net = control_net if schema == SCHEMA and width == 128 else train_net.remote(
            data_text, schema, input_dimension, width, epochs, run_id, seed)
        input_hash = _sha256(data_text + f"{schema}:{input_dimension}:{width}:{epochs}:{seed}")
        report = read_report.remote(run_id, width, input_hash)
        print(f"manifest SHA {manifest_sha256}; seed {seed}; width {width}: {json.dumps(report, sort_keys=True)}")
        return net, report

    def evaluate_candidate(width: int, candidate: tuple[bytes, dict]) -> bool:
        net, report = candidate
        offline_ok = (
            report["validation_wdl_loss"] <= control["validation_wdl_loss"] * 0.98
            and report["test_wdl_loss"] <= control["test_wdl_loss"] * 0.99
            and report["quantization_max_error_cp"] <= 32
        )
        if not offline_ok:
            print(f"width {width} promotion: rejected before screen")
            return False
        rust_parity_check.remote(net)
        screen = run_screen.remote(net, run_id, report, control, manifest_sha256, config_sha256)
        screen_wdl = screen["wdl"]
        print(f"width {width} screen (384 games): {screen_wdl['wins']}W {screen_wdl['draws']}D {screen_wdl['losses']}L")
        screen_score = (screen_wdl["wins"], screen_wdl["draws"], screen_wdl["losses"])
        if not promotes(report, control, screen_score):
            print(f"width {width} promotion: rejected")
            return False
        verdict = run_full_gate.remote(net, run_id, report, control, manifest_sha256, config_sha256)
        assert verdict["wdl"]["games"] == 2304
        print(f"width {width} full gate: {json.dumps(verdict, sort_keys=True)}")
        return verdict["promotion_decision"] == "adoption-design-required"

    _run_ordered_ladder(
        candidate_widths, train_candidate, evaluate_candidate,
        stop_on_failure=(schema == HALFKA_SCHEMA),
    )
