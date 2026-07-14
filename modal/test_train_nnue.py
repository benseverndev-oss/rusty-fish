import json
import pathlib
import struct
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_nnue import _load_samples


def test_rust_image_uses_the_pinned_debian_bookworm_slim_digest():
    import app

    assert app.RUST_IMAGE_BASE == (
        "debian@sha256:60eac759739651111db372c07be67863818726f754804b8707c90979bda511df"
    )


def test_stockfish_image_install_links_to_a_non_colliding_executable_path():
    import app

    assert app.REMOTE_STOCKFISH == "/opt/stockfish/stockfish-bin"
    assert app.STOCKFISH_INSTALL_COMMAND.endswith('ln -s "$stockfish" /opt/stockfish/stockfish-bin')


def test_gate_outcome_records_addressed_inputs_wdl_and_sprt_verdict():
    import app

    sprt = (
        "engine_version\twins\tdraws\tlosses\telo_estimate\telo0\telo1\talpha\tbeta\tllr\tlower_bound\tupper_bound\tdecision\n"
        "0.1\t120\t80\t40\t35.50\t0\t5\t0.05\t0.05\t3.2\t-2.9\t2.9\tAcceptH1\n"
    )
    inputs, outcome = app._gate_outcome(
        stage="full-gate",
        run_id="halfka-128",
        net_bytes=b"candidate-net",
        candidate_report={"model_sha256": "candidate"},
        control_report={"model_sha256": "control"},
        manifest_sha256="manifest",
        config_sha256="config",
        wins=120,
        draws=80,
        losses=40,
        sprt_text=sprt,
        promotion_decision="AcceptH1",
    )

    assert inputs["network_sha256"] == app._sha256(b"candidate-net")
    assert inputs["manifest_sha256"] == "manifest"
    assert outcome["run_id"] == "halfka-128"
    assert outcome["wdl"] == {"wins": 120, "draws": 80, "losses": 40, "games": 240}
    assert outcome["elo_estimate"] == "35.50"
    assert outcome["sprt"]["decision"] == "AcceptH1"
    assert outcome["promotion_decision"] == "AcceptH1"


def test_outcome_artifact_rejects_conflicting_result_at_same_input_address(tmp_path, monkeypatch):
    import app

    class FakeVolume:
        def commit(self):
            pass

    monkeypatch.setattr(app, "artifacts", FakeVolume())
    monkeypatch.setattr(
        app, "_artifact_path",
        lambda _run_id, _stage, _input_hash, name="artifact": str(tmp_path / name),
    )
    path = app._write_artifact("run", "screen", "fixed-input", '{"wins": 1}', "report.json")
    assert pathlib.Path(path).read_text(encoding="utf-8") == '{"wins": 1}'
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        app._write_artifact("run", "screen", "fixed-input", '{"wins": 2}', "report.json")


def test_halfka_ladder_does_not_train_later_width_after_failed_evaluation():
    import app

    trained = []

    def train(width):
        trained.append(width)
        return width

    evaluated = []

    def evaluate(width, net):
        evaluated.append((width, net))
        return False

    assert app._run_ordered_ladder([128, 256], train, evaluate, stop_on_failure=True) == [(128, False)]
    assert trained == [128]
    assert evaluated == [(128, 128)]


@pytest.mark.parametrize("output", ["16\t0", "16\t0\t0\t0", "16\t-1\t1", "16\tno\t0"])
def test_screen_shard_parser_rejects_truncated_or_malformed_parseable_wdl(output):
    import app

    with pytest.raises(ValueError, match="screen shard"):
        app._parse_screen_shard_wdl(output)


def test_screen_rejects_truncated_aggregate_before_sprt_or_report_write(monkeypatch):
    import app

    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[1] == "gen-openings":
            return subprocess.CompletedProcess(command, 0, stdout="\n".join("opening" for _ in range(192)))
        if command[1] == "gate-file":
            return subprocess.CompletedProcess(command, 0, stdout="16\t0\t0\n")
        raise AssertionError(f"unexpected subprocess invocation: {command}")

    monkeypatch.setattr(app.subprocess, "run", fake_run)
    monkeypatch.setattr(app, "_write_gate_outcome", lambda **_kwargs: pytest.fail("screen report was written"))

    with pytest.raises(RuntimeError, match="192 games, not 384"):
        app.run_screen.local(b"candidate", "run", {}, {}, "manifest", "config")

    assert all(command[1] != "sprt" for command in calls)


def test_remote_calibration_uses_the_pinned_binary_and_returns_its_config(tmp_path, monkeypatch):
    import app

    remote_binary = tmp_path / "stockfish"
    remote_binary.write_bytes(b"pinned Stockfish 18")
    monkeypatch.setattr(app, "REMOTE_STOCKFISH", str(remote_binary))

    observed = []

    def fake_run(command, **_kwargs):
        observed.append(command)
        output = pathlib.Path(command[-1])
        output.write_text(
            "stockfish_config\\t1\\n"
            f"binary\\t{remote_binary}\\n"
            f"binary_sha256\\t{app._sha256(remote_binary.read_bytes())}\\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(app.subprocess, "run", fake_run)
    manifest = "manifest body"
    payload = json.dumps({
        "train": "fen\\tsource\\n" + "\\n".join(f"fen-{index}\\trandom" for index in range(1_000)),
        "validation": "fen\\tsource\\n",
        "test": "fen\\tsource\\n",
    })

    config = app._calibrate_remote_stockfish_config(manifest, payload, tmp_path / "work")

    assert observed[0][0:3] == [app.BIN, "stockfish-calibrate", str(tmp_path / "work" / "manifest.tsv")]
    assert observed[0][3] == str(remote_binary)
    assert observed[0][4] == app._sha256(remote_binary.read_bytes())
    assert f"binary_sha256\\t{app._sha256(remote_binary.read_bytes())}" in config


def test_calibration_entrypoint_writes_the_remote_config_to_the_requested_path(tmp_path, monkeypatch):
    import app

    class Remote:
        def __init__(self, result):
            self.result = result

        def remote(self, *_args):
            return self.result

    monkeypatch.setattr(app, "build_corpus", Remote(("manifest", "positions")))
    monkeypatch.setattr(app, "calibrate_stockfish_config", Remote("stockfish_config\\t1\\n"))
    output = tmp_path / "chosen-config.tsv"

    app.calibrate.info.raw_f(run_id="chosen-run", smoke=True, output=str(output))

    assert output.read_text(encoding="utf-8") == "stockfish_config\\t1\\n"


def test_load_samples_rejects_mixed_schema_or_feature_dimension(tmp_path):
    path = tmp_path / "mixed.tsv"
    path.write_text(
        "rfnn_tsv\t1\tv1\t768\n1\t0\t\n"
        "rfnn_tsv\t1\thalfka-v2-64\t40960\n1\t0\t\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema"):
        _load_samples(path, expected_schema="v1", input_dimension=768)


def test_load_samples_rejects_out_of_range_feature_index(tmp_path):
    path = tmp_path / "out-of-range.tsv"
    path.write_text("rfnn_tsv\t1\tv1\t768\n1\t768\t0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="feature dimension"):
        _load_samples(path, expected_schema="v1", input_dimension=768)


def test_quantization_error_is_maximum_sealed_prediction_delta(tmp_path):
    torch = pytest.importorskip("torch")
    import train_nnue

    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, own_values, own_offsets, opp_values, opp_offsets):
            return self.bias.expand(own_offsets.numel())

    path = tmp_path / "sealed-test.tsv"
    path.write_text("rfnn_tsv\t1\tv1\t768\n1\t0\t0\n1\t1\t1\n", encoding="utf-8")
    assert train_nnue.quantization_max_error_cp(
        ConstantModel(), path, "cpu", schema="v1", input_dimension=768
    ) == pytest.approx(0.25)


def test_halfka_export_header_matches_rust_v2_contract(tmp_path):
    pytest.importorskip("torch")
    import train_nnue

    model = train_nnue.tiny_model(
        schema="halfka-v2-64", input_dimension=64 * 2 * 5 * 64, hidden=4
    )
    out = tmp_path / "net.rfnn"
    train_nnue.quantize_and_write(model, "halfka-v2-64", 40960, 64, 4, out)
    header = out.read_bytes()[:18]
    assert header[4:8] == struct.pack("<I", 2)
    assert header[8:10] == bytes((1, 64))
    assert header[10:14] == struct.pack("<I", 40960)
    assert header[14:18] == struct.pack("<I", 4)


def test_schema_aware_train_runs_one_cpu_epoch(tmp_path):
    pytest.importorskip("torch")
    import train_nnue

    data = tmp_path / "train.tsv"
    data.write_text("rfnn_tsv\t1\tv1\t2\n0\t0\t1\n", encoding="utf-8")
    model = train_nnue.train(data, "v1", 2, 2, 1, 1, 1e-3, "cpu")
    assert model.transformer.weight.shape == (2, 2)
