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


def test_calibrate_run_builds_and_calibrates_within_one_remote_function(monkeypatch):
    import app

    calls = []
    monkeypatch.setattr(
        app, "_build_corpus_artifacts",
        lambda run_id, smoke: calls.append(("corpus", run_id, smoke)) or ("manifest", "positions"),
    )
    monkeypatch.setattr(
        app, "_calibrate_remote_stockfish_config",
        lambda manifest, positions, _root: calls.append(("calibrate", manifest, positions)) or "config-tsv\n",
    )
    monkeypatch.setattr(
        app, "_write_artifact",
        lambda run_id, stage, _input, contents, name: calls.append(("artifact", run_id, stage, contents, name)),
    )

    assert app.calibrate_run.local("direct-run", True) == "config-tsv\n"
    assert calls[0] == ("corpus", "direct-run", True)
    assert calls[1] == ("calibrate", "manifest", "positions")
    assert calls[2] == ("artifact", "direct-run", "stockfish-config", "config-tsv\n", "stockfish-config.tsv")


def test_label_batches_are_deterministic_bounded_and_keep_the_split_header():
    import app

    split = "fen\tsource\nfen-0\trandom\nfen-1\trandom\nfen-2\topening\n"
    assert app._partition_label_rows(split, 2) == [
        "fen\tsource\nfen-0\trandom\nfen-1\trandom\n",
        "fen\tsource\nfen-2\topening\n",
    ]


def test_label_scheduler_caps_each_wave_at_eighty_workers():
    import app

    jobs = list(range(161))
    waves = list(app._label_worker_waves(jobs))

    assert [len(wave) for wave in waves] == [80, 80, 1]
    assert max(len(wave) for wave in waves) == 80
    assert [job for wave in waves for job in wave] == jobs


def test_halfka_reencoding_addresses_the_exact_source_label_artifact():
    import app

    assert app._source_label_artifact_path("v1-run", "label-input") == (
        "/artifacts/runs/v1-run/labels-v1-label-input/artifact"
    )


def test_halfka_selection_prefers_the_stronger_held_out_alignment():
    import app

    candidates = [
        (256, b"narrow", {"validation_alignment": {"pearson_correlation": 0.30},
                            "test_alignment": {"pearson_correlation": 0.29}}),
        (512, b"wide", {"validation_alignment": {"pearson_correlation": 0.34},
                          "test_alignment": {"pearson_correlation": 0.33}}),
    ]

    assert app.select_halfka_candidate(candidates)[0] == 512


def test_v1_pipeline_remote_coordinator_owns_stages_and_records_progress(monkeypatch):
    import app

    stages = []

    class Remote:
        def __init__(self, result):
            self.result = result

        def remote(self, *_args):
            return self.result

    labels = {"train": "rfnn_tsv\t1\tv1\t768\n", "validation": "rfnn_tsv\t1\tv1\t768\n", "test": "rfnn_tsv\t1\tv1\t768\n"}
    report = {"validation_wdl_loss": 0.0, "test_wdl_loss": 0.0, "quantization_max_error_cp": 0}
    monkeypatch.setattr(app, "_build_corpus_artifacts", lambda *_args: ("manifest", "positions"))
    monkeypatch.setattr(app, "_calibrate_remote_stockfish_config", lambda *_args: "config")
    monkeypatch.setattr(app, "label_manifest", Remote(labels))
    monkeypatch.setattr(app, "train_net", Remote((b"net", report)))
    monkeypatch.setattr(app, "rust_parity_check", Remote(None))
    monkeypatch.setattr(app, "run_screen", Remote({"wdl": {"wins": 0, "draws": 0, "losses": 384}}))
    monkeypatch.setattr(app, "_write_artifact", lambda *_args, **_kwargs: "artifact")
    monkeypatch.setattr(app, "_write_pipeline_status", lambda _run, stage, **_details: stages.append(stage))

    result = app.run_v1_pipeline.local("durable-v1", True, "128", 1, 1)

    assert result["stage"] == "complete"
    assert stages == ["corpus", "calibration", "labels", "control", "candidate-128", "screen-128", "complete"]


def test_v1_pipeline_uses_training_report_without_rereading_its_artifact(monkeypatch):
    import app

    class Remote:
        def __init__(self, result):
            self.result = result

        def remote(self, *_args):
            return self.result

    class MustNotRead:
        def remote(self, *_args):
            pytest.fail("pipeline must use train_net's report")

    labels = {
        "train": "rfnn_tsv\t1\tv1\t768\n",
        "validation": "rfnn_tsv\t1\tv1\t768\n",
        "test": "rfnn_tsv\t1\tv1\t768\n",
    }
    report = {"validation_wdl_loss": 0.0, "test_wdl_loss": 0.0, "quantization_max_error_cp": 0}
    monkeypatch.setattr(app, "_build_corpus_artifacts", lambda *_args: ("manifest", "positions"))
    monkeypatch.setattr(app, "_calibrate_remote_stockfish_config", lambda *_args: "config")
    monkeypatch.setattr(app, "label_manifest", Remote(labels))
    monkeypatch.setattr(app, "train_net", Remote((b"net", report)))
    monkeypatch.setattr(app, "read_report", MustNotRead())
    monkeypatch.setattr(app, "rust_parity_check", Remote(None))
    monkeypatch.setattr(app, "run_screen", Remote({"wdl": {"wins": 0, "draws": 0, "losses": 384}}))
    monkeypatch.setattr(app, "_write_artifact", lambda *_args, **_kwargs: "artifact")
    monkeypatch.setattr(app, "_write_pipeline_status", lambda *_args, **_kwargs: None)

    assert app.run_v1_pipeline.local("durable-v1", True, "128", 1, 1)["stage"] == "complete"


def test_label_shard_reuses_its_immutable_artifact(tmp_path, monkeypatch):
    import app

    class FakeVolume:
        def __init__(self):
            self.reloads = 0

        def reload(self):
            self.reloads += 1

        def commit(self):
            pass

    volume = FakeVolume()
    monkeypatch.setattr(app, "artifacts", volume)
    monkeypatch.setattr(
        app, "_artifact_path",
        lambda _run_id, _stage, _input_hash, name="artifact": str(tmp_path / name),
    )
    produced = []
    label = "rfnn_tsv\t1\tv1\t768\n31\t0\t1\tfen\t6624\n"

    first = app._reuse_or_write_label_shard(
        "run", "v1", "train", 0, "manifest", "config", "fen\tsource\nfen\trandom\n",
        lambda: produced.append(True) or label,
    )
    second = app._reuse_or_write_label_shard(
        "run", "v1", "train", 0, "manifest", "config", "fen\tsource\nfen\trandom\n",
        lambda: produced.append(True) or label,
    )

    assert first == second
    assert produced == [True]
    assert volume.reloads == 2
    assert pathlib.Path(first).read_text(encoding="utf-8") == label


def test_aggregate_label_shards_orders_rows_and_keeps_one_schema_header():
    import app

    header = "rfnn_tsv\t1\tv1\t768\n"
    files = {
        "train-1": header + "2\t\t\tfen-2\t2\n",
        "train-0": header + "1\t\t\tfen-1\t1\n",
        "validation-0": header + "3\t\t\tfen-3\t3\n",
    }
    results = [("validation", 0, "validation-0"), ("train", 1, "train-1"), ("train", 0, "train-0")]

    aggregate = app._aggregate_label_shards(results, lambda path: files[path], "v1")

    assert aggregate["train"] == header + "1\t\t\tfen-1\t1\n2\t\t\tfen-2\t2\n"
    assert aggregate["validation"] == header + "3\t\t\tfen-3\t3\n"
    assert aggregate["test"] == header


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


def test_wdl_loss_uses_soft_label_logistic_objective(tmp_path):
    torch = pytest.importorskip("torch")
    import train_nnue

    class ConstantModel(torch.nn.Module):
        def forward(self, own_values, own_offsets, opp_values, opp_offsets):
            return torch.zeros(own_offsets.numel())

    path = tmp_path / "soft-label.tsv"
    path.write_text("rfnn_tsv\t1\tv1\t2\n400\t0\t1\n", encoding="utf-8")

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0]), torch.sigmoid(torch.tensor([1.0])),
    ).item()
    assert train_nnue.wdl_loss(ConstantModel(), path, "cpu", "v1", 2) == pytest.approx(expected)


def test_teacher_alignment_metrics_measure_correlation_and_decisive_pair_ordering():
    torch = pytest.importorskip("torch")
    import train_nnue

    metrics = train_nnue.teacher_alignment_metrics_from_predictions(
        torch.tensor([-400.0, -100.0, 100.0, 400.0]),
        torch.tensor([-400.0, -100.0, 100.0, 400.0]),
    )

    assert metrics["pearson_correlation"] == pytest.approx(1.0)
    assert metrics["pairwise_ranking_accuracy"] == pytest.approx(1.0)
    assert metrics["decisive_pair_count"] == 2


def test_pairwise_ranking_loss_prefers_the_teacher_order():
    torch = pytest.importorskip("torch")
    import train_nnue

    targets = torch.tensor([400.0, -400.0])
    correctly_ordered = train_nnue.pairwise_ranking_loss(torch.tensor([100.0, -100.0]), targets)
    reversed_order = train_nnue.pairwise_ranking_loss(torch.tensor([-100.0, 100.0]), targets)

    assert correctly_ordered < reversed_order


def test_offline_promotion_requires_improved_held_out_teacher_alignment():
    import app

    control = {
        "validation_wdl_loss": 0.70,
        "test_wdl_loss": 0.70,
        "quantization_max_error_cp": 0.0,
        "validation_alignment": {"pearson_correlation": 0.30, "pairwise_ranking_accuracy": 0.70},
        "test_alignment": {"pearson_correlation": 0.30, "pairwise_ranking_accuracy": 0.70},
    }
    candidate = {
        **control,
        "validation_wdl_loss": 0.68,
        "test_wdl_loss": 0.69,
        "validation_alignment": {"pearson_correlation": 0.30, "pairwise_ranking_accuracy": 0.70},
        "test_alignment": {"pearson_correlation": 0.30, "pairwise_ranking_accuracy": 0.70},
    }

    assert not app.offline_promotes(candidate, control)
    candidate["validation_alignment"] = {"pearson_correlation": 0.32, "pairwise_ranking_accuracy": 0.71}
    candidate["test_alignment"] = {"pearson_correlation": 0.32, "pairwise_ranking_accuracy": 0.71}
    assert app.offline_promotes(candidate, control)


def test_quantization_aware_model_has_no_parameter_rounding_drift(tmp_path):
    torch = pytest.importorskip("torch")
    import train_nnue

    path = tmp_path / "sealed-test.tsv"
    path.write_text(
        "rfnn_tsv\t1\tv1\t2\n1\t0\t1\n-1\t1\t0\n",
        encoding="utf-8",
    )
    torch.manual_seed(1)
    model = train_nnue.tiny_model(
        schema="v1", input_dimension=2, hidden=2, quantization_aware=True,
    )

    assert train_nnue.quantization_max_error_cp(
        model, path, "cpu", schema="v1", input_dimension=2
    ) == pytest.approx(0.0)


def test_training_artifact_input_includes_the_quantization_method_version():
    import app

    assert app._training_artifact_input("labels", "v1", 768, 128, 40, 1).endswith(
        app.TRAINING_METHOD
    )


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
