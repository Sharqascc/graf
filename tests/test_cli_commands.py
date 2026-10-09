from pathlib import Path

from graf.cli import (
    main,
    run_compare,
    run_demo_graphs,
    run_status,
    run_train_conflict_pairs,
)


def test_run_status_success(monkeypatch):
    called = {}

    def fake_print_pipeline_status(root, depth):
        called["root"] = root
        called["depth"] = depth

    monkeypatch.setattr("graf.cli.print_pipeline_status", fake_print_pipeline_status)

    exit_code = run_status("/tmp", depth=2)
    assert exit_code == 0
    assert called["root"] == Path("/tmp").resolve()
    assert called["depth"] == 2


def test_run_status_failure(monkeypatch):
    def fake_print_pipeline_status(root, depth):
        raise RuntimeError("boom")

    monkeypatch.setattr("graf.cli.print_pipeline_status", fake_print_pipeline_status)

    exit_code = run_status("/tmp")
    assert exit_code == 1


def test_run_demo_graphs_success(monkeypatch, capsys):
    fake_path = Path("/tmp/sample_graph_pyg.json")

    def fake_export_graph_samples(outdir):
        return fake_path

    monkeypatch.setattr("graf.cli.export_graph_samples", fake_export_graph_samples)

    exit_code = run_demo_graphs("/tmp/out")
    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"Wrote {fake_path}" in captured.out


def test_run_demo_graphs_failure(monkeypatch):
    def fake_export_graph_samples(outdir):
        raise RuntimeError("boom")

    monkeypatch.setattr("graf.cli.export_graph_samples", fake_export_graph_samples)

    exit_code = run_demo_graphs("/tmp/out")
    assert exit_code == 1


def test_main_dispatch_status(monkeypatch):
    fake_return = 42

    def fake_run_status(root, depth):
        return fake_return

    monkeypatch.setattr("graf.cli.run_status", fake_run_status)

    exit_code = main(["status", "--root", ".", "--depth", "2"])
    assert exit_code == fake_return


def test_main_dispatch_demo(monkeypatch):
    fake_return = 7

    def fake_run_demo_graphs(outdir):
        return fake_return

    monkeypatch.setattr("graf.cli.run_demo_graphs", fake_run_demo_graphs)

    exit_code = main(["demo-graphs", "--outdir", "/tmp/out"])
    assert exit_code == fake_return


def test_run_train_conflict_pairs_success(monkeypatch, capsys):
    called = {}

    def fake_run_cross_validation(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("graf.cli.run_cross_validation", fake_run_cross_validation)

    exit_code = run_train_conflict_pairs(
        tracks="tracks.jsonl",
        graphs_dir="graphs",
        homography_config="h.yaml",
        output_dir="/tmp/out",
        window_size=5,
        stride=2,
        distance_threshold=5.0,
        min_interaction_frames=3,
        epochs=10,
        num_folds=3,
        seed=42,
    )
    assert exit_code == 0
    assert called["tracks_path"] == "tracks.jsonl"
    assert called["epochs"] == 10
    assert called["num_folds"] == 3
    assert called["seed"] == 42
    captured = capsys.readouterr()
    assert "metrics_cv.json" in captured.out


def test_run_train_conflict_pairs_failure(monkeypatch):
    def fake_run_cross_validation(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("graf.cli.run_cross_validation", fake_run_cross_validation)

    exit_code = run_train_conflict_pairs(
        tracks="t",
        graphs_dir="g",
        homography_config="h",
        output_dir="/tmp/out",
        window_size=5,
        stride=2,
        distance_threshold=5.0,
        min_interaction_frames=3,
        epochs=10,
        num_folds=3,
        seed=42,
    )
    assert exit_code == 1


def test_main_dispatch_train_conflict_pairs(monkeypatch):
    fake_return = 11

    def fake_run(*args, **kwargs):
        return fake_return

    monkeypatch.setattr("graf.cli.run_train_conflict_pairs", fake_run)

    exit_code = main(
        [
            "train-conflict-pairs",
            "--tracks",
            "tracks.jsonl",
            "--graphs_dir",
            "graphs",
            "--homography_config",
            "h.yaml",
        ]
    )
    assert exit_code == fake_return


def test_train_conflict_pairs_parser_defaults():
    from graf.cli import build_parser

    args = build_parser().parse_args(
        [
            "train-conflict-pairs",
            "--tracks",
            "t.jsonl",
            "--graphs_dir",
            "g",
            "--homography_config",
            "h.yaml",
        ]
    )
    assert args.command == "train-conflict-pairs"
    assert args.window_size == 5
    assert args.stride == 2
    assert args.distance_threshold == 5.0
    assert args.min_interaction_frames == 3
    assert args.epochs == 50
    assert args.num_folds == 5
    assert args.seed == 42
    assert args.output_dir == "outputs/models_conflict_pairs"


def test_main_dispatch_reproduce(monkeypatch):
    fake_return = 11
    captured = {}

    def fake_run_reproduce(recipe_path, dry_run=False, only=None, from_step=None):
        captured["recipe_path"] = recipe_path
        captured["dry_run"] = dry_run
        captured["only"] = only
        captured["from_step"] = from_step
        return fake_return

    monkeypatch.setattr("graf.cli.run_reproduce", fake_run_reproduce)

    exit_code = main(["reproduce", "r.yaml", "--dry-run", "--only", "s1"])
    assert exit_code == fake_return
    assert captured["recipe_path"] == "r.yaml"
    assert captured["dry_run"] is True
    assert captured["only"] == "s1"


def test_main_dispatch_doctor(monkeypatch):
    fake_return = 3
    captured = {}

    def fake_run_doctor(root=None):
        captured["root"] = root
        return fake_return

    monkeypatch.setattr("graf.cli.run_doctor", fake_run_doctor)

    exit_code = main(["doctor", "--root", "/tmp/x"])
    assert exit_code == fake_return
    assert captured["root"] == "/tmp/x"


def test_main_dispatch_fetch_detector(monkeypatch):
    fake_return = 4
    captured = {}

    def fake_run_fetch_detector(model, output_dir):
        captured["model"] = model
        captured["output_dir"] = output_dir
        return fake_return

    monkeypatch.setattr("graf.cli.run_fetch_detector", fake_run_fetch_detector)

    exit_code = main(
        ["fetch-detector", "--model", "YOLOv11-X", "--output-dir", "/tmp/m"]
    )
    assert exit_code == fake_return
    assert captured["model"] == "YOLOv11-X"
    assert captured["output_dir"] == "/tmp/m"


def test_main_dispatch_detect_video(monkeypatch):
    fake_return = 5
    captured = {}

    def fake_run_detect_video(frames_dir, model, output_dir, samples, conf, imgsz):
        captured.update(locals())
        return fake_return

    monkeypatch.setattr("graf.cli.run_detect_video", fake_run_detect_video)

    exit_code = main(
        [
            "detect-video",
            "--frames-dir",
            "/tmp/f",
            "--model",
            "/tmp/m.pt",
            "--output-dir",
            "/tmp/o",
            "--samples",
            "3",
            "--conf",
            "0.4",
            "--imgsz",
            "960",
        ]
    )
    assert exit_code == fake_return
    assert captured["frames_dir"] == "/tmp/f"
    assert captured["samples"] == 3
    assert captured["conf"] == 0.4
    assert captured["imgsz"] == 960


def _fake_run_factory(captured):
    class _Result:
        returncode = 0

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return _Result()

    return fake_run


def test_run_compare_builds_command_without_calibration(monkeypatch):
    captured = {}
    monkeypatch.setattr("graf.cli.subprocess.run", _fake_run_factory(captured))

    exit_code = run_compare(
        tracks="t.jsonl",
        graphs_dir="g",
        homography_config="h.yaml",
        output_dir="/tmp/out",
        models="rf,logreg",
        num_folds=5,
        split="blocked",
        label_source="ttc",
        label_strategy="sustained",
        run_length=3,
        ttc_threshold_seconds=1.5,
        ttc_distance_threshold=3.0,
        calibrate_threshold=False,
        calibration_objective="accuracy",
        exclude_features="",
        single_feature_name="edge_attr_nonzero_frac",
        bootstrap_resamples=100,
    )
    assert exit_code == 0
    cmd = captured["cmd"]
    # underscore variants are the flags scripts/compare_baselines.py expects
    assert "--tracks" in cmd and "t.jsonl" in cmd
    assert "--graphs_dir" in cmd and "g" in cmd
    assert "--homography_config" in cmd and "h.yaml" in cmd
    assert "--output_dir" in cmd and "/tmp/out" in cmd
    assert "--num_folds" in cmd and "5" in cmd
    # calibration flags must NOT appear when calibrate_threshold is False
    assert "--calibrate-threshold" not in cmd
    assert "--calibration-objective" not in cmd
    # empty exclude_features must not be passed
    assert "--exclude-features" not in cmd


def test_run_compare_includes_calibration_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr("graf.cli.subprocess.run", _fake_run_factory(captured))

    exit_code = run_compare(
        tracks="t.jsonl",
        graphs_dir="g",
        homography_config="h.yaml",
        output_dir="/tmp/out",
        models="rf",
        num_folds=10,
        split="blocked",
        label_source="ttc",
        label_strategy="sustained",
        run_length=5,
        ttc_threshold_seconds=1.5,
        ttc_distance_threshold=3.0,
        calibrate_threshold=True,
        calibration_objective="youden",
        exclude_features="",
        single_feature_name="edge_attr_nonzero_frac",
        bootstrap_resamples=10000,
    )
    assert exit_code == 0
    cmd = captured["cmd"]
    assert "--calibrate-threshold" in cmd
    assert "--calibration-objective" in cmd
    assert "youden" in cmd


def test_run_compare_passes_exclude_features(monkeypatch):
    captured = {}
    monkeypatch.setattr("graf.cli.subprocess.run", _fake_run_factory(captured))

    exit_code = run_compare(
        tracks="t.jsonl",
        graphs_dir="g",
        homography_config="h.yaml",
        output_dir="/tmp/out",
        models="rf",
        num_folds=10,
        split="blocked",
        label_source="ttc",
        label_strategy="sustained",
        run_length=5,
        ttc_threshold_seconds=1.5,
        ttc_distance_threshold=3.0,
        calibrate_threshold=False,
        calibration_objective="accuracy",
        exclude_features="edge_attr_nonzero_frac",
        single_feature_name="edge_attr_nonzero_frac",
        bootstrap_resamples=10000,
    )
    assert exit_code == 0
    cmd = captured["cmd"]
    assert "--exclude-features" in cmd
    assert "edge_attr_nonzero_frac" in cmd


def test_run_compare_missing_script(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)

    exit_code = run_compare(
        tracks="t",
        graphs_dir="g",
        homography_config="h",
        output_dir="o",
        models="rf",
        num_folds=1,
        split="blocked",
        label_source="ttc",
        label_strategy="sustained",
        run_length=1,
        ttc_threshold_seconds=1.5,
        ttc_distance_threshold=3.0,
        calibrate_threshold=False,
        calibration_objective="accuracy",
        exclude_features="",
        single_feature_name="edge_attr_nonzero_frac",
        bootstrap_resamples=0,
    )
    assert exit_code == 1


def test_main_dispatch_compare(monkeypatch):
    fake_return = 99

    def fake_run(**kwargs):
        return fake_return

    monkeypatch.setattr("graf.cli.run_compare", fake_run)

    exit_code = main(
        [
            "compare",
            "--tracks",
            "t.jsonl",
            "--graphs-dir",
            "g",
            "--homography-config",
            "h.yaml",
        ]
    )
    assert exit_code == fake_return


def test_compare_parser_defaults():
    from graf.cli import build_parser

    args = build_parser().parse_args(
        [
            "compare",
            "--tracks",
            "t.jsonl",
            "--graphs-dir",
            "g",
            "--homography-config",
            "h.yaml",
        ]
    )
    assert args.command == "compare"
    assert args.output_dir == "outputs/compare"
    assert args.models == "majority,logreg,rf,single_feature"
    assert args.num_folds == 10
    assert args.split == "blocked"
    assert args.label_source == "ttc"
    assert args.label_strategy == "sustained"
    assert args.run_length == 5
    assert args.ttc_threshold_seconds == 1.5
    assert args.ttc_distance_threshold == 3.0
    assert args.calibrate_threshold is False
    assert args.calibration_objective == "accuracy"
    assert args.exclude_features == ""
    assert args.single_feature_name == "edge_attr_nonzero_frac"
    assert args.bootstrap_resamples == 10000
