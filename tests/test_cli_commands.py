from pathlib import Path

from graf.cli import (
    main,
    run_audit,
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

    def fake_run_reproduce(
        recipe_path, dry_run=False, only=None, from_step=None, allow_missing=False
    ):
        captured["recipe_path"] = recipe_path
        captured["dry_run"] = dry_run
        captured["only"] = only
        captured["from_step"] = from_step
        captured["allow_missing"] = allow_missing
        return fake_return

    monkeypatch.setattr("graf.cli.run_reproduce", fake_run_reproduce)

    exit_code = main(["reproduce", "r.yaml", "--dry-run", "--only", "s1"])
    assert exit_code == fake_return
    assert captured["recipe_path"] == "r.yaml"
    assert captured["dry_run"] is True
    assert captured["only"] == "s1"
    assert captured["allow_missing"] is False


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


def test_run_audit_pass(tmp_path, capsys):
    import json

    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "setup": {
                    "label_source": "ttc",
                    "label_strategy": "sustained",
                    "run_length": 5,
                    "ttc_threshold_seconds": 1.5,
                    "ttc_distance_threshold": 3.0,
                    "split": "blocked",
                    "calibrate_threshold": False,
                    "calibration_objective": "accuracy",
                    "purge_gap_frames": 10,
                },
                "labels": {"majority_baseline": 0.779},
                "results": [
                    {
                        "model": "rf",
                        "fold_accuracy": [0.7],
                        "fold_auc": [0.75],
                        "fold_majority": [0.77],
                        "leakage_after_purge": {"leaked": 0, "total": 249},
                    }
                ],
            }
        )
    )
    rc = run_audit(str(p))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Reviewer checklist" in out


def test_run_audit_fail(tmp_path, capsys):
    import json

    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "setup": {"label_source": "ttc"},
                "labels": {"majority_baseline": 0.779},
                "results": [],
            }
        )
    )
    rc = run_audit(str(p))
    assert rc == 1


def test_run_audit_missing_file(tmp_path):
    rc = run_audit(str(tmp_path / "nope.json"))
    assert rc == 1


def test_main_dispatch_audit(monkeypatch, tmp_path):
    fake_return = 77

    def fake_run_audit(comparison_json):
        return fake_return

    monkeypatch.setattr("graf.cli.run_audit", fake_run_audit)

    exit_code = main(["audit", str(tmp_path / "c.json")])
    assert exit_code == fake_return


def test_main_dispatch_recipes(monkeypatch):
    fake_return = 12

    def fake_run_recipes():
        return fake_return

    monkeypatch.setattr("graf.cli.run_recipes", fake_run_recipes)
    assert main(["recipes"]) == fake_return


def test_main_dispatch_validate(monkeypatch):
    fake_return = 13
    captured = {}

    def fake_run_validate(recipe_path):
        captured["recipe_path"] = recipe_path
        return fake_return

    monkeypatch.setattr("graf.cli.run_validate", fake_run_validate)
    assert main(["validate", "r.yaml"]) == fake_return
    assert captured["recipe_path"] == "r.yaml"


def test_main_dispatch_reproduce_allow_missing(monkeypatch):
    captured = {}

    def fake_run_reproduce(
        recipe_path, dry_run=False, only=None, from_step=None, allow_missing=False
    ):
        captured["allow_missing"] = allow_missing
        return 0

    monkeypatch.setattr("graf.cli.run_reproduce", fake_run_reproduce)
    assert main(["reproduce", "r.yaml", "--allow-missing"]) == 0
    assert captured["allow_missing"] is True


# ------------------------------------------------------------------
# Coverage additions for cli.py (targeting uncovered lines)
# ------------------------------------------------------------------


def test_build_parser_handles_missing_package_metadata(monkeypatch):
    """--version falls back to 'unknown' when graf is not installed."""
    from importlib.metadata import PackageNotFoundError

    import graf.cli as cli_mod

    def fake_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("graf.cli.version", fake_version)
    parser = cli_mod.build_parser()
    assert parser is not None


def test_run_doctor_default_root(monkeypatch):
    """run_doctor(None) uses the repo root and calls run_doctor_checks."""
    import graf.cli as cli_mod

    captured = {}

    def fake_checks(root):
        captured["root"] = root
        return 0

    monkeypatch.setattr("graf.cli.run_doctor_checks", fake_checks)
    rc = cli_mod.run_doctor(None)
    assert rc == 0
    assert captured["root"] == cli_mod._repo_root()


def test_run_doctor_explicit_root(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    captured = {}

    def fake_checks(root):
        captured["root"] = root
        return 0

    monkeypatch.setattr("graf.cli.run_doctor_checks", fake_checks)
    rc = cli_mod.run_doctor(str(tmp_path))
    assert rc == 0
    assert captured["root"] == tmp_path.resolve()


def test_run_doctor_exception(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    def fake_checks(root):
        raise RuntimeError("boom")

    monkeypatch.setattr("graf.cli.run_doctor_checks", fake_checks)
    rc = cli_mod.run_doctor(str(tmp_path))
    assert rc == 1


def test_run_fetch_detector_missing_script(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)
    rc = cli_mod.run_fetch_detector("YOLOv11-S", "/tmp/models")
    assert rc == 1


def test_run_fetch_detector_success(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fetch_detector.py").write_text("# placeholder")
    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)

    captured = {}

    class _R:
        returncode = 0

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _R()

    monkeypatch.setattr("graf.cli.subprocess.run", fake_run)
    rc = cli_mod.run_fetch_detector("YOLOv11-S", "/tmp/models")
    assert rc == 0
    joined = " ".join(captured["cmd"])
    assert "fetch_detector.py" in joined
    assert "YOLOv11-S" in captured["cmd"]
    assert "/tmp/models" in captured["cmd"]


def test_run_fetch_detector_subprocess_exception(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fetch_detector.py").write_text("# placeholder")
    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)

    def fake_run(cmd, *args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("graf.cli.subprocess.run", fake_run)
    rc = cli_mod.run_fetch_detector("YOLOv11-S", "/tmp/models")
    assert rc == 1


def test_run_detect_video_missing_script(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)
    rc = cli_mod.run_detect_video("/tmp/f", "/tmp/m.pt", "/tmp/o", 3, 0.4, 640)
    assert rc == 1


def test_run_detect_video_success(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "detect_video.py").write_text("# placeholder")
    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)

    captured = {}

    class _R:
        returncode = 0

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _R()

    monkeypatch.setattr("graf.cli.subprocess.run", fake_run)
    rc = cli_mod.run_detect_video("/tmp/f", "/tmp/m.pt", "/tmp/o", 3, 0.4, 960)
    assert rc == 0
    j = " ".join(captured["cmd"])
    for token in [
        "detect_video.py",
        "--frames_dir",
        "/tmp/f",
        "--model",
        "/tmp/m.pt",
        "--output_dir",
        "/tmp/o",
        "--samples",
        "3",
        "--conf",
        "0.4",
        "--imgsz",
        "960",
    ]:
        assert token in captured["cmd"] or token in j


def test_run_detect_video_subprocess_exception(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "detect_video.py").write_text("# placeholder")
    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)

    def fake_run(cmd, *args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("graf.cli.subprocess.run", fake_run)
    rc = cli_mod.run_detect_video("/tmp/f", "/tmp/m.pt", "/tmp/o", 3, 0.4, 640)
    assert rc == 1


def test_run_compare_subprocess_exception(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "compare_baselines.py").write_text("# placeholder")
    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)

    def fake_run(cmd, *args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("graf.cli.subprocess.run", fake_run)
    rc = cli_mod.run_compare(
        tracks="/tmp/t.jsonl",
        graphs_dir="/tmp/g",
        homography_config="/tmp/h.yaml",
        output_dir="/tmp/o",
        models="rf",
        num_folds=5,
        split="blocked",
        label_source="ttc",
        label_strategy="sustained",
        run_length=5,
        ttc_threshold_seconds=1.5,
        ttc_distance_threshold=3.0,
        calibrate_threshold=False,
        calibration_objective="accuracy",
        exclude_features="",
        single_feature_name="edge_attr_nonzero_frac",
        bootstrap_resamples=100,
    )
    assert rc == 1


def test_run_audit_import_error(monkeypatch):
    """When graf.audit cannot be imported, run_audit returns 1."""
    import sys as _sys

    import graf.cli as cli_mod

    monkeypatch.setitem(_sys.modules, "graf.audit", None)
    rc = cli_mod.run_audit("/tmp/whatever.json")
    assert rc == 1


def test_run_reproduce_file_not_found(tmp_path):
    import graf.cli as cli_mod

    rc = cli_mod.run_reproduce(str(tmp_path / "missing.yaml"))
    assert rc == 1


def test_run_recipes_no_dir(monkeypatch, tmp_path):
    import graf.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)
    rc = cli_mod.run_recipes()
    assert rc == 1  # no configs/recipes/ directory


def test_run_recipes_empty_dir(monkeypatch, tmp_path, capsys):
    import graf.cli as cli_mod

    (tmp_path / "configs" / "recipes").mkdir(parents=True)
    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)
    rc = cli_mod.run_recipes()
    assert rc == 0
    assert "no recipes" in capsys.readouterr().out


def test_run_recipes_lists_files(monkeypatch, tmp_path, capsys):
    import graf.cli as cli_mod

    rdir = tmp_path / "configs" / "recipes"
    rdir.mkdir(parents=True)
    (rdir / "a.yaml").write_text(
        "name: a\ndescription: test recipe\nsteps:\n  - name: s1\n    cmd: [echo, hi]\n"
    )
    (rdir / "b.yaml").write_text(
        "name: b\nsteps:\n  - name: s2\n    cmd: [echo, bye]\n"
    )
    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)
    rc = cli_mod.run_recipes()
    assert rc == 0
    out = capsys.readouterr().out
    assert "a.yaml" in out
    assert "test recipe" in out
    assert "b.yaml" in out
    assert "(no description)" in out


def test_run_recipes_unreadable_file(monkeypatch, tmp_path, capsys):
    import graf.cli as cli_mod

    rdir = tmp_path / "configs" / "recipes"
    rdir.mkdir(parents=True)
    # no steps key -> load_recipe raises
    (rdir / "bad.yaml").write_text("name: bad\n")
    monkeypatch.setattr(cli_mod, "_repo_root", lambda: tmp_path)
    rc = cli_mod.run_recipes()
    assert rc == 0  # skips and prints "unreadable"
    assert "unreadable" in capsys.readouterr().out


def test_run_validate_ok(tmp_path, capsys):
    import graf.cli as cli_mod

    r = tmp_path / "r.yaml"
    r.write_text("name: t\nsteps:\n  - name: s1\n    cmd: [echo, hi]\n")
    rc = cli_mod.run_validate(str(r))
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_run_validate_missing_inputs(tmp_path, capsys):
    import graf.cli as cli_mod

    r = tmp_path / "r.yaml"
    r.write_text(
        "name: t\nsteps:\n  - name: s1\n    cmd: [echo, hi]\n"
        "    requires: [missing-file.txt]\n"
    )
    rc = cli_mod.run_validate(str(r))
    assert rc == 1
    out = capsys.readouterr().out
    assert "problem" in out
    assert "missing-file.txt" in out


def test_run_validate_bad_recipe(tmp_path):
    import graf.cli as cli_mod

    rc = cli_mod.run_validate(str(tmp_path / "does-not-exist.yaml"))
    assert rc == 1


def test_main_reads_sys_argv_when_none(monkeypatch):
    """main(None) reads sys.argv[1:]; bare `graf` prints the welcome."""
    import sys as _sys

    import graf.cli as cli_mod

    monkeypatch.setattr(_sys, "argv", ["graf"])
    called = {}

    def fake_welcome(parser):
        called["welcome"] = True

    monkeypatch.setattr(cli_mod, "_print_welcome", fake_welcome)
    rc = cli_mod.main(None)
    assert rc == 0
    assert called.get("welcome")


def test_run_reproduce_success_direct(tmp_path):
    """Call run_reproduce directly with a valid recipe; exercises the
    success path through to run_recipe (the dispatch test monkeypatches
    run_reproduce, so the function body never runs there)."""
    from graf.cli import run_reproduce

    r = tmp_path / "r.yaml"
    r.write_text('name: t\nsteps:\n  - name: s1\n    cmd: ["echo", "hi"]\n')
    rc = run_reproduce(str(r), dry_run=True)
    assert rc == 0


def test_run_reproduce_invalid_recipe_direct(tmp_path):
    """A recipe that exists but is malformed returns 1."""
    from graf.cli import run_reproduce

    r = tmp_path / "r.yaml"
    r.write_text("name: t\nsteps: not-a-list\n")
    rc = run_reproduce(str(r))
    assert rc == 1


def test_print_welcome_output(capsys):
    """_print_welcome prints the getting-started guidance.

    The dispatch test monkeypatches _print_welcome, so the function body
    never runs there. This test calls it directly and asserts on the
    actual output. That closes the last coverage gap on cli.py.
    """
    from graf.cli import _print_welcome, build_parser

    _print_welcome(build_parser())
    out = capsys.readouterr().out
    assert "GRAF" in out
    assert "Get started" in out
    assert "graf doctor" in out
    assert "graf recipes" in out
    assert "graf reproduce" in out
    assert "Everything else" in out
    assert "graf --help" in out
    assert "graf --version" in out
