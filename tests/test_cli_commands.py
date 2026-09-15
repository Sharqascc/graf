from pathlib import Path

from graf.cli import (
    main,
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


def test_main_json_workaround(monkeypatch, capsys):
    # A leading JSON path should be ignored and print help
    exit_code = main(["/tmp/config.json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "GRAF: graph-based surrogate safety analysis pipeline" in captured.out


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
