"""Numbered pipeline entry point for `graf run`.

Each step wraps an existing script under scripts/. The registry
is thin: it names the steps, computes default paths from a site
identifier, and builds argv. Real work happens in the scripts.

Two pipelines share this registry:

    science  (default)
        1. detect      UVH-26 detection on extracted frames
        2. track       greedy IoU association with gates
        3. trajectory  tracks -> world coordinates
        4. graph       tracks -> per-frame PyG graphs
        5. window      graphs -> rolling temporal windows
        6. train       windows -> fitted GCN + metrics_cv.json
        7. compare     windows -> comparison.json (baselines)
        8. audit       comparison.json -> reviewer checklist

    inspect
        1. detect      UVH-26 detection on extracted frames
        2. grid        metric grid overlay for homography QA
        3. gate        virtual gate overlay for zone entry/exit
        4. track       greedy IoU association with gates
        5. pet         PET events from tracks + gates
        6. visualize   rendered frames / video

The inspect pipeline declares all six steps even when some scripts
do not exist yet. Preflight reports which steps are ready and which
are missing, before any work runs.

Every existing subcommand and recipe is unchanged. `graf run` is
additive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Context:
    site: str
    video: str | None
    root: Path


@dataclass(frozen=True)
class Command:
    script: str
    args: list[str]


@dataclass(frozen=True)
class Step:
    number: int
    name: str
    description: str
    build_commands: Callable[[Context], list[Command]]
    outputs: Callable[[Context], list[Path]]


def _frames_dir(ctx: Context) -> Path:
    return ctx.root / "data" / "interim" / "frames" / ctx.site / ctx.site


def _detections_dir(ctx: Context) -> Path:
    return ctx.root / "data" / "interim" / "detections" / ctx.site


def _tracks_dir(ctx: Context) -> Path:
    return ctx.root / "data" / "interim" / "tracks" / ctx.site


def _trajectories_file(ctx: Context) -> Path:
    return ctx.root / "data" / "interim" / "trajectories" / (ctx.site + ".json")


def _graphs_dir(ctx: Context) -> Path:
    return ctx.root / "data" / "processed" / "graphs" / ctx.site


def _windows_dir(ctx: Context) -> Path:
    return ctx.root / "data" / "processed" / "windows" / ctx.site


def _model_dir(ctx: Context) -> Path:
    return ctx.root / "outputs" / ("model_" + ctx.site)


def _comparison_dir(ctx: Context) -> Path:
    return ctx.root / "outputs" / ("comparison_" + ctx.site)


def _grid_dir(ctx: Context) -> Path:
    return ctx.root / "data" / "interim" / "grid" / ctx.site


def _gates_dir(ctx: Context) -> Path:
    return ctx.root / "data" / "interim" / "gates" / ctx.site


def _pet_dir(ctx: Context) -> Path:
    return ctx.root / "data" / "processed" / "ssm_events" / ctx.site


def _render_dir(ctx: Context) -> Path:
    return ctx.root / "outputs" / ("render_" + ctx.site)


def _homography_default() -> str:
    return "configs/homography/default.yaml"


def _detection_config(ctx: Context) -> str:
    candidate = ctx.root / "configs" / "detection" / (ctx.site + ".yaml")
    if candidate.exists():
        return str(candidate.relative_to(ctx.root))
    return "configs/detection/uvh26.yaml"


def _tracking_config(ctx: Context) -> str:
    candidate = ctx.root / "configs" / "tracking" / (ctx.site + ".yaml")
    if candidate.exists():
        return str(candidate.relative_to(ctx.root))
    return "configs/tracking/bytetrack.yaml"


def _build_1_detect(ctx: Context) -> list[Command]:
    return [
        Command(
            script="run_detection.py",
            args=[
                "--frames_dir",
                str(_frames_dir(ctx)),
                "--output_dir",
                str(_detections_dir(ctx)),
                "--config",
                _detection_config(ctx),
                "--video-id",
                ctx.site,
            ],
        )
    ]


def _build_2_track(ctx: Context) -> list[Command]:
    return [
        Command(
            script="run_tracking.py",
            args=[
                "--detections",
                str(_detections_dir(ctx) / "detections.jsonl"),
                "--output_dir",
                str(_tracks_dir(ctx)),
                "--config",
                _tracking_config(ctx),
            ],
        )
    ]


def _build_3_trajectory(ctx: Context) -> list[Command]:
    return [
        Command(
            script="build_trajectories.py",
            args=[
                "--tracks",
                str(_tracks_dir(ctx) / "tracks.jsonl"),
                "--output",
                str(_trajectories_file(ctx)),
                "--homography_config",
                _homography_default(),
            ],
        )
    ]


def _build_4_graph(ctx: Context) -> list[Command]:
    return [
        Command(
            script="build_graphs.py",
            args=[
                "--tracks",
                str(_tracks_dir(ctx) / "tracks.jsonl"),
                "--output_dir",
                str(_graphs_dir(ctx)),
                "--homography_config",
                _homography_default(),
                "--radius",
                "10.0",
            ],
        )
    ]


def _build_5_window(ctx: Context) -> list[Command]:
    return [
        Command(
            script="make_windows.py",
            args=[
                "--graphs_dir",
                str(_graphs_dir(ctx)),
                "--output_dir",
                str(_windows_dir(ctx)),
            ],
        )
    ]


def _build_6_train(ctx: Context) -> list[Command]:
    return [
        Command(
            script="train_conflict_pairs.py",
            args=[
                "--tracks",
                str(_tracks_dir(ctx) / "tracks.jsonl"),
                "--graphs_dir",
                str(_graphs_dir(ctx)),
                "--homography_config",
                _homography_default(),
                "--output_dir",
                str(_model_dir(ctx)),
            ],
        )
    ]


def _build_7_compare(ctx: Context) -> list[Command]:
    return [
        Command(
            script="compare_baselines.py",
            args=[
                "--tracks",
                str(_tracks_dir(ctx) / "tracks.jsonl"),
                "--graphs_dir",
                str(_graphs_dir(ctx)),
                "--homography_config",
                _homography_default(),
                "--output_dir",
                str(_comparison_dir(ctx)),
                "--calibrate-threshold",
                "--calibration-objective",
                "accuracy",
            ],
        )
    ]


def _build_8_audit(ctx: Context) -> list[Command]:
    return [
        Command(
            script="_audit_inline",
            args=[str(_comparison_dir(ctx) / "comparison.json")],
        )
    ]


def _build_inspect_2_grid(ctx: Context) -> list[Command]:
    return [
        Command(
            script="draw_grid.py",
            args=[
                "--frames_dir",
                str(_frames_dir(ctx)),
                "--homography_config",
                _homography_default(),
                "--output_dir",
                str(_grid_dir(ctx)),
            ],
        )
    ]


def _build_inspect_3_gate(ctx: Context) -> list[Command]:
    return [
        Command(
            script="draw_gates.py",
            args=[
                "--frames_dir",
                str(_frames_dir(ctx)),
                "--homography_config",
                _homography_default(),
                "--output_dir",
                str(_gates_dir(ctx)),
            ],
        )
    ]


def _build_inspect_5_pet(ctx: Context) -> list[Command]:
    return [
        Command(
            script="compute_pet.py",
            args=[
                "--tracks",
                str(_tracks_dir(ctx) / "tracks.jsonl"),
                "--homography_config",
                _homography_default(),
                "--output_dir",
                str(_pet_dir(ctx)),
            ],
        )
    ]


def _build_inspect_6_viz(ctx: Context) -> list[Command]:
    return [
        Command(
            script="render_results.py",
            args=[
                "--detections",
                str(_detections_dir(ctx) / "detections.jsonl"),
                "--tracks",
                str(_tracks_dir(ctx) / "tracks.jsonl"),
                "--pet",
                str(_pet_dir(ctx) / "pet_events.jsonl"),
                "--output_dir",
                str(_render_dir(ctx)),
            ],
        )
    ]


SCIENCE_STEPS: list[Step] = [
    Step(
        1,
        "detect",
        "UVH-26 detection on extracted frames",
        _build_1_detect,
        lambda c: [_detections_dir(c) / "detections.jsonl"],
    ),
    Step(
        2,
        "track",
        "greedy IoU association with gates",
        _build_2_track,
        lambda c: [_tracks_dir(c) / "tracks.jsonl"],
    ),
    Step(
        3,
        "trajectory",
        "tracks -> world coordinates",
        _build_3_trajectory,
        lambda c: [_trajectories_file(c)],
    ),
    Step(
        4,
        "graph",
        "tracks -> per-frame PyG graphs",
        _build_4_graph,
        lambda c: [_graphs_dir(c)],
    ),
    Step(
        5,
        "window",
        "graphs -> rolling temporal windows",
        _build_5_window,
        lambda c: [_windows_dir(c)],
    ),
    Step(
        6,
        "train",
        "windows -> fitted GCN + metrics_cv.json",
        _build_6_train,
        lambda c: [_model_dir(c) / "metrics_cv.json"],
    ),
    Step(
        7,
        "compare",
        "windows -> comparison.json (baselines)",
        _build_7_compare,
        lambda c: [_comparison_dir(c) / "comparison.json"],
    ),
    Step(
        8,
        "audit",
        "comparison.json -> reviewer checklist",
        _build_8_audit,
        lambda c: [],
    ),
]


INSPECT_STEPS: list[Step] = [
    Step(
        1,
        "detect",
        "UVH-26 detection on extracted frames",
        _build_1_detect,
        lambda c: [_detections_dir(c) / "detections.jsonl"],
    ),
    Step(
        2,
        "grid",
        "metric grid overlay for homography QA",
        _build_inspect_2_grid,
        lambda c: [_grid_dir(c)],
    ),
    Step(
        3,
        "gate",
        "virtual gate overlay for zone entry/exit",
        _build_inspect_3_gate,
        lambda c: [_gates_dir(c)],
    ),
    Step(
        4,
        "track",
        "greedy IoU association with gates",
        _build_2_track,
        lambda c: [_tracks_dir(c) / "tracks.jsonl"],
    ),
    Step(
        5,
        "pet",
        "PET events from tracks + gates",
        _build_inspect_5_pet,
        lambda c: [_pet_dir(c) / "pet_events.jsonl"],
    ),
    Step(
        6,
        "visualize",
        "rendered frames / video",
        _build_inspect_6_viz,
        lambda c: [_render_dir(c)],
    ),
]


# Backward-compat: `STEPS` remains the science pipeline
STEPS = SCIENCE_STEPS


PIPELINES: dict[str, list[Step]] = {
    "science": SCIENCE_STEPS,
    "inspect": INSPECT_STEPS,
}


def get_step(n: int, pipeline: str = "science") -> Step:
    steps = PIPELINES[pipeline]
    for s in steps:
        if s.number == n:
            return s
    raise ValueError(
        "no step with number " + str(n) + " in pipeline " + repr(pipeline)
    )


def select_steps(target: str, pipeline: str = "science") -> list[Step]:
    steps = PIPELINES[pipeline]
    if target == "all":
        return list(steps)
    if target.startswith("from-"):
        try:
            n = int(target[5:])
        except ValueError as exc:
            raise ValueError(
                "from-N expects an integer N, got " + repr(target)
            ) from exc
        return [s for s in steps if s.number >= n]
    try:
        n = int(target)
    except ValueError as exc:
        raise ValueError(
            "target must be all, N, or from-N; got " + repr(target)
        ) from exc
    return [get_step(n, pipeline=pipeline)]


def format_listing(pipeline: str = "science") -> str:
    steps = PIPELINES[pipeline]
    out = [
        "",
        f"  pipeline: {pipeline}",
        "",
        "  #   step        description",
        "  --  ----------  " + "-" * 45,
    ]
    out.extend(
        "  " + str(s.number) + "   " + s.name.ljust(11) + s.description for s in steps
    )
    out.append("")
    return chr(10).join(out)


def format_command(cmd: Command) -> str:
    if cmd.script == "_audit_inline":
        return "graf audit " + " ".join(cmd.args)
    return "python scripts/" + cmd.script + " " + " ".join(cmd.args)


@dataclass(frozen=True)
class PreflightIssue:
    step: int
    step_name: str
    detail: str


def preflight(steps: list[Step], ctx: Context) -> list[PreflightIssue]:
    """Check every step's scripts exist on disk.

    Does not execute anything. A script that does not exist is
    reported here so the caller can abort before step 1 runs.
    """
    issues: list[PreflightIssue] = []
    for step in steps:
        for cmd in step.build_commands(ctx):
            if cmd.script == "_audit_inline":
                continue
            script_path = ctx.root / "scripts" / cmd.script
            if not script_path.exists():
                issues.append(
                    PreflightIssue(
                        step=step.number,
                        step_name=step.name,
                        detail=f"scripts/{cmd.script} not found",
                    )
                )
    return issues


def format_preflight(
    issues: list[PreflightIssue], steps: list[Step]
) -> str:
    by_step: dict[int, list[str]] = {}
    for issue in issues:
        by_step.setdefault(issue.step, []).append(issue.detail)

    out = [
        "",
        "  #   step        status",
        "  --  ----------  " + "-" * 45,
    ]
    for step in steps:
        details = by_step.get(step.number, [])
        if details:
            status = "NOT READY: " + "; ".join(details)
        else:
            status = "ready"
        out.append(
            "  " + str(step.number) + "   " + step.name.ljust(11) + status
        )
    ready = len(steps) - len(set(i.step for i in issues))
    out.append("")
    out.append(f"  {ready} of {len(steps)} step(s) ready")
    out.append("")
    return chr(10).join(out)
