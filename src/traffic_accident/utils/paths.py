from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    metrics: Path
    figures: Path
    artifacts: Path


def make_run_paths(output_root: str | Path = "outputs/runs", run_name: str | None = None) -> RunPaths:
    base = Path(output_root)
    if not base.is_absolute():
        base = project_root() / base
    run_id = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base / run_id
    return RunPaths(
        root=root,
        checkpoints=root / "checkpoints",
        metrics=root / "metrics",
        figures=root / "figures",
        artifacts=root / "artifacts",
    )

