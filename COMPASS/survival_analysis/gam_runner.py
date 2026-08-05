"""Notebook-friendly entry points for the two COMPASS R GAM stages."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable


SURVIVAL_DIR = Path(__file__).resolve().parent


def _landmarks(values: Iterable[int]) -> list[int]:
    landmarks = [int(value) for value in values]
    if not landmarks:
        raise ValueError("At least one landmark day is required.")
    return landmarks


def _rscript_executable(rscript: str | Path) -> str:
    requested = str(rscript)
    resolved = shutil.which(requested)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find {requested!r}. Install R or pass the full Rscript path."
        )
    return resolved


def _run(command: list[str], label: str) -> None:
    print(f"[run ] {label}")
    print("       " + " ".join(command))
    subprocess.run(command, check=True)
    print(f"[done] {label}\n")


def run_gam_trajectory_features(
    *,
    inputs_dir: str | Path,
    landmark_days: Iterable[int],
    force: bool = True,
    rscript: str | Path = "Rscript",
    k_pop: int = 10,
    k_pat: int = 5,
    trailing_window_days: int = 180,
    nthreads: int = 1,
    fit_split: str = "all",
) -> list[Path]:
    """Run hierarchical trajectory GAMs and return their feature CSV paths."""
    inputs_dir = Path(inputs_dir)
    landmarks = _landmarks(landmark_days)
    outputs = [
        inputs_dir / f"gam_trajectory_features_landmark{landmark}.csv"
        for landmark in landmarks
    ]
    diagnostics = [
        inputs_dir / f"gam_fit_diagnostics_landmark{landmark}.csv"
        for landmark in landmarks
    ]
    if not force and all(path.exists() for path in outputs + diagnostics):
        print("[skip] GAM trajectory features -> all requested outputs exist")
        return outputs

    command = [
        _rscript_executable(rscript),
        str(SURVIVAL_DIR / "gam_trajectory_features.R"),
        "--inputs-dir",
        str(inputs_dir),
        "--landmark-days",
        ",".join(map(str, landmarks)),
        "--k-pop",
        str(k_pop),
        "--k-pat",
        str(k_pat),
        "--trailing-window-days",
        str(trailing_window_days),
        "--nthreads",
        str(nthreads),
        "--fit-split",
        fit_split,
    ]
    _run(command, f"GAM trajectory features ({','.join(map(str, landmarks))}d)")
    return outputs


def run_gam_cox_nonlinearity(
    *,
    inputs_dir: str | Path,
    model_output_dir: str | Path,
    landmark_days: Iterable[int],
    force: bool = True,
    rscript: str | Path = "Rscript",
    min_events_per_feature: int = 10,
    k_smooth: int = 10,
) -> list[Path]:
    """Run nonlinear Cox GAMs using each landmark's exact univariate feature list."""
    inputs_dir = Path(inputs_dir)
    model_output_dir = Path(model_output_dir)
    output_dir = model_output_dir / "cox" / "gam_nonlinearity"
    output_dir.mkdir(parents=True, exist_ok=True)
    rscript_executable = _rscript_executable(rscript)
    outputs: list[Path] = []

    for landmark in _landmarks(landmark_days):
        selection_path = (
            model_output_dir
            / "cox"
            / f"landmark_{landmark}"
            / "both"
            / "cox_agg_feature_selection.csv"
        )
        if not selection_path.exists():
            raise FileNotFoundError(
                f"Missing {selection_path}. Run the landmark's univariate model first."
            )

        output_path = output_dir / f"gam_cox_nonlinearity_landmark{landmark}.csv"
        outputs.append(output_path)
        if output_path.exists() and not force:
            print(f"[skip] GAM Cox nonlinearity landmark +{landmark}d -> output exists")
            continue

        command = [
            rscript_executable,
            str(SURVIVAL_DIR / "gam_cox_nonlinearity.R"),
            "--inputs-dir",
            str(inputs_dir),
            "--output-dir",
            str(output_dir),
            "--landmark-days",
            str(landmark),
            "--feature-selection-csv",
            str(selection_path),
            "--min-events-per-feature",
            str(min_events_per_feature),
            "--k-smooth",
            str(k_smooth),
        ]
        _run(command, f"GAM Cox nonlinearity landmark +{landmark}d")

    return outputs
