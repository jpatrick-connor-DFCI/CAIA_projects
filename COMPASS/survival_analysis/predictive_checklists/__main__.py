"""Run with python -m COMPASS.survival_analysis.predictive_checklists."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inputs-root", type=Path, default=Path(
        "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis"))
    parser.add_argument("--platinum-inputs", type=Path)
    parser.add_argument("--nepc-inputs", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoints", nargs="+", choices=["platinum", "nepc"])
    parser.add_argument("--landmarks", nargs="+", type=int)
    parser.add_argument("--horizons", nargs="+", type=int, help="Days after the landmark")
    parser.add_argument("--solver", choices=["highs", "cbc", "cplex"])
    parser.add_argument("--solver-time-limit", type=float, help="Seconds per checklist fit")
    parser.add_argument("--n-folds", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-nepc-sensitivity", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_true")
    existing.add_argument("--resume", action="store_true",
                          help="Reuse completed cells of an interrupted run with identical settings")
    parser.add_argument("--demo", action="store_true", help="Generate clearly labeled synthetic inputs")
    args = parser.parse_args(argv)
    config = RunConfig.load(args.config)
    for key in ("endpoints", "landmarks", "horizons", "solver", "solver_time_limit",
                "n_folds", "bootstrap_samples"):
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    if args.no_baselines:
        config.baselines = ()
    if args.no_nepc_sensitivity:
        config.resolved_nepc_sensitivity = False
    config.validate()
    if (args.output_dir / "run_manifest.json").exists() and not (args.overwrite or args.resume):
        parser.error("Output run exists; choose a new --output-dir, pass --overwrite (archives prior "
                     "results) or --resume (reuses completed cells)")
    if args.demo:
        from .demo import build_demo_inputs
        args.inputs_root = args.output_dir / "synthetic_inputs"
        build_demo_inputs(args.inputs_root, config.landmarks, seed=config.seed)
    from .runner import run
    input_dirs = {
        "platinum": args.platinum_inputs or args.inputs_root / "prediction_inputs_adt",
        "nepc": args.nepc_inputs or args.inputs_root / "prediction_inputs_adt_nepc",
    }
    return run(config, input_dirs, args.output_dir, preflight_only=args.preflight_only,
               overwrite=args.overwrite, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
