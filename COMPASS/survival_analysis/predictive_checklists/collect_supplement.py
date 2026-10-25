"""Flatten finished checklist runs into the tables behind the checklist supplement.

Torch-free and read-only on the run directories. One call covers the feature-set
runs (for example labs only and labs + Gleason + somatic) side by side and writes:

* checklist_cells.csv  - one row per (feature set, endpoint, landmark): fit status,
  M-of-N threshold, item count and held-out cohort size.
* checklist_items_long.csv - one row per checklist item.
* checklist_metrics.csv - held-out metrics for the checklist and each baseline.

`render_supplement.R` draws the figures from these CSVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CELL_COLUMNS = ["feature_set", "endpoint", "landmark_days", "horizon_days", "population",
                "status", "reason", "threshold", "n_items", "n_test", "n_events_total"]
ITEM_COLUMNS = ["feature_set", "endpoint", "landmark_days", "horizon_days", "population",
                "item_order", "name", "feature", "op", "threshold", "group"]


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def collect(runs: dict[str, Path], horizon: int, population: str = "full"):
    """Return (cells, items, metrics) frames for `runs` ({feature_set: run_dir})."""
    cells, items, metrics = [], [], []
    for feature_set, run_dir in runs.items():
        run_dir = Path(run_dir)
        run_metrics = _read_json(run_dir / "metrics.json") or []
        for row in pd.json_normalize(run_metrics).to_dict("records"):
            if row.get("horizon_days") != horizon or row.get("population") != population:
                continue
            row = {key.replace("bootstrap.intervals.", "").replace(".", "_"): value
                   for key, value in row.items() if key != "calibration"}
            # The run labels its own feature set; the supplement keys on the run it came from.
            metrics.append({**row, "feature_set": feature_set})
        for endpoint_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name in ("platinum", "nepc")):
            for landmark_dir in sorted(endpoint_dir.glob("landmark_*"),
                                       key=lambda p: int(p.name.split("_")[1])):
                landmark = int(landmark_dir.name.split("_")[1])
                folder = landmark_dir / population / f"horizon_{horizon}"
                audit = _read_json(landmark_dir / population / "cohort_audit.json") or {}
                key = dict(feature_set=feature_set, endpoint=endpoint_dir.name,
                           landmark_days=landmark, horizon_days=horizon, population=population)
                cell = {**key, "n_test": audit.get("n_test"), "n_events_total": audit.get("n_events")}
                checklist = _read_json(folder / "checklist.json")
                if checklist is None:
                    failure = _read_json(folder / "failure.json") or {}
                    cell_metrics = _read_json(folder / "metrics.json") or []
                    status = next((r for r in cell_metrics if r.get("model") == "checklist"), failure)
                    cells.append({**cell, "status": status.get("status", "not_fit"),
                                  "reason": status.get("reason", "No checklist exported")})
                    continue
                cells.append({**cell, "status": "ok", "reason": None,
                              "threshold": checklist["threshold"], "n_items": len(checklist["items"])})
                for order, item in enumerate(checklist["items"], start=1):
                    items.append({**key, "item_order": order, **{
                        field: item.get(field) for field in ("name", "feature", "op", "threshold", "group")}})
    return (pd.DataFrame(cells, columns=CELL_COLUMNS),
            pd.DataFrame(items, columns=ITEM_COLUMNS),
            pd.DataFrame(metrics))


def write(runs: dict[str, Path], output_dir: Path, horizon: int = 1825, population: str = "full"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, items, metrics = collect(runs, horizon, population)
    cells.to_csv(output_dir / "checklist_cells.csv", index=False)
    items.to_csv(output_dir / "checklist_items_long.csv", index=False)
    metrics.to_csv(output_dir / "checklist_metrics.csv", index=False)
    return cells, items, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labs-run", type=Path, required=True,
                        help="checklist run directory fit on labs only")
    parser.add_argument("--trio-run", type=Path, required=True,
                        help="checklist run directory fit on labs + Gleason + somatic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1825)
    args = parser.parse_args(argv)
    cells, _, _ = write({"labs": args.labs_run, "labs_gleason_somatic": args.trio_run},
                        args.output_dir, args.horizon)
    print(cells.to_string(index=False))


if __name__ == "__main__":
    main()
