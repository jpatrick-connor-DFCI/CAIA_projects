"""CLI: ``python -m COMPASS.multivariable_models.survival_tree --output-dir <dir>``."""
from __future__ import annotations

import argparse
import json

from survival_common import federated_endpoint_transfer as fet

from .local_tree import TreeConfig, run, write_outputs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(fet.DEFAULT_DATA))
    ap.add_argument("--lab-mapping", default=str(fet.DEFAULT_LAB_MAPPING))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", help="JSON file overriding TreeConfig fields")
    ap.add_argument("--n-bootstrap", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)

    overrides = json.load(open(args.config)) if args.config else {}
    for key in ("n_bootstrap", "seed"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    cfg = TreeConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()})
    local = fet.read_local_frame(args.data, cfg.endpoints)
    result = run(local, fet.load_lab_mapping(args.lab_mapping), cfg)
    write_outputs(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
