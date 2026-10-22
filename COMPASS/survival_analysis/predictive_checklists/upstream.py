"""Optional reference adapter for a separately checked-out authors' repository.

No upstream source is vendored or downloaded at import time. Weighted survival
training uses optimizer.py; this adapter executes the actual pinned upstream
implementation for unweighted parity checks. Upstream declares 'BSD' in setup.py
but does not include a license-text file at the pinned commit.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import numpy as np

from .optimizer import ChecklistModel, _binary_matrix


UPSTREAM_ORIGIN = "https://github.com/MLforHealth/predictive_checklists"
UPSTREAM_COMMIT = "d205245b9f62dd2cfd977171f0e5073c13d6aebf"
PAPER_URL = "https://arxiv.org/html/2112.01020v1"


def provenance() -> Dict[str, Any]:
    return {
        "paper": PAPER_URL, "upstream_origin": UPSTREAM_ORIGIN,
        "upstream_commit": UPSTREAM_COMMIT, "upstream_version": "0.3",
        "upstream_declared_license": "BSD (setup.py metadata)",
        "upstream_license_text_present": False,
        "weighted_implementation": "independent local extension of paper formulation",
        "upstream_source_bundled": False,
    }


def _git(checkout: Path, *arguments: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(checkout), *arguments], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Unable to inspect upstream checkout: %s" % checkout) from exc


def inspect_checkout(checkout: Any) -> Dict[str, str]:
    """Require the exact pinned official checkout and unchanged tracked files."""
    path = Path(checkout).expanduser().resolve()
    if not (path / "IPChecklists" / "dataset.py").is_file():
        raise ValueError("Upstream checkout must contain IPChecklists/dataset.py")
    if Path(_git(path, "rev-parse", "--show-toplevel")).resolve() != path:
        raise ValueError("The supplied directory is not the upstream repository root")
    commit = _git(path, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Upstream commit mismatch: expected %s, got %s" % (UPSTREAM_COMMIT, commit))
    origin = _git(path, "remote", "get-url", "origin")
    normalized = origin.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized not in {UPSTREAM_ORIGIN, "git@github.com:MLforHealth/predictive_checklists"}:
        raise ValueError("Upstream origin is not the pinned authors' repository")
    if _git(path, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Upstream checkout has modified tracked files; parity requires clean source")
    return {"path": str(path), "origin": origin, "commit": commit}


@dataclass
class UpstreamAPI:
    BinaryDataset: Any
    ChecklistMIP: Any
    MaxNumFeatureConstraint: Any
    MConstraint: Any
    backend: str
    checkout: Dict[str, str]


def load_upstream(checkout: Any, backend: str = "cplex") -> UpstreamAPI:
    """Load real upstream classes after verifying their source checkout.

    The upstream dataset imports CPLEX callbacks even for CBC. Dependency
    failures are reported explicitly; no local replacement classes or fake
    solver modules are injected into the authors' code. Run reference checks in
    a fresh process: CBC and OR-Tools wheels can load conflicting native solver
    library names on some platforms.
    """
    if backend not in {"cplex", "cbc"}:
        raise ValueError("The authors' implementation supports cplex or cbc, not highs")
    details = inspect_checkout(checkout)
    existing = sys.modules.get("IPChecklists")
    if existing is not None and Path(existing.__file__).resolve().parent != Path(details["path"]) / "IPChecklists":
        raise RuntimeError("Another IPChecklists package is already imported; use a fresh Python process")
    sys.path.insert(0, details["path"])
    try:
        dataset = importlib.import_module("IPChecklists.dataset")
        suffix = "cplex" if backend == "cplex" else "pythonmip"
        model = importlib.import_module("IPChecklists.model_" + suffix)
        constraints = importlib.import_module("IPChecklists.constraints_" + suffix)
    except ImportError as exc:
        raise ImportError(
            "Pinned upstream reference dependencies are unavailable (%s). "
            "Install the upstream dependencies in an isolated reference environment; "
            "its dataset module requires cplex even when backend='cbc'." % exc
        ) from exc
    finally:
        sys.path.remove(details["path"])
    return UpstreamAPI(dataset.BinaryDataset, model.ChecklistMIP,
                       constraints.MaxNumFeatureConstraint, constraints.MConstraint, backend, details)


def fit_upstream_reference(X: np.ndarray, y: np.ndarray, checkout: Any,
                           max_items: int = 5, time_limit: float = 60,
                           backend: str = "cplex", seed: int = 42) -> ChecklistModel:
    """Execute authors' unweighted model on already-binary independent items.

    This deliberately has no sample_weight argument: upstream supports class
    weights, not arbitrary patient weights. Use only as an unweighted reference,
    never as an IPCW substitute. Source-feature grouping is handled by the local
    weighted optimizer and is outside this small reference adapter's scope.
    """
    import pandas as pd

    matrix = _binary_matrix(X)
    target = np.asarray(y)
    if target.shape != (len(matrix),) or not np.all((target == 0) | (target == 1)):
        raise ValueError("y must be a binary vector aligned with X")
    api = load_upstream(checkout, backend)
    columns = ["item_%d" % i for i in range(matrix.shape[1])]
    # Upstream negates X coefficients directly; unsigned inputs would wrap
    # negative values to 255 and change its mathematical constraints.
    frame = pd.DataFrame(matrix.astype(np.int64), columns=columns)
    frame["outcome"] = target
    dataset = api.BinaryDataset(frame, target_name="outcome", pos_label=1,
                                already_binarized=True, add_complements=False,
                                use_missing_features=False)
    kwargs = {"solver": "CBC"} if backend == "cbc" else {}
    model = api.ChecklistMIP(dataset, cost_func="01", one_feature_per_group=True, **kwargs)
    model.build_problem(N_constraint=api.MaxNumFeatureConstraint("<=", min(max_items, len(columns))),
                        M_constraint=api.MConstraint(">=", 1))
    model.solve(max_seconds=time_limit, threads=1, seed=seed)
    if not model.solved:
        raise RuntimeError("The authors' solver did not find a feasible reference checklist")
    checklist = model.to_checklist()
    selected_names = checklist.get_selected_cols()
    selected = [columns.index(name) for name in selected_names]
    diagnostics = provenance()
    diagnostics.update({"implementation": "authors_pinned_unweighted_reference",
                        "solver": backend, "checkout": api.checkout,
                        "n_selected": len(selected)})
    return ChecklistModel(selected, int(round(checklist.M)), len(columns), diagnostics)
