"""Behavioral checks against exhaustive feasible checklist enumeration."""

from itertools import combinations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from COMPASS.survival_analysis.predictive_checklists import optimizer
from COMPASS.survival_analysis.predictive_checklists.optimizer import ChecklistModel, fit_checklist
from COMPASS.survival_analysis.predictive_checklists.upstream import (
    UPSTREAM_COMMIT, inspect_checkout, provenance,
)


def enumerate_optimum(X, y, weights, groups, max_items):
    candidates = []
    for count in range(1, min(max_items, X.shape[1]) + 1):
        for indices in combinations(range(X.shape[1]), count):
            if len({groups[i] for i in indices}) != count:
                continue
            scores = X[:, indices].sum(axis=1)
            for threshold in range(1, count + 1):
                loss = float(weights @ ((scores >= threshold) != y)) / weights.sum()
                candidates.append((loss, count, threshold))
    best_loss = min(row[0] for row in candidates)
    return min((row for row in candidates if row[0] <= best_loss + 1e-8), key=lambda row: (row[1], row[2]))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("compress", [False, True])
def test_weighted_global_optimum_matches_exhaustive_search(seed, compress):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(15, 5), dtype=np.uint8)
    y = rng.integers(0, 2, size=15, dtype=np.uint8)
    weights = rng.uniform(0.15, 3.9, size=15)
    weights[0] = 0
    groups = ["a", "a", "b", "c", "d"]
    expected = enumerate_optimum(X, y, weights, groups, 3)
    model = fit_checklist(X, y, weights, groups, max_items=3, time_limit=10, compress=compress)
    actual = (weights @ (model.predict(X) != y) / weights.sum(),
              len(model.selected_indices), model.threshold)
    assert actual[0] == pytest.approx(expected[0], abs=1e-8)
    assert actual[1:] == expected[1:]
    assert len({groups[i] for i in model.selected_indices}) == len(model.selected_indices)
    assert model.diagnostics["optimal"]
    assert model.diagnostics["loss_lower_bound"] == pytest.approx(expected[0], abs=1e-8)


def test_fractional_weights_change_selection_and_duplicate_weights_sum():
    X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8)
    y = np.array([1, 1, 1, 0], dtype=np.uint8)
    weights = np.array([0.10, 0.13, 2.71, 0.19])
    unweighted = fit_checklist(X, y, max_items=1)
    weighted = fit_checklist(X, y, weights, max_items=1)
    assert unweighted.selected_indices == [0]
    assert weighted.selected_indices == [1]
    assert weighted.diagnostics["n_compressed_rows"] == 3
    assert weighted.diagnostics["weighted_error"] == pytest.approx(0.42)
    rescaled = fit_checklist(X, y, weights * 1000.13, max_items=1)
    assert rescaled.selected_indices == weighted.selected_indices
    assert rescaled.diagnostics["weighted_error_rate"] == pytest.approx(weighted.diagnostics["weighted_error_rate"])


def test_unit_score_and_json_roundtrip():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y = np.array([0, 1, 1, 1], dtype=np.uint8)
    model = fit_checklist(X, y, max_items=2)
    assert model.selected_indices == [0, 1]
    assert model.threshold == 1
    np.testing.assert_array_equal(model.predict_score(X), [0, 1, 1, 2])
    restored = ChecklistModel.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(restored.predict(X), y)
    assert restored.predict_score(np.empty((0, 2), dtype=np.uint8)).shape == (0,)
    with pytest.raises(ValueError, match="different number"):
        restored.predict(np.zeros((2, 3), dtype=np.uint8))


def test_group_constraint_prevents_two_thresholds_from_one_source():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y = np.array([0, 1, 1, 1], dtype=np.uint8)
    model = fit_checklist(X, y, groups=["same", "same"], max_items=2)
    assert len(model.selected_indices) == 1
    assert model.diagnostics["weighted_error"] == 1


def test_timeout_retains_feasible_singleton_and_does_not_claim_optimal(monkeypatch):
    monkeypatch.setattr(optimizer, "_solve", lambda *args: optimizer._Result(
        None, "limit", False, None, 0.0, None, "test timeout"))
    X = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.array([0, 1, 1], dtype=np.uint8)
    model = fit_checklist(X, y)
    assert model.diagnostics["solution_source"] == "single_item_fallback"
    assert not model.diagnostics["optimal"]
    assert len(model.diagnostics["stages"]) == 1
    np.testing.assert_array_equal(model.predict(X), y)


def test_partial_primary_solve_stops_before_tiebreaks(monkeypatch):
    actual_solve = optimizer._solve

    def limited(*args):
        result = actual_solve(*args)
        result.optimal, result.status = False, "limit"
        return result

    monkeypatch.setattr(optimizer, "_solve", limited)
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    model = fit_checklist(X, np.array([0, 1, 1, 1]), max_items=2)
    assert not model.diagnostics["lexicographic_complete"]
    assert model.diagnostics["solution_source"] == "solver"
    assert [stage["name"] for stage in model.diagnostics["stages"]] == ["weighted_error"]


def test_solver_optimal_label_does_not_override_requested_loss_tolerance(monkeypatch):
    actual_solve = optimizer._solve

    def loose_certificate(*args):
        result = actual_solve(*args)
        result.bound = result.objective - 1e-4
        return result

    monkeypatch.setattr(optimizer, "_solve", loose_certificate)
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    model = fit_checklist(X, np.array([0, 1, 1, 0]), max_items=2)
    assert not model.diagnostics["loss_optimal"]
    assert not model.diagnostics["optimal"]
    assert len(model.diagnostics["stages"]) == 1


@pytest.mark.parametrize("backend,dependency", [("cbc", "mip"), ("cplex", "cplex")])
def test_optional_solver_matches_weighted_exhaustive_optimum(backend, dependency):
    pytest.importorskip(dependency)
    rng = np.random.default_rng(117)
    X = rng.integers(0, 2, size=(20, 5), dtype=np.uint8)
    y = rng.integers(0, 2, size=20, dtype=np.uint8)
    weights = rng.uniform(0.03, 4.5, size=20)
    groups = ["a", "a", "b", "c", "d"]
    expected = enumerate_optimum(X, y, weights, groups, 3)
    model = fit_checklist(X, y, weights, groups, max_items=3, solver=backend)
    assert model.diagnostics["weighted_error_rate"] == pytest.approx(expected[0], abs=1e-8)
    assert (len(model.selected_indices), model.threshold) == expected[1:]
    assert model.diagnostics["optimal"]


@pytest.mark.parametrize("kwargs", [
    {"sample_weight": [-1, 1]}, {"sample_weight": [0, 0]},
    {"sample_weight": [1, float("nan")]}, {"sample_weight": [1]},
    {"groups": ["too_short"]}, {"max_items": 0}, {"time_limit": 0},
    {"loss_tolerance": -0.1}, {"solver": "unknown"},
])
def test_invalid_training_input_fails(kwargs):
    with pytest.raises(ValueError):
        fit_checklist(np.array([[0, 1], [1, 0]], dtype=np.uint8), np.array([0, 1]), **kwargs)


def test_nonbinary_input_is_not_silently_cast():
    with pytest.raises(ValueError, match="binary"):
        fit_checklist(np.array([[0.3, 1], [1, 0]]), np.array([0, 1]))


def test_provenance_identifies_local_extension_and_exact_commit():
    details = provenance()
    assert details["upstream_source_bundled"] is False
    assert details["upstream_license_text_present"] is False
    assert "independent" in details["weighted_implementation"]
    pin = Path(optimizer.__file__).with_name("UPSTREAM_COMMIT").read_text().strip()
    assert pin == UPSTREAM_COMMIT


def test_reference_rejects_noncheckout(tmp_path):
    with pytest.raises(ValueError, match="IPChecklists"):
        inspect_checkout(tmp_path)


@pytest.mark.parametrize("labels", [[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]])
def test_optional_actual_upstream_parity(labels):
    checkout = os.environ.get("PREDICTIVE_CHECKLISTS_UPSTREAM")
    if not checkout:
        pytest.skip("Set PREDICTIVE_CHECKLISTS_UPSTREAM to the clean pinned authors' checkout")
    if any(importlib.util.find_spec(name) is None for name in ("cplex", "optbinning")):
        pytest.skip("Upstream reference dependencies are not installed")
    # CBC/OR-Tools wheels can ship conflicting native library names on macOS.
    # Reference parity intentionally uses a fresh interpreter, just as an
    # isolated authors' reference environment would.
    script = """
import json, sys
import numpy as np
from COMPASS.survival_analysis.predictive_checklists.optimizer import fit_checklist
from COMPASS.survival_analysis.predictive_checklists.upstream import fit_upstream_reference
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
y = np.array(json.loads(sys.argv[2]), dtype=np.uint8)
ours = fit_checklist(X, y, max_items=2)
authors = fit_upstream_reference(X, y, sys.argv[1], max_items=2)
np.testing.assert_array_equal(authors.predict(X), ours.predict(X))
assert (len(authors.selected_indices), authors.threshold) == (len(ours.selected_indices), ours.threshold)
"""
    result = subprocess.run([sys.executable, "-c", script, checkout, json.dumps(labels)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
