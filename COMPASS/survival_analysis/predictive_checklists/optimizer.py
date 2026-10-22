"""Weighted M-of-N checklists, independently implementing Zhang et al. (2021).

This is the local IPCW-capable extension, not the authors' implementation.
The objective is lexicographic: weighted classification error, item count,
then the integer decision threshold. The first objective uses normalized,
real-valued weights; duplicate observations sum their weights. There is no
rounding/replication of weights and no conversion of unknown outcomes to zero.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import math
import time
from typing import Any, Dict, Hashable, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class ChecklistModel:
    selected_indices: List[int]
    threshold: int
    n_features: int
    diagnostics: Dict[str, Any]

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return raw unit-weight item counts, not probabilities."""
        matrix = _binary_matrix(X, allow_empty=True)
        if matrix.shape[1] != self.n_features:
            raise ValueError("Prediction matrix has a different number of features")
        return matrix[:, self.selected_indices].sum(axis=1, dtype=np.int64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_score(X) >= self.threshold).astype(np.uint8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "weighted_m_of_n_checklist",
            "schema_version": 1,
            "selected_indices": [int(i) for i in self.selected_indices],
            "threshold": int(self.threshold),
            "n_features": int(self.n_features),
            "diagnostics": deepcopy(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ChecklistModel":
        if value.get("model_type") != "weighted_m_of_n_checklist" or value.get("schema_version") != 1:
            raise ValueError("Unsupported checklist artifact")
        raw = value["selected_indices"]
        selected = [int(i) for i in raw]
        threshold, n_features = int(value["threshold"]), int(value["n_features"])
        if (selected != raw or threshold != value["threshold"]
                or n_features != value["n_features"] or n_features < 1
                or not selected or len(set(selected)) != len(selected)
                or min(selected) < 0 or max(selected) >= n_features
                or not 1 <= threshold <= len(selected)):
            raise ValueError("Invalid checklist parameters")
        return cls(selected, threshold, n_features, deepcopy(value.get("diagnostics", {})))


@dataclass
class _Problem:
    matrix: Any
    lower: np.ndarray
    upper: np.ndarray
    variable_lower: np.ndarray
    variable_upper: np.ndarray

    def constrain(self, coefficients: np.ndarray, lower: float, upper: float) -> "_Problem":
        return _Problem(
            sparse.vstack((self.matrix, sparse.csr_matrix(coefficients[None, :])), format="csc"),
            np.append(self.lower, lower), np.append(self.upper, upper),
            self.variable_lower, self.variable_upper,
        )


@dataclass
class _Result:
    x: Optional[np.ndarray]
    status: str
    optimal: bool
    objective: Optional[float]
    bound: Optional[float]
    gap: Optional[float]
    message: str
    nodes: Optional[int] = None


def _finite(value: Any) -> Optional[float]:
    return float(value) if value is not None and np.isfinite(value) else None


def _binary_matrix(X: np.ndarray, allow_empty: bool = False) -> np.ndarray:
    matrix = np.asarray(X)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or (not allow_empty and not matrix.shape[0]):
        raise ValueError("X must be a nonempty two-dimensional binary matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("X must contain only binary 0/1 values")
    return matrix.astype(np.uint8, copy=False)


def _validate_backend(solver: str) -> None:
    if solver not in {"highs", "cbc", "cplex"}:
        raise ValueError("solver must be 'highs', 'cbc', or 'cplex'")
    dependency = {"cbc": "mip", "cplex": "cplex"}.get(solver)
    if dependency:
        try:
            importlib.import_module(dependency)
        except ImportError as exc:
            raise ImportError(
                "The requested %s solver requires the optional '%s' package; "
                "use solver='highs' for the SciPy backend" % (solver, dependency)
            ) from exc


def _solve(problem: _Problem, c: np.ndarray, solver: str, seconds: float,
           seed: int, incumbent: np.ndarray) -> _Result:
    deadline = time.monotonic() + seconds
    if solver == "highs":
        result = milp(
            c, integrality=np.ones(len(c), dtype=np.uint8),
            bounds=Bounds(problem.variable_lower, problem.variable_upper),
            constraints=LinearConstraint(problem.matrix, problem.lower, problem.upper),
            options={"time_limit": seconds, "mip_rel_gap": 0.0, "presolve": True},
        )
        return _Result(
            result.x, {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(result.status, "error"),
            result.status == 0, _finite(result.fun), _finite(getattr(result, "mip_dual_bound", None)),
            _finite(getattr(result, "mip_gap", None)), str(result.message),
            int(result.mip_node_count) if getattr(result, "mip_node_count", None) is not None else None,
        )
    if solver == "cbc":
        import mip

        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.verbose, model.threads, model.seed = 0, 1, seed
        model.max_mip_gap, model.max_mip_gap_abs = 0.0, 0.0
        variables = [model.add_var(var_type=mip.INTEGER, lb=float(lo), ub=float(hi))
                     for lo, hi in zip(problem.variable_lower, problem.variable_upper)]
        rows = problem.matrix.tocsr()
        for i, (lo, hi) in enumerate(zip(problem.lower, problem.upper)):
            begin, end = rows.indptr[i:i + 2]
            expression = mip.xsum(float(v) * variables[int(j)]
                                 for j, v in zip(rows.indices[begin:end], rows.data[begin:end]))
            if np.isfinite(lo):
                model += expression >= float(lo)
            if np.isfinite(hi):
                model += expression <= float(hi)
        model.objective = mip.xsum(float(v) * variables[j] for j, v in enumerate(c) if v)
        model.start = [(variable, float(value)) for variable, value in zip(variables, incumbent)]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _Result(None, "limit", False, None, None, None, "Budget exhausted building CBC problem")
        status = model.optimize(max_seconds=remaining)
        has_solution = model.num_solutions > 0
        optimal = status == mip.OptimizationStatus.OPTIMAL
        return _Result(
            np.asarray([v.x for v in variables], dtype=float) if has_solution else None,
            "optimal" if optimal else ("limit" if status in {mip.OptimizationStatus.FEASIBLE, mip.OptimizationStatus.NO_SOLUTION_FOUND} else status.name.lower()),
            optimal, _finite(model.objective_value), _finite(model.objective_bound),
            _finite(model.gap), status.name,
        )

    import cplex

    model = cplex.Cplex()
    for setter in (model.set_log_stream, model.set_results_stream, model.set_warning_stream):
        setter(None)
    model.parameters.threads.set(1)
    model.parameters.randomseed.set(seed)
    model.parameters.mip.tolerances.mipgap.set(0.0)
    model.parameters.mip.tolerances.absmipgap.set(0.0)
    model.variables.add(obj=c.tolist(), lb=problem.variable_lower.tolist(),
                        ub=problem.variable_upper.tolist(), types="I" * len(c))
    rows = problem.matrix.tocsr()
    expressions, senses, rhs = [], [], []
    for i, (lo, hi) in enumerate(zip(problem.lower, problem.upper)):
        begin, end = rows.indptr[i:i + 2]
        expression = cplex.SparsePair(rows.indices[begin:end].tolist(), rows.data[begin:end].tolist())
        if np.isfinite(lo):
            expressions.append(expression); senses.append("G"); rhs.append(float(lo))
        if np.isfinite(hi):
            expressions.append(expression); senses.append("L"); rhs.append(float(hi))
    model.linear_constraints.add(lin_expr=expressions, senses="".join(senses), rhs=rhs)
    model.MIP_starts.add(cplex.SparsePair(list(range(len(c))), incumbent.tolist()),
                         model.MIP_starts.effort_level.auto)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        model.end()
        return _Result(None, "limit", False, None, None, None, "Budget exhausted building CPLEX problem")
    model.parameters.timelimit.set(remaining)
    model.solve()
    solution = model.solution
    code, message = solution.get_status(), solution.get_status_string()
    optimal = code == solution.status.MIP_optimal
    has_solution = solution.is_primal_feasible()
    status = "optimal" if optimal else ("limit" if "limit" in message.lower() or "abort" in message.lower() or has_solution else "error")
    result = _Result(
        np.asarray(solution.get_values()) if has_solution else None, status, optimal,
        _finite(solution.get_objective_value()) if has_solution else None,
        _finite(solution.MIP.get_best_objective()),
        _finite(solution.MIP.get_mip_relative_gap()) if has_solution else None, message,
        int(solution.progress.get_num_nodes_processed()),
    )
    model.end()
    return result


def fit_checklist(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
                  groups: Optional[Sequence[Hashable]] = None, max_items: int = 5,
                  time_limit: float = 60, solver: str = "highs", seed: int = 42,
                  loss_tolerance: float = 1e-8, compress: bool = True) -> ChecklistModel:
    """Fit a weighted checklist with at most one selected item per source group.

    ``groups`` contains one hashable group ID per binary feature. Loss is
    normalized by the total sample weight. Later stages may increase that loss
    by at most ``loss_tolerance``; this is an explicit numerical tie tolerance.
    The time limit is shared across all stages. A deterministic best singleton
    is retained if a solver reaches its limit before finding an incumbent; its
    diagnostics explicitly identify the heuristic fallback and missing proof.
    HiGHS is deterministic through SciPy's documented API; ``seed`` configures
    CBC/CPLEX only. This function neither fits censoring weights nor calibrates
    scores: both belong in the training-fold pipeline.
    """
    started = time.monotonic()
    matrix = _binary_matrix(X)
    target = np.asarray(y)
    n_samples, n_features = matrix.shape
    if target.shape != (n_samples,) or not np.all((target == 0) | (target == 1)):
        raise ValueError("y must be a one-dimensional binary vector aligned with X")
    weights = np.ones(n_samples, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weight must be finite, nonnegative, and aligned with X")
    total_weight = float(weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("sample_weight must have a positive finite sum")
    if isinstance(max_items, bool) or int(max_items) != max_items or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    if not np.isfinite(loss_tolerance) or not 0 <= loss_tolerance < 1:
        raise ValueError("loss_tolerance must be finite and in [0, 1)")
    _validate_backend(solver)
    group_ids = list(range(n_features)) if groups is None else list(groups)
    if len(group_ids) != n_features:
        raise ValueError("groups must have one source group ID per feature")
    grouped: Dict[Hashable, List[int]] = {}
    try:
        for j, group in enumerate(group_ids):
            grouped.setdefault(group, []).append(j)
    except TypeError as exc:
        raise ValueError("Feature group IDs must be hashable") from exc
    max_items = min(int(max_items), len(grouped), n_features)
    keep = weights > 0
    training_X, training_y = matrix[keep], target[keep].astype(np.uint8)
    normalized = weights[keep] / total_weight
    if compress:
        unique, inverse = np.unique(np.column_stack((training_X, training_y)), axis=0, return_inverse=True)
        normalized = np.bincount(inverse, weights=normalized, minlength=len(unique))
        training_X, training_y = unique[:, :-1], unique[:, -1]
    n_rows = len(training_y)
    # Variables: selected items lambda[0:p], mistake indicators z[0:n], M.
    n_vars = n_features + n_rows + 1
    sign = np.where(training_y == 1, -1.0, 1.0)
    event_constraints = sparse.hstack((
        sparse.csr_matrix(training_X).multiply(sign[:, None]),
        -max_items * sparse.eye(n_rows, format="csr"),
        sparse.csr_matrix((-sign)[:, None]),
    ), format="csc")
    extra_rows, extra_upper = [], []
    size = np.zeros(n_vars); size[:n_features] = 1
    threshold = np.zeros(n_vars); threshold[-1] = 1
    extra_rows.extend((size, threshold - size))
    extra_upper.extend((max_items, 0))
    for indices in grouped.values():
        if len(indices) > 1:
            row = np.zeros(n_vars); row[indices] = 1
            extra_rows.append(row); extra_upper.append(1)
    problem_matrix = sparse.vstack((event_constraints, sparse.csr_matrix(np.asarray(extra_rows))), format="csc")
    upper = np.concatenate((np.where(training_y == 1, 0.0, -1.0), extra_upper))
    variable_lower, variable_upper = np.zeros(n_vars), np.ones(n_vars)
    variable_lower[-1], variable_upper[-1] = 1, max_items
    problem = _Problem(problem_matrix, np.full(len(upper), -np.inf), upper, variable_lower, variable_upper)
    loss = np.zeros(n_vars); loss[n_features:-1] = normalized

    # Always retain a feasible incumbent; never describe this as a solver proof.
    singleton_losses = normalized @ (training_X != training_y[:, None])
    best_index = int(np.argmin(singleton_losses))
    incumbent = np.zeros(n_vars); incumbent[best_index] = 1; incumbent[-1] = 1
    incumbent[n_features:-1] = training_X[:, best_index] != training_y
    source, stages = "single_item_fallback", []
    original_bound, loss_limit = None, None
    loss_optimal, lexicographic_complete = False, False
    deadline = started + float(time_limit)

    for stage_index, (name, objective) in enumerate((("weighted_error", loss), ("item_count", size), ("threshold", threshold))):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        stage_started = time.monotonic()
        result = _solve(problem, objective, solver, remaining, int(seed), incumbent)
        if result.status not in {"optimal", "limit"}:
            raise RuntimeError("Checklist %s stage failed (%s): %s" % (name, result.status, result.message))
        accepted = False
        if result.x is not None and np.all(np.isfinite(result.x)):
            candidate = np.rint(result.x)
            selected = np.flatnonzero(candidate[:n_features])
            M = int(candidate[-1])
            valid = (np.all(np.abs(result.x - candidate) <= 1e-5)
                     and np.all((candidate[:n_features] == 0) | (candidate[:n_features] == 1))
                     and 1 <= M <= len(selected) <= max_items
                     and all(candidate[indices].sum() <= 1 for indices in grouped.values()))
            if valid:
                candidate[n_features:-1] = (training_X[:, selected].sum(axis=1) >= M) != training_y
                candidate_loss = float(loss @ candidate)
                # Recheck against actual predictions, independent of solver z.
                valid = (loss_limit is None or candidate_loss <= loss_limit + 1e-12)
                if stage_index == 2:
                    valid = valid and int(size @ candidate) == int(size @ incumbent)
                if valid and float(objective @ candidate) <= float(objective @ incumbent) + 1e-12:
                    incumbent, source, accepted = candidate, "solver", True
        stage = {
            "name": name, "status": result.status, "optimal": result.optimal,
            "accepted_incumbent": accepted, "objective": result.objective,
            "lower_bound": result.bound, "relative_gap": result.gap,
            "message": result.message, "nodes": result.nodes,
            "elapsed_seconds": time.monotonic() - stage_started,
        }
        stages.append(stage)
        if stage_index == 0:
            original_bound = result.bound
            loss_optimal = bool(result.optimal and accepted and result.bound is not None
                                and float(loss @ incumbent) - result.bound <= loss_tolerance + 1e-12)
        # Do not trade primary loss for simplicity while its optimum is unproved.
        if not result.optimal or not accepted or (stage_index == 0 and not loss_optimal):
            break
        if stage_index == 0:
            # Anchor the tolerance to the certified lower bound as well as the
            # incumbent so solver and tie tolerances cannot accumulate.
            loss_limit = min(float(loss @ incumbent), max(0.0, result.bound)) + loss_tolerance
            problem = problem.constrain(loss, -np.inf, loss_limit)
        elif stage_index == 1:
            item_count = int(size @ incumbent)
            problem = problem.constrain(size, item_count, item_count)
        else:
            lexicographic_complete = True

    selected_indices = np.flatnonzero(incumbent[:n_features]).astype(int).tolist()
    error_rate = float(loss @ incumbent)
    bound = max(0.0, original_bound) if original_bound is not None else None
    absolute_gap = max(0.0, error_rate - bound) if bound is not None else None
    relative_gap = (absolute_gap / error_rate if error_rate else (0.0 if absolute_gap == 0 else None)) if absolute_gap is not None else None
    diagnostics = {
        "implementation": "local_weighted_paper_formulation",
        "solver": solver, "status": "optimal" if lexicographic_complete else "feasible_unproved",
        "has_feasible_solution": True, "solution_source": source,
        "optimal": lexicographic_complete, "loss_optimal": loss_optimal,
        "lexicographic_complete": lexicographic_complete,
        "loss_tolerance": float(loss_tolerance), "loss_normalization": "sum_of_sample_weights",
        "weighted_error": error_rate * total_weight, "weighted_error_rate": error_rate,
        "loss_lower_bound": bound, "loss_upper_bound": error_rate,
        "absolute_gap": absolute_gap, "optimality_gap": relative_gap,
        "n_samples": n_samples, "n_positive_weight_samples": int(keep.sum()),
        "n_compressed_rows": n_rows, "compression": bool(compress),
        "total_weight": total_weight, "n_features": n_features,
        "n_selected": len(selected_indices), "max_items": max_items,
        "requested_seed": int(seed), "solver_seed": None if solver == "highs" else int(seed),
        "time_limit_seconds": float(time_limit), "elapsed_seconds": time.monotonic() - started,
        "stages": stages,
    }
    return ChecklistModel(selected_indices, int(incumbent[-1]), n_features, diagnostics)
