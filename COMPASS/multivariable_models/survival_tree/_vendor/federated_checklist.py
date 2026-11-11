"""Federation math for the predictive checklist (server side, framework-free).

Nothing here touches NVFlare or patient rows; the controller feeds it pooled
counts and tables.  Four pieces:

* ``OrderStatSolver`` -- exact pooled order statistics by bracket refinement, so
  quantile thresholds and medians equal ``np.quantile`` on the pooled rows.
* ``plan_binarizer`` / ``assemble_candidates`` -- PROFILE's ``FeatureBinarizer``
  fit expressed over pooled counts.
* ``best_threshold`` / ``select_best`` / ``search`` -- pooled scoring of
  candidate checklists and the beam search that refines the sites' local optima.
* ``site_bound`` / ``combine_bounds`` -- the certificate: the sum of the sites'
  local loss lower bounds is a lower bound on the pooled optimum, so
  ``gap = pooled_loss - sum(bounds)`` bounds the distance to the centralized fit.
"""
from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np

from checklist_core import feature_group

LOSS_TOL = 1e-8


# --------------------------------------------------------------------------- #
# Exact pooled order statistics
# --------------------------------------------------------------------------- #

def quantile_ranks(n: int, q: float) -> tuple[int, int, float]:
    """The two 0-based order statistics ``np.quantile`` (linear) blends and the
    weight on the upper one (same as ``federated_deephit.quantile_ranks``)."""
    h = (int(n) - 1) * float(q)
    return int(np.floor(h)), int(np.ceil(h)), float(h - np.floor(h))


def numpy_lerp(a: float, b: float, t: float) -> float:
    """``numpy``'s linear-interpolation formula, so rounding matches it."""
    diff = b - a
    value = a + diff * t
    return float(b - diff * (1.0 - t)) if t >= 0.5 else float(value)


def bracket_edges(lo: float, hi: float, bins: int) -> list[float]:
    """Interior bin edges of the bracket ``[lo, hi]``; the last bin is closed.

    Edges lie in ``(lo, hi]``, so every bin is strictly narrower than the
    bracket and each round shrinks it.  A degenerate numeric range falls back to
    the single edge ``hi`` (bins ``[lo, hi)`` and ``[hi, hi]``).
    """
    edges = np.unique(np.linspace(lo, hi, int(bins) + 1)[1:-1])
    edges = edges[(edges > lo) & (edges <= hi)]
    return [float(e) for e in edges] or [float(hi)]


def bracket_counts(sorted_values: np.ndarray, lo: float, hi: float, edges) -> tuple[list, list, list]:
    """Site side: counts and in-bin min/max for values in ``[lo, hi]``.

    Bin 0 is ``[lo, e1)``, bin j is ``[e_j, e_{j+1})`` and the last is
    ``[e_m, hi]``.  Empty bins report NaN extremes.
    """
    edges = np.asarray(edges, dtype=float)
    bounds = np.concatenate((
        [np.searchsorted(sorted_values, lo, side="left")],
        np.searchsorted(sorted_values, edges, side="left"),
        [np.searchsorted(sorted_values, hi, side="right")],
    ))
    counts = np.diff(bounds)
    counts = np.maximum(counts, 0)
    mins = np.full(len(counts), np.nan)
    maxs = np.full(len(counts), np.nan)
    nonempty = counts > 0
    if nonempty.any():
        mins[nonempty] = sorted_values[bounds[:-1][nonempty]]
        maxs[nonempty] = sorted_values[bounds[1:][nonempty] - 1]
    return counts.astype(int).tolist(), mins.tolist(), maxs.tolist()


class OrderStatSolver:
    """Resolve exact pooled order statistics over repeated rounds.

    ``targets[key] = (lo, hi, ranks)`` where ``[lo, hi]`` is the pooled
    observed range.  Each unresolved rank carries a bracket ``(lo, hi, base)``:
    its value lies in ``[lo, hi]`` and ``base`` pooled values lie below ``lo``.
    A site answers with bin counts and in-bin extremes; the server picks the bin
    holding the rank and, when that bin's pooled min equals its max, the value is
    known exactly -- otherwise the bracket shrinks to the bin's [min, max].
    """

    def __init__(self, targets: dict, bins: int = 32):
        self.bins = int(bins)
        self.brackets: dict = {}
        self.resolved: dict = {key: {} for key in targets}
        for key, (lo, hi, ranks) in targets.items():
            for rank in ranks:
                if lo == hi:
                    self.resolved[key][rank] = float(lo)
                else:
                    self.brackets[(key, rank)] = [float(lo), float(hi), 0]

    @property
    def done(self) -> bool:
        return not self.brackets

    def queries(self) -> dict:
        """``{key: [[lo, hi, edges], ...]}`` -- one entry per distinct bracket."""
        grouped: dict = {}
        for (key, _), (lo, hi, _) in self.brackets.items():
            grouped.setdefault(key, {})[(lo, hi)] = None
        return {key: [[lo, hi, bracket_edges(lo, hi, self.bins)] for (lo, hi) in boxes]
                for key, boxes in grouped.items()}

    def absorb(self, queries: dict, replies: list[dict]) -> None:
        """``replies`` are the sites' ``{key: [[counts, mins, maxs], ...]}``."""
        for key, boxes in queries.items():
            for index, (lo, hi, edges) in enumerate(boxes):
                parts = [reply[key][index] for reply in replies if key in reply]
                counts = np.sum([p[0] for p in parts], axis=0).astype(int)
                mins = np.array([p[1] for p in parts], dtype=float)
                maxs = np.array([p[2] for p in parts], dtype=float)
                with np.errstate(all="ignore"):
                    pooled_min = np.where(np.isnan(mins).all(axis=0), np.nan, np.nanmin(np.where(np.isnan(mins), np.inf, mins), axis=0))
                    pooled_max = np.where(np.isnan(maxs).all(axis=0), np.nan, np.nanmax(np.where(np.isnan(maxs), -np.inf, maxs), axis=0))
                cumulative = np.cumsum(counts)
                for rank in [r for (k, r), b in list(self.brackets.items()) if k == key and b[0] == lo and b[1] == hi]:
                    base = self.brackets[(key, rank)][2]
                    local = rank - base
                    if local < 0 or local >= cumulative[-1]:
                        raise RuntimeError(f"order statistic {key} rank {rank} escaped its bracket")
                    chosen = int(np.searchsorted(cumulative, local, side="right"))
                    if pooled_min[chosen] == pooled_max[chosen]:
                        self.resolved[key][rank] = float(pooled_min[chosen])
                        del self.brackets[(key, rank)]
                    else:
                        self.brackets[(key, rank)] = [
                            float(pooled_min[chosen]), float(pooled_max[chosen]),
                            int(base + cumulative[chosen] - counts[chosen]),
                        ]


# --------------------------------------------------------------------------- #
# Binarizer over pooled counts
# --------------------------------------------------------------------------- #

def plan_binarizer(presence: dict, n_rows: int, *, age_col: str, min_coverage: float,
                   quantiles) -> dict:
    """The eligible features of one scope and the ranks needed for each.

    ``presence[feature] = (n_observed, lo, hi)`` pooled over sites.  Mirrors
    ``FeatureBinarizer.fit``: a lab needs ``coverage >= min_coverage`` and age
    only needs an observed value; a feature always gets a median, and threshold
    items only when it takes at least two distinct values.
    """
    plan = {}
    for feature in sorted(presence):
        if feature_group(feature, age_col) is None:
            continue
        n_obs, lo, hi = presence[feature]
        if n_obs <= 0:
            continue
        is_lab = feature != age_col
        if is_lab and n_obs / n_rows < min_coverage:
            continue
        ranks = {(n_obs - 1) // 2, n_obs // 2}
        if lo < hi:
            for q in quantiles:
                low, high, _ = quantile_ranks(n_obs, q)
                ranks.update((low, high))
        plan[feature] = {"n_obs": int(n_obs), "lo": float(lo), "hi": float(hi),
                         "ranks": sorted(ranks)}
    return plan


def guideline_threshold(value) -> float:
    """Round a checklist threshold by magnitude, to the nearest step (halves round away from zero).

    Copied from PROFILE-testing ``federated_clinical_comparison.guideline_threshold``, the rule
    behind the generalized checklists: |x| < 1: quarter (0.25, 0.5, 0.75); 1 <= |x| < 10: whole
    number; tens: multiple of 5; hundreds: multiple of 10; thousands: multiple of 100; each further
    decade multiplies the step by 10.  The sign is kept, so a negative threshold rounds by its
    magnitude.
    """
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"cannot round a non-finite threshold: {value!r}")
    a = abs(x)
    if a < 1:
        step = 0.25
    elif a < 10:
        step = 1.0
    else:
        decade = int(math.floor(math.log10(a)))   # 1 = tens, 2 = hundreds, 3 = thousands
        step = 5.0 if decade == 1 else 10.0 ** (decade - 1)
    rounded = math.floor(a / step + 0.5) * step
    return math.copysign(rounded, x) if rounded else 0.0


def assemble_candidates(plan: dict, resolved: dict, n_rows: int, *, age_col: str,
                        quantiles, round_thresholds: bool = True) -> dict:
    """The binarizer spec (all candidate items, before the constant-item drop).

    With ``round_thresholds`` each pooled quantile is passed through
    ``guideline_threshold`` before it becomes an item, so training searches the
    generalized (rounded) cutoffs directly; quantiles that round to the same
    value give one item pair.
    """
    columns, medians, coverage, items = [], {}, {}, []
    for feature in sorted(plan):
        info = plan[feature]
        values = resolved[feature]
        n_obs = info["n_obs"]
        group = feature_group(feature, age_col)
        columns.append(feature)
        medians[feature] = float((values[(n_obs - 1) // 2] + values[n_obs // 2]) / 2.0)
        coverage[feature] = float(n_obs / n_rows)
        if info["lo"] < info["hi"]:
            thresholds = set()
            for q in quantiles:
                low, high, weight = quantile_ranks(n_obs, q)
                cut = numpy_lerp(values[low], values[high], weight)
                thresholds.add(guideline_threshold(cut) if round_thresholds else float(cut))
            for threshold in sorted(thresholds):
                for op in ("<=", ">="):
                    items.append({"name": f"{feature} {op} {threshold:.12g}", "feature": feature,
                                  "op": op, "threshold": float(threshold), "group": group})
    return {"feature_columns": columns, "medians": medians, "coverage": coverage,
            "items": items, "n_rows": int(n_rows)}


def drop_constant_items(spec: dict, true_counts: list[int]) -> tuple[dict, list[int]]:
    """Keep items that are neither always nor never true on the scope's rows.

    PROFILE skips this check for ``is_missing`` items; so does this.
    """
    n = spec["n_rows"]
    keep = [i for i, item in enumerate(spec["items"])
            if item["op"] == "is_missing" or 0 < true_counts[i] < n]
    result = dict(spec)
    result["items"] = [spec["items"][i] for i in keep]
    return result, keep


# --------------------------------------------------------------------------- #
# Pooled scoring
# --------------------------------------------------------------------------- #

def best_threshold(case_weight: np.ndarray, control_weight: np.ndarray) -> tuple[float, int]:
    """Exact best M for one candidate from its balanced ``(score, class)`` tables.

    ``loss(M) = sum_{score<M} w_case + sum_{score>=M} w_control``.  Among M within
    ``LOSS_TOL`` of the minimum the smallest is returned, matching PROFILE's
    third (threshold) stage.
    """
    n_items = len(case_weight) - 1
    miss = np.cumsum(case_weight)[:n_items]
    false_alarm = control_weight.sum() - np.cumsum(control_weight)[:n_items]
    loss = miss + false_alarm
    best = float(loss.min())
    m = int(np.flatnonzero(loss <= best + LOSS_TOL)[0]) + 1
    return float(loss[m - 1]), m


def select_best(scored: dict, names: list[str]) -> tuple[tuple, float, int]:
    """Lexicographic pick: min loss, then fewest items, smallest M, sorted names."""
    floor = min(loss for loss, _ in scored.values())
    eligible = [c for c, (loss, _) in scored.items() if loss <= floor + LOSS_TOL]
    best = min(eligible, key=lambda c: (len(c), scored[c][1], tuple(sorted(names[i] for i in c)), c))
    return best, scored[best][0], scored[best][1]


def _feasible(candidate: tuple, groups: list[str], size: int) -> bool:
    chosen = [groups[i] for i in candidate]
    return 1 <= len(candidate) <= size and len(set(chosen)) == len(chosen)


def neighbours(candidate: tuple, pool: list[int], groups: list[str], size: int) -> set:
    """Add, drop and swap moves that keep one item per group and N <= size."""
    result: set = set()
    members = set(candidate)
    for item in pool:
        if item in members:
            continue
        if len(candidate) < size and groups[item] not in {groups[i] for i in candidate}:
            result.add(tuple(sorted(candidate + (item,))))
        for removed in candidate:
            rest = tuple(i for i in candidate if i != removed)
            if groups[item] not in {groups[i] for i in rest}:
                result.add(tuple(sorted(rest + (item,))))
    if len(candidate) > 1:
        for removed in candidate:
            result.add(tuple(i for i in candidate if i != removed))
    return {c for c in result if _feasible(c, groups, size)}


def search(*, size: int, allowed: list[int], groups: list[str], names: list[str],
           seeds: list[tuple], lower_bound: float, beam_width: int = 5,
           max_rounds: int = 25, pool_size: int = 100) -> Iterator[list]:
    """Federated beam search as a generator.

    Yields lists of candidate item tuples to score and receives
    ``{candidate: (loss, M)}`` for them; returns ``(best, loss, M, info)``.
    Seeds are the sites' local optima; added are the best single item and a
    greedy forward path.  Only the ``pool_size`` best singleton items (plus seed
    items) are used as neighbours, which bounds message size and affects the
    search only -- never the lower bound.
    """
    scored: dict = {}

    def absorb(result: dict) -> None:
        scored.update(result)

    singles = [(i,) for i in allowed]
    absorb((yield singles))
    ranked = sorted(allowed, key=lambda i: (scored[(i,)][0], i))
    best_single = (ranked[0],)

    path, current = [], ()
    for _ in range(size):
        options = [tuple(sorted(current + (i,))) for i in allowed
                   if i not in current and _feasible(tuple(sorted(current + (i,))), groups, size)]
        if not options:
            break
        absorb((yield options))
        current = min(options, key=lambda c: (scored[c][0], c))
        path.append(current)

    seed_sets = {best_single, *path}
    seed_sets.update(tuple(sorted(s)) for s in seeds if s and _feasible(tuple(sorted(s)), groups, size))
    needed = [c for c in seed_sets if c not in scored]
    if needed:
        absorb((yield needed))
    pool = list(dict.fromkeys(ranked[:pool_size] + [i for s in seed_sets for i in s]))

    def key(c):
        return (scored[c][0], len(c), scored[c][1], c)

    beam = sorted(seed_sets, key=key)[:beam_width]
    best, best_loss, best_m = select_best({c: scored[c] for c in seed_sets}, names)
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        if best_loss - lower_bound <= LOSS_TOL:
            break
        fresh = set()
        for member in beam:
            fresh |= neighbours(member, pool, groups, size)
        fresh = [c for c in fresh if c not in scored]
        if not fresh:
            break
        absorb((yield fresh))
        beam = sorted(set(beam) | set(fresh), key=key)[:beam_width]
        new_best, new_loss, new_m = select_best(scored, names)
        improved = new_loss < best_loss - 1e-12
        best, best_loss, best_m = new_best, new_loss, new_m
        if not improved:
            break
    # Polish: drop moves can only trade nothing for fewer items.
    for _ in range(size):
        drops = [c for c in neighbours(best, [], groups, size) if c not in scored]
        if drops:
            absorb((yield drops))
        new_best, new_loss, new_m = select_best(scored, names)
        if new_best == best:
            break
        best, best_loss, best_m = new_best, new_loss, new_m
    return best, best_loss, best_m, {"rounds": rounds, "n_scored": len(scored),
                                     "greedy_path": [list(p) for p in path]}


# --------------------------------------------------------------------------- #
# Lower-bound certificate
# --------------------------------------------------------------------------- #

def site_bound(reply: dict | None) -> tuple[float, str]:
    """One site's contribution to the certificate, on the global weight scale.

    ``reply`` carries the stage-1 solver bound and incumbent loss normalized by
    the site's own weight total ``total_weight``.  Any unusable bound degrades to
    zero (a weaker, still valid certificate) with the reason recorded.
    """
    if reply is None or reply.get("error"):
        return 0.0, "nonnegative_loss_fallback"
    if reply.get("empty"):
        return 0.0, "empty_known"
    raw = reply.get("lb_raw")
    total = reply.get("total_weight")
    if raw is None or total is None or not math.isfinite(raw) or not math.isfinite(total):
        return 0.0, "nonnegative_loss_fallback"
    bound = max(0.0, float(raw)) * float(total)
    losses = [v for v in (reply.get("incumbent_loss"), reply.get("solution_loss")) if v is not None]
    ceiling = min(losses) * float(total) if losses else None
    if ceiling is not None and bound > ceiling + 1e-7 * max(1.0, float(total)):
        return 0.0, "numerical_consistency_fallback"
    return bound, "solver"


def combine_bounds(replies: list[dict | None]) -> tuple[float, list[str]]:
    parts = [site_bound(r) for r in replies]
    return float(sum(b for b, _ in parts)), [s for _, s in parts]


def certify(pooled_loss: float, lower_bound: float, sources: list[str]) -> dict:
    """Gap and status for one fit.  A bound above the pooled loss is impossible
    for a valid bound, so it is discarded rather than trusted."""
    if lower_bound > pooled_loss + 1e-7:
        lower_bound, sources = 0.0, ["numerical_consistency_fallback"] * len(sources)
    gap = max(0.0, pooled_loss - lower_bound)
    return {"pooled_loss": float(pooled_loss), "lower_bound": float(lower_bound),
            "loss_gap": float(gap), "bound_sources": sources,
            "loss_certified": bool(gap <= LOSS_TOL),
            "status": "loss_certified" if gap <= LOSS_TOL else "feasible_gap"}


# --------------------------------------------------------------------------- #
# Scopes and blocks (shared by the controller and the executor)
# --------------------------------------------------------------------------- #

def fit_scopes(n_folds: int) -> list[str]:
    """Scopes a binarizer and censoring curve are fitted on."""
    return ["train_val"] + [f"fold{k}_train" for k in range(n_folds)]


def all_scopes(n_folds: int) -> list[str]:
    return ["train_val", "test"] + [f"fold{k}_{part}" for k in range(n_folds) for part in ("train", "val")]


def block_id(fit_scope: str, eval_scope: str) -> str:
    """Rows of ``eval_scope`` binarized (and IPCW-weighted) by ``fit_scope``."""
    return f"{fit_scope}>{eval_scope}"


def split_block(block: str) -> tuple[str, str]:
    fit_scope, _, eval_scope = block.partition(">")
    return fit_scope, eval_scope


def all_blocks(n_folds: int) -> list[str]:
    blocks = [block_id("train_val", "train_val"), block_id("train_val", "test")]
    for k in range(n_folds):
        blocks += [block_id(f"fold{k}_train", f"fold{k}_train"), block_id(f"fold{k}_train", f"fold{k}_val")]
    return blocks
