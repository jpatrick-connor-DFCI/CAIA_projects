"""Shallow survival tree: rpart ``method="exp"`` on binary items (no NVFlare).

Everything here is plain numpy so tests import it directly.  The federated
controller and the pooled reference :func:`fit_pooled_tree` share
:func:`choose_split` and :func:`finalize_tree`; they differ only in where the
node sums come from (site replies vs. local rows).

rpart parity (rpart 4.1.27, ``src/poisson.c``, ``src/partition.c``,
``R/rpart.exp.R``):

* **Time scaling** -- :func:`rpart_exp_scale` reproduces ``rpart.exp``: unique
  death times thinned by ``rpartexp2``, at most 1001 type-7 quantile knots,
  right-closed intervals, piecewise-constant hazard, linear interpolation of the
  cumulative hazard.  It takes the pooled ``(time, n, events)`` table, which is
  all ``rpart.exp`` needs.
* **Split gain** -- ``2 [d_T log(d_T/z_T) + d_F log(d_F/z_F) - d log(d/z)]``
  (``poisson()``); a node without deaths has gain 0; each child needs at least
  ``min_leaf`` rows (``edge``).
* **Stopping** -- a node splits only with ``n >= min_split`` rows and depth
  ``< max_depth`` (rpart's ``nodenum > maxnode``), and only on a gain
  ``> 1e-10 * reference`` (rpart's ``iscale`` rule; here the reference is the
  root's best gain rather than the running maximum, which differs from rpart
  only for gains within 1e-10 of zero).
* **cp = 0 pruning** -- :func:`finalize_tree` ports the end of ``partition()``:
  a split whose subtree does not lower the shrunken-rate deviance is collapsed.
  The deviance is computed without the ``sum log z_i`` over events, which cancels
  in every difference rpart takes.
* **Ties** -- rpart scans columns in order and keeps the first strict maximum.
  Here gains within ``TIE_TOL * max(1, d)`` of the best are ties and the lowest
  item index wins, so the choice does not depend on how site sums were rounded.
  The tolerance scales with the node's events ``d``, not with the gain: the gain
  is a difference of ``d log(d/z)`` terms, so its rounding error is ~eps * d.
  Complementary items (``age <= c`` and ``age >= c`` with no missing ages and no
  value at ``c``) have gains equal up to that error.
* **Leaf rate** -- ``(d + 1) / (z + z_root/d_root)``, rpart's ``yval`` with
  ``shrink = 1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

TIE_TOL = 1e-10
MIN_GAIN_RTOL = 1e-10
MAX_KNOTS = 1000
LEFT, RIGHT = 0, 1  # child taken when the split item is false / true


# --------------------------------------------------------------------------- #
# Time scaling (rpart.exp)
# --------------------------------------------------------------------------- #

def _rpartexp2(dtimes: np.ndarray, eps: float = np.finfo(float).eps) -> np.ndarray:
    """``rpartexp2.c``: drop death times within ``eps * IQR`` of the last kept."""
    n = len(dtimes)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    delta = eps * (dtimes[(3 * n) // 4] - dtimes[n // 4])
    lasty = dtimes[0]
    keep[0] = True
    for i in range(1, n):
        if dtimes[i] - lasty > delta:
            keep[i] = True
            lasty = dtimes[i]
    return keep


def _quantile_type7(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R's ``quantile(x, probs)`` (type 7), with R's arithmetic."""
    n = len(x)
    index = 1.0 + max(n - 1, 0) * probs
    lo = np.floor(index).astype(int)
    hi = np.ceil(index).astype(int)
    qs = x[lo - 1].astype(float)
    h = index - lo
    mask = (index > lo) & (x[hi - 1] != qs)
    qs[mask] = (1.0 - h[mask]) * qs[mask] + h[mask] * x[hi[mask] - 1]
    return qs


def rpart_exp_scale(times: Sequence[float], totals: Sequence[float],
                    events: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Knots and cumulative hazard of ``rpart.exp`` from a pooled outcome table.

    ``times`` are distinct positive follow-up times with ``totals`` rows and
    ``events`` deaths each.  Returns ``(itable, cumhaz)``; the rescaled time of a
    row is ``np.interp(time, itable, cumhaz)`` (:func:`rescale`).
    """
    times = np.asarray(times, dtype=float)
    totals = np.asarray(totals, dtype=float)
    events = np.asarray(events, dtype=float)
    order = np.argsort(times, kind="stable")
    times, totals, events = times[order], totals[order], events[order]
    if times.size == 0 or (times <= 0).any():
        raise ValueError("observation times must be > 0")
    if events.sum() <= 0:
        raise ValueError("no deaths in data set")
    dtimes = times[events > 0]
    dtimes = dtimes[_rpartexp2(dtimes)]
    if len(dtimes) > MAX_KNOTS:
        dtimes = _quantile_type7(dtimes, np.arange(MAX_KNOTS + 1) / MAX_KNOTS)
    itable = np.concatenate([[0.0], dtimes[:-1], [times.max()]])
    ilength = np.diff(itable)
    ngrp = len(ilength)
    # cut(time, itable, include.lowest=TRUE): intervals (a, b], the first [0, b].
    index = np.clip(np.searchsorted(itable, times, side="left") - 1, 0, ngrp - 1)
    tab1 = np.bincount(index, weights=totals, minlength=ngrp)
    later = np.concatenate([np.cumsum(tab1[::-1])[::-1][1:], [0.0]])
    itime = np.bincount(index, weights=(times - itable[index]) * totals, minlength=ngrp)
    pyears = ilength * later + itime
    deaths = np.bincount(index, weights=events, minlength=ngrp)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(pyears > 0, deaths / pyears, 0.0)
    cumhaz = np.cumsum(np.concatenate([[0.0], rate * ilength]))
    return itable, cumhaz


def rescale(duration: np.ndarray, knots: Sequence[float], cumhaz: Sequence[float]) -> np.ndarray:
    """``approx(itable, cumhaz, time)$y``."""
    return np.interp(np.asarray(duration, dtype=float), np.asarray(knots, dtype=float),
                     np.asarray(cumhaz, dtype=float))


def outcome_table(duration: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(times, totals, events)`` as ``pc_fl_init`` reports them."""
    duration = np.asarray(duration, dtype=float)
    times, inverse, totals = np.unique(duration, return_inverse=True, return_counts=True)
    events = np.bincount(inverse.reshape(-1), weights=np.asarray(event, dtype=float), minlength=len(times))
    return times, totals.astype(float), events


# --------------------------------------------------------------------------- #
# Node sums and splits
# --------------------------------------------------------------------------- #

def node_sums(X: np.ndarray, z: np.ndarray, event: np.ndarray, mask: np.ndarray) -> dict:
    """Totals ``(n, d, z)`` and per-item ``(n_T, d_T, z_T)`` over ``mask`` rows.

    This is exactly what a site returns for one node in ``st_fl_grow``.
    """
    Xm = X[mask].astype(np.float64)
    zm, em = z[mask], event[mask].astype(np.float64)
    return {
        "n": int(mask.sum()), "d": float(em.sum()), "z": float(zm.sum()),
        "nT": Xm.sum(axis=0).round().astype(np.int64),
        "dT": em @ Xm if Xm.size else np.zeros(X.shape[1]),
        "zT": zm @ Xm if Xm.size else np.zeros(X.shape[1]),
    }


def _xlogx_over(d: np.ndarray, z: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float)
    out = np.zeros(np.broadcast(d, z).shape)
    ok = (d > 0) & (z > 0)
    out[ok] = (d * np.log(np.where(ok, d, 1.0) / np.where(ok, z, 1.0)))[ok]
    return out


def split_gains(n: float, d: float, z: float, nT: np.ndarray, dT: np.ndarray,
                zT: np.ndarray) -> np.ndarray:
    """rpart ``poisson()`` improvement for every binary item at one node."""
    nT, dT, zT = (np.asarray(a, dtype=float) for a in (nT, dT, zT))
    if d <= 0:
        return np.zeros(nT.shape)
    dF, zF = d - dT, z - zT
    best = _xlogx_over(dT, zT) + _xlogx_over(dF, zF)
    return 2.0 * (best - float(_xlogx_over(d, z)))


def choose_split(stats: dict, *, min_leaf: int, min_split: int, allowed: np.ndarray | None,
                 withheld: np.ndarray | None = None, reference_gain: float | None = None
                 ) -> tuple[int | None, float, np.ndarray]:
    """Best admissible item at a node, or ``None``.

    Returns ``(item, gain, gains)``; ``gains`` is ``-inf`` where inadmissible.
    """
    nT = np.asarray(stats["nT"], dtype=float)
    gains = np.full(nT.shape, -np.inf)
    n = stats["n"]
    if n < min_split or stats["d"] <= 0 or nT.size == 0:
        return None, 0.0, gains
    ok = (nT >= min_leaf) & (n - nT >= min_leaf)
    if allowed is not None:
        ok &= allowed
    if withheld is not None:
        ok &= ~withheld
    if not ok.any():
        return None, 0.0, gains
    raw = split_gains(n, stats["d"], stats["z"], stats["nT"], stats["dT"], stats["zT"])
    gains[ok] = raw[ok]
    top = float(gains.max())
    floor = MIN_GAIN_RTOL * (top if reference_gain is None else max(reference_gain, top))
    if not top > floor or not top > 0:
        return None, 0.0, gains
    tied = np.flatnonzero(gains >= top - TIE_TOL * max(1.0, float(stats["d"])))
    item = int(tied.min())
    return item, float(gains[item]), gains


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #

Path = tuple[tuple[int, int], ...]  # ((item, 0/1), ...) from the root


@dataclass
class Node:
    id: int                      # rpart numbering: root 1, children 2k (false) / 2k+1 (true)
    path: Path
    n: int
    d: float
    z: float
    split: int | None = None
    gain: float = 0.0

    @property
    def depth(self) -> int:
        return len(self.path)


@dataclass
class SurvivalTree:
    """Binary-item survival tree; ``nodes`` keyed by rpart node number."""

    nodes: dict[int, Node]
    root_rate: float
    items: list[dict] = field(default_factory=list)

    # -- structure -------------------------------------------------------- #
    def children(self, nid: int) -> tuple[int, int]:
        return 2 * nid, 2 * nid + 1

    def is_leaf(self, nid: int) -> bool:
        return self.nodes[nid].split is None

    def leaves(self) -> list[int]:
        return sorted(k for k in self.nodes if self.is_leaf(k))

    def depth(self) -> int:
        return max((self.nodes[k].depth for k in self.leaves()), default=0)

    def leaf_rate(self, nid: int) -> float:
        node = self.nodes[nid]
        return (node.d + 1.0) / (node.z + 1.0 / self.root_rate)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Leaf node number of every row of the item matrix ``X``."""
        out = np.ones(len(X), dtype=np.int64)
        for _ in range(64):
            moved = False
            for nid in np.unique(out):
                split = self.nodes[int(nid)].split
                if split is None:
                    continue
                rows = out == nid
                out[rows] = 2 * nid + (X[rows, split] != 0)
                moved = True
            if not moved:
                return out
        raise RuntimeError("tree deeper than 64")

    def truncate(self, max_depth: int) -> "SurvivalTree":
        """Grown tree cut at ``max_depth`` (before :func:`finalize_tree`)."""
        nodes = {}
        for k, node in self.nodes.items():
            if node.depth > max_depth:
                continue
            copy = Node(node.id, node.path, node.n, node.d, node.z, node.split, node.gain)
            if copy.depth == max_depth:
                copy.split, copy.gain = None, 0.0
            nodes[k] = copy
        return SurvivalTree(nodes, self.root_rate, list(self.items))

    def leaf_of_ancestor(self, other_leaf: int) -> int:
        """The leaf of this tree containing a node of a deeper tree."""
        nid = int(other_leaf)
        while nid > 1 and nid not in self.nodes:
            nid //= 2
        if nid not in self.nodes or not self.is_leaf(nid):
            raise KeyError(f"node {other_leaf} is not inside a leaf of this tree")
        return nid

    def rules(self, nid: int) -> list[dict]:
        out = []
        for item, value in self.nodes[nid].path:
            spec = self.items[item] if item < len(self.items) else {"name": f"item{item}"}
            out.append({**spec, "item": item, "value": int(value)})
        return out

    def rule_text(self, nid: int) -> str:
        parts = []
        for r in self.rules(nid):
            name = r.get("name", f"item{r['item']}")
            parts.append(name if r["value"] else f"NOT({name})")
        return " AND ".join(parts) or "all patients"

    # -- serialization ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "root_rate": self.root_rate,
            "items": self.items,
            "nodes": [{"id": n.id, "path": [list(p) for p in n.path], "n": n.n, "d": n.d, "z": n.z,
                       "split": n.split, "gain": n.gain} for n in sorted(self.nodes.values(), key=lambda v: v.id)],
        }

    @classmethod
    def from_dict(cls, state: dict) -> "SurvivalTree":
        nodes = {int(s["id"]): Node(int(s["id"]), tuple((int(a), int(b)) for a, b in s["path"]), int(s["n"]),
                                    float(s["d"]), float(s["z"]), None if s["split"] is None else int(s["split"]),
                                    float(s.get("gain", 0.0))) for s in state["nodes"]}
        return cls(nodes, float(state["root_rate"]), list(state.get("items", [])))

    def signature(self) -> tuple:
        """Structure only: ``((node, split), ...)`` over internal nodes."""
        return tuple(sorted((k, v.split) for k, v in self.nodes.items() if v.split is not None))


def _deviance(d: float, z: float, root_rate: float) -> float:
    """rpart ``poissondev`` risk with shrink=1, minus ``-2 sum_events log z_i``."""
    lam = (d + 1.0) / (z + 1.0 / root_rate)
    return -2.0 * ((d * np.log(lam) if d > 0 else 0.0) - lam * z + d)


def finalize_tree(tree: SurvivalTree) -> SurvivalTree:
    """rpart's cp=0 collapse (end of ``partition()``), on a copy of ``tree``."""
    out = SurvivalTree({k: Node(v.id, v.path, v.n, v.d, v.z, v.split, v.gain) for k, v in tree.nodes.items()},
                       tree.root_rate, list(tree.items))
    risk = {k: _deviance(v.d, v.z, out.root_rate) for k, v in out.nodes.items()}
    cp: dict[int, float] = {}

    def rec(nid: int) -> tuple[float, int]:
        node = out.nodes[nid]
        if node.split is None:
            cp[nid] = 0.0
            return risk[nid], 0
        left, right = out.children(nid)
        left_risk, left_split = rec(left)
        right_risk, right_split = rec(right)
        temp = (risk[nid] - (left_risk + right_risk)) / (left_split + right_split + 1)
        if cp[right] > cp[left]:
            if temp > cp[left]:
                left_risk, left_split = risk[left], 0
                temp = (risk[nid] - (left_risk + right_risk)) / (left_split + right_split + 1)
                if temp > cp[right]:
                    right_risk, right_split = risk[right], 0
        elif temp > cp[right]:
            right_risk, right_split = risk[right], 0
            temp = (risk[nid] - (left_risk + right_risk)) / (left_split + right_split + 1)
            if temp > cp[left]:
                left_risk, left_split = risk[left], 0
        cp[nid] = (risk[nid] - (left_risk + right_risk)) / (left_split + right_split + 1)
        if cp[nid] <= 0.0:
            _drop_below(out, nid)
            return risk[nid], 0
        return left_risk + right_risk, left_split + right_split + 1

    rec(1)
    return out


def _drop_below(tree: SurvivalTree, nid: int) -> None:
    tree.nodes[nid].split, tree.nodes[nid].gain = None, 0.0
    stack = list(tree.children(nid))
    while stack:
        k = stack.pop()
        if k in tree.nodes:
            del tree.nodes[k]
            stack.extend(tree.children(k))


def child_stats(stats: dict, item: int) -> tuple[tuple[int, float, float], tuple[int, float, float]]:
    """``((n, d, z) false child, (n, d, z) true child)`` of a split."""
    nT, dT, zT = int(stats["nT"][item]), float(stats["dT"][item]), float(stats["zT"][item])
    return (int(stats["n"]) - nT, float(stats["d"]) - dT, float(stats["z"]) - zT), (nT, dT, zT)


def grow_level(tree: SurvivalTree, frontier_stats: dict[int, dict], *, min_leaf: int, min_split: int,
               max_depth: int, allowed: np.ndarray | None, withheld: dict[int, np.ndarray] | None = None,
               reference_gain: float | None = None) -> list[int]:
    """Split every frontier node from its sums; returns the new frontier."""
    new = []
    for nid in sorted(frontier_stats):
        stats = frontier_stats[nid]
        node = tree.nodes[nid]
        if node.depth >= max_depth:
            continue
        item, gain, _ = choose_split(stats, min_leaf=min_leaf, min_split=min_split, allowed=allowed,
                                     withheld=None if withheld is None else withheld.get(nid),
                                     reference_gain=reference_gain)
        if item is None:
            continue
        node.split, node.gain = item, gain
        (nf, df, zf), (nt, dt, zt) = child_stats(stats, item)
        for child, value, (cn, cd, cz) in ((2 * nid, 0, (nf, df, zf)), (2 * nid + 1, 1, (nt, dt, zt))):
            tree.nodes[child] = Node(child, node.path + ((item, value),), cn, cd, cz)
            new.append(child)
    return new


def path_mask(X: np.ndarray, path: Iterable[Sequence[int]]) -> np.ndarray:
    mask = np.ones(len(X), dtype=bool)
    for item, value in path:
        mask &= (X[:, int(item)] != 0) == bool(value)
    return mask


def fit_pooled_tree(X: np.ndarray, duration: np.ndarray, event: np.ndarray, *, max_depth: int,
                    min_leaf: int, min_split: int | None = None, allowed: np.ndarray | None = None,
                    items: list[dict] | None = None, finalize: bool = True) -> SurvivalTree:
    """Central reference fit on pooled rows (the federated fit must equal this)."""
    X = np.asarray(X)
    event = np.asarray(event, dtype=float)
    knots, cumhaz = rpart_exp_scale(*outcome_table(duration, event))
    z = rescale(duration, knots, cumhaz)
    return grow_from_rows(X, z, event, max_depth=max_depth, min_leaf=min_leaf,
                          min_split=2 * min_leaf if min_split is None else min_split,
                          allowed=allowed, items=items, finalize=finalize)


def grow_from_rows(X, z, event, *, max_depth, min_leaf, min_split, allowed=None, items=None,
                   finalize=True) -> SurvivalTree:
    root = node_sums(X, z, event, np.ones(len(X), dtype=bool))
    tree = SurvivalTree({1: Node(1, (), root["n"], root["d"], root["z"])}, root["d"] / root["z"], list(items or []))
    frontier = {1: root}
    reference = None
    while frontier:
        new = grow_level(tree, frontier, min_leaf=min_leaf, min_split=min_split, max_depth=max_depth,
                         allowed=allowed, reference_gain=reference)
        if reference is None:
            reference = tree.nodes[1].gain
        frontier = {k: node_sums(X, z, event, path_mask(X, tree.nodes[k].path)) for k in new}
    return finalize_tree(tree) if finalize else tree


# --------------------------------------------------------------------------- #
# Leaf survival curves
# --------------------------------------------------------------------------- #

def kaplan_meier(time_units: np.ndarray, events: np.ndarray, counts: np.ndarray) -> dict:
    """KM (with log-log Greenwood 95% CI) and Nelson-Aalen from ``(t, event, n)`` cells."""
    t = np.asarray(time_units, dtype=float)
    e = np.asarray(events, dtype=int)
    c = np.asarray(counts, dtype=float)
    times = np.unique(t)
    deaths = np.array([c[(t == u) & (e == 1)].sum() for u in times])
    leaving = np.array([c[t == u].sum() for u in times])
    at_risk = c.sum() - np.concatenate([[0.0], np.cumsum(leaving)[:-1]])
    keep = deaths > 0
    times, deaths, at_risk = times[keep], deaths[keep], at_risk[keep]
    surv = np.cumprod(1.0 - deaths / at_risk)
    chf = np.cumsum(deaths / at_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        gw = np.cumsum(np.where(at_risk > deaths, deaths / (at_risk * (at_risk - deaths)), np.inf))
        log_s = np.log(surv)
        se = np.sqrt(gw) / np.abs(log_s)
        # S < 1, so the larger exponent gives the lower bound.
        lower = np.where(surv > 0, surv ** np.exp(1.959963984540054 * se), 0.0)
        upper = np.where(surv > 0, surv ** np.exp(-1.959963984540054 * se), 0.0)
    lower = np.where(np.isfinite(lower), lower, 0.0)
    upper = np.where(np.isfinite(upper), upper, 1.0)
    return {"time_unit": times.tolist(), "survival": surv.tolist(), "lower": lower.tolist(),
            "upper": upper.tolist(), "cumhaz": chf.tolist(), "n": float(c.sum()), "events": float(deaths.sum())}


def step_at(curve: dict, horizons: np.ndarray, key: str = "survival", start: float = 1.0) -> np.ndarray:
    """Right-continuous step value of ``curve[key]`` at each horizon (time units)."""
    times = np.asarray(curve["time_unit"], dtype=float)
    values = np.concatenate([[start], np.asarray(curve[key], dtype=float)])
    return values[np.searchsorted(times, np.asarray(horizons, dtype=float), side="right")]
