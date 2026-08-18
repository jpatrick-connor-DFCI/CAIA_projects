"""Target/label construction for the multivariate_longitudinal arm.

Torch-free by design: this is the competing-risk semantics for Dynamic-DeepHit
and SurvLatent ODE, factored out of the model code so it is unit-testable
without torch or the external SurvLatent repo. Both models import
``LONGITUDINAL_CONFIGS`` / ``resolve_config`` / ``patient_targets`` from here
and take identical ``--config`` values.

Six configs, deliberately no "death-alone" config. Each names a primary
cause of interest, optionally paired with death as a competing cause:

  platinum            -- n_events=1. Death is never read; a patient who dies
                         without platinum is already censored at death via
                         t_platinum's fallback to t_last_contact (see
                         cohort.make_outcome_df). This is cause-specific
                         censoring with zero special-casing, and is what makes
                         this config directly comparable to the existing
                         Cox/XGBoost arms (same cohort, same censoring).
  competing           -- n_events=2. label = 0 (censored) / 1 (platinum) /
                         2 (death), fixed by the order of event_cols below. A
                         patient with both events observed is labeled by
                         whichever has the smaller post-landmark duration
                         (argmin ties break toward the first listed cause,
                         i.e. platinum).
  nepc                -- the platinum config's analogue for the LLM-adjudicated
                         NEPC diagnosis endpoint. Same cause-specific censoring.
  nepc_competing      -- the competing analogue: label 1 = NEPC, 2 = death.
  avpc_nepc           -- the platinum config's analogue for the broader
                         AVPC_NEPC criteria-timeline endpoint (>=3 Aparicio
                         criteria or any NEPC feature, NEPC-precedence timing).
                         Same cause-specific censoring.
  avpc_nepc_competing -- the competing analogue: label 1 = AVPC_NEPC, 2 = death.

The nepc/avpc_nepc configs each read a cohort built with their own incident
gate (``--require-nepc`` / ``endpoint=avpc_nepc``), so neither is the same
patient set as the platinum configs, nor as each other -- their metrics are
not a like-for-like comparison. See the README's "Arms and endpoints" section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Maps each config to the cox_aggregated.ENDPOINTS key whose AUC(t) horizon
# grid it evaluates on, so a discrete-time head is scored on the same timeline
# as the Cox/XGBoost arm for the same event. Kept beside the configs because
# adding one without the other yields a KeyError at horizon-resolution time.
CONFIG_ENDPOINTS: dict[str, str] = {
    "platinum": "platinum",
    "competing": "platinum",
    "nepc": "nepc",
    "nepc_competing": "nepc",
    "avpc_nepc": "avpc_nepc",
    "avpc_nepc_competing": "avpc_nepc",
}

LONGITUDINAL_CONFIGS: dict[str, dict[str, list[str]]] = {
    "platinum": {
        "event_cols": ["PLATINUM"],
        "time_cols": ["t_platinum"],
        "event_names": ["platinum"],
    },
    "competing": {
        "event_cols": ["PLATINUM", "DEATH"],
        "time_cols": ["t_platinum", "t_death"],
        "event_names": ["platinum", "death"],
    },
    "nepc": {
        "event_cols": ["NEPC"],
        "time_cols": ["t_nepc"],
        "event_names": ["nepc"],
    },
    "nepc_competing": {
        "event_cols": ["NEPC", "DEATH"],
        "time_cols": ["t_nepc", "t_death"],
        "event_names": ["nepc", "death"],
    },
    "avpc_nepc": {
        "event_cols": ["AVPC_NEPC"],
        "time_cols": ["t_avpc_nepc"],
        "event_names": ["avpc_nepc"],
    },
    "avpc_nepc_competing": {
        "event_cols": ["AVPC_NEPC", "DEATH"],
        "time_cols": ["t_avpc_nepc", "t_death"],
        "event_names": ["avpc_nepc", "death"],
    },
}
# Fixed cause ordering: a future reorder of any competing config would
# silently swap which label (1 vs 2) every downstream risk column means. The
# cause of interest must stay first, and death second.
assert LONGITUDINAL_CONFIGS["competing"]["event_cols"] == ["PLATINUM", "DEATH"]
assert LONGITUDINAL_CONFIGS["nepc_competing"]["event_cols"] == ["NEPC", "DEATH"]
assert LONGITUDINAL_CONFIGS["avpc_nepc_competing"]["event_cols"] == ["AVPC_NEPC", "DEATH"]
# Every config must declare the horizon grid it evaluates on.
assert set(CONFIG_ENDPOINTS) == set(LONGITUDINAL_CONFIGS)


class LongitudinalEventConfig:
    """Resolved (event_cols, time_cols, event_names) for one --config value."""

    __slots__ = ("name", "event_cols", "time_cols", "event_names")

    def __init__(
        self,
        name: str,
        event_cols: list[str],
        time_cols: list[str],
        event_names: list[str],
    ) -> None:
        self.name = name
        self.event_cols = event_cols
        self.time_cols = time_cols
        self.event_names = event_names

    @property
    def n_events(self) -> int:
        return len(self.event_cols)


def resolve_config(config: str, manifest: dict) -> LongitudinalEventConfig:
    """Resolve ``--config`` against the fixed two-entry registry.

    ``manifest`` (the per-landmark longitudinal manifest, carrying the full
    ``event_cols``/``time_to_event_cols`` pair) is read for validation only —
    it does not select the config, so a stale or hand-edited manifest cannot
    silently redefine what "competing" means.
    """
    normalized = config.strip().lower()
    if normalized not in LONGITUDINAL_CONFIGS:
        valid = ", ".join(sorted(LONGITUDINAL_CONFIGS))
        raise ValueError(f"Unsupported --config {config!r}. Choose from: {valid}")

    spec = LONGITUDINAL_CONFIGS[normalized]
    event_cols = list(spec["event_cols"])
    time_cols = list(spec["time_cols"])
    event_names = list(spec["event_names"])

    manifest_event_cols = list(manifest.get("event_cols", []))
    missing = [c for c in event_cols if c not in manifest_event_cols]
    if missing:
        raise ValueError(
            f"--config {config!r} needs event columns {missing} but the manifest's "
            f"event_cols is {manifest_event_cols}. Rebuild prediction inputs."
        )

    return LongitudinalEventConfig(
        name=normalized, event_cols=event_cols, time_cols=time_cols, event_names=event_names
    )


def patient_targets(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    event_cols: list[str],
    time_cols: list[str],
    max_pred_window: int,
) -> pd.DataFrame:
    """Collapse a person-period frame to one discrete-time target row per patient.

    ``landmark_time`` = max(``time_col``) per patient (the synthetic all-NaN
    landmark row planted by :func:`survival_common.cohort.build_person_period_wide`
    at ``TIME == landmark_time``). ``duration`` = ``event_time - landmark_time``
    is therefore post-landmark, in the same width/origin as the manifest AUC(t)
    horizons -- see :func:`horizon_units_to_bins`.

    Competing-cause label: ``argmin`` over per-cause candidate durations among
    observed, positive-duration causes; ``0`` if none observed. Events beyond
    ``max_pred_window`` are censored *at* the window (label -> 0, duration ->
    window) -- this is a second, independent censoring pass on top of any
    cohort-level administrative censoring (Step 0's ``max_followup_days``),
    scoped to what the discrete-time head can actually reach.

    Returns one row per patient (``id_col``-indexed) with ``landmark_time``,
    ``duration``, ``duration_bin`` (>=1, <=max_pred_window), ``label``, and the
    pre-window-censoring ``uncensored_duration``/``uncensored_label`` pair for
    diagnostics.
    """
    if max_pred_window < 1:
        raise ValueError(f"max_pred_window must be >= 1, got {max_pred_window}.")

    rows: list[dict] = []
    n_censored_at_window = 0
    for mrn, group in df.groupby(id_col, sort=False):
        landmark = float(pd.to_numeric(group[time_col], errors="coerce").max())
        durations: list[float] = []
        observed: list[bool] = []
        for event_col, event_time_col in zip(event_cols, time_cols):
            event = int(pd.to_numeric(group[event_col], errors="coerce").fillna(0).iloc[0])
            event_time = float(pd.to_numeric(group[event_time_col], errors="coerce").iloc[0])
            durations.append(event_time - landmark)
            observed.append(event == 1)

        durations_arr = np.asarray(durations, dtype=float)
        observed_arr = np.asarray(observed, dtype=bool)
        valid_times = np.isfinite(durations_arr) & (durations_arr > 0)
        if not valid_times.any():
            continue

        if observed_arr.any():
            candidate = np.where(observed_arr & valid_times, durations_arr, np.inf)
            event_idx = int(np.argmin(candidate))
            duration = float(candidate[event_idx])
            label = event_idx + 1 if np.isfinite(duration) else 0
        else:
            duration = float(np.nanmin(durations_arr[valid_times]))
            label = 0

        uncensored_duration = duration
        uncensored_label = label

        if duration > max_pred_window:
            duration = float(max_pred_window)
            label = 0
            n_censored_at_window += 1

        duration_bin = int(np.clip(np.ceil(duration), 1, max_pred_window))
        if duration_bin < 1:
            raise AssertionError(
                f"patient {mrn}: duration_bin={duration_bin} < 1 after clipping "
                f"(duration={duration}, landmark={landmark}). A negative "
                "duration must not be silently masked into bin 1."
            )
        rows.append(
            {
                id_col: mrn,
                "landmark_time": landmark,
                "duration": duration,
                "duration_bin": duration_bin,
                "label": label,
                "uncensored_duration": uncensored_duration,
                "uncensored_label": uncensored_label,
            }
        )

    print(
        f"[patient_targets] {len(rows)} patients with a valid target; "
        f"{n_censored_at_window} censored at max_pred_window={max_pred_window}."
    )
    columns = [
        id_col,
        "landmark_time",
        "duration",
        "duration_bin",
        "label",
        "uncensored_duration",
        "uncensored_label",
    ]
    return pd.DataFrame(rows, columns=columns).set_index(id_col)


def manifest_horizons_for_config(
    build_manifest: dict,
    landmark_day: int,
    event_names: list[str],
    *,
    max_pred_window: int,
    endpoint: str = "platinum",
) -> dict[str, np.ndarray]:
    """Read the shared AUC(t) horizon grid for each cause, clamped to the window.

    Sourced from ``build_manifest["auc_horizons_by_landmark"][str(landmark_day)]``
    (README invariant #6) so DeepHit/SurvLatent evaluate on the identical grid
    Cox/XGBoost do.

    ``endpoint`` selects which endpoint's grid to read -- ``CONFIG_ENDPOINTS``
    maps each config to it, so the ``nepc*`` configs score against the NEPC
    grid and the ``platinum*`` configs against the platinum one. Within a
    config, **every** cause reads that one grid: the cause-of-interest row is
    then comparable across all model arms for the same endpoint, and the death
    row (competing configs only) is reported on the same grid as a secondary
    diagnostic.
    """
    landmark_horizons = build_manifest.get("auc_horizons_by_landmark", {}).get(str(int(landmark_day)))
    if landmark_horizons is None:
        raise KeyError(
            f"build_manifest.json has no auc_horizons_by_landmark entry for landmark "
            f"+{landmark_day}d."
        )
    endpoint_horizons = landmark_horizons.get(endpoint)
    if endpoint_horizons is None:
        raise KeyError(
            f"build_manifest.json's auc_horizons_by_landmark[{landmark_day}] has no "
            f"{endpoint!r} entry to source horizons from; got keys "
            f"{list(landmark_horizons)}. A cohort built without --require-nepc "
            "carries no NEPC horizon grid -- rebuild prediction inputs for that "
            "endpoint."
        )

    horizons = np.asarray(endpoint_horizons, dtype=float)
    horizons = horizons[(horizons > 0) & (horizons <= float(max_pred_window))]

    horizons_by_event: dict[str, np.ndarray] = {}
    for event_name in event_names:
        if len(horizons):
            horizons_by_event[event_name] = horizons
    return horizons_by_event


def horizon_units_to_bins(
    horizons: np.ndarray,
    *,
    manifest_time_unit_days: int,
    input_time_unit_days: int,
) -> np.ndarray:
    """Map manifest AUC(t) horizons (time_unit_days units, from the landmark)
    onto person-period ``duration_bin`` indices (also time_unit_days units,
    from the landmark; see :func:`patient_targets`'s ``landmark_time``).

    The two axes share an origin and width by construction (the person-period
    builder reuses ``--time-unit-days`` for its binning rather than a
    separate ``--time-bin-days``), so this is the identity map -- provided the
    widths actually match. Raises loudly on mismatch rather than rescaling: a
    fractional bin index is meaningless to a discrete-time head.
    """
    if manifest_time_unit_days != input_time_unit_days:
        raise ValueError(
            f"Horizon unit mismatch: build_manifest.json's time_unit_days="
            f"{manifest_time_unit_days} != the person-period input's "
            f"time_unit_days={input_time_unit_days}. These must be built with "
            "the same --time-unit-days so horizons map 1:1 onto duration bins."
        )
    return np.asarray(horizons, dtype=int)
