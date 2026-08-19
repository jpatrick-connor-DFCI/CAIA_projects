"""Shared cohort construction and pre-landmark lab feature engineering.

This module is the single home for the landmarked-cohort builders that several
survival-analysis entry points consume: ``build_prediction_inputs`` (the source
of truth for splits / aggregated + longitudinal inputs), ``build_genomic_inputs``
(sample-anchored variant + lab features), and the analysis scripts that re-export
these via ``cox_aggregated``.

Patient identifiers are configurable at runtime. ``ID_COL`` / ``AGE_COL`` default
to the PROFILE schema; entry points can call :func:`configure_id_columns` once in
``main`` if they need alternate column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Runtime-configurable patient-identifier / age columns. Default to the PROFILE
# schema; reconfigured via configure_id_columns() by entry points that need
# alternate names.
ID_COL = "DFCI_MRN"
AGE_COL = "AGE_AT_TREATMENTSTART"

PLATINUM_MEDS = {"CARBOPLATIN", "CISPLATIN"}
MIN_DELTA_OBS = 2

# Causes beyond the always-present platinum/death pair, as {event_col:
# time_col}. Carried through the person-period builder only when the cohort
# supplies both columns, mirroring make_outcome_df's presence gate. Keep in
# sync with build_prediction_inputs.LONGITUDINAL_OPTIONAL_EVENT_COLS -- the
# manifest is written from that one and the data from this one, so a cause
# listed in only one of them yields a manifest that promises a column the
# person-period frame does not carry.
OPTIONAL_LONGITUDINAL_CAUSES = {
    "NEPC": "t_nepc",
    "AVPC": "t_avpc",
    "AVPC_NEPC": "t_avpc_nepc",
}

# Optional per-patient endpoints make_outcome_df derives generically, as
# {endpoint_key: (event_col, duration_col, date_col)}. Each is independently
# present or absent in the input (an upstream cohort can carry NEPC, AVPC,
# AVPC_NEPC, any combination, or none) -- presence is detected per-key, not
# via a single shared flag. `platinum` is not in this registry: it has its
# own bespoke, always-present derivation path feeding t_either/EITHER below.
# `nepc` and `avpc` are the two components of the `avpc_nepc` union endpoint,
# all three sourced from the same AVPC/NEPC criteria-timeline labels file.
OPTIONAL_ENDPOINT_SPECS = {
    "nepc": ("NEPC", "t_nepc", "NEPC_DATE"),
    "avpc": ("AVPC", "t_avpc", "AVPC_DATE"),
    "avpc_nepc": ("AVPC_NEPC", "t_avpc_nepc", "AVPC_NEPC_DATE"),
}


def configure_id_columns(id_col: str, age_col: str) -> None:
    """Set the patient-id / age columns used by every builder in this module.

    Entry points call this once in ``main`` (alongside setting their own module
    globals) so the cohort builders below operate on the requested schema.
    """
    global ID_COL, AGE_COL
    ID_COL = id_col
    AGE_COL = age_col


def normalize_landmark_days(raw_landmark_days: list[int]) -> list[int]:
    landmark_days: list[int] = []
    for raw_day in raw_landmark_days:
        day = int(raw_day)
        if day < 0:
            raise ValueError(f"Landmark days must be non-negative, got {day}.")
        if day not in landmark_days:
            landmark_days.append(day)
    return sorted(landmark_days)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _coerce_duration(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce").astype(float)


def _derive_duration(
    patient_df: pd.DataFrame,
    *,
    duration_col: str,
    event_date_col: str,
    fallback_duration_col: str | None = None,
) -> pd.Series:
    if duration_col in patient_df.columns:
        existing = _coerce_duration(patient_df[duration_col])
        if existing is not None:
            return existing

    derived = pd.Series(np.nan, index=patient_df.index, dtype=float)
    if event_date_col in patient_df.columns and "FIRST_RECORD_DATE" in patient_df.columns:
        event_date = _coerce_datetime(patient_df[event_date_col])
        first_record = _coerce_datetime(patient_df["FIRST_RECORD_DATE"])
        derived = (event_date - first_record).dt.days.astype(float)

    if fallback_duration_col and fallback_duration_col in patient_df.columns:
        fallback = _coerce_duration(patient_df[fallback_duration_col])
        if fallback is not None:
            derived = derived.fillna(fallback)

    return derived


def _coerce_platinum(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    platinum = series.astype(str).str.upper().isin(PLATINUM_MEDS)
    return numeric.fillna(platinum.astype(int)).fillna(0).astype(int)


def make_outcome_df(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int = 0,
    anchor_col: str | None = "t_first_treatment",
    extra_anchor_cols: tuple[str, ...] = (),
    require_first_treatment: bool = True,
    max_followup_days: float | None = None,
    endpoint: str = "platinum",
    require_nepc: bool = False,
) -> pd.DataFrame:
    """Build the per-patient outcome table rebased to a landmark.

    Args:
        landmark_offset_days: days added to ``anchor_col`` to define the landmark.
        anchor_col: column whose value (in days from first record) anchors the
            landmark. Defaults to ``t_first_treatment``; the genomic arm passes
            ``t_sample``. Pass ``None`` when the durations are already measured
            from the index date (e.g. COMPASS's treatment-anchor-relative clock):
            the landmark is then the pure offset ``landmark_offset_days`` from time
            0, with no anchor term and no anchor-based patient filtering.
        extra_anchor_cols: additional patient-level columns to preserve through
            the dedup (e.g. ``("t_sample", "SAMPLE_COLLECTION_DT")`` for the
            genomic arm). These are kept in the returned frame.
        require_first_treatment: whether the cohort filter requires
            ``FIRST_TREATMENT == 1``. Off for the genomic arm where treatment
            timing is irrelevant to the outcome window.
        max_followup_days: administrative-censoring horizon on the
            landmark-relative clock (days from time 0). Patients whose platinum
            or death event lands beyond this horizon are censored at the horizon
            (event flag -> 0, duration -> horizon); censored follow-up
            (``t_last_contact``) is clipped to the horizon too. Defaults to
            ``None`` (no cap, full follow-up used). Previously defaulted to
            3650 (10 years) to guard against a sparse tail destabilizing the
            Cox fits; pass an explicit value to restore that horizon.
        endpoint: endpoint whose duration must be non-null and positive. This is
            deliberately exclusive: platinum timing never gates an NEPC cohort,
            and NEPC timing never gates a platinum cohort.
        require_nepc: deprecated compatibility alias for ``endpoint="nepc"``.
    """
    patient_level_cols = [
        ID_COL,
        AGE_COL,
        "GENDER",
        "FIRST_RECORD_DATE",
        "DIAGNOSIS_DATE",
        "FIRST_TREATMENT_DATE",
        "FIRST_TREATMENT",
        "LAST_CONTACT_DATE",
        "PLATINUM_DATE",
        "PLATINUM",
        "DEATH",
        "t_diagnosis",
        "t_first_treatment",
        "t_platinum",
        "t_last_contact",
        "t_death",
        # NEPC endpoint. Absent unless the cohort was built with the LLM
        # annotations mounted; every use below is guarded on presence.
        "NEPC_DATE",
        "NEPC",
        "t_nepc",
        "NEPC_DATE_SOURCE",
        "NEPC_DATE_PRECISION",
        "NEPC_LABEL_SOURCE",
        # AVPC_NEPC endpoint. Absent unless the cohort was built with the LLM
        # annotations mounted; every use below is guarded on presence.
        "AVPC_NEPC_DATE",
        "AVPC_NEPC",
        "t_avpc_nepc",
        # Audit/provenance columns for AVPC_NEPC, not used in duration math.
        "AVPC_NEPC_DATE_SOURCE",
        "AVPC_NEPC_DATE_PRECISION",
        "AVPC_NEPC_LABEL_SOURCE",
        # AVPC endpoint (>=3 Aparicio criteria alone, independent of any NEPC
        # feature). Absent unless the cohort was built with the LLM
        # annotations mounted; every use below is guarded on presence.
        "AVPC",
        "AVPC_DATE",
        "t_avpc",
        "AVPC_N_CRITERIA",
        "NEPC_TIMELINE",
        "NEPC_TIMELINE_DATE",
        *extra_anchor_cols,
    ]
    available_cols = [col for col in patient_level_cols if col in df.columns]
    if ID_COL not in available_cols:
        raise ValueError(f"Input data must contain the id column {ID_COL!r}.")

    pat = df[available_cols].drop_duplicates(ID_COL).set_index(ID_COL)

    if "FIRST_RECORD_DATE" not in pat.columns:
        if "LAB_DATE" not in df.columns:
            raise ValueError("Input data must contain FIRST_RECORD_DATE or LAB_DATE.")
        first_record = _coerce_datetime(df["LAB_DATE"]).groupby(df[ID_COL]).min()
        pat["FIRST_RECORD_DATE"] = first_record

    for date_col in [
        "FIRST_RECORD_DATE",
        "DIAGNOSIS_DATE",
        "FIRST_TREATMENT_DATE",
        "LAST_CONTACT_DATE",
        "PLATINUM_DATE",
        "NEPC_DATE",
        "AVPC_NEPC_DATE",
    ]:
        if date_col in pat.columns:
            pat[date_col] = _coerce_datetime(pat[date_col])

    if AGE_COL in pat.columns:
        pat[AGE_COL] = pd.to_numeric(pat[AGE_COL], errors="coerce")
    else:
        pat[AGE_COL] = np.nan
    if "GENDER" in pat.columns:
        pat["GENDER"] = pat["GENDER"].astype("string").str.strip()
    pat["DEATH"] = pd.to_numeric(pat.get("DEATH"), errors="coerce").fillna(0).astype(int)
    pat["PLATINUM"] = _coerce_platinum(pat.get("PLATINUM", pd.Series(0, index=pat.index)))
    pat["FIRST_TREATMENT"] = pd.to_numeric(
        pat.get(
            "FIRST_TREATMENT",
            pat.get("FIRST_TREATMENT_DATE", pd.Series(index=pat.index)).notna(),
        ),
        errors="coerce",
    ).fillna(0).astype(int)

    pat["t_last_contact"] = _derive_duration(
        pat,
        duration_col="t_last_contact",
        event_date_col="LAST_CONTACT_DATE",
    )
    pat["t_death"] = _derive_duration(
        pat,
        duration_col="t_death",
        event_date_col="LAST_CONTACT_DATE",
        fallback_duration_col="t_last_contact",
    )
    pat["t_diagnosis"] = _derive_duration(
        pat,
        duration_col="t_diagnosis",
        event_date_col="DIAGNOSIS_DATE",
        fallback_duration_col="t_last_contact",
    )
    pat["t_first_treatment"] = _derive_duration(
        pat,
        duration_col="t_first_treatment",
        event_date_col="FIRST_TREATMENT_DATE",
        fallback_duration_col="t_last_contact",
    )
    pat["t_platinum"] = _derive_duration(
        pat,
        duration_col="t_platinum",
        event_date_col="PLATINUM_DATE",
    )
    pat["t_platinum"] = pat["t_platinum"].where(
        pat["PLATINUM"].eq(1),
        pat["t_platinum"].fillna(pat["t_last_contact"]),
    )

    # Optional endpoints (NEPC, AVPC_NEPC, ...), each derived exactly like
    # platinum but only when the upstream cohort carried it. ``has_optional``
    # gates every endpoint-touching block below per key, so an input missing
    # one or both optional endpoints flows through unchanged for the rest.
    has_optional: dict[str, bool] = {}
    for key, (event_col, duration_col, date_col) in OPTIONAL_ENDPOINT_SPECS.items():
        has_col = event_col in pat.columns or duration_col in pat.columns
        has_optional[key] = has_col
        if not has_col:
            continue
        pat[event_col] = (
            pd.to_numeric(pat.get(event_col), errors="coerce").fillna(0).astype(int)
        )
        pat[duration_col] = _derive_duration(
            pat,
            duration_col=duration_col,
            event_date_col=date_col,
        )
        pat[duration_col] = pat[duration_col].where(
            pat[event_col].eq(1),
            pat[duration_col].fillna(pat["t_last_contact"]),
        )

    if anchor_col is None:
        # Durations are already measured from the index date; the landmark is a
        # pure offset from time 0 (no anchor term, no anchor-based filtering).
        landmark_time = float(landmark_offset_days)
    else:
        if anchor_col not in pat.columns:
            raise ValueError(f"make_outcome_df: anchor_col {anchor_col!r} missing from input.")
        landmark_time = pat[anchor_col].astype(float) + float(landmark_offset_days)
    rebased_duration_cols = ["t_last_contact", "t_death", "t_platinum"]
    for key, (_event_col, duration_col, _date_col) in OPTIONAL_ENDPOINT_SPECS.items():
        if has_optional[key]:
            rebased_duration_cols.append(duration_col)
    for duration_col in rebased_duration_cols:
        pat[f"{duration_col}_from_first_record"] = pat[duration_col]
        pat[duration_col] = pat[duration_col].astype(float) - landmark_time

    # Administrative censoring at max_followup_days (on the landmark-relative
    # clock). An event whose time exceeds the horizon becomes a censored
    # observation AT the horizon; censored follow-up is clipped to the horizon
    # too. Done before the first_event_time derivation below so
    # PLATINUM/DEATH/EITHER all stay consistent with the clipped
    # durations. The *_from_first_record columns are left uncapped (they record
    # the raw timing for diagnostics, not the modeled outcome).
    if max_followup_days is not None:
        horizon = float(max_followup_days)
        platinum_past = pat["PLATINUM"].eq(1) & pat["t_platinum"].gt(horizon)
        death_past = pat["DEATH"].eq(1) & pat["t_death"].gt(horizon)
        n_platinum_censored = int(platinum_past.sum())
        n_death_censored = int(death_past.sum())
        pat.loc[platinum_past, "PLATINUM"] = 0
        pat.loc[death_past, "DEATH"] = 0
        optional_msg = ""
        for key, (event_col, duration_col, _date_col) in OPTIONAL_ENDPOINT_SPECS.items():
            if not has_optional[key]:
                continue
            optional_past = pat[event_col].eq(1) & pat[duration_col].gt(horizon)
            pat.loc[optional_past, event_col] = 0
            optional_msg += f" and {int(optional_past.sum())} {event_col} events"
        for duration_col in rebased_duration_cols:
            pat[duration_col] = pat[duration_col].clip(upper=horizon)
        print(
            f"[make_outcome_df @ landmark +{landmark_offset_days}d] administrative "
            f"censoring at {horizon:g}d: {n_platinum_censored} PLATINUM events and "
            f"{n_death_censored} DEATH events{optional_msg} pushed past the horizon censored."
        )

    platinum_event_time = np.where(pat["PLATINUM"].eq(1), pat["t_platinum"], np.inf)
    death_event_time = np.where(pat["DEATH"].eq(1), pat["t_death"], np.inf)
    first_event_time = np.minimum(platinum_event_time, death_event_time)

    pat["EITHER"] = np.isfinite(first_event_time).astype(int)
    pat["t_either"] = np.where(pat["EITHER"].eq(1), first_event_time, pat["t_death"])

    if require_nepc:
        endpoint = "nepc"
    endpoint = str(endpoint).lower()
    if endpoint not in {"platinum", *OPTIONAL_ENDPOINT_SPECS}:
        expected = "', '".join(["platinum", *OPTIONAL_ENDPOINT_SPECS])
        raise ValueError(
            f"make_outcome_df: unknown endpoint {endpoint!r}; expected '{expected}'."
        )
    if endpoint in OPTIONAL_ENDPOINT_SPECS and not has_optional[endpoint]:
        event_col, duration_col, date_col = OPTIONAL_ENDPOINT_SPECS[endpoint]
        if endpoint == "nepc":
            # Preserve the "(require_nepc)" substring some callers/tests
            # match on for the one endpoint that predates this generic
            # registry; the rest of the text reflects the current label
            # source (the avpc_nepc criteria-timeline file, not the old
            # strict NEPC diagnosis labels).
            raise ValueError(
                "make_outcome_df: endpoint='nepc' (require_nepc) but the input carries no NEPC "
                "columns. Rebuild the survival cohort with the LLM AVPC/NEPC "
                "criteria-timeline labels (--nepc-labels) before running the nepc endpoint."
            )
        raise ValueError(
            f"make_outcome_df: endpoint={endpoint!r} but the input carries no "
            f"{event_col} columns. Rebuild the survival cohort with the LLM "
            f"{event_col} labels before running the {endpoint} endpoint."
        )

    # Individual validity conditions, kept separate so the attrition each one
    # causes can be reported below. Durations here are already landmark-rebased
    # (shifted by landmark_time above), so a non-positive value means the event /
    # last contact falls at or before the landmark.
    conditions: dict[str, pd.Series] = {
        "FIRST_RECORD_DATE notna": pat["FIRST_RECORD_DATE"].notna(),
        "t_death notna": pat["t_death"].notna(),
        "t_last_contact notna": pat["t_last_contact"].notna(),
        "t_death > 0": pat["t_death"].gt(0),
        "t_last_contact > 0": pat["t_last_contact"].gt(0),
    }
    if endpoint == "platinum":
        conditions["t_platinum notna"] = pat["t_platinum"].notna()
        conditions["t_platinum > 0"] = pat["t_platinum"].gt(0)
        conditions["t_either notna"] = pat["t_either"].notna()
        conditions["t_either > 0"] = pat["t_either"].gt(0)
    else:
        # Incident semantics for optional endpoints (NEPC, AVPC_NEPC, ...):
        # only that endpoint's own timing joins the endpoint-specific gate. In
        # particular, pre-anchor platinum exposure is irrelevant here.
        _, duration_col, _ = OPTIONAL_ENDPOINT_SPECS[endpoint]
        conditions[f"{duration_col} notna"] = pat[duration_col].notna()
        conditions[f"{duration_col} > 0"] = pat[duration_col].gt(0)

    if anchor_col is not None:
        # A real anchor column must be present and on-or-after first record; with
        # anchor_col=None the durations are already index-relative and there is no
        # anchor to gate on.
        conditions[f"{anchor_col} notna"] = pat[anchor_col].notna()
        conditions[f"{anchor_col} >= 0"] = pat[anchor_col].ge(0)
    if require_first_treatment:
        conditions["FIRST_TREATMENT == 1"] = pat["FIRST_TREATMENT"].eq(1)

    valid = pd.Series(True, index=pat.index)
    for cond in conditions.values():
        valid = valid & cond

    # Per-condition attrition report: for each condition, how many patients it
    # fails on its own ("failed alone") and how many are lost cumulatively as the
    # conditions are ANDed in order ("dropped this step"). "failed alone" columns
    # sum to more than the total dropped because a patient can fail several.
    n_in = len(pat)
    print(f"[make_outcome_df @ landmark +{landmark_offset_days}d] "
          f"validity attrition from {n_in} patients:")
    running = pd.Series(True, index=pat.index)
    for name, cond in conditions.items():
        failed_alone = int((~cond).sum())
        before = int(running.sum())
        running = running & cond
        dropped_step = before - int(running.sum())
        print(f"    {name:<24} failed alone={failed_alone:<5} "
              f"dropped this step={dropped_step:<5} remaining={int(running.sum())}")
    print(f"    => {int(valid.sum())}/{n_in} patients pass all conditions.")

    return pat.loc[valid].copy()


def build_pre_treatment_lab_long(
    df: pd.DataFrame,
    *,
    cohort_index: pd.Index | None = None,
    landmark_offset_days: int = 0,
    anchor_col: str | None = "t_first_treatment",
    anchor_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Long-format pre-landmark lab observations used for canonical-lab selection.

    Returns columns DFCI_MRN, LAB_NAME, LAB_VALUE, t_lab[, <anchor_col>].
    Restricts to observations with t_lab < <anchor_col> + landmark_offset_days
    so the lab presence used for coverage matches the aggregated feature
    engineering and Dynamic DeepHit person-period builder windows.

    ``anchor_series`` (MRN-indexed) overrides the column lookup, supporting the
    genomic arm where ``t_sample`` is per-patient and not carried on every row.

    Pass ``anchor_col=None`` when ``t_lab`` is already measured from the index
    date (COMPASS's treatment-anchor clock): the window is then the pure offset
    ``t_lab < landmark_offset_days`` with no anchor column required.
    """
    base_required = {ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab"}
    missing = base_required - set(df.columns)
    if missing:
        raise ValueError(
            f"build_pre_treatment_lab_long missing columns: {sorted(missing)}"
        )

    cols = list(base_required)
    if anchor_col is not None and anchor_series is None:
        if anchor_col not in df.columns:
            raise ValueError(
                f"build_pre_treatment_lab_long needs anchor column {anchor_col!r} on df or via anchor_series."
            )
        cols.append(anchor_col)
    out = df[cols].copy()
    out["LAB_NAME"] = out["LAB_NAME"].astype(str).str.strip()
    out["LAB_VALUE"] = pd.to_numeric(out["LAB_VALUE"], errors="coerce")
    out["t_lab"] = pd.to_numeric(out["t_lab"], errors="coerce")
    dropna_cols = [ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab"]
    if anchor_col is not None:
        if anchor_series is not None:
            out[anchor_col] = out[ID_COL].map(anchor_series.astype(float)).astype(float)
        else:
            out[anchor_col] = pd.to_numeric(out[anchor_col], errors="coerce")
        dropna_cols.append(anchor_col)
    out = out.dropna(subset=dropna_cols)
    if anchor_col is None:
        # t_lab is already index-relative; window is the pure offset from 0.
        landmark_t = float(landmark_offset_days)
    else:
        landmark_t = out[anchor_col] + float(landmark_offset_days)
    out = out.loc[out["t_lab"] < landmark_t].copy()
    if cohort_index is not None:
        out = out.loc[out[ID_COL].isin(cohort_index)].copy()
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int = 0,
    anchor_col: str | None = "t_first_treatment",
    anchor_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-patient lab summary features for the pre-landmark window.

    Args:
        landmark_offset_days: days added to ``anchor_col`` to define the landmark.
        anchor_col: column on ``df`` whose value (in days from first record) is
            the anchor. Default ``t_first_treatment``; the genomic arm uses
            ``t_sample``. Pass ``None`` when ``t_lab`` is already index-relative
            (COMPASS's treatment-anchor clock): the window is the pure offset
            ``t_lab < landmark_offset_days`` with no anchor column required.
        anchor_series: optional MRN-indexed Series providing per-patient anchor
            values when the column isn't carried on every lab row (e.g.
            ``t_sample`` joined externally). Takes precedence over ``anchor_col``.
    """
    working = df.copy()
    required_cols = {ID_COL, "LAB_NAME", "LAB_VALUE"}
    missing_required = required_cols - set(working.columns)
    if missing_required:
        missing_str = ", ".join(sorted(missing_required))
        raise ValueError(f"Input data is missing required columns for feature engineering: {missing_str}")

    working["LAB_NAME"] = working["LAB_NAME"].astype(str).str.strip()
    working["LAB_VALUE"] = pd.to_numeric(working["LAB_VALUE"], errors="coerce")

    if "t_lab" not in working.columns:
        if "LAB_DATE" not in working.columns:
            raise ValueError("Input data must contain t_lab or LAB_DATE.")
        if "FIRST_RECORD_DATE" not in working.columns:
            working["FIRST_RECORD_DATE"] = _coerce_datetime(working["LAB_DATE"]).groupby(working[ID_COL]).transform("min")
        working["t_lab"] = (
            _coerce_datetime(working["LAB_DATE"]) - _coerce_datetime(working["FIRST_RECORD_DATE"])
        ).dt.days.astype(float)
    else:
        working["t_lab"] = _coerce_duration(working["t_lab"])

    dropna_cols = [ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab"]
    if anchor_col is None:
        # t_lab is already index-relative; the pre-landmark window is the pure
        # offset from time 0, with no anchor column required.
        working = working.dropna(subset=dropna_cols)
        landmark_time = float(landmark_offset_days)
    else:
        if anchor_series is not None:
            anchor_map = anchor_series.astype(float)
            working[anchor_col] = working[ID_COL].map(anchor_map).astype(float)
        elif anchor_col not in working.columns:
            if anchor_col == "t_first_treatment" and {"FIRST_TREATMENT_DATE", "FIRST_RECORD_DATE"}.issubset(working.columns):
                working["t_first_treatment"] = (
                    _coerce_datetime(working["FIRST_TREATMENT_DATE"])
                    - _coerce_datetime(working["FIRST_RECORD_DATE"])
                ).dt.days.astype(float)
            else:
                raise ValueError(
                    f"build_feature_matrix needs anchor column {anchor_col!r} on the df or via anchor_series."
                )
        else:
            working[anchor_col] = _coerce_duration(working[anchor_col])
        working = working.dropna(subset=[*dropna_cols, anchor_col])
        landmark_time = working[anchor_col].astype(float) + float(landmark_offset_days)

    pre_treatment = working.loc[working["t_lab"].lt(landmark_time)].copy()
    if pre_treatment.empty:
        raise ValueError("No pre-landmark lab rows were available to build lab summary features.")

    sort_cols = [ID_COL, "LAB_NAME", "t_lab"]
    if "LAB_DATE" in pre_treatment.columns:
        pre_treatment["LAB_DATE"] = _coerce_datetime(pre_treatment["LAB_DATE"])
        sort_cols.append("LAB_DATE")
    pre_treatment = pre_treatment.sort_values(sort_cols)

    feature_long = (
        pre_treatment.groupby([ID_COL, "LAB_NAME"])["LAB_VALUE"]
        .agg(
            mean="mean",
            min="min",
            max="max",
            first="first",
            last="last",
            n_observations="count",
        )
        .reset_index()
    )
    feature_long["delta"] = np.where(
        feature_long["n_observations"] >= MIN_DELTA_OBS,
        feature_long["last"] - feature_long["first"],
        np.nan,
    )
    feature_long = feature_long.drop(columns=["first"])
    feature_df = (
        feature_long.set_index([ID_COL, "LAB_NAME"])
        .stack()
        .rename("value")
        .reset_index()
        .rename(columns={"level_2": "feature_stat"})
    )
    feature_df["feature_name"] = feature_df["LAB_NAME"] + "__" + feature_df["feature_stat"]
    feature_df = feature_df.pivot(index=ID_COL, columns="feature_name", values="value")
    feature_df = feature_df.sort_index(axis=1)

    print(f"Raw feature matrix: {feature_df.shape[0]} patients x {feature_df.shape[1]} summary-lab features")
    return feature_df


def select_longitudinal_labs(
    labs: pd.DataFrame,
    *,
    train_mrns: set,
    min_coverage: float,
    max_labs: int | None,
    canonical_labs: list[str] | None,
) -> list[str]:
    """Choose the lab panel for the person-period sequence, coverage-ranked.

    If ``canonical_labs`` is given (the same train+valid-selected panel Cox/
    XGBoost use, via :func:`survival_common.helper.select_canonical_labs`),
    intersect it with what is actually present in ``labs`` rather than
    re-deriving coverage — this keeps the longitudinal panel and the
    aggregated-feature panel in lockstep. Otherwise rank by train-only
    coverage (fraction of ``train_mrns`` with >=1 observation) among labs with
    more than one distinct observed value (a constant lab carries no sequence
    signal).
    """
    if canonical_labs is not None:
        present = set(labs["LAB_NAME"].astype(str))
        selected = [lab for lab in canonical_labs if lab in present]
        if not selected:
            raise ValueError("No canonical labs are present in the longitudinal lab table.")
        if max_labs is not None:
            selected = selected[:max_labs]
        return selected
    if not train_mrns:
        raise ValueError("Training set is empty; cannot select labs.")
    train_labs = labs.loc[labs[ID_COL].isin(train_mrns)]
    coverage = train_labs.groupby("LAB_NAME")[ID_COL].nunique() / len(train_mrns)
    variability = train_labs.groupby("LAB_NAME")["LAB_VALUE"].nunique()
    eligible = coverage.index[(coverage >= min_coverage) & (variability.reindex(coverage.index) > 1)]
    if not len(eligible):
        raise ValueError(f"No labs passed coverage >= {min_coverage} on training set.")
    ranked = coverage.loc[eligible].sort_values(ascending=False)
    if max_labs is not None and len(ranked) > max_labs:
        ranked = ranked.head(max_labs)
    return ranked.index.tolist()


def fit_clip_bounds(
    agg: pd.DataFrame,
    *,
    train_mrns: set,
    labs: list[str],
    q_lo: float,
    q_hi: float,
) -> dict[str, tuple[float, float]]:
    """Per-lab (lo, hi) outlier clip bounds fit on TRAIN only.

    Fit on the person-period long frame (one row per patient/TIME/lab), not
    the static aggregated table, so the bounds reflect the actual sequence
    values the model will see. Skips a lab if it has fewer than 10 train
    observations or a degenerate (hi <= lo) range.
    """
    train_agg = agg.loc[agg[ID_COL].isin(train_mrns)]
    bounds: dict[str, tuple[float, float]] = {}
    for lab in labs:
        vals = train_agg.loc[train_agg["LAB_NAME"] == lab, "LAB_VALUE"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            continue
        lo = float(np.quantile(vals, q_lo))
        hi = float(np.quantile(vals, q_hi))
        if hi <= lo:
            continue
        bounds[lab] = (lo, hi)
    return bounds


def apply_clip_bounds(agg: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Clip ``LAB_VALUE`` in a long (id, TIME, LAB_NAME, LAB_VALUE) frame to
    per-lab ``bounds`` (unbounded labs pass through unchanged)."""
    if not bounds:
        return agg
    vals = agg["LAB_VALUE"].to_numpy(dtype=float)
    lo_series = agg["LAB_NAME"].map({k: v[0] for k, v in bounds.items()}).to_numpy(dtype=float)
    hi_series = agg["LAB_NAME"].map({k: v[1] for k, v in bounds.items()}).to_numpy(dtype=float)
    has_bound = np.isfinite(lo_series) & np.isfinite(hi_series)
    clipped = vals.copy()
    clipped[has_bound] = np.clip(vals[has_bound], lo_series[has_bound], hi_series[has_bound])
    out = agg.copy()
    out["LAB_VALUE"] = clipped
    return out


def build_person_period_wide(
    lab_long: pd.DataFrame,
    outcome_df: pd.DataFrame,
    *,
    landmark_day: int,
    train_mrns: set,
    canonical_labs: list[str] | None,
    time_unit_days: int,
    min_coverage: float,
    max_labs: int | None,
    outlier_lo: float,
    outlier_hi: float,
    anchor_col: str | None = None,
) -> tuple[pd.DataFrame, dict, list[str], dict[str, tuple[float, float]]]:
    """Build the person-period ("wide", binned-time) sequence input consumed
    by Dynamic-DeepHit and SurvLatent ODE.

    Args:
        lab_long: pre-landmark lab observations, e.g. from
            :func:`build_pre_treatment_lab_long` called at the same
            ``landmark_day``/``anchor_col``. Already windowed to
            ``t_lab < landmark`` -- this function does not re-filter.
        outcome_df: the already-built per-patient outcome+feature+split table
            (:func:`build_landmark_merged`'s ``merged``, post administrative
            censoring). Outcome columns (``PLATINUM``/``DEATH``/``t_platinum``/
            ``t_death``/``split``) are read from here rather than a bespoke
            static table, so the person-period cohort is exactly the
            aggregated cohort -- the precondition for cross-model
            comparability with Cox/XGBoost.
        anchor_col: must match what ``lab_long`` was built with. ``None``
            (COMPASS) means ``t_lab`` is already landmark-relative, so
            ``REL_BIN = floor((t_lab - landmark_day) / time_unit_days)``. A
            real anchor column is not supported here (COMPASS is the only
            caller today; a future anchored caller must pass anchor-relative
            ``t_lab`` and adapt this formula rather than silently reusing it).

    Returns ``(wide_df, extras, selected_labs, bounds)`` where ``extras`` has
    ``max_landmark_time`` (TRAIN-only, for SurvLatent's absolute prediction
    window) and ``n_patients_without_selected_labs``.
    """
    if anchor_col is not None:
        raise NotImplementedError(
            "build_person_period_wide only supports anchor_col=None "
            "(index-relative durations, COMPASS's clock) today."
        )

    required_outcome_cols = {AGE_COL, "PLATINUM", "DEATH", "t_platinum", "t_death", "split"}
    missing_outcome = required_outcome_cols - set(outcome_df.columns)
    if missing_outcome:
        raise ValueError(
            f"build_person_period_wide: outcome_df is missing columns {sorted(missing_outcome)}."
        )
    # Optional causes are carried only when the cohort actually has them, the
    # same gate make_outcome_df applies. This must stay in step with
    # build_prediction_inputs.longitudinal_event_columns(), which advertises
    # the cause in the longitudinal manifest off the *same* outcome frame: if
    # the manifest declares NEPC but the person-period frame drops it,
    # resolve_config("nepc") passes and patient_targets then dies on a
    # KeyError for the missing column.
    optional_outcome_cols = {
        event_col: time_col
        for event_col, time_col in OPTIONAL_LONGITUDINAL_CAUSES.items()
        if event_col in outcome_df.columns and time_col in outcome_df.columns
    }
    carried_cols = set(required_outcome_cols)
    for event_col, time_col in optional_outcome_cols.items():
        carried_cols.update((event_col, time_col))
    static = outcome_df[sorted(carried_cols)].copy()

    labs = lab_long[[ID_COL, "LAB_NAME", "LAB_VALUE", "t_lab"]].copy()
    labs = labs.loc[labs[ID_COL].isin(static.index)]

    selected_labs = select_longitudinal_labs(
        labs,
        train_mrns=train_mrns,
        min_coverage=min_coverage,
        max_labs=max_labs,
        canonical_labs=canonical_labs,
    )
    selected_lab_rows = labs.loc[labs["LAB_NAME"].isin(selected_labs)].copy()

    # Anchor-agnostic REL_BIN: t_lab is already landmark-relative (anchor_col
    # is None), so the bin index is a pure offset from landmark_day -- no
    # t_first_treatment term. Getting this formula wrong silently shifts
    # every patient's time axis.
    selected_lab_rows["REL_BIN"] = np.floor(
        (selected_lab_rows["t_lab"].to_numpy(dtype=float) - float(landmark_day))
        / float(time_unit_days)
    ).astype(int)

    landmark_time = pd.Series(0, index=static.index, dtype=int, name="landmark_time")
    selected_lab_landmarks = (-selected_lab_rows.groupby(ID_COL)["REL_BIN"].min()).astype(int)
    landmark_time.loc[selected_lab_landmarks.index] = selected_lab_landmarks
    n_without_selected_labs = int((landmark_time == 0).sum())
    print(
        f"[build_person_period_wide @ landmark +{landmark_day}d] "
        f"{n_without_selected_labs}/{len(static)} patients have no pre-landmark "
        "observation of a selected lab (landmark row only)."
    )

    selected_lab_rows = selected_lab_rows.merge(
        landmark_time.reset_index(), on=ID_COL, how="inner"
    )
    selected_lab_rows["TIME"] = selected_lab_rows["REL_BIN"] + selected_lab_rows["landmark_time"]

    landmark_time = (
        selected_lab_rows.groupby(ID_COL)["landmark_time"].first()
        .reindex(static.index)
        .fillna(0)
        .astype(int)
        .rename("landmark_time")
    )

    agg = (
        selected_lab_rows.groupby([ID_COL, "TIME", "LAB_NAME"], sort=False)["LAB_VALUE"]
        .mean()
        .reset_index()
    )
    bounds = fit_clip_bounds(
        agg, train_mrns=train_mrns, labs=selected_labs, q_lo=outlier_lo, q_hi=outlier_hi
    )
    agg = apply_clip_bounds(agg, bounds)

    wide = agg.pivot_table(
        index=[ID_COL, "TIME"], columns="LAB_NAME", values="LAB_VALUE", aggfunc="mean"
    ).reset_index()
    wide.columns.name = None
    for lab in selected_labs:
        if lab not in wide.columns:
            wide[lab] = np.nan
    wide = wide.merge(landmark_time.reset_index(), on=ID_COL, how="inner")

    # Synthetic all-NaN landmark row at TIME == landmark_time, so every
    # patient (even one with zero pre-landmark labs) has an anchor row for
    # the sequence model.
    landmark_rows = pd.DataFrame(
        {
            ID_COL: landmark_time.index,
            "TIME": landmark_time.to_numpy(dtype=int),
            "landmark_time": landmark_time.to_numpy(dtype=int),
        }
    )
    for lab in selected_labs:
        landmark_rows[lab] = np.nan
    wide = pd.concat([wide, landmark_rows], ignore_index=True, sort=False)

    optional_time_cols = [optional_outcome_cols[e] for e in sorted(optional_outcome_cols)]
    static_cols = (
        [AGE_COL, "t_platinum", "t_death"]
        + optional_time_cols
        + ["PLATINUM", "DEATH"]
        + sorted(optional_outcome_cols)
        + ["split"]
    )
    wide = wide.merge(static[static_cols], left_on=ID_COL, right_index=True, how="inner")

    # Event-time rebasing: the cause time columns (landmark-relative, already
    # administratively censored via outcome_df) become bin indices on the
    # same landmark_time-origin clock as TIME, so patient_targets's
    # `duration = event_time - landmark` lines up with the person-period axis.
    for time_col in ["t_platinum", "t_death"] + optional_time_cols:
        after_landmark = np.ceil(
            (wide[time_col].to_numpy(dtype=float) - float(landmark_day)) / float(time_unit_days)
        )
        wide[time_col] = wide["landmark_time"].to_numpy(dtype=float) + after_landmark

    # Deliberately platinum/death-only, and *not* widened to the optional
    # causes. Every row here is at TIME <= landmark_time (pre-landmark labs
    # plus the synthetic landmark anchor) while an optional cause's event is
    # post-landmark by cohort construction, so adding it to this min() cannot
    # drop a row it should drop -- it can only drop rows from the platinum
    # arm, whose person-period frame must stay byte-identical to today's.
    event_time = wide[["t_platinum", "t_death"]].min(axis=1).to_numpy(dtype=float)
    n_before_event_filter = len(wide)
    wide = wide.loc[wide["TIME"].to_numpy(dtype=float) < event_time].copy()
    print(
        f"[build_person_period_wide @ landmark +{landmark_day}d] pre-event row "
        f"filter: {n_before_event_filter - len(wide)}/{n_before_event_filter} rows "
        "dropped (at/after the patient's event or censoring time)."
    )

    counts = wide.groupby(ID_COL).size()
    surviving = counts[counts > 0].index
    wide = wide.loc[wide[ID_COL].isin(surviving)].copy()

    # max_landmark_time is TRAIN-only: SurvLatent needs it to size its
    # absolute prediction window, and sizing it from TEST would leak
    # information about the (held-out) test cohort's history length.
    max_landmark_time = (
        int(wide.loc[wide["split"] == "train", "landmark_time"].max())
        if "landmark_time" in wide.columns and not wide.empty
        else 0
    )
    # Optional causes are appended after the always-present platinum/death
    # block so a platinum-only cohort's column order is unchanged.
    column_order = (
        [ID_COL, "TIME"]
        + selected_labs
        + [AGE_COL, "PLATINUM", "DEATH", "t_platinum", "t_death"]
        + sorted(optional_outcome_cols)
        + optional_time_cols
        + ["split"]
    )
    wide = wide.sort_values([ID_COL, "TIME"])[column_order]

    extras = {
        "max_landmark_time": max_landmark_time,
        "n_patients_without_selected_labs": n_without_selected_labs,
    }
    return wide, extras, selected_labs, bounds


def build_landmark_merged(
    df: pd.DataFrame,
    *,
    landmark_offset_days: int,
    anchor_col: str | None = "t_first_treatment",
    require_first_treatment: bool = True,
    max_followup_days: float | None = None,
    endpoint: str = "platinum",
    require_nepc: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the landmarked outcome + feature merge for a given anchor.

    ``anchor_col`` defaults to ``t_first_treatment`` (the standard pipeline). A
    non-standard anchor (e.g. ``t_sample``) is not among the patient-level columns
    make_outcome_df keeps by default, so it is passed through ``extra_anchor_cols``
    to survive the per-patient dedup — mirroring how the genomic arm carries
    ``t_sample``. Pass ``anchor_col=None`` when durations are already index-relative
    (COMPASS): the landmark is then a pure offset and no anchor column is needed.

    ``max_followup_days`` forwards to :func:`make_outcome_df`'s administrative
    censoring horizon; defaults to ``None`` (no cap) so callers must opt in
    explicitly. ``endpoint`` selects the only endpoint-specific validity gate.
    ``require_nepc`` remains as a deprecated compatibility alias.
    """
    extra_anchor_cols = (
        () if anchor_col in (None, "t_first_treatment") else (anchor_col,)
    )
    outcome_df = make_outcome_df(
        df,
        landmark_offset_days=landmark_offset_days,
        anchor_col=anchor_col,
        extra_anchor_cols=extra_anchor_cols,
        require_first_treatment=require_first_treatment,
        max_followup_days=max_followup_days,
        endpoint=endpoint,
        require_nepc=require_nepc,
    )
    print(f"Outcome table @ landmark +{landmark_offset_days}d: {len(outcome_df)} patients")

    print(f"Building raw aggregated lab summary feature matrix through landmark +{landmark_offset_days}d...")
    feature_df = build_feature_matrix(
        df, landmark_offset_days=landmark_offset_days, anchor_col=anchor_col
    )

    n_outcome_only = len(outcome_df.index.difference(feature_df.index))
    n_feature_only = len(feature_df.index.difference(outcome_df.index))
    print(
        f"Feature/outcome index overlap @ landmark +{landmark_offset_days}d: "
        f"outcome_df n={len(outcome_df)}, feature_df n={len(feature_df)}; "
        f"{n_outcome_only} outcome-valid patients have no pre-landmark lab feature "
        f"(kept, features left all-NaN for downstream imputation); "
        f"{n_feature_only} feature-only patients dropped (no valid outcome)."
    )
    # Left join on outcome_df: every patient with a valid landmark outcome is kept
    # even if they have zero pre-landmark labs (build_feature_matrix's pivot only
    # emits rows for patients with >=1 qualifying lab, so an inner join here would
    # silently drop otherwise-eligible, just lab-sparse patients). Missing feature
    # columns are handled downstream via per-feature mean imputation (see
    # SimpleImputer usage in cox_models.py), so an all-NaN feature row is valid
    # input, not a defect.
    merged = outcome_df.join(feature_df, how="left")
    n_before_age_filter = len(merged)
    n_missing_age = int(merged[AGE_COL].isna().sum())
    merged = merged.loc[merged[AGE_COL].notna()].copy()
    print(
        f"Feature+outcome merge @ landmark +{landmark_offset_days}d: {n_before_age_filter} patients; "
        f"{AGE_COL} missing for {n_missing_age}; {len(merged)} remaining."
    )
    if merged.empty:
        raise ValueError("No patients have both engineered features and valid outcomes.")
    return outcome_df, feature_df, merged


def build_landmark_availability_table(
    merged_by_landmark: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Index]:
    if not merged_by_landmark:
        raise ValueError("No landmark cohorts were provided.")

    all_mrns = pd.Index([])
    common_mrns: pd.Index | None = None
    for merged in merged_by_landmark.values():
        all_mrns = all_mrns.union(merged.index)
        common_mrns = merged.index if common_mrns is None else common_mrns.intersection(merged.index)

    availability = pd.DataFrame(index=all_mrns)
    landmark_cols: list[str] = []
    for landmark_day in sorted(merged_by_landmark):
        col = f"eligible_landmark_{landmark_day}"
        landmark_cols.append(col)
        availability[col] = availability.index.isin(merged_by_landmark[landmark_day].index)
    availability["eligible_all_landmarks"] = availability[landmark_cols].all(axis=1)
    availability = availability.rename_axis(ID_COL).reset_index()
    return availability, (common_mrns if common_mrns is not None else pd.Index([]))
