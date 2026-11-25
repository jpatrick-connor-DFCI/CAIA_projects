"""Specimen-level IHC annotations rebuilt from the pre-existing step-1 intermediates (no raw data reads).

Inputs are the three parquets ``extract_prostate_ihc`` wrote to
``ihc_extraction/intermediate/``: ``prostate_cohort``, ``prostate_ihc_long`` and
``prostate_ihc_specimen_all``. Nothing is re-extracted or re-parsed.

**Specimen.** One pathology part, keyed by (MRN_INT, ACCESSION, PART), with >= 1 lesional stain row in the long
table (so Ki-67-only workups count as IHC).

**Marker calls** are recomputed per specimen x marker from the long table's lesional rows.
Positive/retained and negative/lost mentions on the same specimen give ``discordant``.
A discordant NE marker counts as positive (focal NE staining is NE-positive); a discordant AR-axis marker counts
as neither positive nor negative. Mentions with neither result give ``equivocal``.

**Flags** (per specimen):

* ``ihc_any``   -- always True (the specimen has a lesional stain row).
* ``ne_pos``    -- >= 1 of synaptophysin, chromogranin, INSM1, CD56 called positive or discordant.
* ``nepc_like`` -- ``ne_pos``, >= 1 AR-axis marker called negative and none called positive.
* ``ar_low``    -- >= 1 AR-axis marker (AR, PSA, NKX3.1, PSAP, PSMA, prostein/P501S) called negative and none
                   called positive, whatever the NE stains show.

``prostate_relevant`` / ``prostate_evidence`` / ``ihc_phenotype`` are taken from ``prostate_ihc_specimen_all``. The
table's ``NE_any_pos`` / ``ARaxis_any_pos`` are checked against the recomputed calls: each must be True when a marker
is called positive and False when none is positive or discordant (only a discordant call may go either way, so both
the original intermediates and a rerun of the current extraction load); otherwise loading raises.

**Date.** ``specimen_date`` is the earliest lesional ``specimen_date`` in the long table (pathology
OUTSIDE_SPECIMEN_DATE, else SPEC_DATE); missing stays missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from COMPASS.prostate_ihc import paths as ihc_paths

SOURCE_VERSION = "prostate-ihc-intermediate-annotations-v2"
SPECIMEN_KEYS = ["MRN_INT", "ACCESSION", "PART"]
# Same marker sets as extract_prostate_ihc (NE, ARX); tests/test_prostate_ihc_annotations.py checks they agree.
NE_MARKERS = ("Synaptophysin", "Chromogranin", "INSM1", "CD56")
AR_AXIS_MARKERS = ("AR", "PSA", "NKX3.1", "PSAP", "PSMA", "Prostein/P501S")
POSITIVE_RESULTS = ("positive", "retained")
NEGATIVE_RESULTS = ("negative", "lost")

LONG_COLUMNS = [*SPECIMEN_KEYS, "target", "marker_norm", "group", "result", "source", "specimen_date"]
SPEC_ALL_COLUMNS = [*SPECIMEN_KEYS, "prostate_evidence", "prostate_relevant", "ihc_phenotype", "NE_histology",
                    "NE_any_pos", "ARaxis_any_pos"]
COHORT_COLUMNS = ["MRN_INT"]
ANNOTATION_COLUMNS = [
    *SPECIMEN_KEYS, "SPECIMEN_ID", "specimen_date", "prostate_relevant", "prostate_evidence", "ihc_phenotype",
    "NE_histology", "ihc_any", "ne_pos", "nepc_like", "ar_low", "ARaxis_any_pos", "ARaxis_any_neg",
]
CALL_COLUMNS = [*SPECIMEN_KEYS, "marker_norm", "call"]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_intermediates(intermediate_dir=None) -> dict[str, pd.DataFrame]:
    """Read the step-1 parquets: ``{'cohort', 'long', 'spec'}``."""
    d = Path(intermediate_dir or ihc_paths.default_paths().intermediate_dir)
    missing = [f for f in ihc_paths.INTERMEDIATE_FILES if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing step-1 intermediates in {d}: {missing}.")
    return {
        "cohort": pd.read_parquet(d / ihc_paths.COHORT_FILE),
        "long": pd.read_parquet(d / ihc_paths.LONG_FILE, columns=LONG_COLUMNS),
        "spec": pd.read_parquet(d / ihc_paths.SPECIMEN_ALL_FILE, columns=SPEC_ALL_COLUMNS),
    }


def _require(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} lacks columns {missing}")


def _dates(s: pd.Series, name: str) -> pd.Series:
    raw = s.astype("string").str.slice(0, 10)
    parsed = pd.to_datetime(raw, errors="coerce", format="%Y-%m-%d")
    bad = raw.notna() & raw.str.strip().ne("") & parsed.isna()
    if bad.any():
        raise ValueError(f"Unparseable {name} values, e.g. {raw[bad].iloc[0]!r}")
    return parsed


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the specimen keys (MRN int; accession / part as strings, missing -> '')."""
    out = df.copy()
    out["MRN_INT"] = pd.to_numeric(out["MRN_INT"], errors="raise").astype("int64")
    for c in ("ACCESSION", "PART"):
        out[c] = out[c].astype("string").fillna("").astype(str)
    return out


def _lesional(long: pd.DataFrame) -> pd.DataFrame:
    _require(long, [*SPECIMEN_KEYS, "target", "marker_norm", "result", "specimen_date"], "IHC long table")
    les = long[long["target"].astype(str).eq("lesional") & long["marker_norm"].notna()]
    return _keyed(les)


# --------------------------------------------------------------------------- #
# Calls and specimen flags
# --------------------------------------------------------------------------- #
def marker_calls(long: pd.DataFrame) -> pd.DataFrame:
    """One row per specimen x marker: ``call`` in positive / negative / discordant / equivocal."""
    les = _lesional(long)
    res = les["result"].astype("string")          # exact match, as in the original script
    les = les.assign(_p=res.isin(POSITIVE_RESULTS).fillna(False).astype(bool),
                     _n=res.isin(NEGATIVE_RESULTS).fillna(False).astype(bool))
    g = les.groupby([*SPECIMEN_KEYS, "marker_norm"], sort=True).agg(p=("_p", "any"), n=("_n", "any")).reset_index()
    g["call"] = np.select([g["p"] & g["n"], g["p"], g["n"]], ["discordant", "positive", "negative"], "equivocal")
    return g[CALL_COLUMNS]


def specimen_annotations(long: pd.DataFrame, spec_all: pd.DataFrame) -> pd.DataFrame:
    """One row per stained specimen (all specimens, relevant or not) with the flags in the module docstring."""
    _require(spec_all, SPEC_ALL_COLUMNS, "prostate_ihc_specimen_all")
    les = _lesional(long)
    calls = marker_calls(long)
    ne, arx = calls["marker_norm"].isin(NE_MARKERS), calls["marker_norm"].isin(AR_AXIS_MARKERS)
    pos, disc = calls["call"].eq("positive"), calls["call"].eq("discordant")
    c = calls.assign(
        ne_pos=ne & (pos | disc),                 # discordant NE -> positive
        ARaxis_any_pos=arx & pos,                 # discordant AR-axis -> neither
        ARaxis_any_neg=arx & calls["call"].eq("negative"),
        _ne_strict=ne & pos, _ar_loose=arx & (pos | disc),   # bounds for the spec_all check
    )
    flags = c.groupby(SPECIMEN_KEYS)[["ne_pos", "ARaxis_any_pos", "ARaxis_any_neg", "_ne_strict", "_ar_loose"]].any()
    dates = (les.assign(specimen_date=_dates(les["specimen_date"], "specimen_date"))
             .groupby(SPECIMEN_KEYS)["specimen_date"].min())
    out = flags.join(dates).reset_index()
    out["ihc_any"] = True
    out["ar_low"] = out["ARaxis_any_neg"] & ~out["ARaxis_any_pos"]
    out["nepc_like"] = out["ne_pos"] & out["ar_low"]

    spec = _keyed(spec_all)
    if spec.duplicated(SPECIMEN_KEYS).any():
        raise ValueError("Duplicate (MRN_INT, ACCESSION, PART) rows in prostate_ihc_specimen_all.")
    merged = out.merge(spec, on=SPECIMEN_KEYS, how="outer", indicator=True, suffixes=("", "_spec_all"))
    only = merged["_merge"].value_counts()
    if only.get("left_only", 0) or only.get("right_only", 0):
        raise ValueError(f"Specimens do not line up between the long table and prostate_ihc_specimen_all: "
                         f"{only.get('left_only', 0)} only in long (lesional), "
                         f"{only.get('right_only', 0)} only in specimen_all.")
    # spec_all may count a discordant marker either way, but must agree wherever no marker is discordant.
    for lo, hi, theirs in (("_ne_strict", "ne_pos", "NE_any_pos"),
                           ("ARaxis_any_pos", "_ar_loose", "ARaxis_any_pos_spec_all")):
        flag = merged[theirs].fillna(False).astype(bool)
        bad = (merged[lo].astype(bool) & ~flag) | (~merged[hi].astype(bool) & flag)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} specimens disagree on {theirs.replace('_spec_all', '')} between the "
                             "recomputed marker calls and prostate_ihc_specimen_all (beyond discordant calls); the "
                             "long and specimen tables do not come from the same extraction.")
    merged["prostate_relevant"] = merged["prostate_relevant"].fillna(False).astype(bool)
    merged["NE_histology"] = merged["NE_histology"].fillna(False).astype(bool)
    merged["SPECIMEN_ID"] = merged["MRN_INT"].astype(str) + "|" + merged["ACCESSION"] + "|" + merged["PART"]
    for col in ("ihc_any", "ne_pos", "nepc_like", "ar_low", "ARaxis_any_pos", "ARaxis_any_neg"):
        merged[col] = merged[col].astype(bool)
    return merged[ANNOTATION_COLUMNS].sort_values(SPECIMEN_KEYS).reset_index(drop=True)


def build(intermediate_dir=None) -> dict[str, pd.DataFrame]:
    """Load the intermediates and return ``{'cohort', 'calls', 'specimens'}``."""
    t = load_intermediates(intermediate_dir)
    return {"cohort": t["cohort"], "calls": marker_calls(t["long"]),
            "specimens": specimen_annotations(t["long"], t["spec"])}


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
FLAG_LABELS = (("stained (ihc_any)", "ihc_any"), ("NE-positive", "ne_pos"), ("NEPC-like", "nepc_like"),
               ("AR-low (AR-axis negative)", "ar_low"))


def summary(specimens: pd.DataFrame) -> pd.DataFrame:
    """Specimens / patients per flag, among all stained specimens and among prostate-relevant ones."""
    rows = []
    for label, col in FLAG_LABELS:
        row = {"flag": label}
        for scope, df in (("all", specimens), ("relevant", specimens[specimens["prostate_relevant"]])):
            hit = df[df[col]]
            row[f"specimens_{scope}"] = len(hit)
            row[f"patients_{scope}"] = hit["MRN_INT"].nunique()
            row[f"undated_{scope}"] = int(hit["specimen_date"].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def marker_frequency(calls: pd.DataFrame, specimens: pd.DataFrame, *, relevant_only: bool = True) -> pd.DataFrame:
    """Specimens per marker: stained, positive, negative, discordant, and stained among NE-positive specimens."""
    spec = specimens[specimens["prostate_relevant"]] if relevant_only else specimens
    c = _keyed(calls).merge(spec[[*SPECIMEN_KEYS, "ne_pos"]], on=SPECIMEN_KEYS, how="inner")
    cols = ["marker", "specimens_stained", "positive", "negative", "discordant", "stained_among_ne_pos"]
    if c.empty:
        return pd.DataFrame(columns=cols)
    out = c.groupby("marker_norm").agg(
        specimens_stained=("call", "size"),
        positive=("call", lambda s: int(s.eq("positive").sum())),
        negative=("call", lambda s: int(s.eq("negative").sum())),
        discordant=("call", lambda s: int(s.eq("discordant").sum())),
        stained_among_ne_pos=("ne_pos", "sum"),
    ).reset_index().rename(columns={"marker_norm": "marker"})
    out["stained_among_ne_pos"] = out["stained_among_ne_pos"].astype(int)
    return out.sort_values(["specimens_stained", "marker"], ascending=[False, True]).reset_index(drop=True)[cols]
