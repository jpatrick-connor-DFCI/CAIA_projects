"""Cohort loaders for the COMPASS PROFILE survival-analysis pipeline."""

from __future__ import annotations

import pandas as pd


def record_span_days(labs_df: pd.DataFrame, *, id_col: str, date_col: str = "LAB_DATE") -> pd.Series:
    """Compute per-patient record span (max - min of `date_col`) in days."""
    g = labs_df.groupby(id_col)[date_col]
    span = (g.max() - g.min()).dt.days.astype("Float64")
    span.name = "record_span_days"
    return span
