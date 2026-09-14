"""Small label bridge for 05_figures.Rmd; never reads longitudinal lab data.

Reuse the pipeline's medication-derived ADT intent labels, and compare them
with the metastatic-diagnosis LLM task and dated regex stages. Stage I–III is
local; stage IV is metastatic. Missing/invalid values remain unclassified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ID = "DFCI_MRN"
LOCAL = "Local"
MET = "Metastatic"


def normalize_id(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col(ID).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    ).drop_nulls(ID)


def collapse_stage(column: str) -> pl.Expr:
    stage = pl.col(column).cast(pl.String).str.strip_chars().str.to_uppercase()
    return (
        pl.when(stage.is_in(["1", "2", "3", "1.0", "2.0", "3.0", "I", "II", "III"]))
        .then(pl.lit(LOCAL))
        .when(stage.is_in(["4", "4.0", "IV"]))
        .then(pl.lit(MET))
        .otherwise(None)
    )


def build_labels(intent: pl.DataFrame, notes: pl.DataFrame,
                 llm: pl.DataFrame) -> pl.DataFrame:
    intent = normalize_id(intent).with_columns(
        pl.col("ADT_FIRST_DATE").cast(pl.String).str.to_datetime(strict=False),
        pl.col("ADT_INTENT").replace_strict(
            {"LOCALIZED_ADJUVANT": LOCAL, "METASTATIC": MET}, default=None
        ).alias("ADT_LABEL"),
    )
    if intent[ID].n_unique() != intent.height:
        raise ValueError("ADT intent input must contain exactly one row per patient")

    # The input is the metastatic-diagnosis task, not the NEPC subtype task.
    verdict = pl.col("has_metastatic_disease").cast(pl.String).str.to_lowercase()
    llm = normalize_id(llm).with_columns(
        pl.when(verdict.is_in(["true", "1", "1.0"])).then(pl.lit(MET))
        .when(verdict.is_in(["false", "0", "0.0"])).then(pl.lit(LOCAL))
        .otherwise(None).alias("LLM_LABEL")
    ).group_by(ID).agg(
        # Match compass_pipeline: any positive wins, missing stays missing.
        pl.when((pl.col("LLM_LABEL") == MET).any()).then(pl.lit(MET))
        .when((pl.col("LLM_LABEL") == LOCAL).any()).then(pl.lit(LOCAL))
        .otherwise(None).alias("LLM_LABEL")
    )
    stage_text = pl.col("DERIVED_STAGE_MERGED").cast(pl.String).str.strip_chars().str.to_uppercase()
    notes = normalize_id(notes).select(
        ID,
        pl.col("EVENT_DATE").cast(pl.String).str.to_datetime(strict=False).alias("stage_date"),
        stage_text.replace_strict(
            {"1": 1, "2": 2, "3": 3, "4": 4,
             "I": 1, "II": 2, "III": 3, "IV": 4,
             "1.0": 1, "2.0": 2, "3.0": 3, "4.0": 4}, default=None,
        ).alias("stage"),
    ).drop_nulls(["stage_date", "stage"])
    joined = notes.join(intent.select(ID, "ADT_FIRST_DATE"), on=ID).with_columns(
        (pl.col("stage_date") - pl.col("ADT_FIRST_DATE")).dt.total_days().alias("days")
    )
    before = joined.filter(pl.col("days") <= 0)
    nearest = before.filter(pl.col("days") >= -365).sort(
        ["days", "stage"], descending=[True, True]
    ).group_by(ID).agg(pl.col("stage").first().alias("REGEX_STAGE"))
    result = intent.join(llm, on=ID, how="left").join(nearest, on=ID, how="left")
    for suffix, frame in [("BEFORE", before), ("AFTER", joined.filter(pl.col("days") > 0))]:
        result = result.join(frame.group_by(ID).agg(
            pl.col("stage").max().alias(f"REGEX_MAX_{suffix}_STAGE")
        ), on=ID, how="left")
    return result.with_columns(
        collapse_stage("REGEX_STAGE").alias("REGEX_LABEL"),
        collapse_stage("REGEX_MAX_BEFORE_STAGE").alias("REGEX_MAX_BEFORE"),
        collapse_stage("REGEX_MAX_AFTER_STAGE").alias("REGEX_MAX_AFTER"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--icd")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    intent = pl.read_csv(args.intent, infer_schema_length=0)
    notes = pl.read_parquet(args.stage, columns=[ID, "EVENT_DATE", "DERIVED_STAGE_MERGED"])
    llm = pl.read_parquet(args.llm, columns=[ID, "has_metastatic_disease"])
    labels = build_labels(intent, notes, llm)
    if args.icd and Path(args.icd).exists():
        from COMPASS.data_preprocessing.validate_adt_intent import compute_met_burden_at_adt
        icds = pl.read_csv(args.icd, infer_schema_length=0,
                           columns=[ID, "DIAGNOSIS_ICD10_CD", "START_DT"])
        burden = compute_met_burden_at_adt(icds, labels)
        labels = labels.join(burden, on=ID, how="left")
    # Only the fields used by R cross the bridge. R restricts to the canonical
    # time-zero analysis cohort before computing any summaries.
    columns = [ID, "ADT_FIRST_DATE", "ADT_LABEL", "LLM_LABEL", "REGEX_LABEL",
               "REGEX_MAX_BEFORE", "REGEX_MAX_AFTER"]
    columns += [c for c in labels.columns if c == "N_MET_SITES" or c.startswith("MET_SITE_")]
    labels.select(columns).write_csv(args.output)


if __name__ == "__main__":
    main()
