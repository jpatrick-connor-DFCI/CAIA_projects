"""
Shared polars-based fast I/O helpers for cohort-filtering large raw OncDRS
CSVs (labs alone is 7.3M rows).

Both `COMPASS/data_preprocessing/compile_COMPASS_cohort_data.py` and
`IPIO/data_preprocessing/compile_irae_data.py` need to: stream a big raw CSV,
keep only rows whose DFCI_MRN is in a cohort set, optionally project to a
column subset, and recover "dirty" free-text numeric columns (e.g. RESULT_NBR
values like '05055/D') back to real numeric dtypes wherever every filtered,
non-null value in that column actually parses.

`pl.scan_csv(..., infer_schema_length=0)` reads every column as Utf8 (no
per-block dtype sniffing / no raising on a later `'05055/D'`-style row), which
mirrors the pyarrow `column_types={c: pa.string() for c in col_names}` trick
the pandas-based `filter_and_save` implementations used. `.gz` inputs (the
somatic file) are read transparently by polars' scan_csv, same as pyarrow.

This module is intentionally polars-only (no pandas import) and does no
writing itself -- each caller's own `filter_and_save` wrapper adds
`.write_csv()` so output paths/signatures stay stable.
"""

import polars as pl

ID_COL = "DFCI_MRN"


def scan_filter(path, cohort_mrns, cols=None) -> pl.LazyFrame:
    """Lazily scan `path` (all-String schema) and filter to `cohort_mrns`.

    - `cols`, if given, is applied as a `.select()` (DFCI_MRN is always
      included even if the caller forgot it).
    - Cohort filtering is done by casting DFCI_MRN to a numeric type and
      testing membership in `cohort_mrns` (a set/iterable of ints), mirroring
      the pandas `pd.to_numeric(..., errors='coerce').isin(cohort_mrns)` used
      by the original `filter_and_save` implementations.
    """
    lf = pl.scan_csv(path, infer_schema_length=0)

    if cols:
        select_cols = list(cols) if ID_COL in cols else [ID_COL] + list(cols)
        lf = lf.select(select_cols)

    cohort_mrns = set(int(m) for m in cohort_mrns)
    mrn_num = pl.col(ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)
    lf = lf.filter(mrn_num.is_in(list(cohort_mrns)))

    return lf


def recover_numeric(df: pl.DataFrame, exclude=()) -> pl.DataFrame:
    """Cast each column not in `exclude` to Float64 iff every non-null,
    stripped value in that (already-filtered, small) column parses as a
    number. Mirrors the pandas post-filter recovery:
        as_num = pd.to_numeric(filtered[col], errors='coerce')
        if non_null.any() and as_num[non_null].notna().all():
            filtered[col] = as_num

    Empty-string cells are treated as null (mirrors the pandas
    `filtered.mask(filtered == '', np.nan)` step before recovery) so a blank
    CSV cell doesn't block numeric recovery for an otherwise-numeric column.
    ID_COL is left untouched here; callers decide whether to include it in
    `exclude` (COMPASS/IPIO both keep DFCI_MRN as-is downstream).
    """
    exclude = set(exclude)
    exprs = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in exclude:
            continue
        if dtype != pl.Utf8:
            continue

        col = pl.col(name)
        stripped = col.str.strip_chars()
        is_blank = stripped == ""
        non_null = col.is_not_null() & ~is_blank
        as_num = stripped.cast(pl.Float64, strict=False)

        n_non_null = df.select(non_null.sum()).item()
        if n_non_null == 0:
            continue
        n_parses = df.select((non_null & as_num.is_not_null()).sum()).item()
        if n_parses == n_non_null:
            exprs.append(
                pl.when(is_blank).then(None).otherwise(as_num).alias(name)
            )

    if exprs:
        df = df.with_columns(exprs)
    return df
