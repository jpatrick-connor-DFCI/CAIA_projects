"""Step 2: IHC calls for prostate OncoPanel specimens.

OncoPanel specimens with CANCER_TYPE == 'Prostate Cancer' (parquet/CDM_SPEC_CON_ids.parquet) are matched to the
pathology report of the sequenced block, then to the IHC calls built by ``extract_prostate_ihc`` (intermediate/).
Outputs (``results_dir``):
  oncopanel_prostate_specimen_ihc.csv  one row per sequenced specimen: match status, NE / AR-axis flags, call per marker
  oncopanel_prostate_ihc_long.csv      one row per stain mention on the matched pathology specimens, with source sentence

The original step-2 script (``prostate_ihc/2_oncopanel_ihc.py``), logic unchanged, wrapped in ``run()``.

Run: ``python -m COMPASS.prostate_ihc.oncopanel_ihc`` from the repo root, after step 1.
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from COMPASS.prostate_ihc import paths as ihc_paths

NE = ['Synaptophysin', 'Chromogranin', 'INSM1', 'CD56']
ARX = ['AR', 'PSA', 'NKX3.1', 'PSAP', 'PSMA', 'Prostein/P501S']
FLAGS = ['NE_stained', 'NE_any_pos', 'ARaxis_any_pos', 'ARaxis_any_neg', 'NE_histology']
MARKER_ORDER = ['AR', 'PSA', 'NKX3.1', 'PSAP', 'PSMA', 'Prostein/P501S', 'ERG', 'Synaptophysin', 'Chromogranin', 'INSM1',
                'CD56', 'ASCL1', 'NEUROD1', 'DLL3', 'SSTR2', 'TTF-1', 'YAP1', 'SOX2', 'RB1', 'p53', 'PTEN', 'p63',
                'HMWCK (34bE12)', 'CK5/6', 'PIN4 cocktail', 'AMACR', 'GATA3', 'CK7', 'CK20']
LONG_COLUMNS = ['MRN_INT', 'SAMPLE_ID', 'UNIQUE_SAMPLE_ID', 'ACCESSION', 'PART', 'specimen_date', 'organ', 'histology',
                'group', 'marker_norm', 'result', 'qualifier', 'pct', 'target', 'source', 'RAW_TEXT']


def key(s):
    """Normalise an accession for matching: BS-13-23027-C12, BS-23-W46905-A1, BWHPowerpath_BS1323027 -> BS13-23027.
    The letter after the year is dropped (newer pathology reports omit it); patient is also required to match."""
    if s is None:
        return None
    s = s.upper().replace('BWHPOWERPATH_', '')
    m = re.search(r'\b([A-Z]{1,3})-?(\d{2})-?([A-Z]?)-?0*(\d{1,6})', s)
    return f'{m.group(1)}{m.group(2)}-{m.group(4)}' if m else None


def run(parquet_dir=None, pathology_dir=None, intermediate_dir=None, results_dir=None, *, write: bool = True):
    """Build the two OncoPanel tables; returns ``(specimen, long)`` and writes them as CSV to ``results_dir``."""
    d = ihc_paths.default_paths()
    parquet_dir = Path(parquet_dir or d.parquet_dir)
    pathology_dir = Path(pathology_dir or d.pathology_dir)
    intermediate_dir = Path(intermediate_dir or d.intermediate_dir)
    results_dir = Path(results_dir or d.results_dir)

    g = (pl.read_parquet(parquet_dir / 'CDM_SPEC_CON_ids.parquet').filter(pl.col('CANCER_TYPE') == 'Prostate Cancer')
         .with_columns(pl.col('BLOCK_ACCESSION_NBR').map_elements(key, return_dtype=pl.Utf8).alias('k')))
    S = (pl.scan_parquet(pathology_dir / 'samples.parquet').filter(pl.col('LATEST_VERSION'))
         .select('MRN_INT', 'ACCESSION', 'OUTSIDE_SPECIMEN_ID').unique().collect())
    S = pl.concat([S.with_columns(pl.col('ACCESSION').map_elements(key, return_dtype=pl.Utf8).alias('k')),
                   S.with_columns(pl.col('OUTSIDE_SPECIMEN_ID').map_elements(key, return_dtype=pl.Utf8).alias('k'))]).drop_nulls('k')
    m = g.join(S.select('MRN_INT', 'k', 'ACCESSION').unique(), on=['MRN_INT', 'k'], how='left')
    pairs = m.select('UNIQUE_SAMPLE_ID', 'ACCESSION').drop_nulls().unique()

    # ---- stain-level rows for the matched pathology specimens ----
    long = (pl.read_parquet(intermediate_dir / ihc_paths.LONG_FILE).join(pairs, on='ACCESSION')
            .join(g.select('UNIQUE_SAMPLE_ID', 'SAMPLE_ID'), on='UNIQUE_SAMPLE_ID'))
    long_out = long.select(*LONG_COLUMNS).sort('MRN_INT', 'SAMPLE_ID', 'group', 'marker_norm')

    # ---- one call per sequenced specimen x marker (tumour / lesional stains, all parts of the matched report) ----
    les = long.filter(pl.col('target') == 'lesional')
    call = (les.group_by('UNIQUE_SAMPLE_ID', 'marker_norm')
            .agg(pl.col('result').is_in(['positive', 'retained']).any().alias('p'), pl.col('result').is_in(['negative', 'lost']).any().alias('n'))
            .with_columns(pl.when(pl.col('p') & pl.col('n')).then(pl.lit('discordant')).when('p').then(pl.lit('positive'))
                          .when('n').then(pl.lit('negative')).otherwise(pl.lit('equivocal')).alias('call')))
    call = call.with_columns(pl.when(pl.col('marker_norm').is_in(['RB1', 'PTEN']))
                             .then(pl.col('call').replace({'positive': 'retained', 'negative': 'lost'})).otherwise('call'))
    wide = call.pivot(on='marker_norm', index='UNIQUE_SAMPLE_ID', values='call')
    ki = les.filter(pl.col('marker_norm') == 'Ki-67').group_by('UNIQUE_SAMPLE_ID').agg(pl.col('pct').max().alias('Ki67_pct'))

    has = lambda cols: [c for c in cols if c in wide.columns]  # noqa: E731

    def anyv(cols, v):
        cols = has(cols)
        return pl.any_horizontal([pl.col(c) == v for c in cols]).fill_null(False) if cols else pl.lit(False)

    ne_cols = has(NE)
    ne_stained = pl.any_horizontal([pl.col(c).is_not_null() for c in ne_cols]) if ne_cols else pl.lit(False)
    wide = wide.with_columns(ne_stained.alias('NE_stained'),
                             anyv(NE, 'positive').alias('NE_any_pos'), anyv(ARX, 'positive').alias('ARaxis_any_pos'),
                             anyv(ARX, 'negative').alias('ARaxis_any_neg'))

    ctx = (pl.scan_parquet(pathology_dir / 'samples.parquet').filter(pl.col('LATEST_VERSION'))
           .select('ACCESSION', 'organ', 'histology', pl.col('RAW_TEXT').alias('diagnosis_text')).collect()
           .join(pairs, on='ACCESSION'))
    nehist = ctx.group_by('UNIQUE_SAMPLE_ID').agg(
        (pl.col('histology').fill_null('').str.contains('(?i)small cell|neuroendocrine')
         | pl.col('diagnosis_text').fill_null('').str.contains('(?i)small cell carcinoma|neuroendocrine carcinoma|neuroendocrine differentiation')).any().alias('NE_histology'),
        pl.col('organ').drop_nulls().unique().sort().str.join('; ').alias('path_organs'),
        pl.col('histology').drop_nulls().unique().sort().str.join('; ').alias('path_histologies'))

    base = (m.group_by('UNIQUE_SAMPLE_ID').agg(
                pl.col('MRN_INT').first(), pl.col('SAMPLE_ID').first(), pl.col('BLOCK_ACCESSION_NBR').first(),
                pl.col('PRIMARY_CANCER_DIAGNOSIS').first(), pl.col('ORIGINAL_PATH_DIAGNOSIS').first(),
                pl.col('SAMPLE_COLLECTION_DT').first().str.slice(0, 10).alias('collection_date'),
                pl.col('ACCESSION').drop_nulls().unique().sort().str.join('; ').alias('path_accession'))
            .with_columns((pl.col('path_accession') != '').alias('path_matched')))
    out = (base.join(nehist, on='UNIQUE_SAMPLE_ID', how='left').join(wide, on='UNIQUE_SAMPLE_ID', how='left')
           .join(ki, on='UNIQUE_SAMPLE_ID', how='left'))
    out = out.with_columns([pl.col(c).fill_null(False) for c in FLAGS])
    out = out.with_columns(
        (pl.col('NE_any_pos') & ~pl.col('ARaxis_any_pos') & pl.col('ARaxis_any_neg')).alias('NEPC_ihc'),
        (pl.col('NE_any_pos') & pl.col('ARaxis_any_pos')).alias('amphicrine_ihc'),
        pl.col('ORIGINAL_PATH_DIAGNOSIS').fill_null('').str.contains(r'(?i)small cell|neuroendo|NEPC|\bNET\b|PRSCC').alias('NE_label_oncopanel'))
    out = out.with_columns(
        pl.when(~pl.col('path_matched')).then(pl.lit('no pathology report matched'))
        .when(pl.col('NEPC_ihc')).then(pl.lit('NE+, AR-axis negative (NEPC-like)'))
        .when(pl.col('amphicrine_ihc')).then(pl.lit('NE+, AR-axis positive (amphicrine)'))
        .when(pl.col('NE_any_pos')).then(pl.lit('NE+, AR axis not stained'))
        .when(pl.col('NE_stained')).then(pl.lit('NE stained, negative'))
        .otherwise(pl.lit('no NE stain')).alias('ihc_phenotype'))
    order = has(MARKER_ORDER)
    out = out.select('MRN_INT', 'SAMPLE_ID', 'UNIQUE_SAMPLE_ID', 'collection_date', 'BLOCK_ACCESSION_NBR', 'path_accession', 'path_matched',
                     'PRIMARY_CANCER_DIAGNOSIS', 'ORIGINAL_PATH_DIAGNOSIS', 'NE_label_oncopanel', 'path_organs', 'path_histologies',
                     'ihc_phenotype', 'NEPC_ihc', 'amphicrine_ihc', *FLAGS, 'Ki67_pct', *order).sort('MRN_INT', 'collection_date')
    if write:
        results_dir.mkdir(parents=True, exist_ok=True)
        long_out.write_csv(results_dir / ihc_paths.ONCOPANEL_LONG_FILE)
        out.write_csv(results_dir / ihc_paths.ONCOPANEL_SPECIMEN_FILE)
    return out, long_out


def load(results_dir=None):
    """Read the step-2 CSVs already written to ``results_dir``: ``(specimen, long)``."""
    d = Path(results_dir or ihc_paths.default_paths().results_dir)
    missing = [f for f in ihc_paths.RESULT_FILES if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing step-2 outputs in {d}: {missing}. Run oncopanel_ihc.run().")
    spec = pl.read_csv(d / ihc_paths.ONCOPANEL_SPECIMEN_FILE, infer_schema_length=None)
    long = pl.read_csv(d / ihc_paths.ONCOPANEL_LONG_FILE, infer_schema_length=None)
    return spec, long


def summary_tables(out: pl.DataFrame) -> dict:
    """The README counts: specimens / patients per step and flag, phenotype mix, label agreement."""
    def n(df, label):
        return {'step': label, 'specimens': df.height, 'patients': df['MRN_INT'].n_unique()}
    p = out.filter('path_matched')
    rows = [n(out, 'OncoPanel prostate'), n(p, 'matched to pathology report')]
    for lab, f in [('any NE stain', 'NE_stained'), ('NE positive', 'NE_any_pos'), ('NEPC-like', 'NEPC_ihc'),
                   ('amphicrine', 'amphicrine_ihc'), ('NE histology', 'NE_histology'),
                   ('OncoPanel NE label', 'NE_label_oncopanel')]:
        rows.append(n(p.filter(f), f'  {lab}'))
    rows.append(n(p.filter(pl.col('NE_any_pos') | pl.col('NE_histology')), '  NE+ or NE histology'))
    return {
        'counts': pl.DataFrame(rows),
        'phenotype': p.group_by('ihc_phenotype').len().sort('len', descending=True),
        'label_agreement': p.group_by('NE_label_oncopanel').agg(pl.len(), pl.col('NE_any_pos').sum(), pl.col('NEPC_ihc').sum()),
    }


def main():
    out, long = run()
    for name, table in summary_tables(out).items():
        print(f'--- {name}'); print(table)
    print('stain rows:', long.height)


if __name__ == '__main__':
    main()
