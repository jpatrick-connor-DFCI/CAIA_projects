"""IHC calls for prostate cancer patients (for the neuroendocrine prostate cancer project).

Cohort  = registry prostate primary (IDM_CARG_DIAGNOSIS SITE_CD C61*)  OR  a pathology specimen part with
          organ == prostate, category == malignant and non-urothelial / non-heme histology,
          OR an OncoPanel specimen with CANCER_TYPE == 'Prostate Cancer'.
Markers = prostate lineage / AR axis, neuroendocrine, tumour suppressor, basal-cell and proliferation stains,
          taken from pathology_extracts/ihc.parquet (rule-based extraction), plus a supplementary re-parse of the
          same source lines for AMACR / PSMA / PTEN / spelled-out PSA, which the extractor does not name.
Step 1 of 2. Intermediate outputs (``intermediate_dir``), read by ``oncopanel_ihc``:
  prostate_ihc_long.parquet          one row per specimen part x marker mention, with source text
  prostate_ihc_specimen_all.parquet  one row per specimen part: call per marker, NE / AR-axis phenotype, prostate_evidence
  prostate_cohort.parquet            cohort patients with source flags and first prostate diagnosis date

Run: ``python -m COMPASS.prostate_ihc.extract_prostate_ihc`` from the repo root (paths from ``paths.default_paths``).
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from COMPASS.prostate_ihc import paths as ihc_paths

# ---------------- marker normalisation ----------------
PANEL = {  # extractor marker name -> (normalised name, group)
    'AR': ('AR', 'AR axis'), 'PSA': ('PSA', 'AR axis'), 'PSAP': ('PSAP', 'AR axis'), 'NKX31': ('NKX3.1', 'AR axis'),
    'NKX3': ('NKX3.1', 'AR axis'), 'P501': ('Prostein/P501S', 'AR axis'), 'PROSTEIN': ('Prostein/P501S', 'AR axis'),
    'ERG': ('ERG', 'AR axis'), 'P504': ('AMACR', 'lineage/other'),
    'SYNAPTOPHYSIN': ('Synaptophysin', 'NE'), 'CHROMOGRANIN': ('Chromogranin', 'NE'), 'INSM1': ('INSM1', 'NE'),
    'CD56': ('CD56', 'NE'), 'ASCL1': ('ASCL1', 'NE'), 'NEUROD1': ('NEUROD1', 'NE'), 'DLL3': ('DLL3', 'NE'),
    'SSTR2': ('SSTR2', 'NE'), 'SSTR2A': ('SSTR2', 'NE'), 'TTF1': ('TTF-1', 'NE'), 'YAP1': ('YAP1', 'NE'),
    'SOX2': ('SOX2', 'NE'), 'KI67': ('Ki-67', 'proliferation'),
    'RB1': ('RB1', 'tumour suppressor'), 'P53': ('p53', 'tumour suppressor'),
    'P63': ('p63', 'basal'), 'CK34B': ('HMWCK (34bE12)', 'basal'), 'CK903': ('HMWCK (34bE12)', 'basal'),
    'CK5/6': ('CK5/6', 'basal'), 'CK5': ('CK5/6', 'basal'), 'CK56': ('CK5/6', 'basal'), 'PIN4': ('PIN4 cocktail', 'basal'),
    'GATA3': ('GATA3', 'lineage/other'), 'CK7': ('CK7', 'lineage/other'), 'CK20': ('CK20', 'lineage/other'),
}
SUPP = {  # re-parsed from RAW_TEXT: regex -> (normalised name, group)
    r'\bAMACR\b|\bracemase\b|\bP504S?\b': ('AMACR', 'lineage/other'),
    r'\bPSMA\b|prostate[- ]specific membrane antigen': ('PSMA', 'AR axis'),
    r'\bPTEN\b': ('PTEN', 'tumour suppressor'),
    r'prostate[- ]specific antigen': ('PSA', 'AR axis'),
}
POS = r'positiv\w*|express(?:es|ed|ing)?\b|retained|intact|reactive'
# Consume the entire negated assay phrase so a later "expression of" cannot
# override it. Limit intervening words to assay modifiers, not other markers.
ASSAY_MODIFIERS = (r'(?:(?:significant|detectable|convincing|appreciable|definite|any|'
                   r'nuclear|cytoplasmic|membranous|protein|tumou?r|cell|strong|weak|focal|diffuse)\s+)*')
NEG = (rf'\b(?:(?:loss|lack|absence)\s+of|no|without)\s+{ASSAY_MODIFIERS}'
       rf'(?:staining|expression)\b(?:\s+of\b)?|negativ\w*|lost|loss|absent|non-?reactive|'
       rf'not\s+{ASSAY_MODIFIERS}(?:express\w*|stain\w*)')
# Before the marker, verbal stain / expression phrases are positive ("strong expression of PSMA", "stained by AMACR");
# after it they only name the assay ("PSMA staining is negative"), so FOLLOW needs a polarity word.
# ``spos`` phrases assert staining but are not result words; ``pos`` / ``neg`` are.
STAIN_PHRASE = r'\bexpression of\b|\bstained\b(?:\s+(?:for|with|by)\b)?|\bstains?\s+(?:positively\s+)?(?:for|with)\b'
POS_BEFORE = rf'{POS}|{STAIN_PHRASE}'
POL = re.compile(rf'(?P<neg>{NEG})|(?P<pos>{POS})|(?P<spos>{STAIN_PHRASE})', re.I)
# Between a result word and its own marker only assay wording may intervene ("positive staining for AMACR").
OWN_RESULT_GAP = re.compile(r'^\s*(?:(?:immuno)?(?:staining|stains?|stained|expression)\s+)?'
                            r'(?:(?:for|of|with|by)\s+)?(?:the\s+)?$', re.I)
# "Staining" and "expression" name the assay, not its polarity.
FOLLOW = re.compile(
    rf'^\s*(?:\([^)]*\))?\s*[:\-–(]?\s*'
    rf'(?:(?:stains?|stained|staining|expression)\b\s*)?'
    rf'(?:(?:is|are|was|were|shows?)\b\s*)?'
    rf'(?:(?P<neg>{NEG})|(?P<pos>{POS}))'
    # Accept "stains" as a verb only with an observed cellular/tissue object.
    rf'|^\s*(?P<vpos>stains\s+(?:(?:the|these|both|and|benign|malignant|tumou?r|'
    rf'neoplastic|lesional|epithelial|strongly|weakly|diffusely|focally)\s+)*'
    rf'(?:cells|glands|tissue)\b)', re.I)
# Optionally skip a coordinated marker list first ("PSMA and AMACR were ordered", "PSMA, AMACR, results to follow").
MARKER_LIST = r'(?:\s*(?:,|/|&|\band\b|\bor\b)\s*(?:(?:and|or)\s+)?[\w.+/-]+){0,4}'
UNRESULTED_AFTER = re.compile(
    rf'^{MARKER_LIST}\s*(?:(?:stains?|staining|expression|immunostains?)\s*)?'
    r'(?:(?:(?:is|are|was|were|has|have|had|been|being)\s+)*'
    r'(?:will|would|may|might|can|could|should|pending|planned|ordered|requested|recommended|performed|done)\b'
    r'|[,:;(]?\s*(?:results?\s+)?(?:to\s+follow|pending|forthcoming)\b)', re.I)
UNRESULTED_BEFORE = re.compile(
    r'\b(?:pending|planned|order(?:ed)?|request(?:ed)?|recommend(?:ed)?)\s+'
    r'(?:(?:an?|the|additional)\s+)?(?:(?:stains?|staining)\s+for\s+)?$', re.I)

NONPROS = 'urothelial|lymphoma|leukemia|myelo|ductal carcinoma|melanoma|squamous'
KEYS = ['MRN_INT', 'DFCI_MRN', 'ACCESSION', 'PART', 'SPEC_DATE', 'OUTSIDE_SPECIMEN_DATE', 'RPT_DATE', 'DOC_ID', 'method', 'target']
POSV = ['positive', 'retained']
NE = ['Synaptophysin', 'Chromogranin', 'INSM1', 'CD56']
ARX = ['AR', 'PSA', 'NKX3.1', 'PSAP', 'PSMA', 'Prostein/P501S']
# Is this specimen prostate cancer? Patients here often have other cancers (lung, renal, bladder ...) whose work-ups
# include negative prostate stains, so a stain being done is not evidence. Evidence, in priority order (see run()).
LINEAGE = ['NKX3.1', 'PSA', 'PSAP', 'PSMA', 'Prostein/P501S']        # AR alone is not prostate-specific
OTHER_HIST = 'renal cell|urothelial|squamous|lymphoma|leukemia|myelo|melanoma|ductal carcinoma|lobular|hepatocellular|sarcoma|glioma|mesothelioma|thyroid|germ cell|seminoma|plasma cell|basal cell|neuroendocrine tumor'
MARKER_ORDER = ['AR', 'PSA', 'NKX3.1', 'PSAP', 'PSMA', 'Prostein/P501S', 'ERG', 'Synaptophysin', 'Chromogranin', 'INSM1',
                'CD56', 'ASCL1', 'NEUROD1', 'DLL3', 'SSTR2', 'TTF-1', 'YAP1', 'SOX2', 'RB1', 'p53', 'PTEN', 'p63',
                'HMWCK (34bE12)', 'CK5/6', 'PIN4 cocktail', 'AMACR', 'GATA3', 'CK7', 'CK20']


def polarity(sentence, start, end):
    """Result for a marker mention at [start, end): a polarity word right after it, else the last one before it."""
    if UNRESULTED_BEFORE.search(sentence[:start]):
        return None
    m = FOLLOW.match(sentence[end:end + 40])
    if m:
        return 'negative' if m.group('neg') else 'positive'
    prev = [p for p in POL.finditer(sentence[:start])]
    last = prev[-1] if prev else None
    # Do not borrow another marker's earlier result for an unresulted assay; a result word that belongs to this
    # marker ("positive staining for AMACR was performed") still counts.
    if UNRESULTED_AFTER.match(sentence[end:]):
        own = (last is not None and (last.group('neg') or last.group('pos'))
               and OWN_RESULT_GAP.match(sentence[last.end():start]))
        if not own:
            return None
    if last is not None:
        return 'negative' if last.group('neg') else 'positive'
    return None


TISSUE_TARGET = re.compile(
    r'\b(?P<background>benign|normal|background|stromal|non[- ]neoplastic)\b|'
    r'\b(?P<lesional>tumou?r|malignant|neoplastic|lesional|carcinoma)\b', re.I)
CLAUSE_BREAK = re.compile(r'[;,]|\b(?:but|whereas|while)\b', re.I)


def supplement_target(sentence, start, end, *, text, fallback):
    """Use the mention's clause; mixed or nonlocal tissue evidence is unknown.

    Only inherit the extractor target when the source text has no explicit tissue
    cues. In particular, a benign stain must not inherit another marker's lesional target.
    """
    left = list(CLAUSE_BREAK.finditer(sentence[:start]))
    right = CLAUSE_BREAK.search(sentence, end)
    clause = sentence[left[-1].end() if left else 0:right.start() if right else len(sentence)]
    targets = {m.lastgroup for m in TISSUE_TARGET.finditer(clause)}
    if len(targets) == 1:
        return targets.pop()
    if targets or TISSUE_TARGET.search(text):
        return 'unknown'
    return fallback


NE_MENTION = re.compile(r'(?<![\w-])(?:small[- ]cell|neuroendocrine)\b', re.I)
NE_DIAGNOSIS = re.compile(
    r'(?<![\w-])(?:small[- ]cell carcinoma|neuroendocrine (?:carcinoma|differentiation))\b', re.I)
NE_NEGATED_BEFORE = re.compile(
    r'\b(?:no|not|without|negative\s+for|absence\s+of|absent|rule[ds]?\s+out|'
    r'non[- ]|unlikely|possible|suspected|cannot\s+exclude)\b[^.;]{0,70}$', re.I)
NE_NEGATED_AFTER = re.compile(
    r'^\s*(?:(?:carcinoma|differentiation|features?|component|tumou?r)\s*)?'
    r'(?:(?:is|are|was|were)\s+)?(?:absent|not\s+(?:seen|identified|present)|ruled\s+out)\b', re.I)


def has_ne_histology(histology, diagnosis_text):
    """Affirmed NE histology in either field, with negation scoped to a clause."""
    for text, pattern in ((histology or '', NE_MENTION), (diagnosis_text or '', NE_DIAGNOSIS)):
        for clause in re.split(r'[.;\n]|\b(?:but|however|whereas)\b', text, flags=re.I):
            for mention in pattern.finditer(clause):
                before, after = clause[:mention.start()], clause[mention.end():]
                if re.search(r'\bnon[- ]$', before, re.I):
                    continue
                if not NE_NEGATED_BEFORE.search(before) and not NE_NEGATED_AFTER.search(after):
                    return True
    return False


def ne_histology_expr():
    return pl.struct('histology', 'diagnosis_text').map_elements(
        lambda r: has_ne_histology(r['histology'], r['diagnosis_text']), return_dtype=pl.Boolean)


def supplement(rows):
    out = []
    # A source line can have several extractor mentions with different targets.
    # Do not choose an arbitrary mention's target when no local cue resolves it.
    line_keys = ['MRN_INT', 'ACCESSION', 'DOC_ID', 'PART', 'RAW_TEXT']
    rows = rows.group_by(line_keys, maintain_order=True).agg(
        *[pl.col(k).first() for k in KEYS if k not in line_keys and k != 'target'],
        pl.when(pl.col('target').n_unique() == 1).then(pl.col('target').first())
        .otherwise(pl.lit('unknown')).alias('target'))
    for r in rows.iter_rows(named=True):
        txt = r['RAW_TEXT'] or ''
        for sent in re.split(r'(?<=[.;])\s+|\n', txt):
            for pat, (name, grp) in SUPP.items():
                for m in re.finditer(pat, sent, re.I):
                    res = polarity(sent, m.start(), m.end())
                    if res is None:
                        continue
                    if name == 'PTEN':
                        res = 'retained' if res == 'positive' else 'lost'
                    target = supplement_target(sent, m.start(), m.end(), text=txt, fallback=r['target'])
                    out.append({**{k: r[k] for k in KEYS}, 'target': target,
                                'marker_norm': name, 'group': grp, 'result': res,
                                'qualifier': None, 'pct': None, 'score': None, 'source': 'supplement', 'RAW_TEXT': txt})
    return pl.DataFrame(out, schema_overrides={'pct': pl.Float64}) if out else None


def build_cohort(parquet_dir, S):
    reg = (pl.scan_parquet(f'{parquet_dir}/IDM_CARG_DIAGNOSIS.parquet')
           .filter(pl.col('SITE_CD').str.starts_with('C61'))
           .select('MRN_INT', pl.col('DIAGNOSIS_DT').str.slice(0, 10).alias('dx'), 'HISTOLOGY_DSC')
           .collect())
    reg_pts = reg.group_by('MRN_INT').agg(pl.col('dx').min().alias('registry_dx_date'),
                                          pl.col('HISTOLOGY_DSC').unique().str.join('; ').alias('registry_histology'))
    pros_parts = S.filter((pl.col('organ') == 'prostate') & (pl.col('category') == 'malignant')
                          & ~pl.col('histology').fill_null('').str.contains(NONPROS))
    path_pts = pros_parts.group_by('MRN_INT').agg(
        pl.coalesce('OUTSIDE_SPECIMEN_DATE', 'SPEC_DATE').min().str.slice(0, 10).alias('path_first_prostate_ca'))
    onco_pts = (pl.scan_parquet(f'{parquet_dir}/CDM_SPEC_CON_ids.parquet')
                .filter(pl.col('CANCER_TYPE') == 'Prostate Cancer')
                .select('MRN_INT').unique().collect()
                .with_columns(pl.lit(True).alias('in_oncopanel')))
    return (reg_pts.join(path_pts, on='MRN_INT', how='full', coalesce=True)
            .join(onco_pts, on='MRN_INT', how='full', coalesce=True)
            .with_columns(pl.col('registry_dx_date').is_not_null().alias('in_registry'),
                          pl.col('path_first_prostate_ca').is_not_null().alias('in_pathology'),
                          pl.col('in_oncopanel').fill_null(False)))


def load_samples(pathology_dir):
    return (pl.scan_parquet(f'{pathology_dir}/samples.parquet').filter(pl.col('LATEST_VERSION'))
            .select('MRN_INT', 'DFCI_MRN', 'ACCESSION', 'PART', 'SPEC_DATE', 'OUTSIDE_SPECIMEN_DATE', 'site', 'organ',
                    'procedure', 'category', 'histology', 'metastatic', 'grade_gleason', 'grade_grade_group',
                    pl.col('RAW_TEXT').alias('diagnosis_text'))
            .collect())


def run(pathology_dir=None, parquet_dir=None, out_dir=None, *, write: bool = True) -> dict:
    """Build the step-1 tables; returns {'cohort', 'long', 'spec'} and writes them to ``out_dir``."""
    d = ihc_paths.default_paths()
    pathology_dir = Path(pathology_dir or d.pathology_dir)
    parquet_dir = Path(parquet_dir or d.parquet_dir)
    out_dir = Path(out_dir or d.intermediate_dir)

    # ---------------- cohort ----------------
    S = load_samples(pathology_dir)
    cohort = build_cohort(parquet_dir, S)

    # ---------------- marker rows ----------------
    I = (pl.scan_parquet(f'{pathology_dir}/ihc.parquet').filter(pl.col('LATEST_VERSION') & (pl.col('method') == 'ihc'))
         .join(cohort.select('MRN_INT').lazy(), on='MRN_INT').collect())
    mp = pl.DataFrame({'marker': list(PANEL), 'marker_norm': [v[0] for v in PANEL.values()], 'group': [v[1] for v in PANEL.values()]})
    main = (I.join(mp, on='marker')
            .with_columns(pl.lit('extractor').alias('source'))
            .select(*KEYS, 'marker_norm', 'group', 'result', 'qualifier', 'pct', 'score', 'source', 'RAW_TEXT'))
    mention_keys = ['MRN_INT', 'ACCESSION', 'DOC_ID', 'PART', 'RAW_TEXT']
    lines = I.select(*KEYS, 'RAW_TEXT').unique(subset=[*mention_keys, 'target'])
    sup = supplement(lines)
    if sup is not None:
        # extractor rows win where the same line already yielded that marker
        sup = sup.join(main.select(*mention_keys, 'marker_norm').unique(),
                       on=[*mention_keys, 'marker_norm'], how='anti', nulls_equal=True)
        main = pl.concat([main, sup.select(main.columns).cast(main.schema)])
    long = main.unique(subset=[*mention_keys, 'marker_norm', 'result', 'target'], keep='first')

    # Accession and part are only matched within the same patient.
    specimen_keys = ['MRN_INT', 'ACCESSION', 'PART']
    ctx = S.select(*specimen_keys, 'site', 'organ', 'procedure', 'category', 'histology', 'metastatic',
                   'grade_gleason', 'grade_grade_group', 'diagnosis_text').unique(subset=specimen_keys)
    long = (long.join(ctx, on=specimen_keys, how='left', nulls_equal=True)
            .join(cohort.select('MRN_INT', 'registry_dx_date', 'path_first_prostate_ca'), on='MRN_INT', how='left')
            .with_columns(pl.coalesce('OUTSIDE_SPECIMEN_DATE', 'SPEC_DATE').str.slice(0, 10).alias('specimen_date'))
            .sort('MRN_INT', 'specimen_date', 'ACCESSION', 'PART', 'group', 'marker_norm'))

    # ---------------- specimen-level summary ----------------
    les = long.filter(pl.col('target') == 'lesional')
    call = (les.group_by('MRN_INT', 'ACCESSION', 'PART', 'marker_norm')
            .agg(pl.col('result').is_in(POSV).any().alias('p'), pl.col('result').is_in(['negative', 'lost']).any().alias('n'),
                 pl.col('pct').max().alias('pct'))
            .with_columns(pl.when(pl.col('p') & pl.col('n')).then(pl.lit('discordant'))
                          .when(pl.col('p')).then(pl.lit('positive')).when(pl.col('n')).then(pl.lit('negative'))
                          .otherwise(pl.lit('equivocal')).alias('call')))
    call = call.with_columns(pl.when(pl.col('marker_norm').is_in(['RB1', 'PTEN']))
                             .then(pl.col('call').replace({'positive': 'retained', 'negative': 'lost'})).otherwise('call'))
    wide = call.pivot(on='marker_norm', index=['MRN_INT', 'ACCESSION', 'PART'], values='call')
    ki = call.filter(pl.col('marker_norm') == 'Ki-67').select('MRN_INT', 'ACCESSION', 'PART', pl.col('pct').alias('Ki67_pct'))

    def anyval(cols, v):
        cols = [c for c in cols if c in wide.columns]
        # Discordant retains both positive and negative evidence.
        return pl.any_horizontal([pl.col(c).is_in([v, 'discordant']) for c in cols]).fill_null(False) if cols else pl.lit(False)

    spec = (wide.join(ki, on=['MRN_INT', 'ACCESSION', 'PART'], how='left', nulls_equal=True)
            .with_columns(anyval(NE, 'positive').alias('NE_any_pos'), anyval(NE, 'negative').alias('NE_any_neg'),
                          anyval(ARX, 'positive').alias('ARaxis_any_pos'), anyval(ARX, 'negative').alias('ARaxis_any_neg'))
            .join(ctx, on=specimen_keys, how='left', nulls_equal=True))
    nehist = ne_histology_expr()
    spec = spec.with_columns(
        nehist.alias('NE_histology'),
        pl.when(pl.col('NE_any_pos') & ~pl.col('ARaxis_any_pos') & pl.col('ARaxis_any_neg')).then(pl.lit('NE+, AR-axis negative (NEPC-like)'))
        .when(pl.col('NE_any_pos') & pl.col('ARaxis_any_pos')).then(pl.lit('NE+, AR-axis positive (amphicrine / focal NE)'))
        .when(pl.col('NE_any_pos')).then(pl.lit('NE+, AR axis not stained'))
        .when(pl.col('NE_any_neg') & pl.col('ARaxis_any_neg') & ~pl.col('ARaxis_any_pos')).then(pl.lit('NE-, AR-axis negative (double negative)'))
        .when(pl.col('ARaxis_any_pos')).then(pl.lit('AR-axis positive, NE negative/not stained'))
        .when(pl.col('NE_any_neg')).then(pl.lit('NE negative, AR axis not stained'))
        .otherwise(pl.lit('no NE or AR-axis stain')).alias('ihc_phenotype'))
    dx = pl.col('diagnosis_text').fill_null('')
    stated = dx.str.contains(r'(?i)prostat\w*\s+(?:origin|primary|adeno|carcinoma|acinar|ductal|cancer)'
                             r'|(?:consistent|compatible) with[^.;]{0,60}prostat|support\w*[^.;]{0,60}prostat|favou?r\w*[^.;]{0,60}prostat')
    negated = dx.str.contains(r'(?i)(?:\bnot\b|unlikely|against|rather than|exclude\w*|negative for)[^.;]{0,50}prostat')
    other = (pl.col('histology').fill_null('').str.contains('(?i)' + OTHER_HIST)
             | dx.str.contains(r'(?i)(?:consistent|compatible) with[^.;]{0,50}(?:lung|pulmonary|renal|kidney|urothelial|bladder|colo|rectal|breast|pancrea|melanoma|thyroid|gastric|esophag|ovar|endometri|head and neck|tonsil|hepat|cholangio|merkel)'
                              r'|\b(?:lung|pulmonary|renal cell|urothelial|colorectal|breast|pancreatic) (?:adeno)?carcinoma|non-?small cell|\bNSCLC\b|well[- ]differentiated (?:\w+ )?neuroendocrine tumou?r|carcinoid|merkel cell'))
    spec = spec.with_columns(anyval(LINEAGE, 'positive').alias('prostate_lineage_pos'))
    spec = spec.with_columns(
        pl.when(pl.col('prostate_lineage_pos')).then(pl.lit('1 prostate lineage IHC positive'))
        .when((pl.col('organ') == 'prostate') & ~other).then(pl.lit('2 prostate site'))
        .when(stated & ~negated).then(pl.lit('3 pathologist states prostate origin'))
        .when(other).then(pl.lit('other primary'))
        .when((pl.col('NE_any_pos') | nehist) & (pl.col('category').fill_null('malignant') == 'malignant')).then(pl.lit('4 NE tumour, origin not stated (review)'))
        .otherwise(pl.lit('unrelated / unclear')).alias('prostate_evidence'))
    spec = (spec.with_columns(pl.col('prostate_evidence').str.contains('^[1-4]').alias('prostate_relevant'))
            .join(S.select(*specimen_keys, pl.coalesce('OUTSIDE_SPECIMEN_DATE', 'SPEC_DATE').str.slice(0, 10).alias('specimen_date'))
                  .unique(subset=specimen_keys), on=specimen_keys, how='left', nulls_equal=True)
            .join(cohort.select('MRN_INT', 'registry_dx_date', 'path_first_prostate_ca'), on='MRN_INT', how='left'))
    first = ['MRN_INT', 'ACCESSION', 'PART', 'specimen_date', 'registry_dx_date', 'path_first_prostate_ca', 'site', 'organ',
             'procedure', 'category', 'histology', 'metastatic', 'grade_grade_group', 'prostate_evidence', 'prostate_relevant', 'NE_histology',
             'ihc_phenotype', 'NE_any_pos', 'ARaxis_any_pos', 'Ki67_pct']
    order = [c for c in MARKER_ORDER if c in spec.columns]
    spec = spec.select(first + order + ['diagnosis_text']).sort('MRN_INT', 'specimen_date', 'ACCESSION', 'PART')

    cohort = cohort.with_columns(pl.col('MRN_INT').is_in(long['MRN_INT'].unique().implode()).alias('has_panel_ihc'))
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        cohort.write_parquet(out_dir / ihc_paths.COHORT_FILE)
        long.write_parquet(out_dir / ihc_paths.LONG_FILE)
        spec.write_parquet(out_dir / ihc_paths.SPECIMEN_ALL_FILE)
    return {'cohort': cohort, 'long': long, 'spec': spec}


def load(intermediate_dir=None) -> dict:
    """Read the step-1 tables already written to ``intermediate_dir``."""
    d = Path(intermediate_dir or ihc_paths.default_paths().intermediate_dir)
    missing = [f for f in ihc_paths.INTERMEDIATE_FILES if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing step-1 outputs in {d}: {missing}. Run extract_prostate_ihc.run().")
    return {'cohort': pl.read_parquet(d / ihc_paths.COHORT_FILE), 'long': pl.read_parquet(d / ihc_paths.LONG_FILE),
            'spec': pl.read_parquet(d / ihc_paths.SPECIMEN_ALL_FILE)}


def summary_tables(out: dict) -> dict:
    """The counts the original script printed, as tables."""
    cohort, long, spec = out['cohort'], out['long'], out['spec']
    r = spec.filter('prostate_relevant')
    return {
        'cohort': pl.DataFrame({'cohort': [cohort.height], 'registry': [int(cohort['in_registry'].sum())],
                                'pathology': [int(cohort['in_pathology'].sum())],
                                'oncopanel': [int(cohort['in_oncopanel'].sum())],
                                'with_panel_ihc': [int(cohort['has_panel_ihc'].sum())],
                                'long_rows': [long.height],
                                'supplement_rows': [int((long['source'] == 'supplement').sum())]}),
        'markers': (long.group_by('group', 'marker_norm')
                    .agg(pl.col('MRN_INT').n_unique().alias('patients'), pl.len().alias('rows'))
                    .sort('group', 'patients', descending=[False, True])),
        'evidence': (spec.group_by('prostate_evidence')
                     .agg(pl.len().alias('specimens'), pl.col('MRN_INT').n_unique().alias('patients'))
                     .sort('prostate_evidence')),
        'phenotype_relevant': (r.group_by('ihc_phenotype')
                               .agg(pl.len().alias('specimens'), pl.col('MRN_INT').n_unique().alias('patients'))
                               .sort('specimens', descending=True)),
        'relevant': pl.DataFrame({'prostate_relevant_specimens': [r.height],
                                  'patients': [r['MRN_INT'].n_unique()],
                                  'NE_histology_specimens': [int(r['NE_histology'].sum())],
                                  'NE_histology_patients': [r.filter('NE_histology')['MRN_INT'].n_unique()]}),
    }


def main():
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
    for name, table in summary_tables(run()).items():
        print(f'--- {name}'); print(table)


if __name__ == '__main__':
    main()
