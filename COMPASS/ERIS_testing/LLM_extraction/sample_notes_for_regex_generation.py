"""
Samples a diverse subset of clinical notes per NOTE_TYPE for sending to the
enterprise GPT to generate regex-based cleaning rules.

Sampling strategy:
  - Up to N notes per NOTE_TYPE (default 15)
  - Stratified by text length (short/medium/long terciles) to capture boilerplate
    variation across note sizes
  - One note per patient to maximize diversity of authorship/formatting
  - Outputs a TSV with NOTE_TYPE, EVENT_DATE, text length, and CLINICAL_TEXT
"""

import os
import pandas as pd
import numpy as np

DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'
OUTPUT_PATH = os.path.join(DATA_PATH, 'regex_generation_note_samples.tsv')

NOTES_PER_TYPE = 15  # how many notes to sample per NOTE_TYPE
RANDOM_SEED = 42

# Load full text data (has NOTE_TYPE)
text_df = pd.read_csv(os.path.join(DATA_PATH, 'prostate_text_data.csv'))
text_df['TEXT_LENGTH'] = text_df['CLINICAL_TEXT'].astype(str).apply(len)

# Assign length tercile within each NOTE_TYPE
text_df['LENGTH_BIN'] = text_df.groupby('NOTE_TYPE')['TEXT_LENGTH'].transform(
    lambda x: pd.qcut(x, q=3, labels=['short', 'medium', 'long'], duplicates='drop')
)

sampled_frames = []
for note_type in text_df['NOTE_TYPE'].unique():
    type_df = text_df.loc[text_df['NOTE_TYPE'] == note_type]

    # One note per patient for diversity
    deduped = type_df.drop_duplicates(subset='DFCI_MRN')

    # Stratified sample across length bins
    per_bin = max(1, NOTES_PER_TYPE // deduped['LENGTH_BIN'].nunique())
    sampled = (deduped
               .groupby('LENGTH_BIN', group_keys=False, observed=True)
               .apply(lambda g: g.sample(n=min(per_bin, len(g)), random_state=RANDOM_SEED))
               .head(NOTES_PER_TYPE))
    sampled_frames.append(sampled)

sample_df = pd.concat(sampled_frames, ignore_index=True)
sample_df = sample_df[['NOTE_TYPE', 'EVENT_DATE', 'TEXT_LENGTH', 'LENGTH_BIN', 'CLINICAL_TEXT']]
sample_df = sample_df.sort_values(by=['NOTE_TYPE', 'TEXT_LENGTH'])

sample_df.to_csv(OUTPUT_PATH, sep='\t', index=False)

print(f'Sampled {len(sample_df)} notes across {text_df["NOTE_TYPE"].nunique()} NOTE_TYPEs')
for nt in sample_df['NOTE_TYPE'].unique():
    n = (sample_df['NOTE_TYPE'] == nt).sum()
    print(f'  {nt}: {n} notes')
print(f'Saved to {OUTPUT_PATH}')
