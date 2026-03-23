"""
Evaluates how much text/tokens the regex cleaning removes from the
LLM candidate text dataset. Reports reduction by NOTE_TYPE and overall.
"""

import os
import tiktoken
import pandas as pd
from tqdm.auto import tqdm

from note_cleaning import clean_note

DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'

# tiktoken encoder for gpt-4o token counting
enc = tiktoken.encoding_for_model('gpt-4o')

def count_tokens(text):
    return len(enc.encode(str(text)))

# Load candidate text (now includes NOTE_TYPE)
df = pd.read_csv(os.path.join(DATA_PATH, 'LLM_candidate_text_data.csv'))
print(f"Loaded {len(df)} notes across {df['NOTE_TYPE'].nunique()} NOTE_TYPEs\n")

tqdm.pandas(desc='Cleaning notes')

# Raw metrics
df['raw_chars'] = df['CLINICAL_TEXT'].fillna('').astype(str).str.len()
df['raw_tokens'] = df['CLINICAL_TEXT'].fillna('').astype(str).progress_apply(count_tokens)

# Clean and measure
df['cleaned_text'] = df.progress_apply(
    lambda row: clean_note(row['CLINICAL_TEXT'], note_type=row.get('NOTE_TYPE')),
    axis=1
)
df['cleaned_chars'] = df['cleaned_text'].str.len()
df['cleaned_tokens'] = df['cleaned_text'].progress_apply(count_tokens)

# Compute reductions
df['char_reduction'] = df['raw_chars'] - df['cleaned_chars']
df['token_reduction'] = df['raw_tokens'] - df['cleaned_tokens']
df['char_pct_reduction'] = (df['char_reduction'] / df['raw_chars'].clip(lower=1)) * 100
df['token_pct_reduction'] = (df['token_reduction'] / df['raw_tokens'].clip(lower=1)) * 100

# --- Report ---
def print_stats(label, subset):
    n = len(subset)
    raw_tok = subset['raw_tokens'].sum()
    clean_tok = subset['cleaned_tokens'].sum()
    tok_red = raw_tok - clean_tok
    raw_char = subset['raw_chars'].sum()
    clean_char = subset['cleaned_chars'].sum()
    char_red = raw_char - clean_char

    print(f"  {label} ({n} notes)")
    print(f"    Characters: {raw_char:,} -> {clean_char:,}  "
          f"(-{char_red:,}, {char_red/max(raw_char,1)*100:.1f}%)")
    print(f"    Tokens:     {raw_tok:,} -> {clean_tok:,}  "
          f"(-{tok_red:,}, {tok_red/max(raw_tok,1)*100:.1f}%)")
    print(f"    Per-note median token reduction: "
          f"{subset['token_pct_reduction'].median():.1f}%")
    print()

print("=" * 60)
print("TEXT REDUCTION SUMMARY")
print("=" * 60)
print()

print_stats("OVERALL", df)

for note_type in sorted(df['NOTE_TYPE'].dropna().unique()):
    subset = df.loc[df['NOTE_TYPE'] == note_type]
    print_stats(note_type, subset)

# Per-patient aggregate (how this affects LLM input cost)
px = df.groupby('DFCI_MRN').agg(
    raw_tokens=('raw_tokens', 'sum'),
    cleaned_tokens=('cleaned_tokens', 'sum'),
    num_notes=('raw_tokens', 'count')
).reset_index()
px['token_reduction_pct'] = ((px['raw_tokens'] - px['cleaned_tokens']) / px['raw_tokens'].clip(lower=1)) * 100

print("-" * 60)
print("PER-PATIENT SUMMARY")
print("-" * 60)
print(f"  Patients: {len(px)}")
print(f"  Median notes per patient: {px['num_notes'].median():.0f}")
print(f"  Median raw tokens per patient: {px['raw_tokens'].median():,.0f}")
print(f"  Median cleaned tokens per patient: {px['cleaned_tokens'].median():,.0f}")
print(f"  Median token reduction per patient: {px['token_reduction_pct'].median():.1f}%")
print(f"  Total raw tokens (all patients): {px['raw_tokens'].sum():,}")
print(f"  Total cleaned tokens (all patients): {px['cleaned_tokens'].sum():,}")
