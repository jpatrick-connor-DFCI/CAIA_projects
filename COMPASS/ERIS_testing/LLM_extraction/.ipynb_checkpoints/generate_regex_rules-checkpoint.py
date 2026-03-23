"""
End-to-end script: sample notes per NOTE_TYPE, send to GPT-4o for per-type
regex rule generation, then synthesize into a unified set of rules.

Outputs:
  - Per-type GPT responses saved to regex_generation/ for reference
  - Final synthesized rules written to regex_generation/generated_rules.py
    (ready to paste into note_cleaning.py)
"""

import os
import pandas as pd
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

import prompt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = '/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/'
OUTPUT_DIR = os.path.join(DATA_PATH, 'regex_generation/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOTES_PER_TYPE = 15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version="2024-04-01-preview",
    azure_endpoint="https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
    azure_ad_token_provider=token_provider
)

model_name = "gpt-4o"

# ---------------------------------------------------------------------------
# Step 1: Sample notes
# ---------------------------------------------------------------------------
print("Step 1: Sampling notes per NOTE_TYPE...")

text_df = pd.read_csv(os.path.join(DATA_PATH, 'prostate_text_data.csv'))
text_df['TEXT_LENGTH'] = text_df['CLINICAL_TEXT'].fillna('').astype(str).str.len()

text_df['LENGTH_BIN'] = text_df.groupby('NOTE_TYPE')['TEXT_LENGTH'].transform(
    lambda x: pd.qcut(x, q=3, labels=['short', 'medium', 'long'], duplicates='drop')
)

# Sample and format notes per type
sampled_notes_by_type = {}
for note_type in sorted(text_df['NOTE_TYPE'].unique()):
    type_df = text_df.loc[text_df['NOTE_TYPE'] == note_type]
    deduped = type_df.drop_duplicates(subset='DFCI_MRN')

    per_bin = max(1, NOTES_PER_TYPE // deduped['LENGTH_BIN'].nunique())
    sampled = (deduped
               .groupby('LENGTH_BIN', group_keys=False, observed=True)
               .apply(lambda g: g.sample(n=min(per_bin, len(g)), random_state=RANDOM_SEED))
               .head(NOTES_PER_TYPE)
               .sort_values('TEXT_LENGTH'))

    note_blocks = []
    for i, (_, row) in enumerate(sampled.iterrows(), 1):
        note_blocks.append(
            f"===NOTE {i} START===\n"
            f"[Length: {row['TEXT_LENGTH']} chars]\n\n"
            f"{row['CLINICAL_TEXT']}\n"
            f"===NOTE {i} END==="
        )

    sampled_notes_by_type[note_type] = '\n\n'.join(note_blocks)
    print(f"  {note_type}: {len(sampled)} notes sampled")

# ---------------------------------------------------------------------------
# Step 2: Per-type regex generation
# ---------------------------------------------------------------------------
print("\nStep 2: Generating per-type regex rules via GPT-4o...")

per_type_outputs = {}
for note_type, notes_str in sampled_notes_by_type.items():
    print(f"  Sending {note_type} notes...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'user', 'content': prompt.build_regex_prompt(note_type, notes_str)}
            ],
            temperature=0
        ).choices[0].message.content.strip()

        per_type_outputs[note_type] = response

        # Save individual response for reference
        out_path = os.path.join(OUTPUT_DIR, f'gpt_response_{note_type.lower()}.txt')
        with open(out_path, 'w') as f:
            f.write(response)
        print(f"    -> saved to {out_path}")

    except Exception as e:
        print(f"    FAILED for {note_type}: {e}")
        per_type_outputs[note_type] = ""

# Check all types succeeded
if not all(per_type_outputs.values()):
    failed = [k for k, v in per_type_outputs.items() if not v]
    print(f"\nWARNING: Failed types: {failed}. Synthesis will be incomplete.")

# ---------------------------------------------------------------------------
# Step 3: Synthesis
# ---------------------------------------------------------------------------
print("\nStep 3: Synthesizing rules across NOTE_TYPEs...")

synthesis_prompt = prompt.prompt_regex_synthesis.format(
    clinician_output=per_type_outputs.get('Clinician', '(no output)'),
    imaging_output=per_type_outputs.get('Imaging', '(no output)'),
    pathology_output=per_type_outputs.get('Pathology', '(no output)'),
)

try:
    synthesis_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'user', 'content': synthesis_prompt}
        ],
        temperature=0
    ).choices[0].message.content.strip()

    out_path = os.path.join(OUTPUT_DIR, 'generated_rules.py')
    with open(out_path, 'w') as f:
        f.write(synthesis_response)
    print(f"\nDone. Final rules written to {out_path}")
    print("Review the output, then paste the rule lists into note_cleaning.py.")

except Exception as e:
    print(f"\nSynthesis FAILED: {e}")
    print("Per-type outputs are still saved in regex_generation/ for manual synthesis.")
