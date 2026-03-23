# ===============================================================
# Validate Platinum Concept Coverage
# ===============================================================
# Compares carboplatin/cisplatin concepts found via:
#   1. concept_ancestor table (standard OMOP hierarchy)
#   2. Direct name search in the concept table
#   3. Actual drug_exposure records in the clinical data
# to ensure no platinum exposures are missed.
# ===============================================================

import os
from pyspark.sql import functions as F

# ---------------------------------------------------------------
# Load vocabulary CSVs (same as compass_preprocessing.py)
# ---------------------------------------------------------------
_cwd = "file:" + os.getcwd()
spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true") \
    .csv(os.path.join(_cwd, "concept_tables", "concept_subset.csv")) \
    .createOrReplaceTempView("concept")
spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true") \
    .csv(os.path.join(_cwd, "concept_tables", "concept_ancestor_subset.csv")) \
    .createOrReplaceTempView("concept_ancestor")

# ---------------------------------------------------------------
# Method 1: Ancestor-based extraction (current pipeline approach)
# ---------------------------------------------------------------
df_ancestor = spark.sql("""
    SELECT DISTINCT
        descendant_concept_id AS concept_id,
        CASE
            WHEN ancestor_concept_id = 1344905 THEN 'Carboplatin'
            WHEN ancestor_concept_id = 1397599 THEN 'Cisplatin'
        END AS drug_type
    FROM concept_ancestor
    WHERE ancestor_concept_id IN (1344905, 1397599)
""")
df_ancestor.createOrReplaceTempView("ancestor_platinum_ids")

# ---------------------------------------------------------------
# Method 2: Name-based search in concept table
# ---------------------------------------------------------------
df_name = spark.sql("""
    SELECT concept_id, concept_name, vocabulary_id, concept_class_id
    FROM concept
    WHERE LOWER(concept_name) LIKE '%carboplatin%'
       OR LOWER(concept_name) LIKE '%cisplatin%'
       OR LOWER(concept_name) LIKE '%paraplatin%'
       OR LOWER(concept_name) LIKE '%platinol%'
""")
df_name.createOrReplaceTempView("name_platinum_ids")

# ---------------------------------------------------------------
# Compare: concepts found by name but NOT by ancestor
# ---------------------------------------------------------------
df_missing_from_ancestor = spark.sql("""
    SELECT n.concept_id, n.concept_name, n.vocabulary_id, n.concept_class_id
    FROM name_platinum_ids n
    LEFT JOIN ancestor_platinum_ids a ON n.concept_id = a.concept_id
    WHERE a.concept_id IS NULL
""")

# ---------------------------------------------------------------
# Compare: concepts found by ancestor but NOT by name
# ---------------------------------------------------------------
df_missing_from_name = spark.sql("""
    SELECT a.concept_id, a.drug_type, c.concept_name, c.vocabulary_id, c.concept_class_id
    FROM ancestor_platinum_ids a
    LEFT JOIN concept c ON a.concept_id = c.concept_id
    LEFT JOIN name_platinum_ids n ON a.concept_id = n.concept_id
    WHERE n.concept_id IS NULL
""")

# ---------------------------------------------------------------
# Print results
# ---------------------------------------------------------------
print("=" * 80)
print("PLATINUM CONCEPT VALIDATION")
print("=" * 80)

print(f"\nMethod 1 — Ancestor-based (pipeline): {df_ancestor.count()} concepts")
print(f"  Carboplatin descendants: {df_ancestor.filter(F.col('drug_type') == 'Carboplatin').count()}")
print(f"  Cisplatin descendants:   {df_ancestor.filter(F.col('drug_type') == 'Cisplatin').count()}")

print(f"\nMethod 2 — Name-based search: {df_name.count()} concepts")

n_missing_ancestor = df_missing_from_ancestor.count()
print(f"\n{'=' * 80}")
print(f"Concepts found by NAME but missing from ANCESTOR hierarchy: {n_missing_ancestor}")
print("=" * 80)
if n_missing_ancestor > 0:
    print(">>> REVIEW THESE — they may be platinum concepts not captured by the pipeline:")
    df_missing_from_ancestor.show(n_missing_ancestor, truncate=False)
else:
    print("None — ancestor hierarchy fully covers name-matched concepts.")

n_missing_name = df_missing_from_name.count()
print(f"\n{'=' * 80}")
print(f"Concepts found by ANCESTOR but missing from NAME search: {n_missing_name}")
print("=" * 80)
if n_missing_name > 0:
    print("(Expected — ancestor hierarchy includes coded descendants without")
    print(" 'carboplatin'/'cisplatin' in the name, e.g. NDC or branded codes.)")
    df_missing_from_name.show(min(n_missing_name, 50), truncate=False)
else:
    print("None.")

# ---------------------------------------------------------------
# Method 3: Check actual drug exposures against both sets
# ---------------------------------------------------------------
spark.sql("USE CATALOG dfci_ia_aistudio")
spark.sql("USE SCHEMA omop_caia_denorm")

print(f"\n{'=' * 80}")
print("CROSS-CHECK AGAINST ACTUAL DRUG EXPOSURES")
print("=" * 80)

# All distinct drug_concept_ids with 'platin' in the concept name
df_exposure_platin = spark.sql("""
    SELECT DISTINCT
        de.drug_concept_id,
        c.concept_name,
        c.vocabulary_id,
        c.concept_class_id,
        CASE WHEN a.concept_id IS NOT NULL THEN 'Y' ELSE 'N' END AS in_ancestor_set,
        CASE WHEN n.concept_id IS NOT NULL THEN 'Y' ELSE 'N' END AS in_name_set
    FROM dn_drug_exposure_20251219 de
    LEFT JOIN concept c ON de.drug_concept_id = c.concept_id
    LEFT JOIN ancestor_platinum_ids a ON de.drug_concept_id = a.concept_id
    LEFT JOIN name_platinum_ids n ON de.drug_concept_id = n.concept_id
    WHERE a.concept_id IS NOT NULL
       OR n.concept_id IS NOT NULL
    ORDER BY c.concept_name
""")
df_exposure_platin.createOrReplaceTempView("exposure_platin")

n_exposure = df_exposure_platin.count()
print(f"\nDistinct platinum drug_concept_ids in drug_exposure: {n_exposure}")
df_exposure_platin.show(n_exposure, truncate=False)

# Flag any actual exposures missed by the ancestor method
df_exposure_gaps = spark.sql("""
    SELECT * FROM exposure_platin WHERE in_ancestor_set = 'N'
""")
n_gaps = df_exposure_gaps.count()
print(f"\n{'=' * 80}")
print(f"Drug exposures matched by name but MISSED by ancestor hierarchy: {n_gaps}")
print("=" * 80)
if n_gaps > 0:
    print(">>> ACTION REQUIRED — these exposures are not captured by the pipeline:")
    df_exposure_gaps.show(n_gaps, truncate=False)
else:
    print("None — all platinum exposures are captured by the current pipeline.")

# Patient-level summary
df_patient_counts = spark.sql("""
    SELECT
        COUNT(DISTINCT de.person_id) AS patients_with_platinum
    FROM dn_drug_exposure_20251219 de
    WHERE de.drug_concept_id IN (SELECT concept_id FROM ancestor_platinum_ids)
""")
print(f"\n{'=' * 80}")
print("PATIENT COUNTS")
print("=" * 80)
df_patient_counts.show(truncate=False)
