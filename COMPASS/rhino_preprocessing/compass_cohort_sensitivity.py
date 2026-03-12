# ===============================================================
# COMPASS Cohort Sensitivity Analysis
# ===============================================================
# Sweeps exclusion filters and PSA count thresholds to show how
# many patients are included/excluded at each filtering step.
#
# Fixed:
#   - PSA concept set: all 27 variants
# Parameters varied:
#   1. PSA count threshold: 1, 5, 10
#   2. Exclusion filters: none, no primary non-prostate cancer,
#      no PARP, both
# ===============================================================

# ---------------------------------------------------------------
# Part 1: Concept set generation (vocabulary catalog)
# ---------------------------------------------------------------
spark.sql("USE CATALOG snowflake_aistudio_full_catalog")
spark.sql("USE SCHEMA omop_cdm_deid")

CONCEPT_TABLE = "snowflake_aistudio_full_catalog.omop_cdm_deid.concept"
spark.table(CONCEPT_TABLE).cache().createOrReplaceTempView("concept")

# === Prostate cancer concepts ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_prostate_cancer_concepts AS
SELECT descendant_concept_id AS concept_id
FROM concept_ancestor
WHERE ancestor_concept_id = 4163261
""")

# === Non-prostate PRIMARY malignant neoplasm concepts ===
# All descendants of "Malignant neoplastic disease" (443392) MINUS:
#   - prostate cancer descendants (4163261)
#   - secondary/metastatic neoplasm descendants (432851)
# This prevents metastatic prostate cancer (e.g. "Secondary malignant neoplasm
# of bone") from being misclassified as a separate primary malignancy.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_non_prostate_cancer_concepts AS
SELECT descendant_concept_id AS concept_id
FROM concept_ancestor
WHERE ancestor_concept_id = 443392
  AND descendant_concept_id NOT IN (
    SELECT descendant_concept_id
    FROM concept_ancestor
    WHERE ancestor_concept_id = 4163261
  )
  AND descendant_concept_id NOT IN (
    SELECT descendant_concept_id
    FROM concept_ancestor
    WHERE ancestor_concept_id = 432851
  )
""")

# === Non-melanoma skin cancer (NMSC) concepts ===
# Basal cell carcinoma (4112752) and squamous cell carcinoma of skin (4111921)
# and all their descendants. These are generally indolent and unlikely to
# confound prostate cancer outcomes.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_nmsc_concepts AS
SELECT DISTINCT descendant_concept_id AS concept_id
FROM concept_ancestor
WHERE ancestor_concept_id IN (
    4112752,  -- Basal cell carcinoma of skin
    4111921   -- Squamous cell carcinoma of skin
)
""")

# === NOS malignant neoplasm concept ===
# Concept 439392 ("Primary malignant neoplasm" with no site specified)
# is likely a coding artifact and not a true second primary.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_nos_malignant_neoplasm AS
SELECT 439392 AS concept_id
""")

# === Relaxed non-prostate cancer views (allowing back NMSC and/or NOS) ===
# Version 1: Allow NMSC back in (exclude other cancers except skin BCC/SCC)
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_non_prostate_cancer_concepts_allow_nmsc AS
SELECT concept_id FROM temp_non_prostate_cancer_concepts
WHERE concept_id NOT IN (SELECT concept_id FROM temp_nmsc_concepts)
""")

# Version 2: Allow NMSC + NOS back in
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_non_prostate_cancer_concepts_allow_nmsc_nos AS
SELECT concept_id FROM temp_non_prostate_cancer_concepts
WHERE concept_id NOT IN (SELECT concept_id FROM temp_nmsc_concepts)
  AND concept_id NOT IN (SELECT concept_id FROM temp_nos_malignant_neoplasm)
""")

# === PARP inhibitor drug concepts ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_parp_ingredients AS
SELECT concept_id
FROM concept
WHERE LOWER(concept_name) IN ('olaparib', 'rucaparib', 'niraparib', 'talazoparib')
  AND concept_class_id = 'Ingredient'
  AND standard_concept = 'S'
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_parp_drugs AS
SELECT DISTINCT descendant_concept_id AS drug_concept_id
FROM concept_ancestor
WHERE ancestor_concept_id IN (SELECT concept_id FROM temp_parp_ingredients)
""")

# === PSA concepts: all 27 variants ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_psa_concepts AS
SELECT concept_id
FROM concept
WHERE concept_id IN (
  3013603, 3002131, 3034548,
  3037249, 3037774, 3002178, 44811982, 40762314, 3038011, 3032915, 42529229,
  40484164, 715972, 35917418, 40762312, 40480170, 3007273, 44811980, 35918474,
  715971, 44811981, 4272032, 4194418, 4215704, 40762321, 3052038, 44793131
)
""")

# ---------------------------------------------------------------
# Part 2: Clinical data (denormalized catalog)
# ---------------------------------------------------------------
spark.sql("USE CATALOG dfci_ia_aistudio")
spark.sql("USE SCHEMA omop_caia_denorm")

# === First prostate cancer diagnosis per patient ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW first_prostate_diagnosis AS
SELECT
  co.person_id,
  MIN(co.condition_start_date) AS prostate_cancer_diagnosis_date
FROM dn_condition_occurrence_20251219 co
WHERE co.condition_concept_id IN (SELECT concept_id FROM temp_prostate_cancer_concepts)
GROUP BY co.person_id
""")

# === Baseline: total prostate cancer patients ===
baseline_count = spark.sql("SELECT COUNT(*) AS n FROM first_prostate_diagnosis").collect()[0]["n"]
print(f"Total patients with any prostate cancer diagnosis: {baseline_count:,}")
print()

# === Exclusion sets (computed once) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_patients_with_other_cancer AS
SELECT DISTINCT co.person_id
FROM dn_condition_occurrence_20251219 co
WHERE co.condition_concept_id IN (SELECT concept_id FROM temp_non_prostate_cancer_concepts)
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_patients_with_parp AS
SELECT DISTINCT de.person_id
FROM dn_drug_exposure_20251219 de
WHERE de.drug_concept_id IN (SELECT drug_concept_id FROM temp_parp_drugs)
""")

# Relaxed exclusion sets: exclude patients with other cancer EXCEPT NMSC (and optionally NOS)
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_patients_with_other_cancer_allow_nmsc AS
SELECT DISTINCT co.person_id
FROM dn_condition_occurrence_20251219 co
WHERE co.condition_concept_id IN (SELECT concept_id FROM temp_non_prostate_cancer_concepts_allow_nmsc)
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_patients_with_other_cancer_allow_nmsc_nos AS
SELECT DISTINCT co.person_id
FROM dn_condition_occurrence_20251219 co
WHERE co.condition_concept_id IN (SELECT concept_id FROM temp_non_prostate_cancer_concepts_allow_nmsc_nos)
""")

# ---------------------------------------------------------------
# Part 3: Parameter sweep
# ---------------------------------------------------------------
# Pre-compute PSA measurement counts per patient (post-dx only).

PSA_THRESHOLDS = [1, 5, 10]

# Exclusion filter configs: (label, SQL WHERE clause to apply to person_id)
exclusion_configs = [
    ("none",              ""),
    ("no_other_cancer",   "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer)"),
    ("no_parp",           "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_parp)"),
    ("no_cancer_no_parp", "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer) "
                          "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_parp)"),
    # Relaxed: allow NMSC back in
    ("no_cancer_allow_nmsc",
     "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer_allow_nmsc)"),
    ("no_cancer_allow_nmsc_no_parp",
     "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer_allow_nmsc) "
     "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_parp)"),
    # Relaxed: allow NMSC + NOS malignant neoplasm back in
    ("no_cancer_allow_nmsc_nos",
     "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer_allow_nmsc_nos)"),
    ("no_cancer_allow_nmsc_nos_no_parp",
     "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer_allow_nmsc_nos) "
     "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_parp)"),
]

# Count PSA measurements per patient (single scan)
spark.sql("""
CREATE OR REPLACE TEMP VIEW psa_counts AS
SELECT f.person_id, COUNT(*) AS psa_count
FROM first_prostate_diagnosis f
JOIN dn_measurement_20251219 m
  ON f.person_id = m.person_id
WHERE m.measurement_concept_id IN (SELECT concept_id FROM temp_psa_concepts)
  AND m.measurement_date > f.prostate_cancer_diagnosis_date
GROUP BY f.person_id
""")

results = []

for threshold in PSA_THRESHOLDS:
    for excl_label, excl_filter in exclusion_configs:
        row = spark.sql(f"""
            SELECT COUNT(*) AS n_patients
            FROM psa_counts
            WHERE psa_count >= {threshold}
              {excl_filter}
        """).collect()[0]

        results.append({
            "exclusions": excl_label,
            "threshold": threshold,
            "n_patients": row["n_patients"],
        })

# ---------------------------------------------------------------
# Part 4: Display results
# ---------------------------------------------------------------
from pyspark.sql import Row

results_df = spark.createDataFrame([Row(**r) for r in results])
results_df.createOrReplaceTempView("sensitivity_results")

# Pivot: thresholds as columns
print("=" * 90)
print("PIVOT TABLE: Patients by (exclusion filter, PSA count threshold)")
print("=" * 90)

spark.sql("""
    SELECT
        exclusions,
        MAX(CASE WHEN threshold = 1  THEN n_patients END) AS `>=1`,
        MAX(CASE WHEN threshold = 5  THEN n_patients END) AS `>=5`,
        MAX(CASE WHEN threshold = 10 THEN n_patients END) AS `>=10`
    FROM sensitivity_results
    GROUP BY exclusions
    ORDER BY exclusions
""").show(truncate=False)

# ---------------------------------------------------------------
# Part 5: PSA count distribution (no exclusions)
# ---------------------------------------------------------------
print("=" * 90)
print("PSA COUNT DISTRIBUTION (post_dx, no exclusions)")
print("=" * 90)

spark.sql("""
    SELECT
        CASE
            WHEN psa_count BETWEEN 1  AND 2  THEN '01-02'
            WHEN psa_count BETWEEN 3  AND 4  THEN '03-04'
            WHEN psa_count BETWEEN 5  AND 9  THEN '05-09'
            WHEN psa_count BETWEEN 10 AND 14 THEN '10-14'
            WHEN psa_count BETWEEN 15 AND 19 THEN '15-19'
            WHEN psa_count BETWEEN 20 AND 29 THEN '20-29'
            WHEN psa_count BETWEEN 30 AND 49 THEN '30-49'
            WHEN psa_count >= 50              THEN '50+'
        END AS psa_count_bin,
        COUNT(*) AS n_patients,
        MIN(psa_count) AS min_psa,
        MAX(psa_count) AS max_psa
    FROM psa_counts
    GROUP BY 1
    ORDER BY 1
""").show(truncate=False)

print("--- PSA count summary ---")
spark.sql("""
    SELECT
        COUNT(*) AS n_patients_with_any_psa,
        ROUND(AVG(psa_count), 1) AS mean_psa_count,
        PERCENTILE_APPROX(psa_count, 0.25) AS q1,
        PERCENTILE_APPROX(psa_count, 0.50) AS median,
        PERCENTILE_APPROX(psa_count, 0.75) AS q3,
        MIN(psa_count) AS min,
        MAX(psa_count) AS max
    FROM psa_counts
""").show(truncate=False)

# ---------------------------------------------------------------
# Part 6: Exclusion filter impact (detailed)
# ---------------------------------------------------------------
print("=" * 90)
print("EXCLUSION FILTER IMPACT")
print("=" * 90)

for threshold in PSA_THRESHOLDS:
    baseline_n = [r for r in results
                  if r["exclusions"] == "none"
                  and r["threshold"] == threshold][0]["n_patients"]

    print(f"\n--- Threshold: >={threshold} PSA measurements ---")
    print(f"{'Filter':<25} {'Patients':>10} {'Excluded':>10} {'% Remaining':>14}")
    print("-" * 60)
    for excl_label, _ in exclusion_configs:
        n = [r for r in results
             if r["exclusions"] == excl_label
             and r["threshold"] == threshold][0]["n_patients"]
        excluded = baseline_n - n
        pct = round(n * 100.0 / baseline_n, 1) if baseline_n > 0 else 0
        print(f"{excl_label:<25} {n:>10,} {excluded:>10,} {pct:>13.1f}%")
print()

# Full breakdown across all thresholds
print("--- Exclusion impact across all thresholds ---")
spark.sql("""
    SELECT
        b.threshold,
        b.n_patients AS no_filter,
        c.n_patients AS no_other_cancer,
        p.n_patients AS no_parp,
        cp.n_patients AS no_cancer_no_parp,
        (b.n_patients - c.n_patients)  AS lost_to_cancer,
        (b.n_patients - p.n_patients)  AS lost_to_parp,
        (b.n_patients - cp.n_patients) AS lost_to_both
    FROM sensitivity_results b
    JOIN sensitivity_results c  ON b.threshold = c.threshold
    JOIN sensitivity_results p  ON b.threshold = p.threshold
    JOIN sensitivity_results cp ON b.threshold = cp.threshold
    WHERE b.exclusions  = 'none'
      AND c.exclusions  = 'no_other_cancer'
      AND p.exclusions  = 'no_parp'
      AND cp.exclusions = 'no_cancer_no_parp'
    ORDER BY b.threshold
""").show(truncate=False)

# ---------------------------------------------------------------
# Part 7: Exclusion overlap — how many patients hit both filters?
# ---------------------------------------------------------------
print("=" * 90)
print("EXCLUSION OVERLAP (among prostate cancer patients with >=1 post-dx PSA)")
print("=" * 90)

spark.sql("""
    SELECT
        COUNT(*) AS total_with_psa,
        SUM(CASE WHEN person_id IN (SELECT person_id FROM temp_patients_with_other_cancer) THEN 1 ELSE 0 END) AS has_other_cancer,
        SUM(CASE WHEN person_id IN (SELECT person_id FROM temp_patients_with_parp) THEN 1 ELSE 0 END) AS has_parp,
        SUM(CASE WHEN person_id IN (SELECT person_id FROM temp_patients_with_other_cancer)
                  AND person_id IN (SELECT person_id FROM temp_patients_with_parp) THEN 1 ELSE 0 END) AS has_both,
        SUM(CASE WHEN person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer)
                  AND person_id NOT IN (SELECT person_id FROM temp_patients_with_parp) THEN 1 ELSE 0 END) AS has_neither
    FROM psa_counts
""").show(truncate=False)
