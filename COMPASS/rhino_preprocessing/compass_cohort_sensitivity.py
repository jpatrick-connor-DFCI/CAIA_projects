# ===============================================================
# COMPASS Cohort Sensitivity Analysis
# ===============================================================
# Sweeps key parameters to show how many patients are
# included/excluded at each filtering step.
#
# Parameters varied:
#   1. PSA concept set: canonical only (3 IDs) vs all variants (27 IDs)
#   2. PSA count threshold: 1, 3, 5, 10, 15, 20, 25, 30
#   3. Exclusion filters: none, no other cancer, no PARP, both
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

# === Non-prostate malignant neoplasm concepts ===
# All descendants of "Malignant neoplastic disease" (443392) MINUS prostate cancer descendants.
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

# === PSA concept sets (two versions) ===
# Canonical only: the 3 primary PSA concept IDs
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_psa_canonical AS
SELECT concept_id
FROM concept
WHERE concept_id IN (3013603, 3002131, 3034548)
""")

# All variants: canonical + 24 additional PSA variant IDs
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_psa_all_variants AS
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

# ---------------------------------------------------------------
# Part 3: Parameter sweep
# ---------------------------------------------------------------
# Pre-compute PSA measurement counts per patient for each PSA concept set (post-dx only).
# This avoids re-scanning dn_measurement for every threshold.

PSA_THRESHOLDS = [1, 3, 5, 10, 15, 20, 25, 30]

psa_configs = [
    ("canonical_3",     "temp_psa_canonical"),
    ("all_variants_27", "temp_psa_all_variants"),
]

# Exclusion filter configs: (label, SQL WHERE clause to apply to person_id)
exclusion_configs = [
    ("none",              ""),
    ("no_other_cancer",   "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer)"),
    ("no_parp",           "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_parp)"),
    ("no_cancer_no_parp", "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer) "
                          "AND person_id NOT IN (SELECT person_id FROM temp_patients_with_parp)"),
]

results = []

for psa_label, psa_view in psa_configs:
    # Count PSA measurements per patient (one scan per PSA config)
    view_name = f"psa_counts_{psa_label}"
    spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW {view_name} AS
    SELECT f.person_id, COUNT(*) AS psa_count
    FROM first_prostate_diagnosis f
    JOIN dn_measurement_20251219 m
      ON f.person_id = m.person_id
    WHERE m.measurement_concept_id IN (SELECT concept_id FROM {psa_view})
      AND m.measurement_date > f.prostate_cancer_diagnosis_date
    GROUP BY f.person_id
    """)

    # For each (threshold, exclusion) combo, count qualifying patients
    for threshold in PSA_THRESHOLDS:
        for excl_label, excl_filter in exclusion_configs:
            row = spark.sql(f"""
                SELECT COUNT(*) AS n_patients
                FROM {view_name}
                WHERE psa_count >= {threshold}
                  {excl_filter}
            """).collect()[0]

            results.append({
                "psa_concepts": psa_label,
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

# Pivot: thresholds as columns for easier comparison
print("=" * 90)
print("PIVOT TABLE: Patients by (PSA concepts, exclusions, threshold)")
print("=" * 90)

spark.sql("""
    SELECT
        psa_concepts,
        exclusions,
        MAX(CASE WHEN threshold = 1  THEN n_patients END) AS `>=1`,
        MAX(CASE WHEN threshold = 3  THEN n_patients END) AS `>=3`,
        MAX(CASE WHEN threshold = 5  THEN n_patients END) AS `>=5`,
        MAX(CASE WHEN threshold = 10 THEN n_patients END) AS `>=10`,
        MAX(CASE WHEN threshold = 15 THEN n_patients END) AS `>=15`,
        MAX(CASE WHEN threshold = 20 THEN n_patients END) AS `>=20`,
        MAX(CASE WHEN threshold = 25 THEN n_patients END) AS `>=25`,
        MAX(CASE WHEN threshold = 30 THEN n_patients END) AS `>=30`
    FROM sensitivity_results
    GROUP BY psa_concepts, exclusions
    ORDER BY psa_concepts, exclusions
""").show(truncate=False)

# ---------------------------------------------------------------
# Part 5: PSA count distribution (production config: all_variants, no exclusions)
# ---------------------------------------------------------------
print("=" * 90)
print("PSA COUNT DISTRIBUTION (all_variants_27, post_dx, no exclusions)")
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
    FROM psa_counts_all_variants_27
    GROUP BY 1
    ORDER BY 1
""").show(truncate=False)

# Summary stats for both PSA concept sets
print("--- PSA count summary (all_variants_27) ---")
spark.sql("""
    SELECT
        COUNT(*) AS n_patients_with_any_psa,
        ROUND(AVG(psa_count), 1) AS mean_psa_count,
        PERCENTILE_APPROX(psa_count, 0.25) AS q1,
        PERCENTILE_APPROX(psa_count, 0.50) AS median,
        PERCENTILE_APPROX(psa_count, 0.75) AS q3,
        MIN(psa_count) AS min,
        MAX(psa_count) AS max
    FROM psa_counts_all_variants_27
""").show(truncate=False)

print("--- PSA count summary (canonical_3) ---")
spark.sql("""
    SELECT
        COUNT(*) AS n_patients_with_any_psa,
        ROUND(AVG(psa_count), 1) AS mean_psa_count,
        PERCENTILE_APPROX(psa_count, 0.25) AS q1,
        PERCENTILE_APPROX(psa_count, 0.50) AS median,
        PERCENTILE_APPROX(psa_count, 0.75) AS q3,
        MIN(psa_count) AS min,
        MAX(psa_count) AS max
    FROM psa_counts_canonical_3
""").show(truncate=False)

# ---------------------------------------------------------------
# Part 6: Impact of variant IDs — how many patients are gained?
# ---------------------------------------------------------------
print("=" * 90)
print("VARIANT ID IMPACT: Patients gained by including 24 PSA variants (no exclusions)")
print("=" * 90)

spark.sql("""
    SELECT
        c.threshold,
        c.n_patients AS canonical_only,
        v.n_patients AS with_variants,
        (v.n_patients - c.n_patients) AS gained,
        ROUND((v.n_patients - c.n_patients) * 100.0 / NULLIF(c.n_patients, 0), 1) AS pct_increase
    FROM sensitivity_results c
    JOIN sensitivity_results v
      ON c.threshold = v.threshold AND c.exclusions = v.exclusions
    WHERE c.psa_concepts = 'canonical_3'     AND c.exclusions = 'none'
      AND v.psa_concepts = 'all_variants_27' AND v.exclusions = 'none'
    ORDER BY c.threshold
""").show(truncate=False)

# ---------------------------------------------------------------
# Part 7: Exclusion filter impact (at threshold=10, all_variants)
# ---------------------------------------------------------------
print("=" * 90)
print("EXCLUSION FILTER IMPACT (all_variants_27, >=10 PSA)")
print("=" * 90)

baseline_10 = [r for r in results
               if r["psa_concepts"] == "all_variants_27"
               and r["exclusions"] == "none"
               and r["threshold"] == 10][0]["n_patients"]

print(f"{'Filter':<25} {'Patients':>10} {'Excluded':>10} {'% Remaining':>14}")
print("-" * 60)
for excl_label, _ in exclusion_configs:
    n = [r for r in results
         if r["psa_concepts"] == "all_variants_27"
         and r["exclusions"] == excl_label
         and r["threshold"] == 10][0]["n_patients"]
    excluded = baseline_10 - n
    pct = round(n * 100.0 / baseline_10, 1) if baseline_10 > 0 else 0
    print(f"{excl_label:<25} {n:>10,} {excluded:>10,} {pct:>13.1f}%")
print()

# Full breakdown across all thresholds
print("--- Exclusion impact across all thresholds (all_variants_27) ---")
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
    WHERE b.psa_concepts  = 'all_variants_27' AND b.exclusions  = 'none'
      AND c.psa_concepts  = 'all_variants_27' AND c.exclusions  = 'no_other_cancer'
      AND p.psa_concepts  = 'all_variants_27' AND p.exclusions  = 'no_parp'
      AND cp.psa_concepts = 'all_variants_27' AND cp.exclusions = 'no_cancer_no_parp'
    ORDER BY b.threshold
""").show(truncate=False)

# ---------------------------------------------------------------
# Part 8: Exclusion overlap — how many patients hit both filters?
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
    FROM psa_counts_all_variants_27
""").show(truncate=False)

