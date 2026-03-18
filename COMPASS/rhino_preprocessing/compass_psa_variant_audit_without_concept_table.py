# ===============================================================
# PSA Variant Audit
# ===============================================================
# Extracts measurement counts for the 24 PSA variant concept IDs
# that are used for the >=10 PSA eligibility threshold but are
# currently dropped from the final output (no allowed unit combos).
#
# Goal: identify which variants have usable data (concept name,
# unit distribution, value ranges) so they can be incorporated
# into the main preprocessing pipeline.
# ===============================================================

# ===============================================================
# Load vocabulary subset CSVs (replacing deprecated catalog)
# ===============================================================
import os
_cwd = "file:" + os.getcwd()
spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true") \
    .csv(os.path.join(_cwd, "concept_tables", "concept_subset.csv")) \
    .createOrReplaceTempView("concept")
spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true") \
    .csv(os.path.join(_cwd, "concept_tables", "concept_ancestor_subset.csv")) \
    .createOrReplaceTempView("concept_ancestor")

# The 3 PSA concepts already in the final output (have allowed unit combos)
CANONICAL_PSA_IDS = [3013603, 3002131, 3034548]

# The 24 PSA variant IDs currently dropped from the final output
VARIANT_PSA_IDS = [
    3037249, 3037774, 3002178, 44811982, 40762314, 3038011, 3032915, 42529229,
    40484164, 715972, 35917418, 40762312, 40480170, 3007273, 44811980, 35918474,
    715971, 44811981, 4272032, 4194418, 4215704, 40762321, 3052038, 44793131,
]

ALL_PSA_IDS = CANONICAL_PSA_IDS + VARIANT_PSA_IDS

# ---------------------------------------------------------------
# Part 2: Look up concept metadata for all PSA IDs
# ---------------------------------------------------------------
psa_id_list = ", ".join(str(x) for x in ALL_PSA_IDS)

print("=" * 100)
print("PSA CONCEPT METADATA")
print("=" * 100)
spark.sql(f"""
    SELECT
        concept_id,
        concept_name,
        domain_id,
        vocabulary_id,
        concept_class_id,
        standard_concept,
        CASE
            WHEN concept_id IN ({", ".join(str(x) for x in CANONICAL_PSA_IDS)})
            THEN 'canonical (in output)'
            ELSE 'variant (dropped)'
        END AS status
    FROM concept
    WHERE concept_id IN ({psa_id_list})
    ORDER BY status, concept_name
""").show(50, truncate=False)

# ---------------------------------------------------------------
# Part 3: Measurement counts by concept ID and unit ID
# ---------------------------------------------------------------
spark.sql("USE CATALOG dfci_ia_aistudio")
spark.sql("USE SCHEMA omop_caia_denorm")

# Reuse prostate cancer and first diagnosis from main pipeline
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_prostate_cancer_concepts AS
SELECT descendant_concept_id AS concept_id
FROM concept_ancestor
WHERE ancestor_concept_id = 4163261
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW first_prostate_diagnosis AS
SELECT
  co.person_id,
  MIN(co.condition_start_date) AS prostate_cancer_diagnosis_date
FROM dn_condition_occurrence_20251219 co
WHERE co.condition_concept_id IN (SELECT concept_id FROM temp_prostate_cancer_concepts)
GROUP BY co.person_id
""")

# --- 3a. Overall counts: how many measurements exist per variant concept ID? ---
print("=" * 100)
print("MEASUREMENT COUNTS BY PSA VARIANT (all prostate cancer patients, post-diagnosis)")
print("=" * 100)

spark.sql(f"""
    SELECT
        m.measurement_concept_id,
        c.concept_name,
        COUNT(*) AS n_measurements,
        COUNT(DISTINCT m.person_id) AS n_patients,
        CASE
            WHEN m.measurement_concept_id IN ({", ".join(str(x) for x in CANONICAL_PSA_IDS)})
            THEN 'canonical'
            ELSE 'variant'
        END AS status
    FROM first_prostate_diagnosis f
    JOIN dn_measurement_20251219 m
        ON f.person_id = m.person_id
    JOIN concept c
        ON m.measurement_concept_id = c.concept_id
    WHERE m.measurement_concept_id IN ({psa_id_list})
      AND m.measurement_date > f.prostate_cancer_diagnosis_date
    GROUP BY m.measurement_concept_id, c.concept_name
    ORDER BY n_measurements DESC
""").show(50, truncate=False)

# --- 3b. Unit distribution per variant concept ID ---
print("=" * 100)
print("UNIT DISTRIBUTION BY PSA VARIANT")
print("=" * 100)

spark.sql(f"""
    SELECT
        m.measurement_concept_id,
        mc.concept_name AS psa_variant_name,
        m.unit_concept_id,
        COALESCE(uc.concept_name, '<no unit>') AS unit_name,
        COUNT(*) AS n_measurements,
        COUNT(DISTINCT m.person_id) AS n_patients,
        ROUND(MIN(m.value_as_number), 4) AS min_value,
        ROUND(PERCENTILE_APPROX(m.value_as_number, 0.25), 4) AS q1_value,
        ROUND(PERCENTILE_APPROX(m.value_as_number, 0.50), 4) AS median_value,
        ROUND(PERCENTILE_APPROX(m.value_as_number, 0.75), 4) AS q3_value,
        ROUND(MAX(m.value_as_number), 4) AS max_value,
        SUM(CASE WHEN m.value_as_number IS NULL THEN 1 ELSE 0 END) AS n_null_values
    FROM first_prostate_diagnosis f
    JOIN dn_measurement_20251219 m
        ON f.person_id = m.person_id
    JOIN concept mc
        ON m.measurement_concept_id = mc.concept_id
    LEFT JOIN concept uc
        ON m.unit_concept_id = uc.concept_id
    WHERE m.measurement_concept_id IN ({psa_id_list})
      AND m.measurement_date > f.prostate_cancer_diagnosis_date
    GROUP BY m.measurement_concept_id, mc.concept_name, m.unit_concept_id, uc.concept_name
    ORDER BY m.measurement_concept_id, n_measurements DESC
""").show(200, truncate=False)

# --- 3c. Variant-only: focus on the 24 dropped IDs ---
variant_id_list = ", ".join(str(x) for x in VARIANT_PSA_IDS)

print("=" * 100)
print("VARIANT-ONLY SUMMARY (the 24 IDs currently dropped from output)")
print("=" * 100)

spark.sql(f"""
    SELECT
        m.measurement_concept_id,
        mc.concept_name AS psa_variant_name,
        COUNT(*) AS n_measurements,
        COUNT(DISTINCT m.person_id) AS n_patients,
        SUM(CASE WHEN m.value_as_number IS NOT NULL THEN 1 ELSE 0 END) AS n_with_value,
        SUM(CASE WHEN m.unit_concept_id IS NOT NULL AND m.unit_concept_id != 0 THEN 1 ELSE 0 END) AS n_with_unit,
        SUM(CASE WHEN m.value_as_number IS NOT NULL
                  AND m.unit_concept_id IS NOT NULL
                  AND m.unit_concept_id != 0 THEN 1 ELSE 0 END) AS n_with_value_and_unit
    FROM first_prostate_diagnosis f
    JOIN dn_measurement_20251219 m
        ON f.person_id = m.person_id
    JOIN concept mc
        ON m.measurement_concept_id = mc.concept_id
    WHERE m.measurement_concept_id IN ({variant_id_list})
      AND m.measurement_date > f.prostate_cancer_diagnosis_date
    GROUP BY m.measurement_concept_id, mc.concept_name
    ORDER BY n_measurements DESC
""").show(50, truncate=False)

# --- 3d. Compare canonical vs variant value distributions ---
# For variants that DO have values, compare their distributions to canonical PSA
print("=" * 100)
print("VALUE DISTRIBUTION COMPARISON: canonical vs variant PSA")
print("=" * 100)

spark.sql(f"""
    SELECT
        CASE
            WHEN m.measurement_concept_id IN ({", ".join(str(x) for x in CANONICAL_PSA_IDS)})
            THEN 'canonical'
            ELSE 'variant'
        END AS psa_group,
        COUNT(*) AS n_measurements,
        COUNT(DISTINCT m.person_id) AS n_patients,
        ROUND(AVG(m.value_as_number), 2) AS mean_value,
        ROUND(STDDEV(m.value_as_number), 2) AS std_value,
        ROUND(MIN(m.value_as_number), 4) AS min_value,
        ROUND(PERCENTILE_APPROX(m.value_as_number, 0.25), 4) AS q1_value,
        ROUND(PERCENTILE_APPROX(m.value_as_number, 0.50), 4) AS median_value,
        ROUND(PERCENTILE_APPROX(m.value_as_number, 0.75), 4) AS q3_value,
        ROUND(MAX(m.value_as_number), 4) AS max_value
    FROM first_prostate_diagnosis f
    JOIN dn_measurement_20251219 m
        ON f.person_id = m.person_id
    WHERE m.measurement_concept_id IN ({psa_id_list})
      AND m.measurement_date > f.prostate_cancer_diagnosis_date
      AND m.value_as_number IS NOT NULL
    GROUP BY 1
    ORDER BY 1
""").show(truncate=False)
