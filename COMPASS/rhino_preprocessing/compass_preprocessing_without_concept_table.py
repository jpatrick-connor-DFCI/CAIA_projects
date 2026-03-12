# ===============================================================
# COMPASS Cohort Preprocessing
# ===============================================================
# Inclusion/exclusion criteria:
#   1. Prostate cancer diagnosis
#   2. >= 10 PSA measurements (full 27-variant concept set) after diagnosis
#   3. No other primary cancers (keeping metastatic disease, NMSC, and
#      malignant neoplasm NOS)
#   4. No PARP inhibitor exposure (olaparib, rucaparib, niraparib, talazoparib)
# ===============================================================

# ===============================================================
# Load vocabulary subset CSVs (replacing deprecated catalog)
# ===============================================================
import os
import pandas as pd
_cwd = os.getcwd()
spark.createDataFrame(pd.read_csv(os.path.join(_cwd, "concept_tables", "concept_subset.csv"))).createOrReplaceTempView("concept")
spark.createDataFrame(pd.read_csv(os.path.join(_cwd, "concept_tables", "concept_ancestor_subset.csv"))).createOrReplaceTempView("concept_ancestor")

CONCEPT_TABLE = "concept"

# ===============================================================
# Part 1: Concept set generation (vocabulary catalog)
# ===============================================================

# === 1. Prostate cancer concepts (descendants of OMOP concept 4163261) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_prostate_cancer_concepts AS
SELECT descendant_concept_id AS concept_id
FROM concept_ancestor
WHERE ancestor_concept_id = 4163261
""")

# === 2. PSA measurement concepts (explicit list) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_psa_concepts AS
SELECT concept_id
FROM concept
WHERE concept_id IN (
  3013603, 3002131, 3034548,
  -- The following 24 PSA variant IDs are included here so they count toward the
  -- ≥10 PSA eligibility threshold (step 8). Of these, 4272032 and 3052038 also
  -- appear in the final lab output (remapped to 3013603 in step 16b). The
  -- remaining 22 have no measurement data and are silently dropped by the
  -- allowed_unit_combinations inner join in step 17.
  3037249, 3037774, 3002178, 44811982, 40762314, 3038011, 3032915, 42529229,
  40484164, 715972, 35917418, 40762312, 40480170, 3007273, 44811980, 35918474,
  715971, 44811981, 4272032, 4194418, 4215704, 40762321, 3052038, 44793131
)
""")

# === 3. Non-prostate PRIMARY malignant neoplasm concepts (for exclusion) ===
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

# === 3b. Non-melanoma skin cancer (NMSC) concepts ===
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

# === 3c. NOS malignant neoplasm concept ===
# Concept 439392 ("Primary malignant neoplasm" with no site specified)
# is likely a coding artifact and not a true second primary.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_nos_malignant_neoplasm AS
SELECT 439392 AS concept_id
""")

# === 3d. Relaxed non-prostate cancer view (allowing NMSC + NOS) ===
# Exclude other primary cancers EXCEPT non-melanoma skin cancers and NOS.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_non_prostate_cancer_concepts_allow_nmsc_nos AS
SELECT concept_id FROM temp_non_prostate_cancer_concepts
WHERE concept_id NOT IN (SELECT concept_id FROM temp_nmsc_concepts)
  AND concept_id NOT IN (SELECT concept_id FROM temp_nos_malignant_neoplasm)
""")

# === 3e. PARP inhibitor drug concepts ===
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

# === 4. Other lab measurement concepts (explicit list) ===
# Includes canonical concepts + variant codings that will be remapped after extraction.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_other_lab_concepts AS
SELECT concept_id
FROM concept
WHERE concept_id IN (
  -- Canonical lab concepts
  3006923, 3024561, 3020509, 3035995, 3009306, 3013721, 3027597, 3024128, 3036277, 3020891,
  3025315, 3020460, 3006906, 3037551, 3022914, 3015632, 3003785, 3014576, 3007220, 3016723,
  3012888, 3019897, 3013707, 3020416, 3027970, 3004501, 3027018, 3023314, 3000963, 3022217,
  3016436, 3000905, 3004327, 3012030, 3009744, 3023599, 3033575, 3013650, 3024929,
  3023103, 3020630, 3034426, 3024171, 3019550, 3004249, 3008893, 3013682, 3013466,
  -- Category 1: Variant codings of existing labs (will be remapped post-extraction)
  4272032, 3052038,  -- PSA SNOMED/LOINC variants -> 3013603
  3002131, 3034548,  -- PSA LOINC variants -> 3013603
  3000285,           -- Sodium in Blood -> 3019550
  3000483,           -- Glucose in Blood -> 3004501
  3004295,           -- BUN in Blood -> 3013682
  3005456,           -- Potassium in Blood -> 3023103
  3018572,           -- Chloride in Blood -> 3014576
  3004119, 3027484,  -- Hemoglobin variants -> 3000963
  3009542,           -- Hematocrit -> 3023314
  3003338,           -- MCHC -> 3009744
  3024731,           -- MCV -> 3023599
  3035941,           -- MCH -> 3012030
  3002385,           -- RDW -> 3019897
  3010813,           -- Leukocytes in Blood -> 3000905
  3019198,           -- Lymphocytes in Blood -> 3004327
  3017732,           -- Neutrophils in Blood -> 3013650
  3001604,           -- Monocytes in Blood -> 3033575
  3007461,           -- Platelets in Blood -> 3024929
  3026361,           -- Erythrocytes in Blood -> 3020416
  3015183,           -- ESR -> 3013707
  3010156,           -- CRP high-sensitivity -> 3020460
  3022250,           -- LDH lactate->pyruvate -> 3016436
  3005225,           -- LDH pyruvate->lactate -> 3016436
  3018677,           -- aPTT in PPP -> 3013466
  3005755,           -- ALT with P-5'-P -> 3006923
  3037081,           -- AST with P-5'-P -> 3013721
  3010140,           -- CO2 in Venous -> 3015632
  3002417,           -- PT in Blood -> 3034426
  3032080,           -- INR in Blood -> 3022217
  3049555,           -- Testosterone low-detect -> 3008893
  3021886,           -- Globulin in Serum -> 3027970
  3013762,           -- Body weight Measured -> 3025315
  3023540,           -- Body height Measured -> 3036277
  3003215,           -- Lymphocytes manual count -> 3004327
  3017501,           -- Neutrophils manual count -> 3013650
  3027651,           -- Basophils manual count -> 3013429
  3009932,           -- Eosinophils manual count -> 3028615
  3034107,           -- Monocytes manual count -> 3033575
  3019909,           -- Hematocrit centrifugation -> 3023314
  3034976,           -- Hematocrit venous -> 3023314
  3001657,           -- WBC corrected -> 3000905
  3046900,           -- WBC corrected automated -> 3000905
  3048275,           -- Hemoglobin pre-phlebotomy -> 3000963
  3028286,           -- Albumin electrophoresis -> 3024561
  40763912,          -- Albumin BCG -> 3024561
  3046948,           -- A/G ratio electrophoresis -> 3020509
  3011424,           -- Glucose automated strip -> 3004501
  3014053,           -- Glucose manual strip -> 3004501
  3032986,           -- Glucose 1st specimen -> 3004501
  3013826,           -- Glucose mmol/L -> 3004501
  3027219,           -- BUN venous -> 3013682
  3051825,           -- Creatinine blood -> 3016723
  3035285,           -- Chloride venous -> 3014576
  3041354,           -- Potassium venous -> 3023103
  3030597,           -- Calcium corrected -> 3006906
  3015377,           -- Calcium mmol/L -> 3006906
  3019170,           -- TSH sensitive 0.005 -> 3009201
  3019762,           -- TSH sensitive 0.05 -> 3009201
  3024675,           -- Free T4 dialysis -> 3008598
  3019171,           -- Body height Stated -> 3036277
  3023166,           -- Body weight Stated -> 3025315
  3040891,           -- Heart rate resting -> 3027018
  3042292,           -- Heart rate post-exercise -> 3027018
  40771525,          -- Heart rate sitting -> 3027018
  3016628,           -- Fibrinogen Ag immunoassay -> 3016407
  3037950,           -- Fibrinogen Ag nephelometry -> 3016407
  -- Category 2: New lab types
  3013429, 3006315,  -- Basophils (automated + generic; 3006315 remapped -> 3013429)
  3028615, 3013115,  -- Eosinophils (automated + generic; 3013115 remapped -> 3028615)
  3009201,           -- TSH
  3008598,           -- Free T4
  3016407            -- Fibrinogen
)
""")

# === 5. All labs (PSA UNION other labs) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_all_lab_concepts AS
SELECT concept_id FROM temp_psa_concepts
UNION
SELECT concept_id FROM temp_other_lab_concepts
""")

# === 6. Platinum drugs (descendants of carboplatin/cisplatin ingredients) ===
#    Expose both concept_id and drug_concept_id alias for downstream joins.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_platinum_drugs AS
SELECT
  descendant_concept_id AS concept_id,
  descendant_concept_id AS drug_concept_id,
  CASE
    WHEN ancestor_concept_id = 1344905 THEN 'Carboplatin'
    WHEN ancestor_concept_id = 1397599 THEN 'Cisplatin'
  END AS drug_type
FROM concept_ancestor
WHERE ancestor_concept_id IN (1344905, 1397599)
""")

# ===============================================================
# Part 2: Clinical preprocessing (denormalized catalog)
# ===============================================================
spark.sql("USE CATALOG dfci_ia_aistudio")
spark.sql("USE SCHEMA omop_caia_denorm")

# === 7. First prostate cancer diagnosis per patient ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW first_prostate_diagnosis AS
SELECT
  co.person_id,
  MIN(co.condition_start_date) AS prostate_cancer_diagnosis_date
FROM dn_condition_occurrence_20251219 co
WHERE co.condition_concept_id IN (SELECT concept_id FROM temp_prostate_cancer_concepts)
GROUP BY co.person_id
""")

# === 8. Patients with prostate cancer and ≥10 PSA measurements after diagnosis ===
# FIX: Joins against first_prostate_diagnosis (one row per patient) instead of
# condition_occurrence (many rows per patient), preventing cross-product inflation
# of the PSA count.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_prostate_psa10_patients AS
SELECT f.person_id
FROM first_prostate_diagnosis f
JOIN dn_measurement_20251219 m
  ON f.person_id = m.person_id
WHERE m.measurement_concept_id IN (SELECT concept_id FROM temp_psa_concepts)
  AND m.measurement_date > f.prostate_cancer_diagnosis_date
GROUP BY f.person_id
HAVING COUNT(*) >= 10
""")

# === 9. Exclusion: patients with non-prostate primary cancer (allowing NMSC + NOS) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_patients_with_other_cancer AS
SELECT DISTINCT co.person_id
FROM dn_condition_occurrence_20251219 co
WHERE co.condition_concept_id IN (SELECT concept_id FROM temp_non_prostate_cancer_concepts_allow_nmsc_nos)
""")

# === 10. Exclusion: patients with PARP inhibitor exposure ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_patients_with_parp AS
SELECT DISTINCT de.person_id
FROM dn_drug_exposure_20251219 de
WHERE de.drug_concept_id IN (SELECT drug_concept_id FROM temp_parp_drugs)
""")

# === 11. Combined eligible cohort: ≥10 PSA AND no other cancer AND no PARP ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_eligible_patients AS
SELECT person_id
FROM temp_prostate_psa10_patients
WHERE person_id NOT IN (SELECT person_id FROM temp_patients_with_other_cancer)
  AND person_id NOT IN (SELECT person_id FROM temp_patients_with_parp)
""")

# === 12. First platinum drug exposure per patient (one row per patient) ===
#    Picks the earliest platinum exposure overall. Patients on both carboplatin
#    and cisplatin get a single row for whichever came first, preventing
#    duplicate lab rows in the downstream LEFT JOIN.
spark.sql("""
CREATE OR REPLACE TEMP VIEW first_platinum_exposure AS
WITH ranked AS (
  SELECT
    de.person_id,
    de.drug_concept_id,
    de.drug_exposure_start_date,
    ROW_NUMBER() OVER (
      PARTITION BY de.person_id
      ORDER BY de.drug_exposure_start_date, de.drug_concept_id
    ) AS rn
  FROM dn_drug_exposure_20251219 de
  WHERE de.drug_concept_id IN (SELECT drug_concept_id FROM temp_platinum_drugs)
)
SELECT person_id, drug_concept_id,
       drug_exposure_start_date AS first_drug_exposure_start_date
FROM ranked
WHERE rn = 1
""")

# === 13. Combine PSA measurements with first drug info ===
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW prostate_lab_longitudinal AS
SELECT
  p.person_id,
  f.prostate_cancer_diagnosis_date,

  -- ==== Demographics ====
  CASE
    WHEN per.year_of_birth IS NOT NULL THEN
      make_date(
        per.year_of_birth,
        COALESCE(per.month_of_birth, 6),
        COALESCE(per.day_of_birth, 15)
      )
    ELSE NULL
  END AS date_of_birth,
  per.gender_concept_id,
  gc.concept_name  AS gender,
  per.race_concept_id,
  rc.concept_name  AS race,
  per.ethnicity_concept_id,
  ec.concept_name  AS ethnicity,
  FLOOR(
    DATEDIFF(
      f.prostate_cancer_diagnosis_date,
      CASE
        WHEN per.year_of_birth IS NOT NULL THEN
          make_date(
            per.year_of_birth,
            COALESCE(per.month_of_birth, 6),
            COALESCE(per.day_of_birth, 15)
          )
        ELSE NULL
      END
    ) / 365.25
  ) AS age_at_diagnosis,

  -- ==== Lab info ====
  m.measurement_concept_id,
  c.concept_name AS lab_name,
  m.measurement_date,
  m.value_as_number AS lab_value,
  m.unit_concept_id,
  cu.concept_name AS lab_unit_name,

  -- ==== Drug info ====
  fe.drug_concept_id,
  d.concept_name AS drug_name,
  dp.drug_type,
  fe.first_drug_exposure_start_date AS drug_initiation_date

FROM temp_eligible_patients p
JOIN first_prostate_diagnosis f
  ON p.person_id = f.person_id
JOIN dn_person_20251219 per
  ON p.person_id = per.person_id

-- Demographics concept lookups
LEFT JOIN {CONCEPT_TABLE} gc
  ON per.gender_concept_id = gc.concept_id
LEFT JOIN {CONCEPT_TABLE} rc
  ON per.race_concept_id = rc.concept_id
LEFT JOIN {CONCEPT_TABLE} ec
  ON per.ethnicity_concept_id = ec.concept_id

-- Labs
JOIN dn_measurement_20251219 m
  ON p.person_id = m.person_id
  AND m.measurement_concept_id IN (
    3013603, 3002131, 3034548, 4272032, 3052038,
    3006923, 3024561, 3020509, 3035995, 3009306, 3013721, 3027597, 3024128, 3036277, 3020891,
    3025315, 3020460, 3006906, 3037551, 3022914, 3015632, 3003785, 3014576, 3007220, 3016723,
    3012888, 3019897, 3013707, 3020416, 3027970, 3004501, 3027018, 3023314, 3000963, 3022217,
    3016436, 3000905, 3004327, 3012030, 3009744, 3023599, 3033575, 3013650, 3024929,
    3023103, 3020630, 3034426, 3024171, 3019550, 3004249, 3008893, 3013682, 3013466,
    3000285, 3000483, 3004295, 3005456, 3018572,
    3004119, 3027484, 3009542, 3003338, 3024731, 3035941, 3002385, 3010813,
    3019198, 3017732, 3001604, 3007461, 3026361, 3015183, 3010156, 3022250, 3005225,
    3018677, 3005755, 3037081, 3010140, 3002417, 3032080, 3049555, 3021886,
    3013762, 3023540, 3003215, 3017501, 3027651, 3009932, 3034107,
    3019909, 3034976, 3001657, 3046900, 3048275, 3028286, 40763912, 3046948,
    3011424, 3014053, 3032986, 3013826, 3027219, 3051825, 3035285, 3041354,
    3030597, 3015377, 3019170, 3019762, 3024675, 3019171, 3023166,
    3040891, 3042292, 40771525, 3016628, 3037950,
    3013429, 3006315, 3028615, 3013115, 3009201, 3008598, 3016407
  )
LEFT JOIN {CONCEPT_TABLE} c
  ON m.measurement_concept_id = c.concept_id
LEFT JOIN {CONCEPT_TABLE} cu
  ON m.unit_concept_id = cu.concept_id

-- Drugs
LEFT JOIN first_platinum_exposure fe
  ON p.person_id = fe.person_id
LEFT JOIN {CONCEPT_TABLE} d
  ON fe.drug_concept_id = d.concept_id
LEFT JOIN temp_platinum_drugs dp
  ON fe.drug_concept_id = dp.drug_concept_id
""")

# === 14. Last contact / follow-up per patient ===
#     Derived from latest event across all denormalized tables + death.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_followup AS
WITH cohort_ids AS (
  SELECT person_id FROM temp_eligible_patients
),
last_clinical_event AS (
  SELECT person_id, MAX(event_date) AS last_event_date
  FROM (
    SELECT person_id, condition_start_date     AS event_date FROM dn_condition_occurrence_20251219 WHERE person_id IN (SELECT person_id FROM cohort_ids)
    UNION ALL
    SELECT person_id, measurement_date         AS event_date FROM dn_measurement_20251219          WHERE person_id IN (SELECT person_id FROM cohort_ids)
    UNION ALL
    SELECT person_id, drug_exposure_start_date AS event_date FROM dn_drug_exposure_20251219        WHERE person_id IN (SELECT person_id FROM cohort_ids)
    UNION ALL
    SELECT person_id, procedure_date           AS event_date FROM dn_procedure_occurrence_20251219 WHERE person_id IN (SELECT person_id FROM cohort_ids)
    UNION ALL
    SELECT person_id, visit_start_date         AS event_date FROM dn_visit_occurrence_20251219     WHERE person_id IN (SELECT person_id FROM cohort_ids)
  ) events
  GROUP BY person_id
)
SELECT
    lce.person_id,
    lce.last_event_date,
    d.death_date,
    COALESCE(d.death_date, lce.last_event_date) AS last_followup_date,
    CASE WHEN d.death_date IS NOT NULL THEN 1 ELSE 0 END AS is_deceased
FROM last_clinical_event lce
LEFT JOIN dn_death_20251219 d
    ON lce.person_id = d.person_id
""")

# === 15. Merge longitudinal labs with follow-up info ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW prostate_lab_longitudinal_with_followup AS
SELECT
  l.*,
  f.last_followup_date,
  f.death_date,
  f.is_deceased
FROM prostate_lab_longitudinal l
LEFT JOIN temp_followup f ON l.person_id = f.person_id
""")

# === 16. Materialize ===
final_df = spark.table("prostate_lab_longitudinal_with_followup")

# === 16b. Remap variant measurement concept IDs to canonical IDs ===
# This merges alternate LOINC codings into a single concept per lab type,
# so that downstream unit filtering, conversions, and ranges apply uniformly.
from pyspark.sql import functions as F, types as T

concept_remap = {
    # PSA variants
    3002131: 3013603, 3034548: 3013603, 4272032: 3013603, 3052038: 3013603,
    # Chemistry variants (Blood -> Serum/Plasma)
    3000285: 3019550, 3000483: 3004501, 3004295: 3013682,
    3005456: 3023103, 3018572: 3014576, 3010140: 3015632,
    # Protein/enzyme method variants
    3004119: 3000963, 3027484: 3000963, 3005755: 3006923,
    3037081: 3013721, 3005225: 3016436, 3022250: 3016436, 3010156: 3020460,
    3018677: 3013466,  # aPTT in PPP -> aPTT in Blood
    3021886: 3027970,
    # CBC variants (generic -> automated)
    3009542: 3023314, 3003338: 3009744, 3024731: 3023599,
    3035941: 3012030, 3002385: 3019897, 3010813: 3000905,
    3019198: 3004327, 3017732: 3013650, 3001604: 3033575,
    3007461: 3024929, 3026361: 3020416,
    3006315: 3013429,   # Basophils generic -> automated
    3013115: 3028615,   # Eosinophils generic -> automated
    # Other variants
    3015183: 3013707, 3002417: 3034426, 3032080: 3022217,
    3049555: 3008893, 3013762: 3025315, 3023540: 3036277,
    # Manual count -> automated (CBC differentials)
    3003215: 3004327, 3017501: 3013650, 3027651: 3013429,
    3009932: 3028615, 3034107: 3033575,
    # Hematocrit variants -> automated
    3019909: 3023314, 3034976: 3023314,
    # WBC corrected -> WBC
    3001657: 3000905, 3046900: 3000905,
    # Hemoglobin pre-phlebotomy -> Hemoglobin
    3048275: 3000963,
    # Albumin method variants -> Albumin
    3028286: 3024561, 40763912: 3024561,
    # A/G ratio electrophoresis -> A/G ratio
    3046948: 3020509,
    # Glucose variants -> Glucose
    3011424: 3004501, 3014053: 3004501, 3032986: 3004501, 3013826: 3004501,
    # Chemistry specimen variants (venous/blood -> serum/plasma)
    3027219: 3013682, 3051825: 3016723, 3035285: 3014576, 3041354: 3023103,
    # Calcium variants -> Calcium
    3030597: 3006906, 3015377: 3006906,
    # TSH sensitive variants -> TSH
    3019170: 3009201, 3019762: 3009201,
    # Free T4 dialysis -> Free T4
    3024675: 3008598,
    # Vitals variants (stated, resting, etc.)
    3019171: 3036277, 3023166: 3025315,
    3040891: 3027018, 3042292: 3027018, 40771525: 3027018,
    # Fibrinogen Ag variants -> Fibrinogen
    3016628: 3016407, 3037950: 3016407,
}

remap_df = F.broadcast(
    spark.createDataFrame(
        [(k, v) for k, v in concept_remap.items()],
        schema=["old_concept_id", "new_concept_id"],
    )
)
final_df = (
    final_df
    .join(remap_df, final_df.measurement_concept_id == remap_df.old_concept_id, "left")
    .withColumn("measurement_concept_id",
                F.coalesce(F.col("new_concept_id"), F.col("measurement_concept_id")))
    .drop("old_concept_id", "new_concept_id")
)

# === 17. Limit allowed measurement, unit combinations ===
allowed_unit_combinations = [
    (3006923, 0), (3006923, 8645), (3006923, 8718), (3006923, 8923), (3006923, 8985), (3006923, 9254),
    (3024561, 0), (3024561, 8636), (3024561, 8713), (3024561, 8840),
    (3020509, 0), (3020509, 8523), (3020509, 8596),
    (3035995, 0), (3035995, 8645), (3035995, 8923), (3035995, 9254),
    (3009306, 0), (3009306, 8842),
    (3013721, 0), (3013721, 8645), (3013721, 8923), (3013721, 9254),
    (3027597, 0), (3027597, 8840),
    (3024128, 0), (3024128, 8840), (3024128, 8749),
    (3036277, 9307), (3036277, 9306), (3036277, 9330), (3036277, 8582),
    (3020891, 9289), (3020891, 586323), (3020891, 0),
    (3025315, 0), (3025315, 8739), (3025315, 9529), (3025315, 9346), (3025315, 9373),
    (3020460, 0), (3020460, 8751), (3020460, 8840),
    (3006906, 0), (3006906, 8751), (3006906, 8840), (3006906, 8713), (3006906, 8753), (3006906, 9557),
    (3037551, 0), (3037551, 8763),
    (3022914, 8763), (3022914, 0),
    (3015632, 0), (3015632, 8736), (3015632, 8753), (3015632, 9557),
    (3003785, 0), (3003785, 8842),
    (3014576, 0), (3014576, 8753), (3014576, 9557),
    (3007220, 0), (3007220, 8645), (3007220, 8923),
    (3016723, 0), (3016723, 8751), (3016723, 8840), (3016723, 8749),
    (3012888, 8876), (3012888, 0),
    (3019897, 0), (3019897, 8554),
    (3013707, 0), (3013707, 8752),
    (3020416, 0), (3020416, 8734), (3020416, 8815), (3020416, 8816),
    (3020416, 8848), (3020416, 8931), (3020416, 9442),
    (3020416, 8784), (3020416, 8785),
    (3027970, 8713), (3027970, 0), (3027970, 8840),
    (3004501, 0), (3004501, 8713), (3004501, 8840), (3004501, 8751), (3004501, 8753),
    (3027018, 8541), (3027018, 0), (3027018, 8483),
    (3023314, 0), (3023314, 8554),
    (3000963, 0), (3000963, 8636), (3000963, 8713), (3000963, 8840),
    (3022217, 0), (3022217, 8523),
    (3016436, 0), (3016436, 8645), (3016436, 8923),
    (3016436, 8554),  # LDH percent (from remapped 3022250)
    (3000905, 0), (3000905, 8848), (3000905, 8961), (3000905, 9435),
    (3000905, 9436), (3000905, 9444), (3000905, 9446),
    (3000905, 8784), (3000905, 8785),
    (3004327, 0), (3004327, 8647), (3004327, 8784), (3004327, 8848),
    (3004327, 8961), (3004327, 9444),
    (3012030, 0), (3012030, 8564),
    (3009744, 0), (3009744, 8713),
    (3023599, 0), (3023599, 8583),
    (3033575, 0), (3033575, 8784), (3033575, 8848), (3033575, 8961),
    (3033575, 9444),
    (3013650, 0), (3013650, 8784), (3013650, 8848), (3013650, 8961),
    (3013650, 9444), (3013650, 9665),
    (3024929, 0), (3024929, 8815), (3024929, 8848), (3024929, 8931),
    (3024929, 8961), (3024929, 9435), (3024929, 9436), (3024929, 9444),
    (3024929, 9446), (3024929, 8784), (3024929, 8785),
    (3023103, 0), (3023103, 8736), (3023103, 8753), (3023103, 9557),
    (3013603, 0), (3013603, 8842),
    (3020630, 0), (3020630, 8636), (3020630, 8713), (3020630, 8840),
    (3034426, 0), (3034426, 8555),
    (3024171, 8541), (3024171, 0), (3024171, 8483),
    (3019550, 0), (3019550, 8753), (3019550, 9557),
    (3004249, 8876), (3004249, 0),
    (3008893, 0), (3008893, 8817),
    (3013682, 0), (3013682, 8817), (3013682, 8840), (3013682, 8753),
    (3013466, 0), (3013466, 8555),
    # Basophils (10^3/uL canonical)
    (3013429, 0), (3013429, 8784), (3013429, 8848), (3013429, 8961), (3013429, 9444),
    # Eosinophils (10^3/uL canonical)
    (3028615, 0), (3028615, 8647), (3028615, 8784), (3028615, 8848), (3028615, 8961), (3028615, 9444),
    # TSH (mIU/L canonical)
    (3009201, 0), (3009201, 9040), (3009201, 9093), (3009201, 8985), (3009201, 44777578),
    (3009201, 8860), (3009201, 8719), (3009201, 9550), (3009201, 44777583),
    # Free T4 (ng/dL canonical)
    (3008598, 0), (3008598, 8817), (3008598, 8729), (3008598, 8845),
    # Fibrinogen (mg/dL canonical)
    (3016407, 0), (3016407, 8840),
    # ESR variant unit (mm -> mm/h, remapped from 3015183)
    (3013707, 8588),
    # Globulin variant unit (g/dL calculated, remapped from 3021886)
    (3027970, 8950)
]

# Cast IDs to long to match schema consistently
df_cast = (
    final_df
    .withColumn("measurement_concept_id", F.col("measurement_concept_id").cast("long"))
    .withColumn("unit_concept_id", F.col("unit_concept_id").cast("long"))
)

# Turn allowed combinations into a small DataFrame and join
allowed_df = F.broadcast(spark.createDataFrame(
    allowed_unit_combinations,
    schema=T.StructType([
        T.StructField("measurement_concept_id", T.LongType(), False),
        T.StructField("unit_concept_id", T.LongType(), False),
    ])
))

final_df_w_units_filtered = (
    df_cast
    .join(allowed_df, on=["measurement_concept_id", "unit_concept_id"], how="inner")
).cache()

# === 18. Perform required unit conversions on subset of labs ===
# Lookup table: (concept_id, from_unit, factor, to_unit)
# factor=None for special conversions handled separately (temperature F->C)
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, LongType

_conversion_schema = StructType([
    StructField("conv_concept_id", LongType()),
    StructField("conv_from_unit", LongType()),
    StructField("conv_factor", DoubleType()),
    StructField("conv_to_unit", LongType()),
])

_conversion_data = [
    # ALT -> U/L (8645)
    (3006923, 8985, 1000.0, 8645),       # U/mL -> U/L
    (3006923, 8923, 1.0, 8645),           # IU/L -> U/L
    (3006923, 9254, 1.0, 8645),           # /L -> U/L
    # AST -> U/L (8645)
    (3013721, 8923, 1.0, 8645),
    (3013721, 9254, 1.0, 8645),
    # ALP -> U/L (8645)
    (3035995, 8923, 1.0, 8645),
    (3035995, 9254, 1.0, 8645),
    # CK -> U/L (8645)
    (3007220, 8923, 1.0, 8645),
    # LDH -> U/L (8645)
    (3016436, 8923, 1.0, 8645),
    # Albumin -> g/dL (8713)
    (3024561, 8636, 0.1, 8713),           # g/L -> g/dL
    (3024561, 8840, 0.001, 8713),         # mg/dL -> g/dL
    # Height -> cm (8582)
    (3036277, 9307, 30.48, 8582),         # feet -> cm
    (3036277, 9306, 30.48, 8582),         # feet (international) -> cm
    (3036277, 9330, 2.54, 8582),          # inch -> cm
    # Temperature F->C handled separately (affine, not multiplicative)
    # Body weight -> kg (9529)
    (3025315, 8739, 0.45359237, 9529),    # lbs -> kg
    (3025315, 9373, 0.028349523, 9529),   # oz -> kg
    (3025315, 9346, 0.45359237, 9529),    # lbs (avoirdupois) -> kg
    # CRP -> mg/dL (8840)
    (3020460, 8751, 0.1, 8840),
    # Calcium -> mg/dL (8840)
    (3006906, 8751, 0.1, 8840),           # mg/L -> mg/dL
    (3006906, 8713, 1000.0, 8840),        # g/dL -> mg/dL
    (3006906, 8753, 4.008, 8840),         # mmol/L -> mg/dL
    (3006906, 9557, 2.004, 8840),         # mEq/L -> mg/dL (divalent)
    # CO2 -> mmol/L (8753)
    (3015632, 8736, 0.000001, 8753),      # nmol/L -> mmol/L
    (3015632, 9557, 1.0, 8753),           # mEq/L -> mmol/L (monovalent)
    # Creatinine -> mg/dL (8840)
    (3016723, 8751, 0.1, 8840),           # mg/L -> mg/dL
    (3016723, 8749, 0.01131, 8840),       # umol/L -> mg/dL
    # Erythrocytes -> million/uL (8815)
    (3020416, 8848, 0.001, 8815),         # thousand/uL -> million/uL
    (3020416, 9442, 0.000001, 8815),      # million/L -> million/uL
    (3020416, 8931, 1.0, 8815),           # million/mm^3 -> million/uL
    (3020416, 8784, 0.000001, 8815),      # cells/uL -> million/uL
    (3020416, 8785, 0.000001, 8815),      # /mm^3 -> million/uL
    (3020416, 8816, 0.001, 8815),         # million/mL -> million/uL
    (3020416, 8734, 1.0, 8815),           # trillion/L -> million/uL
    # Globulin -> g/dL (8713)
    (3027970, 8840, 0.001, 8713),         # mg/dL -> g/dL
    (3027970, 8950, 1.0, 8713),           # g/dL calculated -> g/dL
    # Glucose -> mg/dL (8840)
    (3004501, 8713, 1000.0, 8840),        # g/dL -> mg/dL
    (3004501, 8751, 0.1, 8840),           # mg/L -> mg/dL
    (3004501, 8753, 18.016, 8840),        # mmol/L -> mg/dL
    # Hemoglobin -> g/dL (8713)
    (3000963, 8636, 0.1, 8713),           # g/L -> g/dL
    (3000963, 8840, 0.001, 8713),         # mg/dL -> g/dL
    # Leukocytes -> thousand/uL (8848)
    (3000905, 9435, 0.000001, 8848),      # thousand/L -> thousand/uL
    (3000905, 9436, 0.001, 8848),         # thousand/mL -> thousand/uL
    (3000905, 9446, 1000000.0, 8848),     # billion/uL -> thousand/uL
    (3000905, 8961, 1.0, 8848),           # thousand/mm^3 -> thousand/uL
    (3000905, 8784, 0.001, 8848),         # cells/uL -> 10^3/uL
    (3000905, 8785, 0.001, 8848),         # /mm^3 -> 10^3/uL
    (3000905, 9444, 0.001, 8848),         # billion/L -> 10^3/uL
    # Lymphocytes -> thousand/uL (8848)
    (3004327, 8647, 0.001, 8848),
    (3004327, 8784, 0.001, 8848),
    (3004327, 8961, 1.0, 8848),
    (3004327, 9444, 0.001, 8848),
    # Monocytes -> thousand/uL (8848)
    (3033575, 8784, 0.001, 8848),
    (3033575, 8961, 1.0, 8848),
    (3033575, 9444, 0.001, 8848),
    # Neutrophils -> thousand/uL (8848)
    (3013650, 8784, 0.001, 8848),
    (3013650, 8961, 1.0, 8848),
    (3013650, 9444, 0.001, 8848),
    (3013650, 9665, 0.001, 8848),
    # Basophils -> thousand/uL (8848)
    (3013429, 8784, 0.001, 8848),
    (3013429, 9444, 0.001, 8848),
    (3013429, 8961, 1.0, 8848),
    # Eosinophils -> thousand/uL (8848)
    (3028615, 8784, 0.001, 8848),
    (3028615, 8647, 0.001, 8848),
    (3028615, 9444, 0.001, 8848),
    (3028615, 8961, 1.0, 8848),
    # Platelets -> thousand/uL (8848)
    (3024929, 8931, 1000.0, 8848),        # million/mm^3 -> thousand/uL
    (3024929, 8815, 1000.0, 8848),        # million/uL -> thousand/uL
    (3024929, 9435, 0.000001, 8848),
    (3024929, 9446, 1000000.0, 8848),
    (3024929, 8961, 1.0, 8848),
    (3024929, 8784, 0.001, 8848),
    (3024929, 8785, 0.001, 8848),
    (3024929, 9436, 0.001, 8848),
    (3024929, 9444, 0.001, 8848),
    # Potassium -> mmol/L (8753)
    (3023103, 8736, 0.000001, 8753),
    (3023103, 9557, 1.0, 8753),
    # Total protein -> g/dL (8713)
    (3020630, 8636, 0.1, 8713),
    (3020630, 8840, 0.001, 8713),
    # BUN -> mg/dL (8840)
    (3013682, 8817, 0.000001, 8840),      # ng/dL -> mg/dL
    (3013682, 8753, 2.8013, 8840),        # mmol/L -> mg/dL
    # Bilirubin total -> mg/dL (8840)
    (3024128, 8749, 0.05847, 8840),       # umol/L -> mg/dL
    # TSH -> mIU/L (9040)
    (3009201, 9093, 1.0, 9040),           # uIU/mL -> mIU/L (1:1)
    (3009201, 8985, 1000000.0, 9040),     # IU/mL -> mIU/L
    (3009201, 8719, 1000.0, 9040),        # mU/mL -> mIU/L
    (3009201, 8860, 1.0, 9040),           # uU/mL -> mIU/L (1:1)
    (3009201, 9550, 1000.0, 9040),        # mIU/mL -> mIU/L
    (3009201, 44777578, 1.0, 9040),       # mU/L -> mIU/L (1:1)
    (3009201, 44777583, 0.001, 9040),     # uIU/L -> mIU/L
    # Free T4 -> ng/dL (8817)
    (3008598, 8729, 1.0/12.871, 8817),    # pmol/L -> ng/dL
    (3008598, 8845, 0.1, 8817),           # pg/mL -> ng/dL
    # Chloride -> mmol/L (8753)
    (3014576, 9557, 1.0, 8753),
    # Sodium -> mmol/L (8753)
    (3019550, 9557, 1.0, 8753),
]

conversion_df = F.broadcast(
    spark.createDataFrame(_conversion_data, schema=_conversion_schema)
)

final_df_w_conversions = (
    final_df_w_units_filtered
    .join(
        conversion_df,
        (F.col("measurement_concept_id") == F.col("conv_concept_id"))
        & (F.col("unit_concept_id") == F.col("conv_from_unit")),
        "left",
    )
    .withColumn(
        "lab_values_converted",
        # Standard conversions via factor; temperature F->C is special (affine)
        F.when(
            (F.col("measurement_concept_id") == 3020891) & (F.col("unit_concept_id") == 9289),
            (F.col("lab_value") - 32) * 5.0 / 9.0
        ).when(
            F.col("conv_factor").isNotNull(),
            F.col("lab_value") * F.col("conv_factor")
        ).otherwise(F.lit(None))
    )
    .withColumn(
        "unit_converted_id",
        F.when(
            # Temperature F->C special case
            (F.col("measurement_concept_id") == 3020891) & (F.col("unit_concept_id") == 9289),
            F.lit(586323)
        ).when(
            F.col("conv_to_unit").isNotNull(),
            F.col("conv_to_unit")
        ).otherwise(F.col("unit_concept_id"))
    )
    .drop("conv_concept_id", "conv_from_unit", "conv_factor", "conv_to_unit")
    # single "final" lab value: converted where available, otherwise original
    .withColumn(
        "lab_value_final",
        F.when(F.col("lab_values_converted").isNotNull(), F.col("lab_values_converted"))
         .otherwise(F.col("lab_value"))
    )
)

# Look up the human-readable name for the converted unit
unit_converted_names = (
    spark.table(CONCEPT_TABLE)
    .select(
        F.col("concept_id").alias("unit_converted_id"),
        F.col("concept_name").alias("unit_converted_name"),
    )
)
final_df_w_conversions = (
    final_df_w_conversions
    .join(unit_converted_names, on="unit_converted_id", how="left")
)

# === 18b. Exclude null/NaN and non-physiologic lab values ===
# 1) Remove rows where lab_value_final is null or NaN
final_df_w_conversions = final_df_w_conversions.filter(
    F.col("lab_value_final").isNotNull() & ~F.isnan(F.col("lab_value_final"))
)

# 2) Remove values outside physiologic bounds (per measurement concept, after unit conversion)
#    Ranges are intentionally generous to catch data-entry errors, not enforce reference ranges.
physiologic_ranges = [
    # (measurement_concept_id, min_value, max_value)
    # Enzymes (U/L)
    (3006923, 0, 50000),      # ALT (Alanine aminotransferase)
    (3035995, 0, 50000),      # ALP (Alkaline phosphatase)
    (3013721, 0, 50000),      # AST (Aspartate aminotransferase)
    (3007220, 0, 50000),      # CK (Creatine kinase)
    (3016436, 0, 50000),      # LDH (Lactate dehydrogenase)
    # Proteins & ratios
    (3024561, 0.5, 7.0),      # Albumin (g/dL)
    (3020509, 0.1, 5.0),      # Albumin/Globulin ratio
    (3027970, 0.5, 10.0),     # Globulin (g/dL)
    (3020630, 2.0, 15.0),     # Total Protein (g/dL)
    (3000963, 1.0, 25.0),     # Hemoglobin (g/dL)
    # Bilirubin (mg/dL)
    (3027597, 0, 50),         # Direct Bilirubin
    (3024128, 0, 50),         # Total Bilirubin
    # Electrolytes (mmol/L)
    (3014576, 50, 150),       # Chloride
    (3015632, 5, 60),         # CO2 / Bicarbonate
    (3023103, 1, 10),         # Potassium
    (3019550, 100, 200),      # Sodium
    # Tumor markers
    (3037551, 0, 100000),     # CA 125 (U/mL)
    (3022914, 0, 100000),     # CA 19-9 (U/mL)
    (3009306, 0, 100000),     # AFP (ng/mL)
    (3003785, 0, 100000),     # CEA (ng/mL)
    # Other chemistry (mg/dL)
    (3020460, 0, 100),        # CRP
    (3006906, 2, 20),         # Calcium
    (3016723, 0, 30),         # Creatinine
    (3004501, 5, 1500),       # Glucose
    (3013682, 1, 200),        # BUN (Urea nitrogen)
    # CBC - absolute counts (10^3/uL or million/uL)
    (3020416, 0.5, 10),       # RBC (million/uL)
    (3000905, 0, 500),        # WBC (10^3/uL)
    (3004327, 0, 50),         # Lymphocytes abs (10^3/uL)
    (3033575, 0, 30),         # Monocytes abs (10^3/uL)
    (3013650, 0, 100),        # Neutrophils abs (10^3/uL)
    (3024929, 5, 1500),       # Platelets (10^3/uL)
    # CBC - indices
    (3012030, 10, 60),        # MCH (pg)
    (3009744, 20, 45),        # MCHC (g/dL)
    (3023599, 40, 150),       # MCV (fL)
    (3019897, 5, 30),         # RDW (%)
    (3023314, 5, 75),         # Hematocrit (%)
    # Coagulation
    (3022217, 0.1, 20),       # INR
    (3034426, 0, 200),        # PT (seconds)
    (3013466, 0, 200),        # aPTT (seconds)
    # Vitals
    (3036277, 50, 275),       # Height (cm)
    (3020891, 25, 45),        # Body Temperature (°C)
    (3025315, 1, 500),        # Body Weight (kg)
    (3012888, 10, 200),       # Diastolic BP (mmHg)
    (3004249, 30, 300),       # Systolic BP (mmHg)
    (3027018, 20, 250),       # Heart Rate (beats/min)
    (3024171, 2, 60),         # Respiratory Rate (breaths/min)
    (3008893, 0, 2000),       # Testosterone (ng/dL)
    (3013707, 0, 200),        # ESR (mm/hr)
    # New CBC differentials (10^3/uL)
    (3013429, 0, 10),         # Basophils abs (10^3/uL)
    (3028615, 0, 30),         # Eosinophils abs (10^3/uL)
    # Thyroid
    (3009201, 0, 500),        # TSH (mIU/L)
    (3008598, 0, 15),         # Free T4 (ng/dL)
    # Coagulation
    (3016407, 0, 2000),       # Fibrinogen (mg/dL)
    # PSA (ng/mL) — only canonical ID needed; 3002131/3034548 are remapped to 3013603
    (3013603, 0, 10000),      # PSA
    # The remaining 22 PSA variant IDs have no measurement data in the dataset.
    # Pre-remap IDs (3002131, 3034548, 4272032, 3052038) are dead code here since they become 3013603.
    # (3037249, 0, 10000),    # (3037774, 0, 10000),    # (3002178, 0, 10000),
    # (44811982, 0, 10000),   # (40762314, 0, 10000),   # (3002131, 0, 10000),
    # (3034548, 0, 10000),    # (3038011, 0, 10000),    # (3032915, 0, 10000),
    # (42529229, 0, 10000),   # (40484164, 0, 10000),   # (715972, 0, 10000),
    # (35917418, 0, 10000),   # (40762312, 0, 10000),   # (40480170, 0, 10000),
    # (3007273, 0, 10000),    # (44811980, 0, 10000),   # (35918474, 0, 10000),
    # (715971, 0, 10000),     # (44811981, 0, 10000),   # (4272032, 0, 10000),
    # (4194418, 0, 10000),    # (4215704, 0, 10000),    # (40762321, 0, 10000),
    # (3052038, 0, 10000),    # (44793131, 0, 10000),
]

ranges_df = F.broadcast(spark.createDataFrame(
    physiologic_ranges,
    schema=T.StructType([
        T.StructField("measurement_concept_id", T.LongType(), False),
        T.StructField("physio_min", T.DoubleType(), False),
        T.StructField("physio_max", T.DoubleType(), False),
    ])
))

final_df_w_conversions = (
    final_df_w_conversions
    .join(ranges_df, on="measurement_concept_id", how="left")
    .filter(
        F.col("physio_min").isNull()
        | (
            (F.col("lab_value_final") >= F.col("physio_min"))
            & (F.col("lab_value_final") <= F.col("physio_max"))
        )
    )
    .drop("physio_min", "physio_max")
)

# === 19. Compute days relative to events (positive = days after event) ===
df_to_upload = (
    final_df_w_conversions
    # --- Lab-level relative timing ---
    .withColumn(
        "days_relative_to_diagnosis",
        F.datediff(F.col("measurement_date"), F.col("prostate_cancer_diagnosis_date"))
    )
    .withColumn(
        "days_relative_to_platinum_chemo_start",
        F.datediff(F.col("measurement_date"), F.col("drug_initiation_date"))
    )
    .withColumn(
        "days_relative_to_last_followup",
        F.datediff(F.col("measurement_date"), F.col("last_followup_date"))
    )
    # --- Patient-level time-to-event variables ---
    .withColumn(
        "event_platinum",
        F.when(F.col("drug_initiation_date").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )
    .withColumn(
        "time_to_platinum_or_censor",
        F.datediff(
            F.coalesce(F.col("drug_initiation_date"), F.col("last_followup_date")),
            F.col("prostate_cancer_diagnosis_date")
        )
    )
    .withColumn(
        "time_to_death_or_censor",
        F.datediff(
            F.coalesce(F.col("death_date"), F.col("last_followup_date")),
            F.col("prostate_cancer_diagnosis_date")
        )
    )
)

from pyspark.sql import functions as F, types as T

# === 20. Enforce canonical types + schema assertions ===

# ---- Define column groups ----
int_cols = [
    "person_id",
    "gender_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
    "measurement_concept_id",
    "unit_concept_id",
    "unit_converted_id",
    "drug_concept_id",
    "age_at_diagnosis",
    "is_deceased",
    "event_platinum",
    "time_to_platinum_or_censor",
    "time_to_death_or_censor",
    "days_relative_to_diagnosis",
    "days_relative_to_platinum_chemo_start",
    "days_relative_to_last_followup",
]

float_cols = [
    "lab_value",
    "lab_values_converted",
    "lab_value_final",
]

date_cols = [
    "date_of_birth",
    "prostate_cancer_diagnosis_date",
    "measurement_date",
    "drug_initiation_date",
    "last_followup_date",
    "death_date",
]

string_cols = [
    "gender",
    "race",
    "ethnicity",
    "lab_name",
    "lab_unit_name",
    "unit_converted_name",
    "drug_name",
    "drug_type",
]

# ---- 1) Canonical casts ----
# Canonical integer type: BIGINT (LongType)
for col in int_cols:
    if col in df_to_upload.columns:
        df_to_upload = df_to_upload.withColumn(col, F.col(col).cast("bigint"))

# Canonical float type: DOUBLE (DoubleType)
for col in float_cols:
    if col in df_to_upload.columns:
        df_to_upload = df_to_upload.withColumn(col, F.col(col).cast("double"))

# (Optional but nice) enforce DATE for date columns
for col in date_cols:
    if col in df_to_upload.columns:
        df_to_upload = df_to_upload.withColumn(col, F.col(col).cast("date"))

# (Optional) enforce STRING for string columns
for col in string_cols:
    if col in df_to_upload.columns:
        df_to_upload = df_to_upload.withColumn(col, F.col(col).cast("string"))

# ---- 2) Column presence assertion ----
expected_columns = [
    # IDs / keys
    "person_id",
    "prostate_cancer_diagnosis_date",

    # Demographics
    "date_of_birth",
    "gender_concept_id",
    "gender",
    "race_concept_id",
    "race",
    "ethnicity_concept_id",
    "ethnicity",
    "age_at_diagnosis",

    # Lab info
    "measurement_concept_id",
    "lab_name",
    "measurement_date",
    "lab_value",
    "unit_concept_id",
    "lab_unit_name",
    "lab_values_converted",
    "unit_converted_id",
    "unit_converted_name",
    "lab_value_final",

    # Drug info
    "drug_concept_id",
    "drug_name",
    "drug_type",
    "drug_initiation_date",

    # Follow-up
    "last_followup_date",
    "death_date",
    "is_deceased",

    # Time-to-event variables
    "event_platinum",
    "time_to_platinum_or_censor",
    "time_to_death_or_censor",

    # Lab-level relative time variables
    "days_relative_to_diagnosis",
    "days_relative_to_platinum_chemo_start",
    "days_relative_to_last_followup",
]

actual_columns = df_to_upload.columns
missing = set(expected_columns) - set(actual_columns)
extra   = set(actual_columns) - set(expected_columns)

assert not missing, f"df_to_upload is missing columns: {sorted(missing)}"
assert not extra,   f"df_to_upload has unexpected extra columns: {sorted(extra)}"

# ---- 3) Type assertions: all int-like are BIGINT, all float-like are DOUBLE ----
schema_dict = {f.name: f.dataType for f in df_to_upload.schema.fields}

for col in int_cols:
    dt = schema_dict[col]
    assert isinstance(dt, T.LongType), (
        f"Column `{col}` expected LongType (BIGINT) but got {dt.simpleString()}"
    )

for col in float_cols:
    dt = schema_dict[col]
    assert isinstance(dt, T.DoubleType), (
        f"Column `{col}` expected DoubleType (DOUBLE) but got {dt.simpleString()}"
    )

for col in date_cols:
    dt = schema_dict[col]
    assert isinstance(dt, T.DateType), (
        f"Column `{col}` expected DateType (DATE) but got {dt.simpleString()}"
    )

for col in string_cols:
    dt = schema_dict[col]
    assert isinstance(dt, T.StringType), (
        f"Column `{col}` expected StringType but got {dt.simpleString()}"
    )

print("df_to_upload schema checks passed: all columns present and cast to canonical types.")

# ===============================================================
# Part 3: Summary Statistics
# ===============================================================

df_to_upload.cache()
df_to_upload.createOrReplaceTempView("final_cohort")
row_count = df_to_upload.count()
print(f"Cached final_cohort: {row_count:,} rows")

# Pre-compute a patient-level view (one row per person) for demographic/treatment/follow-up queries.
# This avoids repeated COUNT(DISTINCT person_id) over the full lab-level table.
patient_df = spark.sql("""
    SELECT
        person_id,
        FIRST(gender) AS gender,
        FIRST(race) AS race,
        FIRST(ethnicity) AS ethnicity,
        FIRST(age_at_diagnosis) AS age_at_diagnosis,
        FIRST(date_of_birth) AS date_of_birth,
        FIRST(prostate_cancer_diagnosis_date) AS prostate_cancer_diagnosis_date,
        FIRST(drug_concept_id) AS drug_concept_id,
        FIRST(drug_name) AS drug_name,
        FIRST(drug_type) AS drug_type,
        FIRST(drug_initiation_date) AS drug_initiation_date,
        FIRST(last_followup_date) AS last_followup_date,
        FIRST(death_date) AS death_date,
        FIRST(is_deceased) AS is_deceased,
        FIRST(event_platinum) AS event_platinum,
        FIRST(time_to_platinum_or_censor) AS time_to_platinum_or_censor,
        FIRST(time_to_death_or_censor) AS time_to_death_or_censor
    FROM final_cohort
    GROUP BY person_id
""")
patient_df.cache()
patient_df.createOrReplaceTempView("patient_cohort")
patient_count = patient_df.count()
print(f"Cached patient_cohort: {patient_count:,} patients")

# === 1. Overall cohort size ===
print("=" * 60)
print("COHORT OVERVIEW")
print("=" * 60)

spark.sql(f"""
    SELECT
        {patient_count} AS total_patients,
        {row_count} AS total_lab_records,
        (SELECT COUNT(*) FROM patient_cohort WHERE drug_concept_id IS NOT NULL) AS patients_with_platinum,
        (SELECT COUNT(*) FROM patient_cohort WHERE is_deceased = 1) AS deceased_patients,
        ROUND((SELECT COUNT(*) FROM patient_cohort WHERE is_deceased = 1) * 100.0
              / {patient_count}, 1) AS mortality_rate_pct
""").show(truncate=False)

# === 2. Demographics ===
print("=" * 60)
print("DEMOGRAPHICS")
print("=" * 60)

print("\n--- Gender ---")
spark.sql(f"""
    SELECT gender, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {patient_count}, 1) AS pct
    FROM patient_cohort
    GROUP BY gender
    ORDER BY n_patients DESC
""").show(truncate=False)

print("--- Race ---")
spark.sql(f"""
    SELECT race, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {patient_count}, 1) AS pct
    FROM patient_cohort
    GROUP BY race
    ORDER BY n_patients DESC
""").show(truncate=False)

print("--- Ethnicity ---")
spark.sql(f"""
    SELECT ethnicity, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {patient_count}, 1) AS pct
    FROM patient_cohort
    GROUP BY ethnicity
    ORDER BY n_patients DESC
""").show(truncate=False)

print("--- Age at Diagnosis ---")
spark.sql("""
    SELECT
        COUNT(*) AS n_patients,
        ROUND(AVG(age_at_diagnosis), 1) AS mean_age,
        ROUND(STDDEV(age_at_diagnosis), 1) AS std_age,
        MIN(age_at_diagnosis) AS min_age,
        PERCENTILE_APPROX(age_at_diagnosis, 0.25) AS q1_age,
        PERCENTILE_APPROX(age_at_diagnosis, 0.50) AS median_age,
        PERCENTILE_APPROX(age_at_diagnosis, 0.75) AS q3_age,
        MAX(age_at_diagnosis) AS max_age
    FROM patient_cohort
    WHERE age_at_diagnosis IS NOT NULL
""").show(truncate=False)

# === 3. Platinum treatment summary ===
print("=" * 60)
print("PLATINUM TREATMENT")
print("=" * 60)

spark.sql("""
    SELECT
        drug_type,
        drug_name,
        COUNT(*) AS n_patients,
        ROUND(AVG(days_dx_to_drug), 1) AS mean_days_dx_to_drug,
        ROUND(STDDEV(days_dx_to_drug), 1) AS std_days_dx_to_drug,
        MIN(days_dx_to_drug) AS min_days,
        PERCENTILE_APPROX(days_dx_to_drug, 0.25) AS q1_days,
        PERCENTILE_APPROX(days_dx_to_drug, 0.50) AS median_days,
        PERCENTILE_APPROX(days_dx_to_drug, 0.75) AS q3_days,
        MAX(days_dx_to_drug) AS max_days
    FROM (
        SELECT
            person_id,
            drug_type,
            drug_name,
            DATEDIFF(drug_initiation_date, prostate_cancer_diagnosis_date) AS days_dx_to_drug
        FROM patient_cohort
        WHERE drug_concept_id IS NOT NULL
    )
    GROUP BY drug_type, drug_name
    ORDER BY n_patients DESC
""").show(truncate=False)

# === 4. Lab measurement summary (converted values) ===
print("=" * 60)
print("LAB MEASUREMENTS (converted values)")
print("=" * 60)

print("--- Records per Lab Type ---")
spark.sql(f"""
    SELECT
        lab_name,
        measurement_concept_id,
        COUNT(*) AS n_records,
        ROUND(AVG(lab_value_final), 2) AS mean_value,
        ROUND(STDDEV(lab_value_final), 2) AS std_value,
        ROUND(PERCENTILE_APPROX(lab_value_final, 0.25), 2) AS q1_value,
        ROUND(PERCENTILE_APPROX(lab_value_final, 0.50), 2) AS median_value,
        ROUND(PERCENTILE_APPROX(lab_value_final, 0.75), 2) AS q3_value,
        ROUND(MIN(lab_value_final), 2) AS min_value,
        ROUND(MAX(lab_value_final), 2) AS max_value,
        SUM(CASE WHEN lab_values_converted IS NOT NULL THEN 1 ELSE 0 END) AS n_converted
    FROM final_cohort
    WHERE lab_value_final IS NOT NULL
    GROUP BY lab_name, measurement_concept_id
    ORDER BY n_records DESC
""").show(100, truncate=False)

print("--- Labs per Patient ---")
spark.sql("""
    SELECT
        ROUND(AVG(n_labs), 1) AS mean_labs_per_patient,
        PERCENTILE_APPROX(n_labs, 0.25) AS q1_labs,
        PERCENTILE_APPROX(n_labs, 0.50) AS median_labs,
        PERCENTILE_APPROX(n_labs, 0.75) AS q3_labs,
        MIN(n_labs) AS min_labs,
        MAX(n_labs) AS max_labs,
        ROUND(AVG(n_lab_types), 1) AS mean_distinct_lab_types,
        PERCENTILE_APPROX(n_lab_types, 0.25) AS q1_lab_types,
        PERCENTILE_APPROX(n_lab_types, 0.50) AS median_lab_types,
        PERCENTILE_APPROX(n_lab_types, 0.75) AS q3_lab_types,
        MIN(n_lab_types) AS min_lab_types,
        MAX(n_lab_types) AS max_lab_types
    FROM (
        SELECT
            person_id,
            COUNT(*) AS n_labs,
            COUNT(DISTINCT measurement_concept_id) AS n_lab_types
        FROM final_cohort
        GROUP BY person_id
    )
""").show(truncate=False)

# PSA measurements per patient
print("--- PSA Measurements per Patient ---")
spark.sql("""
    SELECT
        COUNT(DISTINCT person_id) AS n_patients,
        ROUND(AVG(psa_count), 1) AS mean_psa_per_patient,
        MIN(psa_count) AS min_psa,
        PERCENTILE_APPROX(psa_count, 0.25) AS q1_psa,
        PERCENTILE_APPROX(psa_count, 0.50) AS median_psa,
        PERCENTILE_APPROX(psa_count, 0.75) AS q3_psa,
        MAX(psa_count) AS max_psa
    FROM (
        SELECT person_id, COUNT(*) AS psa_count
        FROM final_cohort
        WHERE measurement_concept_id = 3013603
        GROUP BY person_id
    )
""").show(truncate=False)

# === 5. Unit conversion audit ===
print("=" * 60)
print("UNIT CONVERSION AUDIT")
print("=" * 60)

spark.sql("""
    SELECT
        lab_name,
        measurement_concept_id,
        unit_concept_id AS original_unit_id,
        lab_unit_name AS original_unit_name,
        unit_converted_id,
        unit_converted_name,
        COUNT(*) AS n_records,
        ROUND(AVG(lab_value), 2) AS mean_original,
        ROUND(AVG(lab_value_final), 2) AS mean_converted
    FROM final_cohort
    WHERE lab_values_converted IS NOT NULL
    GROUP BY lab_name, measurement_concept_id, unit_concept_id, lab_unit_name, unit_converted_id, unit_converted_name
    ORDER BY n_records DESC
""").show(100, truncate=False)

# === 6. Follow-up / survival summary ===
print("=" * 60)
print("FOLLOW-UP & SURVIVAL")
print("=" * 60)

spark.sql("""
    SELECT
        COUNT(*) AS n_patients,
        SUM(is_deceased) AS n_deceased,
        ROUND(SUM(is_deceased) * 100.0 / COUNT(*), 1) AS mortality_pct,
        ROUND(AVG(followup_days), 1) AS mean_followup_days,
        ROUND(AVG(followup_days) / 365.25, 1) AS mean_followup_years,
        ROUND(PERCENTILE_APPROX(followup_days, 0.50), 1) AS median_followup_days,
        ROUND(PERCENTILE_APPROX(followup_days, 0.50) / 365.25, 1) AS median_followup_years,
        MIN(followup_days) AS min_followup_days,
        MAX(followup_days) AS max_followup_days
    FROM (
        SELECT
            person_id,
            is_deceased,
            DATEDIFF(last_followup_date, prostate_cancer_diagnosis_date) AS followup_days
        FROM patient_cohort
        WHERE last_followup_date IS NOT NULL
    )
""").show(truncate=False)

# === 7. Time-to-event variables ===
print("=" * 60)
print("TIME-TO-EVENT VARIABLES")
print("=" * 60)

print("--- Platinum Event ---")
spark.sql(f"""
    SELECT
        event_platinum,
        COUNT(*) AS n_patients,
        ROUND(COUNT(*) * 100.0 / {patient_count}, 1) AS pct
    FROM patient_cohort
    GROUP BY event_platinum
    ORDER BY event_platinum
""").show(truncate=False)

print("--- Time to Platinum or Censor (days) ---")
spark.sql("""
    SELECT
        COUNT(*) AS n_patients,
        ROUND(AVG(time_to_platinum_or_censor), 1) AS mean_days,
        ROUND(STDDEV(time_to_platinum_or_censor), 1) AS std_days,
        MIN(time_to_platinum_or_censor) AS min_days,
        PERCENTILE_APPROX(time_to_platinum_or_censor, 0.25) AS q1_days,
        PERCENTILE_APPROX(time_to_platinum_or_censor, 0.50) AS median_days,
        PERCENTILE_APPROX(time_to_platinum_or_censor, 0.75) AS q3_days,
        MAX(time_to_platinum_or_censor) AS max_days
    FROM patient_cohort
    WHERE time_to_platinum_or_censor IS NOT NULL
""").show(truncate=False)

print("--- Time to Death or Censor (days) ---")
spark.sql("""
    SELECT
        COUNT(*) AS n_patients,
        ROUND(AVG(time_to_death_or_censor), 1) AS mean_days,
        ROUND(STDDEV(time_to_death_or_censor), 1) AS std_days,
        MIN(time_to_death_or_censor) AS min_days,
        PERCENTILE_APPROX(time_to_death_or_censor, 0.25) AS q1_days,
        PERCENTILE_APPROX(time_to_death_or_censor, 0.50) AS median_days,
        PERCENTILE_APPROX(time_to_death_or_censor, 0.75) AS q3_days,
        MAX(time_to_death_or_censor) AS max_days
    FROM patient_cohort
    WHERE time_to_death_or_censor IS NOT NULL
""").show(truncate=False)

# === 8. Null value audit ===
print("=" * 60)
print("NULL VALUE AUDIT")
print("=" * 60)

spark.sql("""
    SELECT
        COUNT(*) AS total_rows,
        SUM(CASE WHEN person_id IS NULL THEN 1 ELSE 0 END) AS null_person_id,
        SUM(CASE WHEN gender IS NULL THEN 1 ELSE 0 END) AS null_gender,
        SUM(CASE WHEN race IS NULL THEN 1 ELSE 0 END) AS null_race,
        SUM(CASE WHEN ethnicity IS NULL THEN 1 ELSE 0 END) AS null_ethnicity,
        SUM(CASE WHEN age_at_diagnosis IS NULL THEN 1 ELSE 0 END) AS null_age_at_diagnosis,
        SUM(CASE WHEN date_of_birth IS NULL THEN 1 ELSE 0 END) AS null_date_of_birth,
        SUM(CASE WHEN prostate_cancer_diagnosis_date IS NULL THEN 1 ELSE 0 END) AS null_diagnosis_date,
        SUM(CASE WHEN measurement_concept_id IS NULL THEN 1 ELSE 0 END) AS null_measurement_concept_id,
        SUM(CASE WHEN lab_name IS NULL THEN 1 ELSE 0 END) AS null_lab_name,
        SUM(CASE WHEN measurement_date IS NULL THEN 1 ELSE 0 END) AS null_measurement_date,
        SUM(CASE WHEN lab_value IS NULL THEN 1 ELSE 0 END) AS null_lab_value,
        SUM(CASE WHEN lab_value_final IS NULL THEN 1 ELSE 0 END) AS null_lab_value_final,
        SUM(CASE WHEN lab_values_converted IS NULL THEN 1 ELSE 0 END) AS null_lab_values_converted,
        SUM(CASE WHEN unit_concept_id IS NULL THEN 1 ELSE 0 END) AS null_unit_concept_id,
        SUM(CASE WHEN lab_unit_name IS NULL THEN 1 ELSE 0 END) AS null_lab_unit_name,
        SUM(CASE WHEN unit_converted_name IS NULL THEN 1 ELSE 0 END) AS null_unit_converted_name,
        SUM(CASE WHEN drug_concept_id IS NULL THEN 1 ELSE 0 END) AS null_drug_concept_id,
        SUM(CASE WHEN last_followup_date IS NULL THEN 1 ELSE 0 END) AS null_last_followup_date,
        SUM(CASE WHEN death_date IS NULL THEN 1 ELSE 0 END) AS null_death_date,
        SUM(CASE WHEN event_platinum IS NULL THEN 1 ELSE 0 END) AS null_event_platinum,
        SUM(CASE WHEN time_to_platinum_or_censor IS NULL THEN 1 ELSE 0 END) AS null_time_to_platinum_or_censor,
        SUM(CASE WHEN time_to_death_or_censor IS NULL THEN 1 ELSE 0 END) AS null_time_to_death_or_censor,
        SUM(CASE WHEN days_relative_to_diagnosis IS NULL THEN 1 ELSE 0 END) AS null_days_rel_diagnosis,
        SUM(CASE WHEN days_relative_to_platinum_chemo_start IS NULL THEN 1 ELSE 0 END) AS null_days_rel_platinum,
        SUM(CASE WHEN days_relative_to_last_followup IS NULL THEN 1 ELSE 0 END) AS null_days_rel_followup,
        SUM(CASE WHEN gender_concept_id IS NULL THEN 1 ELSE 0 END) AS null_gender_concept_id,
        SUM(CASE WHEN race_concept_id IS NULL THEN 1 ELSE 0 END) AS null_race_concept_id,
        SUM(CASE WHEN ethnicity_concept_id IS NULL THEN 1 ELSE 0 END) AS null_ethnicity_concept_id,
        SUM(CASE WHEN unit_converted_id IS NULL THEN 1 ELSE 0 END) AS null_unit_converted_id,
        SUM(CASE WHEN drug_name IS NULL THEN 1 ELSE 0 END) AS null_drug_name,
        SUM(CASE WHEN drug_type IS NULL THEN 1 ELSE 0 END) AS null_drug_type,
        SUM(CASE WHEN drug_initiation_date IS NULL THEN 1 ELSE 0 END) AS null_drug_initiation_date,
        SUM(CASE WHEN is_deceased IS NULL THEN 1 ELSE 0 END) AS null_is_deceased
    FROM final_cohort
""").show(truncate=False)
