# ===============================================================
# COMPASS Cohort Preprocessing
# ===============================================================
# Inclusion/exclusion criteria:
#   1. Prostate cancer diagnosis
#   2. Male sex (gender_concept_id = 8507)
#   3. >= 5 PSA measurements (full 27-variant concept set) after diagnosis
#   4. No other primary cancers (keeping metastatic disease, NMSC, and
#      malignant neoplasm NOS)
#   5. No PARP inhibitor exposure (olaparib, rucaparib, niraparib, talazoparib)
# ===============================================================

# ===============================================================
# Load vocabulary subset CSVs (replacing deprecated catalog)
# ===============================================================
import os
from pyspark.sql import functions as F, types as T
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, LongType
_cwd = "file:" + os.getcwd()
spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true") \
    .csv(os.path.join(_cwd, "concept_tables", "concept_subset.csv")) \
    .createOrReplaceTempView("concept")
spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true") \
    .csv(os.path.join(_cwd, "concept_tables", "concept_ancestor_subset.csv")) \
    .createOrReplaceTempView("concept_ancestor")

# ===============================================================
# Performance settings
# ===============================================================
spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

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
# Hardcoded via VALUES to avoid dependency on concept CSV (LOINC IDs were missing).
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_psa_concepts AS
SELECT concept_id FROM VALUES
  (3013603), (3002131), (3034548),
  -- The following 24 PSA variant IDs are included here so they count toward the
  -- ≥5 PSA eligibility threshold (step 8). Of these, 4272032 and 3052038 also
  -- appear in the final lab output (remapped to 3013603 in step 16b). The
  -- remaining 22 have no measurement data and are silently dropped by the
  -- allowed_unit_combinations inner join in step 17.
  (3037249), (3037774), (3002178), (44811982), (40762314), (3038011), (3032915), (42529229),
  (40484164), (715972), (35917418), (40762312), (40480170), (3007273), (44811980), (35918474),
  (715971), (44811981), (4272032), (4194418), (4215704), (40762321), (3052038), (44793131)
AS t(concept_id)
""")

# === 3. Non-prostate PRIMARY malignant neoplasm concepts (for exclusion) ===
# All descendants of "Malignant neoplastic disease" (443392) MINUS:
#   - prostate cancer descendants (4163261)
#   - secondary/metastatic neoplasm descendants (432851)
# This prevents metastatic prostate cancer (e.g. "Secondary malignant neoplasm
# of bone") from being misclassified as a separate primary malignancy.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_non_prostate_cancer_concepts AS
SELECT ca.descendant_concept_id AS concept_id
FROM concept_ancestor ca
LEFT JOIN concept_ancestor excl_prostate
  ON ca.descendant_concept_id = excl_prostate.descendant_concept_id
  AND excl_prostate.ancestor_concept_id = 4163261
LEFT JOIN concept_ancestor excl_meta
  ON ca.descendant_concept_id = excl_meta.descendant_concept_id
  AND excl_meta.ancestor_concept_id = 432851
WHERE ca.ancestor_concept_id = 443392
  AND excl_prostate.descendant_concept_id IS NULL
  AND excl_meta.descendant_concept_id IS NULL
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
SELECT np.concept_id
FROM temp_non_prostate_cancer_concepts np
LEFT JOIN temp_nmsc_concepts nmsc ON np.concept_id = nmsc.concept_id
LEFT JOIN temp_nos_malignant_neoplasm nos ON np.concept_id = nos.concept_id
WHERE nmsc.concept_id IS NULL
  AND nos.concept_id IS NULL
""")

# === 3e. PARP inhibitor drug concepts ===
# Hardcoded ingredient IDs via VALUES; descendants looked up from local concept_ancestor CSV.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_parp_ingredients AS
SELECT concept_id FROM VALUES
  (45892579),  -- Olaparib
  (1718850),   -- Rucaparib
  (1593861),   -- Niraparib
  (35201068)   -- Talazoparib
AS t(concept_id)
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_parp_drugs AS
SELECT DISTINCT descendant_concept_id AS drug_concept_id
FROM concept_ancestor
WHERE ancestor_concept_id IN (SELECT concept_id FROM temp_parp_ingredients)
""")

# === 4. Other lab measurement concepts (explicit list) ===
# Hardcoded via VALUES to avoid dependency on concept CSV (LOINC IDs were missing).
# Includes canonical concepts + variant codings that will be remapped after extraction.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_other_lab_concepts AS
SELECT concept_id FROM VALUES
  -- Canonical lab concepts
  (3006923), (3024561), (3020509), (3035995), (3009306), (3013721), (3027597), (3024128), (3036277), (3020891),
  (3025315), (3020460), (3006906), (3037551), (3022914), (3015632), (3003785), (3014576), (3007220), (3016723),
  (3012888), (3019897), (3013707), (3020416), (3027970), (3004501), (3027018), (3023314), (3000963), (3022217),
  (3016436), (3000905), (3004327), (3012030), (3009744), (3023599), (3033575), (3013650), (3024929),
  (3023103), (3020630), (3034426), (3024171), (3019550), (3004249), (3008893), (3013682), (3013466),
  -- Category 1: Variant codings of existing labs (will be remapped post-extraction)
  (4272032), (3052038),  -- PSA SNOMED/LOINC variants -> 3013603
  (3002131), (3034548),  -- PSA LOINC variants -> 3013603
  (3000285),           -- Sodium in Blood -> 3019550
  (3000483),           -- Glucose in Blood -> 3004501
  (3004295),           -- BUN in Blood -> 3013682
  (3005456),           -- Potassium in Blood -> 3023103
  (3018572),           -- Chloride in Blood -> 3014576
  (3004119), (3027484),  -- Hemoglobin variants -> 3000963
  (3009542),           -- Hematocrit -> 3023314
  (3003338),           -- MCHC -> 3009744
  (3024731),           -- MCV -> 3023599
  (3035941),           -- MCH -> 3012030
  (3002385),           -- RDW -> 3019897
  (3010813),           -- Leukocytes in Blood -> 3000905
  (3019198),           -- Lymphocytes in Blood -> 3004327
  (3017732),           -- Neutrophils in Blood -> 3013650
  (3001604),           -- Monocytes in Blood -> 3033575
  (3007461),           -- Platelets in Blood -> 3024929
  (3026361),           -- Erythrocytes in Blood -> 3020416
  (3015183),           -- ESR -> 3013707
  (3010156),           -- CRP high-sensitivity -> 3020460
  (3022250),           -- LDH lactate->pyruvate -> 3016436
  (3005225),           -- LDH pyruvate->lactate -> 3016436
  (3018677),           -- aPTT in PPP -> 3013466
  (3005755),           -- ALT with P-5'-P -> 3006923
  (3037081),           -- AST with P-5'-P -> 3013721
  (3010140),           -- CO2 in Venous -> 3015632
  (3002417),           -- PT in Blood -> 3034426
  (3032080),           -- INR in Blood -> 3022217
  (3049555),           -- Testosterone low-detect -> 3008893
  (3021886),           -- Globulin in Serum -> 3027970
  (3013762),           -- Body weight Measured -> 3025315
  (3023540),           -- Body height Measured -> 3036277
  (3003215),           -- Lymphocytes manual count -> 3004327
  (3017501),           -- Neutrophils manual count -> 3013650
  (3027651),           -- Basophils manual count -> 3013429
  (3009932),           -- Eosinophils manual count -> 3028615
  (3034107),           -- Monocytes manual count -> 3033575
  (3019909),           -- Hematocrit centrifugation -> 3023314
  (3034976),           -- Hematocrit venous -> 3023314
  (3001657),           -- WBC corrected -> 3000905
  (3046900),           -- WBC corrected automated -> 3000905
  (3048275),           -- Hemoglobin pre-phlebotomy -> 3000963
  (3028286),           -- Albumin electrophoresis -> 3024561
  (40763912),          -- Albumin BCG -> 3024561
  (3046948),           -- A/G ratio electrophoresis -> 3020509
  (3011424),           -- Glucose automated strip -> 3004501
  (3014053),           -- Glucose manual strip -> 3004501
  (3032986),           -- Glucose 1st specimen -> 3004501
  (3013826),           -- Glucose mmol/L -> 3004501
  (3027219),           -- BUN venous -> 3013682
  (3051825),           -- Creatinine blood -> 3016723
  (3035285),           -- Chloride venous -> 3014576
  (3041354),           -- Potassium venous -> 3023103
  (3030597),           -- Calcium corrected -> 3006906
  (3015377),           -- Calcium mmol/L -> 3006906
  (3019170),           -- TSH sensitive 0.005 -> 3009201
  (3019762),           -- TSH sensitive 0.05 -> 3009201
  (3024675),           -- Free T4 dialysis -> 3008598
  (3019171),           -- Body height Stated -> 3036277
  (3023166),           -- Body weight Stated -> 3025315
  (3040891),           -- Heart rate resting -> 3027018
  (3042292),           -- Heart rate post-exercise -> 3027018
  (40771525),          -- Heart rate sitting -> 3027018
  (3016628),           -- Fibrinogen Ag immunoassay -> 3016407
  (3037950),           -- Fibrinogen Ag nephelometry -> 3016407
  -- Category 2: New lab types
  (3013429), (3006315),  -- Basophils (automated + generic; 3006315 remapped -> 3013429)
  (3028615), (3013115),  -- Eosinophils (automated + generic; 3013115 remapped -> 3028615)
  (3009201),           -- TSH
  (3008598),           -- Free T4
  (3016407)            -- Fibrinogen
AS t(concept_id)
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

# === 8. Patients with prostate cancer and ≥5 PSA measurements after diagnosis ===
# FIX: Joins against first_prostate_diagnosis (one row per patient) instead of
# condition_occurrence (many rows per patient), preventing cross-product inflation
# of the PSA count.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_prostate_psa5_patients AS
SELECT f.person_id
FROM first_prostate_diagnosis f
JOIN dn_measurement_20251219 m
  ON f.person_id = m.person_id
WHERE m.measurement_concept_id IN (SELECT concept_id FROM temp_psa_concepts)
  AND m.measurement_date > f.prostate_cancer_diagnosis_date
GROUP BY f.person_id
HAVING COUNT(*) >= 5
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

# === 11. Combined eligible cohort: male AND ≥5 PSA AND no other cancer AND no PARP ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_eligible_patients AS
SELECT t.person_id
FROM temp_prostate_psa5_patients t
JOIN dn_person_20251219 p ON t.person_id = p.person_id
LEFT JOIN temp_patients_with_other_cancer excl_cancer ON t.person_id = excl_cancer.person_id
LEFT JOIN temp_patients_with_parp excl_parp ON t.person_id = excl_parp.person_id
WHERE p.gender_concept_id = 8507  -- Male
  AND excl_cancer.person_id IS NULL
  AND excl_parp.person_id IS NULL
""")

# Cache the eligibility result so downstream views (labs, drugs, follow-up) all
# read from memory rather than re-evaluating all exclusion logic each time.
_eligible_df = spark.table("temp_eligible_patients").cache()
_eligible_df.count()  # trigger materialization
_eligible_df.createOrReplaceTempView("temp_eligible_patients")

# === 11b. Pre-filter all drug exposures for cohort patients (single table scan) ===
# Caching this avoids repeated full scans of dn_drug_exposure_20251219 in steps 12 and 12c.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_cohort_drug_exposures_all AS
SELECT de.person_id, de.drug_concept_id, de.drug_exposure_start_date
FROM dn_drug_exposure_20251219 de
WHERE de.person_id IN (SELECT person_id FROM temp_eligible_patients)
""")
_cohort_drugs_df = spark.table("temp_cohort_drug_exposures_all").cache()
_cohort_drugs_df.count()
_cohort_drugs_df.createOrReplaceTempView("temp_cohort_drug_exposures_all")

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
  FROM temp_cohort_drug_exposures_all de
  WHERE de.drug_concept_id IN (SELECT drug_concept_id FROM temp_platinum_drugs)
)
SELECT person_id, drug_concept_id,
       drug_exposure_start_date AS first_drug_exposure_start_date
FROM ranked
WHERE rn = 1
""")

# === 12b. Last contact / follow-up per patient ===
#     Derived from latest event across all denormalized tables + death.
#     Computed and cached before step 13 so it can be broadcast-joined there.
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
    SELECT person_id, drug_exposure_start_date AS event_date FROM temp_cohort_drug_exposures_all
    UNION ALL
    SELECT person_id, procedure_date           AS event_date FROM dn_procedure_occurrence_20251219 WHERE person_id IN (SELECT person_id FROM cohort_ids)
    UNION ALL
    SELECT person_id, visit_start_date         AS event_date FROM dn_visit_occurrence_20251219     WHERE person_id IN (SELECT person_id FROM cohort_ids)
  ) events
  GROUP BY person_id
)
SELECT
    lce.person_id,
    COALESCE(d.death_date, lce.last_event_date) AS last_followup_date,
    CASE WHEN d.death_date IS NOT NULL THEN 1 ELSE 0 END AS is_deceased
FROM last_clinical_event lce
LEFT JOIN dn_death_20251219 d
    ON lce.person_id = d.person_id
""")
# temp_followup: not cached separately — used only once in step 13 below

# === 13. Combine labs with demographics, drugs, and follow-up ===
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW prostate_lab_longitudinal AS
SELECT /*+ BROADCAST(p), BROADCAST(f), BROADCAST(per), BROADCAST(dp), BROADCAST(fe), BROADCAST(c), BROADCAST(lc), BROADCAST(fu) */
  -- ==== Patient & Demographics ====
  p.person_id,
  per.race_concept_name  AS race,
  per.ethnicity_concept_name  AS ethnicity,
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

  -- ==== Clinical Timeline ====
  f.prostate_cancer_diagnosis_date AS diagnosis_date,
  dp.drug_type AS platinum_type,
  fe.first_drug_exposure_start_date AS platinum_start_date,
  fu.last_followup_date,
  fu.is_deceased,

  -- ==== Lab Measurements ====
  m.measurement_concept_id,
  c.concept_name AS lab_name,
  m.measurement_date,
  m.value_as_number AS lab_value,
  m.unit_concept_id,
  m.unit_concept_name AS lab_unit_name

FROM temp_eligible_patients p
JOIN first_prostate_diagnosis f
  ON p.person_id = f.person_id
JOIN dn_person_20251219 per
  ON p.person_id = per.person_id

-- Labs (all concepts from temp_all_lab_concepts = PSA + other labs)
JOIN dn_measurement_20251219 m
  ON p.person_id = m.person_id
JOIN temp_all_lab_concepts lc
  ON m.measurement_concept_id = lc.concept_id
LEFT JOIN {CONCEPT_TABLE} c
  ON m.measurement_concept_id = c.concept_id
-- Drugs
LEFT JOIN first_platinum_exposure fe
  ON p.person_id = fe.person_id
LEFT JOIN temp_platinum_drugs dp
  ON fe.drug_concept_id = dp.drug_concept_id
-- Follow-up (cached)
LEFT JOIN temp_followup fu
  ON p.person_id = fu.person_id
""")

# === 14. Materialize ===
# Follow-up is already joined in step 13 (via cached temp_followup).
# No separate step 15 view needed.
final_df = spark.table("prostate_lab_longitudinal").cache()
final_df.count()  # trigger materialization before remap join

# Free cohort drug cache — consumed by platinum + followup views above
_cohort_drugs_df.unpersist()

# === 16b. Remap variant measurement concept IDs to canonical IDs ===
# This merges alternate LOINC codings into a single concept per lab type,
# so that downstream unit filtering, conversions, and ranges apply uniformly.
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
)
# No intermediate cache — flows through conversion + range filter to df_to_upload.cache()

# === 18. Perform required unit conversions on subset of labs ===
# Lookup table: (concept_id, from_unit, factor, to_unit)
# factor=None for special conversions handled separately (temperature F->C)
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
        "lab_value",
        F.when(F.col("lab_values_converted").isNotNull(), F.col("lab_values_converted"))
         .otherwise(F.col("lab_value"))
    )
)

# Look up the human-readable name for the converted unit.
# Unit concepts are not in concept_subset.csv, so we use:
#   - lab_unit_name directly when no conversion was applied (unit_converted_id == unit_concept_id)
#   - a small hardcoded lookup for the fixed set of conversion target units
_target_unit_names = spark.createDataFrame([
    (8582,   "centimeter"),
    (8645,   "international unit per liter"),
    (8713,   "gram per deciliter"),
    (8753,   "milliequivalent per liter"),
    (8815,   "million per microliter"),
    (8817,   "nanogram per deciliter"),
    (8840,   "milligram per deciliter"),
    (8848,   "thousand per microliter"),
    (9040,   "milli-international unit per liter"),
    (9529,   "kilogram"),
    (586323, "degree Celsius"),
], schema=["unit_converted_id", "unit_converted_name"])

final_df_w_conversions = (
    final_df_w_conversions
    .join(F.broadcast(_target_unit_names), on="unit_converted_id", how="left")
    .withColumn(
        "unit_converted_name",
        F.coalesce(F.col("unit_converted_name"), F.col("lab_unit_name")),
    )
)

# === 18b. Exclude null/NaN and non-physiologic lab values ===
# 1) Remove rows where lab_value is null or NaN
final_df_w_conversions = final_df_w_conversions.filter(
    F.col("lab_value").isNotNull() & ~F.isnan(F.col("lab_value"))
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
            (F.col("lab_value") >= F.col("physio_min"))
            & (F.col("lab_value") <= F.col("physio_max"))
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
        F.datediff(F.col("measurement_date"), F.col("diagnosis_date"))
    )
    .withColumn(
        "days_relative_to_platinum_start",
        F.datediff(F.col("measurement_date"), F.col("platinum_start_date"))
    )
    .withColumn(
        "days_relative_to_last_followup",
        F.datediff(F.col("measurement_date"), F.col("last_followup_date"))
    )
    # --- Patient-level time-to-event variables ---
    .withColumn(
        "event_platinum",
        F.when(F.col("platinum_start_date").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )
    .withColumn(
        "time_to_platinum_or_censor",
        F.datediff(
            F.coalesce(F.col("platinum_start_date"), F.col("last_followup_date")),
            F.col("diagnosis_date")
        )
    )
    .withColumn(
        "time_to_death_or_censor",
        F.datediff(
            F.col("last_followup_date"),
            F.col("diagnosis_date")
        )
    )
    .drop(
        "measurement_concept_id",
        "unit_concept_id", "unit_converted_id",
        "lab_values_converted", "lab_unit_name",
    )
    .withColumnRenamed("unit_converted_name", "lab_unit")
)

# === 20. Enforce canonical types + schema assertions ===

# ---- Define column groups ----
int_cols = [
    "person_id",
    "age_at_diagnosis",
    "is_deceased",
    "event_platinum",
    "time_to_platinum_or_censor",
    "time_to_death_or_censor",
    "days_relative_to_diagnosis",
    "days_relative_to_platinum_start",
    "days_relative_to_last_followup",
]

float_cols = [
    "lab_value",
]

date_cols = [
    "diagnosis_date",
    "measurement_date",
    "platinum_start_date",
    "last_followup_date",
]

string_cols = [
    "race",
    "ethnicity",
    "lab_name",
    "lab_unit",
    "platinum_type",
]

# ---- 1) Column order + presence assertion ----
expected_columns = [
    # Patient & Demographics
    "person_id",
    "race",
    "ethnicity",
    "age_at_diagnosis",

    # Clinical Timeline
    "diagnosis_date",
    "platinum_type",
    "platinum_start_date",
    "last_followup_date",
    "is_deceased",

    # Lab Measurements
    "lab_name",
    "measurement_date",
    "lab_value",
    "lab_unit",

    # Derived — Patient-level time-to-event
    "event_platinum",
    "time_to_platinum_or_censor",
    "time_to_death_or_censor",

    # Derived — Lab-level relative timing
    "days_relative_to_diagnosis",
    "days_relative_to_platinum_start",
    "days_relative_to_last_followup",
]

actual_columns = df_to_upload.columns
missing = set(expected_columns) - set(actual_columns)
extra   = set(actual_columns) - set(expected_columns)

assert not missing, f"df_to_upload is missing columns: {sorted(missing)}"
assert not extra,   f"df_to_upload has unexpected extra columns: {sorted(extra)}"

# ---- 2) Canonical casts + reorder ----
# Single select() casts types and enforces column order in one projection stage.
_cast_map = (
    {c: "bigint"  for c in int_cols}
    | {c: "double" for c in float_cols}
    | {c: "date"   for c in date_cols}
    | {c: "string" for c in string_cols}
)
df_to_upload = df_to_upload.select(
    *[F.col(c).cast(_cast_map[c]).alias(c) if c in _cast_map else F.col(c)
      for c in expected_columns]
)

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

df_to_upload.display()
