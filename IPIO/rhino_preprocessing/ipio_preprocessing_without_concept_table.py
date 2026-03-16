# ===============================================================
# Load vocabulary subset CSVs (replacing deprecated catalog)
# ===============================================================
import os
import pandas as pd
_cwd = os.getcwd()
spark.createDataFrame(pd.read_csv(os.path.join(_cwd, "concept_tables", "concept_subset.csv"), sep="\t")).createOrReplaceTempView("concept")
spark.createDataFrame(pd.read_csv(os.path.join(_cwd, "concept_tables", "concept_ancestor_subset.csv"), sep="\t")).createOrReplaceTempView("concept_ancestor")

# ===============================================================
# Performance settings
# ===============================================================
spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(50 * 1024 * 1024))  # 50 MB
spark.sparkContext.setCheckpointDir("dbfs:/tmp/caia_checkpoints")

CONCEPT_TABLE = "concept"

# === 1. Define complete cancer type concepts ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW cancer_concepts AS
SELECT DISTINCT c.concept_id,
       c.concept_name,
       c.domain_id,
       c.vocabulary_id
FROM concept_ancestor ca
JOIN concept c
  ON ca.descendant_concept_id = c.concept_id
WHERE ca.ancestor_concept_id IN (443392)
  AND c.invalid_reason IS NULL
""")

# === 2. Define pre-determined cancer type sets ===
# NSCLC IDs are pre-expanded descendants from the original catalog — use directly.
# Bladder, Kidney, Melanoma use ancestor lookups (their ancestors are in the subset CSV).
spark.sql("""
CREATE OR REPLACE TEMP VIEW cancer_labels AS
SELECT concept_id, 'NSCLC' AS cancer_type FROM (VALUES
  (4308784),(37166276),(37166728),(37166795),(37166615),(37166745),(36562727),(36561870),(42511962),(44502320),(36517425),(44501335),(44503152),(42512380),(36538464),(36548902),(44502390),(42512620),(44502341),(44501585),(36531338),(36542106),(44503034),(36553640),(36558787),(36526019),(36518742),(44503111),(602169),(45766131),(45768879),(4311997),(45766129),(45768884),(36684857),(4314156),(37166759),(37166794),(36550921),(36535903),(36537460),(36561693),(44499794),(36536720),(36521546),(44502949),(44501357),(36529691),(36531819),(44499007),(44502334),(36538056),(42512569),(36530431),(4112738),(4312768),(45772938),(4208307),(44502426),(36558543),(36544137),(36545785),(36557771),(36531347),(36537196),(44499621),(36560118),(44500984),(36566369),(36567972),(36552939),(36541699),(44501794),(36551240),(36536513),(36523702),(36529938),(44500201),(36564508),(44503098),(44502876),(36529740),(36551342),(602167),(4110589),(4197582),(4111805),(46272955),(45768920),(37166330),(37166520),(37166667),(37166775),(37166642),(44501388),(44502764),(36563600),(44501567),(44500855),(44500239),(36558877),(44500072),(44499900),(36518010),(36565803),(36549979),(36712707),(602682),(45768880),(45768930),(45772933),(4307118),(36716426),(37166536),(44501310),(44503018),(42512867),(36547828),(36560716),(44502943),(44501086),(44500303),(36536903),(36555776),(36555703),(36532026),(42512859),(36712709),(36686538),(4115276),(45768916),(4322387),(45768885),(37165772),(37166280),(37166780),(44501926),(36552199),(44503539),(36539842),(36524040),(36565633),(42512246),(44499508),(42511851),(44499488),(44499882),(44500061),(36517902),(36532844),(42511673),(36567491),(36563834),(36532293),(36533348),(36555136),(36556919),(36541364),(36550126),(36543615),(602150),(4313751),(45768928),(37311684),(45769035),(37165675),(37166238),(37166278),(37166608),(37166760),(37166771),(44500841),(36538391),(44500348),(44500287),(36538582),(36536437),(44502769),(44499915),(36525571),(36554751),(42513175),(36535086),(36543960),(36539915),(36548970),(36546130),(609080),(3654297),(4196725),(37165750),(37165786),(37166668),(37166762),(37166605),(37166789),(4110706),(44500048),(44501059),(44499866),(44503136),(44500433),(36517234),(44499947),(44500577),(44499726),(44501191),(36526433),(42512456),(36520827),(44502329),(44501438),(36535558),(44500415),(36560845),(44500730),(36527609),(36686537),(4314040),(602697),(609078),(45768922),(37166604),(44503537),(44500188),(44501516),(36561274),(36543984),(36523265),(36552086),(36548201),(42511979),(42512336),(44502549),(36518654),(36567502),(36556614),(36545286),(36712708),(602171),(605447),(36712981),(36712815),(4310703),(45768923),(45768932),(4314832),(37165756),(37166601),(37166236),(36717017),(609076),(4196724),(4308479),(45772939),(4140471),(45768881),(45768919),(3654352),(44501124),(44499863),(36519829),(36533515),(44502921),(44501709),(36541640),(36551824),(44499010),(44502456),(36531433),(36556975),(44503010),(36549175),(36557751),(36534932),(36545242),(44502623),(42512616),(36530069),(44499513),(36561283),(36535081),(36548193),(36538987),(44501411),(36524730),(36561810),(44503093),(37166768),(44500183),(44500843),(44500343),(44501471),(44503004),(44499688),(36520748),(36532996),(44503138),(36522039),(44501404),(36518808),(44500414),(42512801),(36551818),(36564925),(36522427),(36520679),(602163),(3654301),(4313200),(37109576),(4311452),(37166607),(37166774),(37166577),(36565647),(36528414),(44500713),(44499422),(44501061),(44501188),(36567522),(36522300),(36567150),(36549159),(44502436),(44500290),(36537250),(44499889),(44501394),(36554642),(36541027),(36567060),(36531120),(42511853),(44502952),(44502961),(36521617),(36530839),(44500790),(44502784),(36530344),(36526753),(4197581),(45768931),(609075),(37165673),(36526440),(44501560),(44500710),(44502457),(36531136),(44502109),(36528862),(36552385),(44502703),(44502466),(36556450),(36539786),(36523365),(42512752),(36559892),(36567546),(36549522),(44499741),(44502711),(36536832),(44500302),(605821),(42539251),(609079),(4143825),(609081),(4197583),(4314172),(45768921),(605445),(605446),(36712816),(4312274),(4110590),(4310448),(45769034),(36524261),(36560935),(44501559),(44502856),(44499623),(44500190),(36540691),(36534747),(36531543),(44500356),(42513174),(44500359),(36531901),(36548626),(36563105),(44502238),(36567582),(36553787),(37165766),(37166235),(37166328),(37166624),(37166735),(37166234),(44500480),(44500114),(44501707),(44500448),(44500573),(36559186),(36539497),(44499686),(44500458),(36549254),(44501740),(36546410),(44502176),(42512505),(42512222),(36550205),(4110705),(45768886),(45768918),(37166279),(37165759),(37166277),(37166666),(37166773),(44500918),(44501558),(44501613),(36525316),(44502543),(44503538),(36558592),(36525052),(36532437),(36566204),(36541713),(36549769),(44502885),(36520186),(42512265),(36566849),(36541038),(609077),(45768917)
) AS t(concept_id)
UNION ALL
SELECT descendant_concept_id AS concept_id, 'Bladder' AS cancer_type FROM concept_ancestor WHERE ancestor_concept_id IN (197508)
UNION ALL
SELECT descendant_concept_id AS concept_id, 'Kidney'  AS cancer_type FROM concept_ancestor WHERE ancestor_concept_id IN (196653)
UNION ALL
SELECT descendant_concept_id AS concept_id, 'Melanoma' AS cancer_type FROM concept_ancestor WHERE ancestor_concept_id IN (4162276,40391314)
""")

# === 3. Create full cancer type and subtype table ===
# LEFT JOIN so ALL cancer concepts are retained.
# The 4 pre-defined types (NSCLC, Bladder, Kidney, Melanoma) get their label;
# all other cancers fall back to the OMOP concept_name as cancer_type.
spark.sql("""
CREATE OR REPLACE TEMP VIEW cancer_concepts_labeled AS
SELECT /*+ BROADCAST(l) */
    c.concept_id,
    c.concept_name,
    c.domain_id,
    c.vocabulary_id,
    COALESCE(l.cancer_type, c.concept_name) AS cancer_type
FROM cancer_concepts c
LEFT JOIN cancer_labels l
    ON c.concept_id = l.concept_id
""")

# === 4. ICI ingredient concepts ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ICI_ingredients AS
SELECT concept_id FROM VALUES
  (45892628),  -- Nivolumab
  (45775965),  -- Pembrolizumab
  (42629079),  -- Atezolizumab
  (1594034),   -- Durvalumab
  (40238188),  -- Ipilimumab
  (35200783),  -- Cemiplimab
  (1593273),   -- Avelumab
  (741851),    -- Tremelimumab
  (1536789),   -- Dostarlimab
  (1302024),   -- Retifanlimab
  (747052)     -- Toripalimab
AS t(concept_id)
""")

# === 4b. Expand ICI ingredients to all descendant drug products ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ICI_concepts AS
SELECT DISTINCT ca.descendant_concept_id AS concept_id
FROM concept_ancestor ca
WHERE ca.ancestor_concept_id IN (SELECT concept_id FROM temp_ICI_ingredients)
""")

# === 5. Non-ICI antineoplastic drug concepts (ATC L01 descendants, excluding ICI) ===
# Used to detect therapy switches (progression signal) after ICI discontinuation.
# ATC L01 "Antineoplastic agents" = concept_id 21601387
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_antineoplastic_non_ici AS
SELECT DISTINCT ca.descendant_concept_id AS drug_concept_id
FROM concept_ancestor ca
LEFT JOIN temp_ICI_concepts ici ON ca.descendant_concept_id = ici.concept_id
WHERE ca.ancestor_concept_id = 21601387
  AND ici.concept_id IS NULL
""")

# === 6. Systemic corticosteroid concepts (ATC H02AB descendants) ===
# Used to detect irAE treatment (toxicity signal) around ICI discontinuation.
# ATC H02AB "Glucocorticoids" = concept_id 21602728
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_systemic_corticosteroids AS
SELECT DISTINCT ca.descendant_concept_id AS drug_concept_id
FROM concept_ancestor ca
WHERE ca.ancestor_concept_id = 21602728
""")

# === 7. Lab measurement concepts ===
# Hardcoded via VALUES so the view is independent of the concept CSV for filtering.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_lab_concepts AS
SELECT concept_id FROM VALUES
  -- Canonical lab concepts
  (3006923), (3024561), (3020509), (3035995), (3009306), (3013721), (3027597), (3024128), (3036277), (3020891),
  (3025315), (3020460), (3006906), (3037551), (3022914), (3015632), (3003785), (3014576), (3007220), (3016723),
  (3012888), (3019897), (3013707), (3020416), (3027970), (3004501), (3027018), (3023314), (3000963), (3022217),
  (3016436), (3000905), (3004327), (3012030), (3009744), (3023599), (3033575), (3013650), (3024929),
  (3023103), (3020630), (3034426), (3024171), (3019550), (3004249), (3008893), (3013682), (3013466),
  (3013603),
  -- Category 1: Variant codings of existing labs (will be remapped post-extraction)
  (3002131), (3034548),  -- PSA variants -> 3013603
  (3000285),             -- Sodium in Blood -> 3019550
  (3000483),             -- Glucose in Blood -> 3004501
  (3004295),             -- BUN in Blood -> 3013682
  (3005456),             -- Potassium in Blood -> 3023103
  (3018572),             -- Chloride in Blood -> 3014576
  (3004119), (3027484),  -- Hemoglobin variants -> 3000963
  (3009542),             -- Hematocrit -> 3023314
  (3003338),             -- MCHC -> 3009744
  (3024731),             -- MCV -> 3023599
  (3035941),             -- MCH -> 3012030
  (3002385),             -- RDW -> 3019897
  (3010813),             -- Leukocytes in Blood -> 3000905
  (3019198),             -- Lymphocytes in Blood -> 3004327
  (3017732),             -- Neutrophils in Blood -> 3013650
  (3001604),             -- Monocytes in Blood -> 3033575
  (3007461),             -- Platelets in Blood -> 3024929
  (3026361),             -- Erythrocytes in Blood -> 3020416
  (3015183),             -- ESR -> 3013707
  (3010156),             -- CRP high-sensitivity -> 3020460
  (3022250),             -- LDH lactate->pyruvate -> 3016436
  (3005225),             -- LDH pyruvate->lactate -> 3016436
  (3018677),             -- aPTT in PPP -> 3013466
  (3005755),             -- ALT with P-5'-P -> 3006923
  (3037081),             -- AST with P-5'-P -> 3013721
  (3010140),             -- CO2 in Venous -> 3015632
  (3002417),             -- PT in Blood -> 3034426
  (3032080),             -- INR in Blood -> 3022217
  (3049555),             -- Testosterone low-detect -> 3008893
  (3021886),             -- Globulin in Serum -> 3027970
  (3013762),             -- Body weight Measured -> 3025315
  (3023540),             -- Body height Measured -> 3036277
  (3003215),             -- Lymphocytes manual count -> 3004327
  (3017501),             -- Neutrophils manual count -> 3013650
  (3027651),             -- Basophils manual count -> 3013429
  (3009932),             -- Eosinophils manual count -> 3028615
  (3034107),             -- Monocytes manual count -> 3033575
  (3019909),             -- Hematocrit centrifugation -> 3023314
  (3034976),             -- Hematocrit venous -> 3023314
  (3001657),             -- WBC corrected -> 3000905
  (3046900),             -- WBC corrected automated -> 3000905
  (3048275),             -- Hemoglobin pre-phlebotomy -> 3000963
  (3028286),             -- Albumin electrophoresis -> 3024561
  (40763912),            -- Albumin BCG -> 3024561
  (3046948),             -- A/G ratio electrophoresis -> 3020509
  (3011424),             -- Glucose automated strip -> 3004501
  (3014053),             -- Glucose manual strip -> 3004501
  (3032986),             -- Glucose 1st specimen -> 3004501
  (3013826),             -- Glucose mmol/L -> 3004501
  (3027219),             -- BUN venous -> 3013682
  (3051825),             -- Creatinine blood -> 3016723
  (3035285),             -- Chloride venous -> 3014576
  (3041354),             -- Potassium venous -> 3023103
  (3030597),             -- Calcium corrected -> 3006906
  (3015377),             -- Calcium mmol/L -> 3006906
  (3019170),             -- TSH sensitive 0.005 -> 3009201
  (3019762),             -- TSH sensitive 0.05 -> 3009201
  (3024675),             -- Free T4 dialysis -> 3008598
  (3019171),             -- Body height Stated -> 3036277
  (3023166),             -- Body weight Stated -> 3025315
  (3040891),             -- Heart rate resting -> 3027018
  (3042292),             -- Heart rate post-exercise -> 3027018
  (40771525),            -- Heart rate sitting -> 3027018
  (3016628),             -- Fibrinogen Ag immunoassay -> 3016407
  (3037950),             -- Fibrinogen Ag nephelometry -> 3016407
  -- Category 2: New lab types
  (3013429), (3006315),  -- Basophils (automated + generic; 3006315 remapped -> 3013429)
  (3028615), (3013115),  -- Eosinophils (automated + generic; 3013115 remapped -> 3028615)
  (3009201),             -- TSH
  (3008598),             -- Free T4
  (3016407)              -- Fibrinogen
AS t(concept_id)
""")

# ===============================================================
# Part 2: Clinical preprocessing (denormalized catalog)
# ===============================================================
spark.sql("USE CATALOG dfci_ia_aistudio")
spark.sql("USE SCHEMA omop_caia_denorm")

# === 8. Identify cancer diagnoses ===
# FIX: Deduplicate to one row per patient per cancer_type by picking the earliest
# diagnosis_date and a representative cancer_subtype (alphabetically first for determinism).
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW temp_cancer_dx AS
SELECT
    person_id,
    cancer_type,
    MIN(diagnosis_date) AS diagnosis_date,
    MIN(cancer_subtype) AS cancer_subtype
FROM (
    SELECT
        co.person_id,
        co.condition_start_date AS diagnosis_date,
        cc.concept_name AS cancer_subtype,
        cl.cancer_type
    FROM dn_condition_occurrence_20251219 co
    JOIN cancer_concepts_labeled cl
        ON co.condition_concept_id = cl.concept_id
    JOIN cancer_concepts cc
        ON co.condition_concept_id = cc.concept_id
) sub
GROUP BY person_id, cancer_type
""")

# === 9. Identify ICI treatments ===
# FIX: Deduplicate to one row per patient by picking the earliest ICI exposure.
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW temp_ici_treatment AS
SELECT
    person_id,
    ici_start_date,
    drug_concept_id,
    ici_name
FROM (
    SELECT
        de.person_id,
        de.drug_exposure_start_date AS ici_start_date,
        de.drug_concept_id,
        c.concept_name AS ici_name,
        ROW_NUMBER() OVER (
            PARTITION BY de.person_id
            ORDER BY de.drug_exposure_start_date, de.drug_concept_id
        ) AS rn
    FROM dn_drug_exposure_20251219 de
    JOIN temp_ICI_concepts ici
        ON de.drug_concept_id = ici.concept_id
    JOIN {CONCEPT_TABLE} c
        ON de.drug_concept_id = c.concept_id
) ranked
WHERE rn = 1
""")

# === 10. Patients with ICI after cancer diagnosis ===
# FIX: Deduplicate to one row per person_id. If a patient has multiple cancer_types,
# pick the one with the earliest diagnosis. This prevents fan-out in downstream joins.
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_cancer_ici_patients AS
SELECT
    person_id,
    cancer_type,
    cancer_subtype,
    diagnosis_date,
    ici_start_date,
    ici_type
FROM (
    SELECT
        dx.person_id,
        dx.cancer_type,
        dx.cancer_subtype,
        dx.diagnosis_date,
        ici.ici_start_date,
        ici.ici_name AS ici_type,
        ROW_NUMBER() OVER (
            PARTITION BY dx.person_id
            ORDER BY dx.diagnosis_date, dx.cancer_type
        ) AS rn
    FROM temp_cancer_dx dx
    JOIN temp_ici_treatment ici
        ON dx.person_id = ici.person_id
    WHERE ici.ici_start_date > dx.diagnosis_date
) ranked
WHERE rn = 1
""")

# Checkpoint breaks the multi-level concept/eligibility lineage so all downstream
# views (followup, labs, discontinuation) plan against a materialized result.
_ici_patients_df = spark.table("temp_cancer_ici_patients").checkpoint()
_ici_patients_df.cache()
_ici_patients_df.count()  # trigger materialization
_ici_patients_df.createOrReplaceTempView("temp_cancer_ici_patients")

# === 11. Add death and last observation (LTFU) info ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_followup AS
WITH cohort_ids AS (
    SELECT person_id FROM temp_cancer_ici_patients
),
last_clinical_event AS (
    SELECT person_id, MAX(event_date) AS last_event_date
    FROM (
        SELECT person_id, condition_start_date AS event_date FROM dn_condition_occurrence_20251219 WHERE person_id IN (SELECT person_id FROM cohort_ids)
        UNION ALL
        SELECT person_id, measurement_date AS event_date FROM dn_measurement_20251219              WHERE person_id IN (SELECT person_id FROM cohort_ids)
        UNION ALL
        SELECT person_id, drug_exposure_start_date AS event_date FROM dn_drug_exposure_20251219    WHERE person_id IN (SELECT person_id FROM cohort_ids)
        UNION ALL
        SELECT person_id, procedure_date AS event_date FROM dn_procedure_occurrence_20251219       WHERE person_id IN (SELECT person_id FROM cohort_ids)
        UNION ALL
        SELECT person_id, visit_start_date AS event_date FROM dn_visit_occurrence_20251219         WHERE person_id IN (SELECT person_id FROM cohort_ids)
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
LEFT JOIN dn_death_20251219 d ON lce.person_id = d.person_id
""")

# === 12. Pull target lab measurements, add demographics & follow-up ===
final_df = spark.sql(f"""
SELECT
    -- Demographics
    p.person_id,
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
    per.gender_concept_name AS gender,
    per.race_concept_id,
    per.race_concept_name AS race,
    per.ethnicity_concept_id,
    per.ethnicity_concept_name AS ethnicity,

    FLOOR(
        DATEDIFF(
            p.diagnosis_date,
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

    -- Cancer / ICI info
    p.cancer_type,
    p.cancer_subtype,
    p.diagnosis_date,
    p.ici_start_date,
    p.ici_type,

    -- Follow-up
    f.last_followup_date,
    f.death_date,
    f.is_deceased,

    -- Lab info
    m.measurement_concept_id,
    c.concept_name AS lab_name,
    m.measurement_date,
    m.value_as_number AS lab_value,
    m.unit_concept_id,
    m.unit_concept_name AS lab_unit_name

FROM temp_cancer_ici_patients p
LEFT JOIN temp_followup f
    ON p.person_id = f.person_id

-- Restrict to target lab concepts only, after diagnosis
JOIN dn_measurement_20251219 m
    ON p.person_id = m.person_id
    AND m.measurement_concept_id IN (SELECT concept_id FROM temp_lab_concepts)

LEFT JOIN {CONCEPT_TABLE} c
    ON m.measurement_concept_id = c.concept_id
-- Demographics
JOIN dn_person_20251219 per
    ON p.person_id = per.person_id
""")

# === 13. Create the discontinuation dataframe ===
# Groups ICI exposures into continuous treatment blocks (>60-day gap = new block).
# Returns ALL blocks per patient with a block_number (1 = first block, 2 = second, etc.).
# Classifies discontinuation cause using hierarchical signals:
#   DEATH           — patient died during or before block end
#   PROGRESSION     — new non-ICI antineoplastic started within 90 days after block end
#   TOXICITY        — systemic corticosteroid within ±14 days of block end, no new antineoplastic
#   COMPLETED       — continued clinical activity >90 days after block end (evidence of elective stop)
#   CENSORED        — no continued follow-up after block end (reason for stopping unknown)
#
# Materialized as three intermediate temp views to break the deep CTE lineage and
# allow Spark to optimize each stage independently.

# === 13a. ICI treatment blocks ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ici_block_summary AS
WITH ici_exposures AS (
    SELECT
        de.person_id,
        de.drug_exposure_start_date,
        COALESCE(de.drug_exposure_end_date, de.drug_exposure_start_date) AS drug_exposure_end_date
    FROM dn_drug_exposure_20251219 de
    JOIN temp_ICI_concepts ici ON de.drug_concept_id = ici.concept_id
    WHERE de.person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
),
gaps AS (
    SELECT
        person_id,
        drug_exposure_start_date,
        drug_exposure_end_date,
        CASE
            WHEN LAG(drug_exposure_end_date) OVER (PARTITION BY person_id ORDER BY drug_exposure_start_date)
                < drug_exposure_start_date - INTERVAL 60 DAY THEN 1
            ELSE 0
        END AS new_block_flag
    FROM ici_exposures
),
blocks AS (
    SELECT
        person_id,
        drug_exposure_start_date,
        drug_exposure_end_date,
        SUM(new_block_flag) OVER (
            PARTITION BY person_id ORDER BY drug_exposure_start_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS block_id
    FROM gaps
)
SELECT
    person_id,
    block_id,
    MIN(drug_exposure_start_date) AS ici_block_start_date,
    MAX(drug_exposure_start_date) AS last_ici_cycle_start_date,
    MAX(drug_exposure_end_date)   AS ici_end_date
FROM blocks
GROUP BY person_id, block_id
""")
spark.table("temp_ici_block_summary").cache().count()

# === 13b. Censor dates per patient (non-ICI clinical activity) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ici_censor_dates AS
SELECT
    lce.person_id,
    lce.last_event_date AS observation_period_end_date,
    d.death_date
FROM (
    SELECT person_id, MAX(event_date) AS last_event_date
    FROM (
        SELECT person_id, condition_start_date AS event_date FROM dn_condition_occurrence_20251219 WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
        UNION ALL
        SELECT person_id, measurement_date AS event_date FROM dn_measurement_20251219              WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
        UNION ALL
        SELECT de.person_id, de.drug_exposure_start_date AS event_date
        FROM dn_drug_exposure_20251219 de
        LEFT JOIN temp_ICI_concepts ici ON de.drug_concept_id = ici.concept_id
        WHERE ici.concept_id IS NULL AND de.person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
        UNION ALL
        SELECT person_id, procedure_date AS event_date FROM dn_procedure_occurrence_20251219       WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
        UNION ALL
        SELECT person_id, visit_start_date AS event_date FROM dn_visit_occurrence_20251219         WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
    ) events
    GROUP BY person_id
) lce
LEFT JOIN dn_death_20251219 d ON lce.person_id = d.person_id
""")
spark.table("temp_ici_censor_dates").cache().count()

# === 13c. Cohort drug exposures (pre-filtered to cohort, reused by new_therapy and steroid_use) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_cohort_drug_exposures AS
SELECT person_id, drug_concept_id, drug_exposure_start_date
FROM dn_drug_exposure_20251219
WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
""")
spark.table("temp_cohort_drug_exposures").cache().count()

# === 13d. Classify each block and assign block numbers ===
discontinuation_data = spark.sql("""
    WITH new_therapy AS (
        SELECT DISTINCT bs.person_id, bs.block_id
        FROM temp_ici_block_summary bs
        JOIN temp_cohort_drug_exposures de ON bs.person_id = de.person_id
        JOIN temp_antineoplastic_non_ici anic ON de.drug_concept_id = anic.drug_concept_id
        WHERE de.drug_exposure_start_date > bs.ici_end_date
          AND de.drug_exposure_start_date <= bs.ici_end_date + INTERVAL 90 DAY
    ),
    steroid_use AS (
        SELECT DISTINCT bs.person_id, bs.block_id
        FROM temp_ici_block_summary bs
        JOIN temp_cohort_drug_exposures de ON bs.person_id = de.person_id
        JOIN temp_systemic_corticosteroids cs ON de.drug_concept_id = cs.drug_concept_id
        WHERE de.drug_exposure_start_date >= bs.ici_end_date - INTERVAL 14 DAY
          AND de.drug_exposure_start_date <= bs.ici_end_date + INTERVAL 14 DAY
    ),
    classified AS (
        SELECT
            b.person_id,
            b.block_id,
            b.ici_block_start_date,
            b.last_ici_cycle_start_date,
            b.ici_end_date,
            CASE
                WHEN c.death_date IS NOT NULL AND c.death_date <= b.ici_end_date THEN 'DEATH'
                WHEN nt.person_id IS NOT NULL                                    THEN 'PROGRESSION'
                WHEN su.person_id IS NOT NULL                                    THEN 'TOXICITY'
                WHEN c.observation_period_end_date IS NOT NULL
                     AND c.observation_period_end_date > b.ici_end_date + INTERVAL 90 DAY
                                                                                 THEN 'COMPLETED'
                ELSE 'CENSORED'
            END AS ici_discontinuation_cause,
            LEAST(
                b.ici_end_date,
                COALESCE(c.death_date, DATE '9999-12-31'),
                COALESCE(c.observation_period_end_date, DATE '9999-12-31')
            ) AS ici_discontinuation_date
        FROM temp_ici_block_summary b
        LEFT JOIN temp_ici_censor_dates c ON b.person_id = c.person_id
        LEFT JOIN new_therapy nt ON b.person_id = nt.person_id AND b.block_id = nt.block_id
        LEFT JOIN steroid_use su ON b.person_id = su.person_id AND b.block_id = su.block_id
    ),
    numbered AS (
        SELECT
            person_id,
            ici_block_start_date,
            last_ici_cycle_start_date,
            ici_end_date,
            ici_discontinuation_date,
            ici_discontinuation_cause,
            ROW_NUMBER() OVER (
                PARTITION BY person_id ORDER BY ici_block_start_date ASC
            ) AS block_number
        FROM classified
    )
    SELECT
        person_id,
        block_number,
        ici_block_start_date,
        last_ici_cycle_start_date,
        ici_end_date,
        ici_discontinuation_date,
        ici_discontinuation_cause
    FROM numbered
""")

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# === 14. Merge discontinuation data with complete dataframe ===
# Each lab measurement is assigned to the block it falls within (between block start
# and discontinuation date). Labs outside any block window are assigned to the nearest block.
# This produces one row per (patient, block, lab measurement).
final_df_merged = (
    final_df
    .join(discontinuation_data, on='person_id', how='left')
    .withColumn(
        "_in_block",
        F.when(
            (F.col("measurement_date") >= F.col("ici_block_start_date")) &
            (F.col("measurement_date") <= F.col("ici_discontinuation_date")),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    .withColumn(
        "_dist_to_block",
        F.least(
            F.abs(F.datediff(F.col("measurement_date"), F.col("ici_block_start_date"))),
            F.abs(F.datediff(F.col("measurement_date"), F.col("ici_discontinuation_date")))
        )
    )
    .withColumn(
        "_rank",
        F.row_number().over(
            Window.partitionBy("person_id", "measurement_concept_id", "measurement_date", "lab_value")
            .orderBy(F.desc("_in_block"), F.asc("_dist_to_block"))
        )
    )
    .filter(F.col("_rank") == 1)
    .drop("_in_block", "_dist_to_block", "_rank")
)

# === 14b. Remap variant measurement concepts to canonical IDs ===
from pyspark.sql import functions as F, types as T

concept_remap = {
    3002131: 3013603, 3034548: 3013603,  # PSA variants
    3000285: 3019550, 3000483: 3004501, 3004295: 3013682,  # Chemistry
    3005456: 3023103, 3018572: 3014576, 3010140: 3015632,
    3004119: 3000963, 3027484: 3000963, 3005755: 3006923,  # Protein/enzyme
    3037081: 3013721, 3005225: 3016436, 3022250: 3016436, 3010156: 3020460,
    3018677: 3013466,  # aPTT in PPP -> aPTT in Blood
    3021886: 3027970,
    3009542: 3023314, 3003338: 3009744, 3024731: 3023599,  # CBC
    3035941: 3012030, 3002385: 3019897, 3010813: 3000905,
    3019198: 3004327, 3017732: 3013650, 3001604: 3033575,
    3007461: 3024929, 3026361: 3020416,
    3006315: 3013429, 3013115: 3028615,  # Baso/Eos generic->automated
    3015183: 3013707, 3002417: 3034426, 3032080: 3022217,  # Other
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
final_df_merged = (
    final_df_merged
    .join(remap_df, final_df_merged.measurement_concept_id == remap_df.old_concept_id, "left")
    .withColumn("measurement_concept_id",
                F.coalesce(F.col("new_concept_id"), F.col("measurement_concept_id")))
    .drop("old_concept_id", "new_concept_id")
)

# === 15. Subset measurements based on allowed unit combinations (via join) ===

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
    (3020416, 0), (3020416, 8734), (3020416, 8815), (3020416, 8816), (3020416, 8848), (3020416, 8931), (3020416, 9442),
    (3020416, 8784), (3020416, 8785),
    (3027970, 8713), (3027970, 0), (3027970, 8840),
    (3004501, 0), (3004501, 8713), (3004501, 8840), (3004501, 8751), (3004501, 8753),
    (3027018, 8541), (3027018, 0), (3027018, 8483),
    (3023314, 0), (3023314, 8554),
    (3000963, 0), (3000963, 8636), (3000963, 8713), (3000963, 8840),
    (3022217, 0), (3022217, 8523),
    (3016436, 0), (3016436, 8645), (3016436, 8923),
    (3016436, 8554),  # LDH percent (from remapped 3022250)
    (3000905, 0), (3000905, 8848), (3000905, 8961), (3000905, 9435), (3000905, 9436), (3000905, 9444), (3000905, 9446),
    (3000905, 8784), (3000905, 8785),
    (3004327, 0), (3004327, 8647), (3004327, 8784), (3004327, 8848), (3004327, 8961), (3004327, 9444),
    (3012030, 0), (3012030, 8564),
    (3009744, 0), (3009744, 8713),
    (3023599, 0), (3023599, 8583),
    (3033575, 0), (3033575, 8784), (3033575, 8848), (3033575, 8961), (3033575, 9444),
    (3013650, 0), (3013650, 8784), (3013650, 8848), (3013650, 8961), (3013650, 9444), (3013650, 9665),
    (3024929, 0), (3024929, 8815), (3024929, 8848), (3024929, 8931), (3024929, 8961),
    (3024929, 9435), (3024929, 9436), (3024929, 9444), (3024929, 9446),
    (3024929, 8784), (3024929, 8785),
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
    # New: Basophils (10^3/uL canonical, plus cells/uL and billion/L for conversion)
    (3013429, 0), (3013429, 8848), (3013429, 8961), (3013429, 8784), (3013429, 9444),
    # New: Eosinophils (10^3/uL canonical, plus cells/uL, /uL, billion/L for conversion)
    (3028615, 0), (3028615, 8848), (3028615, 8961), (3028615, 8784), (3028615, 8647), (3028615, 9444),
    # New: TSH (mIU/L canonical, plus uIU/mL, IU/mL, mU/mL, uU/mL, mIU/mL, mU/L, uIU/L)
    (3009201, 0), (3009201, 9040), (3009201, 9093), (3009201, 8985), (3009201, 8719),
    (3009201, 8860), (3009201, 9550), (3009201, 44777578), (3009201, 44777583),
    # New: Free T4 (ng/dL canonical, plus pmol/L, pg/mL)
    (3008598, 0), (3008598, 8817), (3008598, 8729), (3008598, 8845),
    # New: Fibrinogen (mg/dL canonical)
    (3016407, 0), (3016407, 8840),
    # ESR variant unit (mm)
    (3013707, 8588),
    # Globulin calculated variant unit (g/dL = 8713 already covered, add ratio unit)
    (3027970, 8950)
]

allowed_df = spark.createDataFrame(
    allowed_unit_combinations,
    schema=T.StructType([
        T.StructField("measurement_concept_id", T.LongType(), False),
        T.StructField("unit_concept_id", T.LongType(), False),
    ])
)

df_cast = (final_df_merged
    .withColumn("measurement_concept_id", F.col("measurement_concept_id").cast("long"))
    .withColumn("unit_concept_id", F.col("unit_concept_id").cast("long"))
)

final_df_w_units_filtered = (
    df_cast
    .join(F.broadcast(allowed_df), on=["measurement_concept_id", "unit_concept_id"], how="inner")
).cache()

# === 16. Convert units when required ===
# Lookup table: (concept_id, from_unit, factor, to_unit)
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, LongType

_conversion_schema = StructType([
    StructField("conv_concept_id", LongType()),
    StructField("conv_from_unit", LongType()),
    StructField("conv_factor", DoubleType()),
    StructField("conv_to_unit", LongType()),
])

_conversion_data = [
    # ALT -> U/L (8645)
    (3006923, 8985, 1000.0, 8645),
    (3006923, 8923, 1.0, 8645),
    (3006923, 9254, 1.0, 8645),
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
    (3024561, 8636, 0.1, 8713),
    (3024561, 8840, 0.001, 8713),
    # Height -> cm (8582)
    (3036277, 9307, 30.48, 8582),
    (3036277, 9306, 30.48, 8582),
    (3036277, 9330, 2.54, 8582),
    # Temperature F->C handled separately (affine)
    # Body weight -> kg (9529)
    (3025315, 8739, 0.45359237, 9529),
    (3025315, 9373, 0.028349523, 9529),
    (3025315, 9346, 0.45359237, 9529),
    # CRP -> mg/dL (8840)
    (3020460, 8751, 0.1, 8840),
    # Calcium -> mg/dL (8840)
    (3006906, 8751, 0.1, 8840),
    (3006906, 8713, 1000.0, 8840),
    (3006906, 8753, 4.008, 8840),
    (3006906, 9557, 2.004, 8840),
    # CO2 -> mmol/L (8753)
    (3015632, 8736, 0.000001, 8753),
    (3015632, 9557, 1.0, 8753),
    # Creatinine -> mg/dL (8840)
    (3016723, 8751, 0.1, 8840),
    (3016723, 8749, 0.01131, 8840),
    # Erythrocytes -> million/uL (8815)
    (3020416, 8848, 0.001, 8815),
    (3020416, 9442, 0.000001, 8815),
    (3020416, 8931, 1.0, 8815),
    (3020416, 8784, 0.000001, 8815),
    (3020416, 8785, 0.000001, 8815),
    (3020416, 8816, 0.001, 8815),
    (3020416, 8734, 1.0, 8815),
    # Globulin -> g/dL (8713)
    (3027970, 8840, 0.001, 8713),
    (3027970, 8950, 1.0, 8713),
    # Glucose -> mg/dL (8840)
    (3004501, 8713, 1000.0, 8840),
    (3004501, 8751, 0.1, 8840),
    (3004501, 8753, 18.016, 8840),
    # Hemoglobin -> g/dL (8713)
    (3000963, 8636, 0.1, 8713),
    (3000963, 8840, 0.001, 8713),
    # Leukocytes -> thousand/uL (8848)
    (3000905, 9435, 0.000001, 8848),
    (3000905, 9436, 0.001, 8848),
    (3000905, 9446, 1000000.0, 8848),
    (3000905, 8961, 1.0, 8848),
    (3000905, 8784, 0.001, 8848),
    (3000905, 8785, 0.001, 8848),
    (3000905, 9444, 0.001, 8848),
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
    (3024929, 8931, 1000.0, 8848),
    (3024929, 8815, 1000.0, 8848),
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
    (3013682, 8817, 0.000001, 8840),
    (3013682, 8753, 2.8013, 8840),
    # Bilirubin total -> mg/dL (8840)
    (3024128, 8749, 0.05847, 8840),
    # TSH -> mIU/L (9040)
    (3009201, 9093, 1.0, 9040),
    (3009201, 8985, 1000000.0, 9040),
    (3009201, 8719, 1000.0, 9040),
    (3009201, 8860, 1.0, 9040),
    (3009201, 9550, 1000.0, 9040),
    (3009201, 44777578, 1.0, 9040),
    (3009201, 44777583, 0.001, 9040),
    # Free T4 -> ng/dL (8817)
    (3008598, 8729, 1.0/12.871, 8817),
    (3008598, 8845, 0.1, 8817),
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
            (F.col("measurement_concept_id") == 3020891) & (F.col("unit_concept_id") == 9289),
            F.lit(586323)
        ).when(
            F.col("conv_to_unit").isNotNull(),
            F.col("conv_to_unit")
        ).otherwise(F.col("unit_concept_id"))
    )
    .drop("conv_concept_id", "conv_from_unit", "conv_factor", "conv_to_unit")
    .withColumn(
        "lab_value_final",
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

# === 16b. Exclude null/NaN and non-physiologic lab values ===
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
    (3013603, 0, 10000),      # PSA (ng/mL)
]

ranges_df = spark.createDataFrame(
    physiologic_ranges,
    schema=T.StructType([
        T.StructField("measurement_concept_id", T.LongType(), False),
        T.StructField("physio_min", T.DoubleType(), False),
        T.StructField("physio_max", T.DoubleType(), False),
    ])
)

final_df_w_conversions = (
    final_df_w_conversions
    .join(F.broadcast(ranges_df), on="measurement_concept_id", how="left")
    .filter(
        F.col("physio_min").isNull()
        | (
            (F.col("lab_value_final") >= F.col("physio_min"))
            & (F.col("lab_value_final") <= F.col("physio_max"))
        )
    )
    .drop("physio_min", "physio_max")
)

# === 17. Compute days relative to events (positive when lab AFTER event) ===
df_to_upload = (
    final_df_w_conversions
    # --- Lab-level relative timing ---
    .withColumn(
        "days_relative_to_diagnosis",
        F.datediff(F.col("measurement_date"), F.col("diagnosis_date"))
    )
    .withColumn(
        "days_relative_to_ici_start",
        F.datediff(F.col("measurement_date"), F.col("ici_start_date"))
    )
    .withColumn(
        "days_relative_to_ici_discontinuation",
        F.datediff(F.col("measurement_date"), F.col("ici_discontinuation_date"))
    )
    .withColumn(
        "days_relative_to_last_followup",
        F.datediff(F.col("measurement_date"), F.col("last_followup_date"))
    )
    # --- Patient-level time-to-event variables ---
    .withColumn(
        "time_on_ici",
        F.datediff(F.col("ici_discontinuation_date"), F.col("ici_block_start_date"))
    )
    .withColumn(
        "event_ici_discontinued",
        F.when(
            F.col("ici_discontinuation_cause").isin("DEATH", "CENSORED"), F.lit(0)
        ).otherwise(F.lit(1))
    )
    .withColumn(
        "time_to_death_or_censor",
        F.datediff(
            F.coalesce(F.col("death_date"), F.col("last_followup_date")),
            F.col("diagnosis_date")
        )
    )
)

from pyspark.sql import functions as F, types as T

# === 18. Enforce canonical types + schema assertions ===

# ---- Define column groups ----
int_cols = [
    "person_id",
    "gender_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
    "age_at_diagnosis",
    "block_number",
    "is_deceased",
    "event_ici_discontinued",
    "time_on_ici",
    "time_to_death_or_censor",
    "measurement_concept_id",
    "unit_concept_id",
    "unit_converted_id",
    "days_relative_to_diagnosis",
    "days_relative_to_ici_start",
    "days_relative_to_ici_discontinuation",
    "days_relative_to_last_followup",
]

float_cols = [
    "lab_value",
    "lab_values_converted",
    "lab_value_final",
]

# FIX: added death_date
date_cols = [
    "date_of_birth",
    "diagnosis_date",
    "ici_start_date",
    "ici_block_start_date",
    "ici_end_date",
    "last_ici_cycle_start_date",
    "ici_discontinuation_date",
    "last_followup_date",
    "death_date",
    "measurement_date",
]

string_cols = [
    "gender",
    "race",
    "ethnicity",
    "cancer_type",
    "cancer_subtype",
    "ici_type",
    "lab_name",
    "lab_unit_name",
    "unit_converted_name",
    "ici_discontinuation_cause",
]

# ---- 1) Canonical casts ----
# Single select() replaces iterative withColumn() loops — each loop iteration
# added a new plan node; one select() emits a single projection stage.
_cast_map = (
    {c: "bigint"  for c in int_cols}
    | {c: "double" for c in float_cols}
    | {c: "date"   for c in date_cols}
    | {c: "string" for c in string_cols}
)
df_to_upload = df_to_upload.select(
    *[F.col(c).cast(_cast_map[c]).alias(c) if c in _cast_map else F.col(c)
      for c in df_to_upload.columns]
)

# ---- 2) Column presence assertion ----
# FIX: added death_date
expected_columns = [
    # IDs / demographics
    "person_id",
    "date_of_birth",
    "gender_concept_id",
    "gender",
    "race_concept_id",
    "race",
    "ethnicity_concept_id",
    "ethnicity",
    "age_at_diagnosis",

    # Cancer / ICI info
    "cancer_type",
    "cancer_subtype",
    "diagnosis_date",
    "ici_start_date",
    "last_ici_cycle_start_date",
    "ici_type",

    # Treatment block info
    "block_number",
    "ici_block_start_date",
    "ici_end_date",

    # Follow-up & death
    "last_followup_date",
    "death_date",
    "is_deceased",
    "ici_discontinuation_date",
    "ici_discontinuation_cause",

    # Time-to-event variables
    "time_on_ici",
    "event_ici_discontinued",
    "time_to_death_or_censor",

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

    # Lab-level relative timing
    "days_relative_to_diagnosis",
    "days_relative_to_ici_start",
    "days_relative_to_ici_discontinuation",
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
# Part 4: Summary Statistics
# ===============================================================

df_to_upload.cache()
df_to_upload.createOrReplaceTempView("final_cohort")
row_count = df_to_upload.count()
print(f"Cached final_cohort: {row_count:,} rows")

# Pre-compute a patient-block-level view (one row per person per block) for queries.
# This avoids repeated aggregations over the full lab-level table.
patient_df = spark.sql("""
    SELECT
        person_id,
        block_number,
        FIRST(gender) AS gender,
        FIRST(race) AS race,
        FIRST(ethnicity) AS ethnicity,
        FIRST(age_at_diagnosis) AS age_at_diagnosis,
        FIRST(date_of_birth) AS date_of_birth,
        FIRST(cancer_type) AS cancer_type,
        FIRST(cancer_subtype) AS cancer_subtype,
        FIRST(diagnosis_date) AS diagnosis_date,
        FIRST(ici_start_date) AS ici_start_date,
        FIRST(ici_type) AS ici_type,
        FIRST(ici_block_start_date) AS ici_block_start_date,
        FIRST(ici_end_date) AS ici_end_date,
        FIRST(last_ici_cycle_start_date) AS last_ici_cycle_start_date,
        FIRST(ici_discontinuation_date) AS ici_discontinuation_date,
        FIRST(ici_discontinuation_cause) AS ici_discontinuation_cause,
        FIRST(last_followup_date) AS last_followup_date,
        FIRST(death_date) AS death_date,
        FIRST(is_deceased) AS is_deceased,
        FIRST(time_on_ici) AS time_on_ici,
        FIRST(event_ici_discontinued) AS event_ici_discontinued,
        FIRST(time_to_death_or_censor) AS time_to_death_or_censor
    FROM final_cohort
    GROUP BY person_id, block_number
""")
patient_df.cache()
patient_df.createOrReplaceTempView("patient_cohort")
patient_block_count = patient_df.count()
patient_count = patient_df.select("person_id").distinct().count()
print(f"Cached patient_cohort: {patient_count:,} patients, {patient_block_count:,} patient-blocks")

# === 1. Overall cohort size ===
print("=" * 60)
print("COHORT OVERVIEW")
print("=" * 60)

spark.sql(f"""
    SELECT
        {patient_count} AS total_patients,
        {row_count} AS total_lab_records,
        (SELECT COUNT(DISTINCT cancer_type) FROM patient_cohort) AS distinct_cancer_types,
        (SELECT COUNT(DISTINCT ici_type) FROM patient_cohort) AS distinct_ici_types,
        (SELECT COUNT(*) FROM patient_cohort WHERE is_deceased = 1 AND block_number = 1) AS deceased_patients,
        ROUND((SELECT COUNT(*) FROM patient_cohort WHERE is_deceased = 1 AND block_number = 1) * 100.0
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
    WHERE block_number = 1
    GROUP BY gender
    ORDER BY n_patients DESC
""").show(truncate=False)

print("--- Race ---")
spark.sql(f"""
    SELECT race, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {patient_count}, 1) AS pct
    FROM patient_cohort
    WHERE block_number = 1
    GROUP BY race
    ORDER BY n_patients DESC
""").show(truncate=False)

print("--- Ethnicity ---")
spark.sql(f"""
    SELECT ethnicity, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {patient_count}, 1) AS pct
    FROM patient_cohort
    WHERE block_number = 1
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
    WHERE age_at_diagnosis IS NOT NULL AND block_number = 1
""").show(truncate=False)

# === 3. Cancer type & subtype distribution ===
print("=" * 60)
print("CANCER TYPE DISTRIBUTION")
print("=" * 60)

print("--- By Cancer Type ---")
spark.sql(f"""
    SELECT cancer_type, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {patient_count}, 1) AS pct
    FROM patient_cohort
    WHERE block_number = 1
    GROUP BY cancer_type
    ORDER BY n_patients DESC
""").show(50, truncate=False)

print("--- Top 20 Cancer Subtypes ---")
spark.sql("""
    SELECT cancer_type, cancer_subtype, COUNT(*) AS n_patients
    FROM patient_cohort
    WHERE block_number = 1
    GROUP BY cancer_type, cancer_subtype
    ORDER BY n_patients DESC
    LIMIT 20
""").show(20, truncate=False)

# === 4. ICI treatment summary ===
print("=" * 60)
print("ICI TREATMENT")
print("=" * 60)

print("--- By ICI Type ---")
spark.sql(f"""
    SELECT ici_type, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {patient_count}, 1) AS pct
    FROM patient_cohort
    WHERE block_number = 1
    GROUP BY ici_type
    ORDER BY n_patients DESC
""").show(truncate=False)

print("--- Time from Diagnosis to ICI Start ---")
spark.sql("""
    SELECT
        COUNT(*) AS n_patients,
        ROUND(AVG(days_dx_to_ici), 1) AS mean_days,
        ROUND(STDDEV(days_dx_to_ici), 1) AS std_days,
        MIN(days_dx_to_ici) AS min_days,
        PERCENTILE_APPROX(days_dx_to_ici, 0.25) AS q1_days,
        PERCENTILE_APPROX(days_dx_to_ici, 0.50) AS median_days,
        PERCENTILE_APPROX(days_dx_to_ici, 0.75) AS q3_days,
        MAX(days_dx_to_ici) AS max_days
    FROM (
        SELECT
            person_id,
            DATEDIFF(ici_start_date, diagnosis_date) AS days_dx_to_ici
        FROM patient_cohort
        WHERE ici_start_date IS NOT NULL AND block_number = 1
    )
""").show(truncate=False)

# === 5. ICI Discontinuation summary ===
print("=" * 60)
print("ICI DISCONTINUATION")
print("=" * 60)

print("--- Discontinuation Cause ---")
spark.sql("""
    SELECT ici_discontinuation_cause,
           COUNT(*) AS n_blocks,
           ROUND(COUNT(*) * 100.0
                 / (SELECT COUNT(*) FROM patient_cohort
                    WHERE ici_discontinuation_cause IS NOT NULL), 1) AS pct
    FROM patient_cohort
    WHERE ici_discontinuation_cause IS NOT NULL
    GROUP BY ici_discontinuation_cause
    ORDER BY n_blocks DESC
""").show(truncate=False)

print("--- ICI Discontinuation Event Indicator (all blocks) ---")
spark.sql(f"""
    SELECT
        event_ici_discontinued,
        COUNT(*) AS n_blocks,
        ROUND(COUNT(*) * 100.0 / {patient_block_count}, 1) AS pct
    FROM patient_cohort
    GROUP BY event_ici_discontinued
    ORDER BY event_ici_discontinued
""").show(truncate=False)

print("--- Time on ICI (ICI start to discontinuation, days) ---")
spark.sql("""
    SELECT
        COUNT(*) AS n_patients,
        ROUND(AVG(time_on_ici), 1) AS mean_days,
        ROUND(STDDEV(time_on_ici), 1) AS std_days,
        MIN(time_on_ici) AS min_days,
        PERCENTILE_APPROX(time_on_ici, 0.25) AS q1_days,
        PERCENTILE_APPROX(time_on_ici, 0.50) AS median_days,
        PERCENTILE_APPROX(time_on_ici, 0.75) AS q3_days,
        MAX(time_on_ici) AS max_days
    FROM patient_cohort
    WHERE time_on_ici IS NOT NULL
""").show(truncate=False)

print("--- Time on ICI by Discontinuation Cause ---")
spark.sql("""
    SELECT
        ici_discontinuation_cause,
        COUNT(*) AS n_blocks,
        ROUND(AVG(time_on_ici), 1) AS mean_days,
        PERCENTILE_APPROX(time_on_ici, 0.50) AS median_days,
        MIN(time_on_ici) AS min_days,
        MAX(time_on_ici) AS max_days
    FROM patient_cohort
    WHERE time_on_ici IS NOT NULL
    GROUP BY ici_discontinuation_cause
    ORDER BY n_blocks DESC
""").show(truncate=False)

print("--- Time to Death or Censor (days from diagnosis) ---")
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

# === 6. Lab measurement summary (using converted values) ===
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

# === 7. Unit conversion audit ===
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

# === 8. Follow-up / survival summary ===
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
            DATEDIFF(last_followup_date, diagnosis_date) AS followup_days
        FROM patient_cohort
        WHERE last_followup_date IS NOT NULL AND block_number = 1
    )
""").show(truncate=False)

# === 9. Null value audit ===
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
        SUM(CASE WHEN diagnosis_date IS NULL THEN 1 ELSE 0 END) AS null_diagnosis_date,
        SUM(CASE WHEN cancer_type IS NULL THEN 1 ELSE 0 END) AS null_cancer_type,
        SUM(CASE WHEN ici_start_date IS NULL THEN 1 ELSE 0 END) AS null_ici_start_date,
        SUM(CASE WHEN block_number IS NULL THEN 1 ELSE 0 END) AS null_block_number,
        SUM(CASE WHEN ici_block_start_date IS NULL THEN 1 ELSE 0 END) AS null_ici_block_start_date,
        SUM(CASE WHEN ici_end_date IS NULL THEN 1 ELSE 0 END) AS null_ici_end_date,
        SUM(CASE WHEN last_ici_cycle_start_date IS NULL THEN 1 ELSE 0 END) AS null_last_ici_cycle_start_date,
        SUM(CASE WHEN ici_type IS NULL THEN 1 ELSE 0 END) AS null_ici_type,
        SUM(CASE WHEN ici_discontinuation_date IS NULL THEN 1 ELSE 0 END) AS null_ici_discontinuation_date,
        SUM(CASE WHEN ici_discontinuation_cause IS NULL THEN 1 ELSE 0 END) AS null_ici_discontinuation_cause,
        SUM(CASE WHEN last_followup_date IS NULL THEN 1 ELSE 0 END) AS null_last_followup_date,
        SUM(CASE WHEN death_date IS NULL THEN 1 ELSE 0 END) AS null_death_date,
        SUM(CASE WHEN measurement_concept_id IS NULL THEN 1 ELSE 0 END) AS null_measurement_concept_id,
        SUM(CASE WHEN lab_name IS NULL THEN 1 ELSE 0 END) AS null_lab_name,
        SUM(CASE WHEN measurement_date IS NULL THEN 1 ELSE 0 END) AS null_measurement_date,
        SUM(CASE WHEN lab_value IS NULL THEN 1 ELSE 0 END) AS null_lab_value,
        SUM(CASE WHEN lab_value_final IS NULL THEN 1 ELSE 0 END) AS null_lab_value_final,
        SUM(CASE WHEN lab_values_converted IS NULL THEN 1 ELSE 0 END) AS null_lab_values_converted,
        SUM(CASE WHEN unit_concept_id IS NULL THEN 1 ELSE 0 END) AS null_unit_concept_id,
        SUM(CASE WHEN lab_unit_name IS NULL THEN 1 ELSE 0 END) AS null_lab_unit_name,
        SUM(CASE WHEN unit_converted_name IS NULL THEN 1 ELSE 0 END) AS null_unit_converted_name,
        SUM(CASE WHEN time_on_ici IS NULL THEN 1 ELSE 0 END) AS null_time_on_ici,
        SUM(CASE WHEN event_ici_discontinued IS NULL THEN 1 ELSE 0 END) AS null_event_ici_discontinued,
        SUM(CASE WHEN time_to_death_or_censor IS NULL THEN 1 ELSE 0 END) AS null_time_to_death_or_censor,
        SUM(CASE WHEN days_relative_to_diagnosis IS NULL THEN 1 ELSE 0 END) AS null_days_rel_diagnosis,
        SUM(CASE WHEN days_relative_to_ici_start IS NULL THEN 1 ELSE 0 END) AS null_days_rel_ici_start,
        SUM(CASE WHEN days_relative_to_ici_discontinuation IS NULL THEN 1 ELSE 0 END) AS null_days_rel_ici_discontinuation,
        SUM(CASE WHEN days_relative_to_last_followup IS NULL THEN 1 ELSE 0 END) AS null_days_rel_followup,
        SUM(CASE WHEN cancer_subtype IS NULL THEN 1 ELSE 0 END) AS null_cancer_subtype,
        SUM(CASE WHEN gender_concept_id IS NULL THEN 1 ELSE 0 END) AS null_gender_concept_id,
        SUM(CASE WHEN race_concept_id IS NULL THEN 1 ELSE 0 END) AS null_race_concept_id,
        SUM(CASE WHEN ethnicity_concept_id IS NULL THEN 1 ELSE 0 END) AS null_ethnicity_concept_id,
        SUM(CASE WHEN unit_converted_id IS NULL THEN 1 ELSE 0 END) AS null_unit_converted_id,
        SUM(CASE WHEN is_deceased IS NULL THEN 1 ELSE 0 END) AS null_is_deceased
    FROM final_cohort
""").show(truncate=False)
