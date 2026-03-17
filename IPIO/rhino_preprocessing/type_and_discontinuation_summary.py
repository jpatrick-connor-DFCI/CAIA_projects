# ===============================================================
# IPIO Cohort Audit: Cancer Types & Discontinuation Causes
# ===============================================================
# Builds the IPIO cohort (cancer dx + ICI exposure) inline and
# computes ICI treatment blocks with discontinuation classification.
# Outputs summary statistics on cancer types and discontinuation.
#
# Uses the same concept sets and 90-day block logic as the main
# IPIO preprocessing pipeline.
# ===============================================================

import os
import pandas as pd
_cwd = os.getcwd()
spark.createDataFrame(pd.read_csv(os.path.join(_cwd, "concept_tables", "concept_subset.csv"), sep="\t")).createOrReplaceTempView("concept")
spark.createDataFrame(pd.read_csv(os.path.join(_cwd, "concept_tables", "concept_ancestor_subset.csv"), sep="\t")).createOrReplaceTempView("concept_ancestor")

# ===============================================================
# Part 1: Concept sets (same as IPIO pipeline)
# ===============================================================

# === 1. Cancer concepts (all descendants of malignant neoplastic disease 443392) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW cancer_concepts AS
SELECT DISTINCT c.concept_id, c.concept_name
FROM concept_ancestor ca
JOIN concept c ON ca.descendant_concept_id = c.concept_id
WHERE ca.ancestor_concept_id = 443392
  AND c.invalid_reason IS NULL
""")

# === 2. Cancer type labels ===
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

spark.sql("""
CREATE OR REPLACE TEMP VIEW cancer_concepts_labeled AS
SELECT
    c.concept_id,
    c.concept_name,
    COALESCE(l.cancer_type, c.concept_name) AS cancer_type
FROM cancer_concepts c
LEFT JOIN cancer_labels l ON c.concept_id = l.concept_id
""")

# === 3. ICI concepts ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ICI_ingredients AS
SELECT concept_id FROM VALUES
  (45892628),(45775965),(42629079),(1594034),(40238188),
  (35200783),(1593273),(741851),(1536789),(1302024),(747052)
AS t(concept_id)
""")

spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ICI_concepts AS
SELECT DISTINCT ca.descendant_concept_id AS concept_id
FROM concept_ancestor ca
WHERE ca.ancestor_concept_id IN (SELECT concept_id FROM temp_ICI_ingredients)
""")

# === 4. Non-ICI antineoplastic drugs (ATC L01 minus ICI) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_antineoplastic_non_ici AS
SELECT DISTINCT ca.descendant_concept_id AS drug_concept_id
FROM concept_ancestor ca
LEFT JOIN temp_ICI_concepts ici ON ca.descendant_concept_id = ici.concept_id
WHERE ca.ancestor_concept_id = 21601387
  AND ici.concept_id IS NULL
""")

# === 5. Systemic corticosteroids (ATC H02AB) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_systemic_corticosteroids AS
SELECT DISTINCT ca.descendant_concept_id AS drug_concept_id
FROM concept_ancestor ca
WHERE ca.ancestor_concept_id = 21602728
""")

# ===============================================================
# Part 2: Build cohort and treatment blocks
# ===============================================================
spark.sql("USE CATALOG dfci_ia_aistudio")
spark.sql("USE SCHEMA omop_caia_denorm")

# === Cancer diagnoses (deduplicated per patient per cancer_type) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_cancer_dx AS
SELECT person_id, cancer_type, MIN(diagnosis_date) AS diagnosis_date
FROM (
    SELECT co.person_id, co.condition_start_date AS diagnosis_date, cl.cancer_type
    FROM dn_condition_occurrence_20251219 co
    JOIN cancer_concepts_labeled cl ON co.condition_concept_id = cl.concept_id
) sub
GROUP BY person_id, cancer_type
""")

# === First ICI exposure per patient ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ici_treatment AS
SELECT person_id, ici_start_date
FROM (
    SELECT de.person_id, de.drug_exposure_start_date AS ici_start_date,
           ROW_NUMBER() OVER (PARTITION BY de.person_id ORDER BY de.drug_exposure_start_date) AS rn
    FROM dn_drug_exposure_20251219 de
    JOIN temp_ICI_concepts ici ON de.drug_concept_id = ici.concept_id
) ranked
WHERE rn = 1
""")

# === Eligible: cancer + ICI after diagnosis (one row per patient) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_cancer_ici_patients AS
SELECT person_id, cancer_type, diagnosis_date, ici_start_date
FROM (
    SELECT dx.person_id, dx.cancer_type, dx.diagnosis_date, ici.ici_start_date,
           ROW_NUMBER() OVER (PARTITION BY dx.person_id ORDER BY dx.diagnosis_date) AS rn
    FROM temp_cancer_dx dx
    JOIN temp_ici_treatment ici ON dx.person_id = ici.person_id
    WHERE ici.ici_start_date > dx.diagnosis_date
) ranked
WHERE rn = 1
""")

_cohort = spark.table("temp_cancer_ici_patients").cache()
cohort_size = _cohort.count()
_cohort.createOrReplaceTempView("temp_cancer_ici_patients")
print(f"Cohort size: {cohort_size:,} patients")

# === ICI treatment blocks (90-day gap) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ici_block_summary AS
WITH ici_exposures AS (
    SELECT de.person_id, de.drug_exposure_start_date,
           MAX(COALESCE(de.drug_exposure_end_date, de.drug_exposure_start_date)) AS drug_exposure_end_date
    FROM dn_drug_exposure_20251219 de
    JOIN temp_ICI_concepts ici ON de.drug_concept_id = ici.concept_id
    WHERE de.person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
    GROUP BY de.person_id, de.drug_exposure_start_date
),
gaps AS (
    SELECT *, CASE WHEN LAG(drug_exposure_end_date) OVER (PARTITION BY person_id ORDER BY drug_exposure_start_date)
                        < drug_exposure_start_date - INTERVAL 90 DAY THEN 1 ELSE 0 END AS new_block_flag
    FROM ici_exposures
),
blocks AS (
    SELECT *, SUM(new_block_flag) OVER (PARTITION BY person_id ORDER BY drug_exposure_start_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS block_id
    FROM gaps
)
SELECT person_id, block_id,
       MIN(drug_exposure_start_date) AS ici_block_start_date,
       MAX(drug_exposure_end_date) AS ici_end_date
FROM blocks
GROUP BY person_id, block_id
""")

# === Censor dates (last non-ICI clinical event) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_censor_dates AS
WITH events AS (
    SELECT person_id, MAX(event_date) AS last_event_date
    FROM (
        SELECT person_id, condition_start_date AS event_date FROM dn_condition_occurrence_20251219 WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
        UNION ALL
        SELECT person_id, measurement_date FROM dn_measurement_20251219 WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
        UNION ALL
        SELECT de.person_id, de.drug_exposure_start_date FROM dn_drug_exposure_20251219 de
            LEFT JOIN temp_ICI_concepts ici ON de.drug_concept_id = ici.concept_id
            WHERE ici.concept_id IS NULL AND de.person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
        UNION ALL
        SELECT person_id, procedure_date FROM dn_procedure_occurrence_20251219 WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
        UNION ALL
        SELECT person_id, visit_start_date FROM dn_visit_occurrence_20251219 WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
    ) e
    GROUP BY person_id
)
SELECT ev.person_id, ev.last_event_date AS observation_period_end_date, d.death_date
FROM events ev
LEFT JOIN dn_death_20251219 d ON ev.person_id = d.person_id
""")

# === Cohort drug exposures (antineoplastic + corticosteroid only) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_cohort_drug_exposures AS
SELECT person_id, drug_concept_id, drug_exposure_start_date
FROM dn_drug_exposure_20251219
WHERE person_id IN (SELECT person_id FROM temp_cancer_ici_patients)
  AND (drug_concept_id IN (SELECT drug_concept_id FROM temp_antineoplastic_non_ici)
    OR drug_concept_id IN (SELECT drug_concept_id FROM temp_systemic_corticosteroids))
""")

# === Classify discontinuation per block ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_discontinuation AS
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
    SELECT b.person_id, b.block_id, b.ici_block_start_date, b.ici_end_date,
        CASE
            WHEN c.death_date IS NOT NULL AND c.death_date <= b.ici_end_date THEN 'DEATH'
            WHEN nt.person_id IS NOT NULL THEN 'PROGRESSION'
            WHEN su.person_id IS NOT NULL THEN 'TOXICITY'
            WHEN c.observation_period_end_date IS NOT NULL
                 AND c.observation_period_end_date > b.ici_end_date + INTERVAL 90 DAY THEN 'COMPLETED'
            ELSE 'CENSORED'
        END AS ici_discontinuation_cause,
        ROW_NUMBER() OVER (PARTITION BY b.person_id ORDER BY b.ici_block_start_date) AS block_number
    FROM temp_ici_block_summary b
    LEFT JOIN temp_censor_dates c ON b.person_id = c.person_id
    LEFT JOIN new_therapy nt ON b.person_id = nt.person_id AND b.block_id = nt.block_id
    LEFT JOIN steroid_use su ON b.person_id = su.person_id AND b.block_id = su.block_id
)
SELECT * FROM classified
""")

# ===============================================================
# Part 3: Summary Statistics
# ===============================================================
print()
print("=" * 60)
print("CANCER TYPE DISTRIBUTION")
print("=" * 60)
spark.sql(f"""
    SELECT cancer_type, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {cohort_size}, 1) AS pct
    FROM temp_cancer_ici_patients
    GROUP BY cancer_type
    ORDER BY n_patients DESC
""").show(50, truncate=False)

print("=" * 60)
print("DISCONTINUATION CAUSE — ALL BLOCKS")
print("=" * 60)
spark.sql("""
    SELECT ici_discontinuation_cause, COUNT(*) AS n_blocks,
           ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM temp_discontinuation), 1) AS pct
    FROM temp_discontinuation
    GROUP BY ici_discontinuation_cause
    ORDER BY n_blocks DESC
""").show(truncate=False)

print("=" * 60)
print("DISCONTINUATION CAUSE — BLOCK 1 ONLY")
print("=" * 60)
spark.sql(f"""
    SELECT ici_discontinuation_cause, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {cohort_size}, 1) AS pct
    FROM temp_discontinuation
    WHERE block_number = 1
    GROUP BY ici_discontinuation_cause
    ORDER BY n_patients DESC
""").show(truncate=False)

print("=" * 60)
print("DISCONTINUATION CAUSE BY CANCER TYPE (BLOCK 1)")
print("=" * 60)
spark.sql("""
    SELECT p.cancer_type, d.ici_discontinuation_cause,
           COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / ct_total.n, 1) AS pct_within_type
    FROM temp_discontinuation d
    JOIN temp_cancer_ici_patients p ON d.person_id = p.person_id
    JOIN (
        SELECT cancer_type, COUNT(*) AS n
        FROM temp_cancer_ici_patients
        GROUP BY cancer_type
    ) ct_total ON p.cancer_type = ct_total.cancer_type
    WHERE d.block_number = 1
    GROUP BY p.cancer_type, d.ici_discontinuation_cause, ct_total.n
    ORDER BY p.cancer_type, n_patients DESC
""").show(50, truncate=False)

print("=" * 60)
print("BLOCKS PER PATIENT")
print("=" * 60)
spark.sql(f"""
    SELECT n_blocks, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {cohort_size}, 1) AS pct
    FROM (
        SELECT person_id, MAX(block_number) AS n_blocks
        FROM temp_discontinuation
        GROUP BY person_id
    )
    GROUP BY n_blocks
    ORDER BY n_blocks
""").show(20, truncate=False)

print("=" * 60)
print("MULTI-BLOCK RATE BY CANCER TYPE")
print("=" * 60)
spark.sql("""
    SELECT p.cancer_type,
           COUNT(DISTINCT p.person_id) AS n_patients,
           SUM(CASE WHEN blk.n_blocks > 1 THEN 1 ELSE 0 END) AS n_multi_block,
           ROUND(SUM(CASE WHEN blk.n_blocks > 1 THEN 1 ELSE 0 END) * 100.0
                 / COUNT(DISTINCT p.person_id), 1) AS pct_multi_block
    FROM temp_cancer_ici_patients p
    JOIN (
        SELECT person_id, MAX(block_number) AS n_blocks
        FROM temp_discontinuation
        GROUP BY person_id
    ) blk ON p.person_id = blk.person_id
    GROUP BY p.cancer_type
    ORDER BY n_patients DESC
""").show(50, truncate=False)