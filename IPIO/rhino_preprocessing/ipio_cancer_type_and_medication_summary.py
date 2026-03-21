# ===============================================================
# IPIO Cohort Audit: Cancer Types & ICI Medications
# ===============================================================
# Builds the IPIO cohort (cancer dx + ICI exposure) inline and
# outputs summary statistics on cancer types and ICI medications.
# ===============================================================

import os
_cwd = "file:" + os.getcwd()
spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true") \
    .csv(os.path.join(_cwd, "concept_tables", "concept_subset.csv")) \
    .createOrReplaceTempView("concept")
spark.read.option("header", "true").option("sep", "\t").option("inferSchema", "true") \
    .csv(os.path.join(_cwd, "concept_tables", "concept_ancestor_subset.csv")) \
    .createOrReplaceTempView("concept_ancestor")

# ===============================================================
# Part 1: Concept sets
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

# Text-based fallback: cancer concepts whose names match NSCLC patterns
# but weren't in the pre-expanded NSCLC concept ID list.
# Excludes "small cell" to avoid SCLC contamination.
spark.sql("""
CREATE OR REPLACE TEMP VIEW cancer_concepts_labeled AS
SELECT
    c.concept_id,
    c.concept_name,
    COALESCE(
        l.cancer_type,
        CASE
            WHEN (LOWER(c.concept_name) LIKE '%lung%'
               OR LOWER(c.concept_name) LIKE '%bronch%'
               OR LOWER(c.concept_name) LIKE '%pulmonary%'
               OR LOWER(c.concept_name) LIKE '%respiratory%')
              AND LOWER(c.concept_name) NOT LIKE '%small cell%'
            THEN 'NSCLC'
            ELSE c.concept_name
        END
    ) AS cancer_type
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

# ===============================================================
# Part 2: Build cohort
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

# === First ICI drug name per patient (ingredient-level) ===
spark.sql("""
CREATE OR REPLACE TEMP VIEW temp_ici_first_drug AS
SELECT person_id, ici_drug_name
FROM (
    SELECT
        de.person_id,
        ing.concept_name AS ici_drug_name,
        ROW_NUMBER() OVER (PARTITION BY de.person_id ORDER BY de.drug_exposure_start_date, de.drug_concept_id) AS rn
    FROM dn_drug_exposure_20251219 de
    JOIN temp_ICI_concepts ici ON de.drug_concept_id = ici.concept_id
    JOIN concept_ancestor ca ON de.drug_concept_id = ca.descendant_concept_id
    JOIN concept ing ON ca.ancestor_concept_id = ing.concept_id
    WHERE ing.concept_id IN (SELECT concept_id FROM temp_ICI_ingredients)
) ranked
WHERE rn = 1
""")

_cohort = spark.table("temp_cancer_ici_patients").cache()
cohort_size = _cohort.count()
_cohort.createOrReplaceTempView("temp_cancer_ici_patients")
print(f"Cohort size: {cohort_size:,} patients")

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
print("ICI MEDICATION — OVERALL (FIRST ICI PER PATIENT)")
print("=" * 60)
spark.sql(f"""
    SELECT d.ici_drug_name, COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / {cohort_size}, 1) AS pct
    FROM temp_ici_first_drug d
    JOIN temp_cancer_ici_patients p ON d.person_id = p.person_id
    GROUP BY d.ici_drug_name
    ORDER BY n_patients DESC
""").show(20, truncate=False)

print("=" * 60)
print("ICI MEDICATION BY CANCER TYPE")
print("=" * 60)
spark.sql("""
    SELECT p.cancer_type, d.ici_drug_name,
           COUNT(*) AS n_patients,
           ROUND(COUNT(*) * 100.0 / ct.n, 1) AS pct_within_type
    FROM temp_ici_first_drug d
    JOIN temp_cancer_ici_patients p ON d.person_id = p.person_id
    JOIN (
        SELECT cancer_type, COUNT(*) AS n
        FROM temp_cancer_ici_patients
        GROUP BY cancer_type
    ) ct ON p.cancer_type = ct.cancer_type
    GROUP BY p.cancer_type, d.ici_drug_name, ct.n
    ORDER BY p.cancer_type, n_patients DESC
""").show(50, truncate=False)