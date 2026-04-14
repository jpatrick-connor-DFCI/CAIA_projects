import os
from pathlib import Path


DEFAULT_DATA_PATH = Path(
    os.environ.get("CAIA_COMPASS_DATA_PATH", "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("CAIA_COMPASS_V3_OUTPUT_DIR", str(DEFAULT_DATA_PATH / "v3_outputs"))
)
_raw_text_paths_env = os.environ.get("CAIA_ONCDRS_RAW_TEXT_PATHS")
_legacy_raw_text_path_env = os.environ.get("CAIA_ONCDRS_RAW_TEXT_PATH")
if _raw_text_paths_env:
    DEFAULT_RAW_TEXT_PATHS = tuple(
        Path(path_str) for path_str in _raw_text_paths_env.split(os.pathsep) if path_str.strip()
    )
elif _legacy_raw_text_path_env:
    DEFAULT_RAW_TEXT_PATHS = (Path(_legacy_raw_text_path_env),)
else:
    DEFAULT_RAW_TEXT_PATHS = (
        Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2024_03/"),
        Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_03/"),
        Path("/data/gusev/PROFILE/CLINICAL/OncDRS/CLINICAL_TEXTS_2025_11/"),
    )
DEFAULT_RAW_TEXT_PATH = DEFAULT_RAW_TEXT_PATHS[-1]

DEFAULT_AZURE_OPENAI_ENDPOINT = os.environ.get(
    "CAIA_AZURE_OPENAI_ENDPOINT",
    "https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
)
DEFAULT_AZURE_OPENAI_API_VERSION = os.environ.get(
    "CAIA_AZURE_OPENAI_API_VERSION",
    "2024-04-01-preview",
)
DEFAULT_MODEL_NAME = os.environ.get("CAIA_AZURE_OPENAI_MODEL", "gpt-4o")

PLATINUM_MEDS = {"CISPLATIN", "CARBOPLATIN"}
ADT_MEDS = {
    "LEUPROLIDE",
    "LEUPROLIDE ACETATE",
    "LUPRON",
    "ELIGARD",
    "GOSERELIN",
    "ZOLADEX",
    "DEGARELIX",
    "FIRMAGON",
    "RELUGOLIX",
    "ORGOVYX",
    "TRIPTORELIN",
    "TRELSTAR",
    "HISTRELIN",
    "VANTAS",
    "SUPPRELIN",
}
ARSI_MEDS = {
    "ABIRATERONE",
    "ZYTIGA",
    "ENZALUTAMIDE",
    "XTANDI",
    "APALUTAMIDE",
    "ERLEADA",
    "DAROLUTAMIDE",
    "NUBEQA",
}
PARP_MEDS = {
    "OLAPARIB",
    "RUCAPARIB",
    "NIRAPARIB",
    "TALAZOPARIB",
}
TAXANE_MEDS = {
    "DOCETAXEL",
    "TAXOTERE",
    "CABAZITAXEL",
    "JEVTANA",
}

TOTAL_PSA_LABELS = {"PSA", "PSAR", "PSATOTSCRN", "CPSA", "PSAMON", "PSAULT", "PSAT"}

PROSTATE_CONTEXT_REGEX = (
    r"\b(?:"
    r"prostate|prostatic|psa|adenocarcinoma|acinar|ductal|mcrpc|crpc|castration[- ]resistant|"
    r"androgen[- ]independent|metastatic castration resistant"
    r")\b"
)

TRIGGER_REGEX = {
    "nepc_scpc": (
        r"\b(?:"
        r"neuroendocrine|neuro-endocrine|nepc|t-nepc|"
        r"small[\s-]?cell|small[\s-]?cell\s+carcinoma|scpc|scnc|oat[\s-]?cell|"
        r"neuroendocrine carcinoma|small[- ]cell neuroendocrine carcinoma"
        r")\b"
    ),
    "transformation": (
        r"\b(?:"
        r"histolog(?:ic|ical)\s+transform(?:ation|ed|ing)|"
        r"transform(?:ation|ed|ing)(?:\s+(?:to|into))?|"
        r"transdifferentiat(?:e|ed|ion|ing)|"
        r"dedifferentiat(?:e|ed|ion|ing)|"
        r"lineage\s+plasticity|"
        r"evolved\s+to|converted\s+to|emerged\s+as|"
        r"treatment[\s-]?emergent\s+neuroendocrine"
        r")\b"
    ),
    "aggressive_variant": (
        r"\b(?:"
        r"aggressive[\s-]?variant|avpc|anaplastic|variant\s+crpc|"
        r"aggressive\s+subtype|androgen[- ]indifferent"
        r")\b"
    ),
    "visceral": (
        r"\b(?:"
        r"visceral\s+met(?:astases|astasis|astatic)?|liver\s+met(?:astases|astasis|astatic)?|"
        r"hepatic\s+met(?:astases|astasis|astatic)?|lung\s+met(?:astases|astasis|astatic)?|"
        r"pulmonary\s+met(?:astases|astasis|astatic)?|adrenal\s+met(?:astases|astasis|astatic)?|"
        r"brain\s+met(?:astases|astasis|astatic)?|pleural\s+met(?:astases|astasis|astatic)?|"
        r"peritoneal\s+met(?:astases|astasis|astatic)?"
        r")\b"
    ),
    "lytic_bone": (
        r"\b(?:"
        r"lytic\s+bone|lytic\s+lesion|predominantly\s+lytic|osseous\s+lytic|"
        r"destructive\s+bone\s+lesion"
        r")\b"
    ),
    "bulky": (
        r"\b(?:"
        r"bulky\s+lymphadenopathy|bulky\s+adenopathy|bulky\s+pelvic\s+mass|"
        r"bulky\s+prostate\s+mass|large\s+pelvic\s+mass|large\s+prostatic\s+mass|"
        r"bulky\s+nodes?|bulky\s+nodal"
        r")\b"
    ),
    "low_psa_high_burden": (
        r"\b(?:"
        r"low\s+psa|disproportionately\s+low\s+psa|psa\s+discordant|"
        r"high[- ]volume\s+bone|extensive\s+bone\s+met(?:astases|astatic)?|"
        r"diffuse\s+osseous\s+met(?:astases|astatic)?|innumerable\s+bone\s+met(?:astases|astatic)?"
        r")\b"
    ),
    "ne_marker": (
        r"\b(?:"
        r"chromogranin(?:\s+a)?|synaptophysin|bombesin|gastrin[- ]releasing\s+peptide|grp|"
        r"nse|neuron[- ]specific\s+enolase|cd56"
        r")\b"
    ),
    "biomarker": (
        r"\b(?:"
        r"brca1|brca2|atm|cdk12|palb2|hrd|hrr|ddr|homologous\s+recombination|"
        r"dna\s+damage\s+repair|msi[- ]h|msi[- ]high|mmr|mismatch\s+repair|msh2|msh6|mlh1|pms2|tmb"
        r")\b"
    ),
    "treatment_resistance": (
        r"\b(?:"
        r"crpc|mcrpc|castration[- ]resistant|androgen[- ]independent|"
        r"refractory|progressed\s+on|resistant\s+to|rapid\s+progression|"
        r"poor\s+response\s+to\s+adt|despite\s+adt|despite\s+enzalutamide|despite\s+abiraterone"
        r")\b"
    ),
    "platinum": r"\b(?:carboplatin|cisplatin|platinum[- ]based)\b",
}

TRIGGER_SCORE_WEIGHTS = {
    "nepc_scpc": 4,
    "transformation": 5,
    "aggressive_variant": 4,
    "visceral": 3,
    "lytic_bone": 3,
    "bulky": 3,
    "low_psa_high_burden": 3,
    "ne_marker": 3,
    "biomarker": 2,
    "treatment_resistance": 2,
    "platinum": 2,
}

SNIPPET_CONTEXT_CHARS = 240
SNIPPET_MAX_CHARS = 2800
SNIPPET_MAX_MATCHES = 14
FALLBACK_NOTE_MAX_CHARS = 2400

NOTE_TYPE_LIMITS = {
    "Clinician": 28,
    "Imaging": 18,
    "Pathology": 18,
}
PLATINUM_WINDOW_BEFORE_DAYS = 120
PLATINUM_WINDOW_AFTER_DAYS = 60
PLATINUM_FALLBACK_LIMITS = {
    "Clinician": 3,
    "Imaging": 2,
    "Pathology": 2,
}

LDH_ULN = 250.0
CEA_ULN = 5.0
HYPERCALCEMIA_THRESHOLD = 10.5

SOMATIC_TARGET_PATTERNS = {
    "has_brca1_alteration": ("BRCA1",),
    "has_brca2_alteration": ("BRCA2",),
    "has_atm_alteration": ("ATM",),
    "has_cdk12_alteration": ("CDK12",),
    "has_palb2_alteration": ("PALB2",),
    "has_msi_high_signal": ("MSI", "MISMATCH", "MMR"),
    "has_hrd_pathway_alteration": ("HRD", "HRR", "HOMOLOGOUS_RECOMBINATION"),
    "has_ddr_pathway_alteration": ("DDR", "DNA_DAMAGE_REPAIR"),
}
