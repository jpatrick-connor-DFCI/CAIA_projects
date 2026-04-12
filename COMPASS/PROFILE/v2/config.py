import os
from pathlib import Path

DEFAULT_DATA_PATH = Path(
    os.environ.get("CAIA_COMPASS_DATA_PATH", "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("CAIA_COMPASS_V2_OUTPUT_DIR", str(DEFAULT_DATA_PATH / "v2_outputs"))
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

TOTAL_PSA_LABELS = {"PSA", "PSAR", "PSATOTSCRN", "CPSA", "PSAMON", "PSAULT", "PSAT"}

FOCUSED_NOTE_GREP_REGEX = {
    "neuroendocrine": (
        r"\b(?:"
        r"neuroendocrine|neuro-endocrine|nepc|t-nepc|anaplastic"
        r")\b"
    ),
    "small_cell": (
        r"\b(?:"
        r"small[\s-]?cell|small[\s-]?cell\s+carcinoma|scpc|scnc|oat[\s-]?cell"
        r")\b"
    ),
    "transformation": (
        r"\b(?:"
        r"histolog(?:ic|ical)\s+transform(?:ation|ed|ing)|"
        r"transform(?:ation|ed|ing)(?:\s+(?:to|into))?|"
        r"transdifferentiat(?:e|ed|ion|ing)|"
        r"dedifferentiat(?:e|ed|ion|ing)|"
        r"lineage\s+plasticity|"
        r"treatment[\s-]?emergent\s+neuroendocrine"
        r")\b"
    ),
}

FOCUSED_PROSTATE_CONTEXT_REGEX = (
    r"\b(?:"
    r"prostate|prostatic|psa|adenocarcinoma|acinar|ductal|mcrpc|crpc|castration[- ]resistant"
    r")\b"
)

FOCUSED_SNIPPET_CONTEXT_CHARS = 240
FOCUSED_SNIPPET_MAX_CHARS = 2400
FOCUSED_SNIPPET_MAX_MATCHES = 12

NOTE_TYPE_LIMITS = {
    "Clinician": 24,
    "Imaging": 16,
    "Pathology": 16,
}

DEFAULT_PLATINUM_WINDOW_DAYS = 120
