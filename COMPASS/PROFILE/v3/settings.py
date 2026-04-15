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

DEFAULT_AZURE_OPENAI_ENDPOINT = os.environ.get(
    "CAIA_AZURE_OPENAI_ENDPOINT",
    "https://je76vfkda5tn4-openai-eastus2.openai.azure.com/",
)
DEFAULT_AZURE_OPENAI_API_VERSION = os.environ.get(
    "CAIA_AZURE_OPENAI_API_VERSION",
    "2024-04-01-preview",
)
DEFAULT_MODEL_NAME = os.environ.get("CAIA_AZURE_OPENAI_MODEL", "gpt-4o")

ARM_NAMES = ("nepc", "avpc", "biomarker")

PROSTATE_CONTEXT_REGEX = (
    r"\b(?:"
    r"prostate|prostatic|psa|adenocarcinoma|acinar|ductal|mcrpc|crpc|"
    r"castration[- ]resistant|metastatic\s+castration[- ]resistant"
    r")\b"
)

SNIPPET_CONTEXT_CHARS = 300
SNIPPET_MAX_CHARS = 3200
SNIPPET_MAX_MATCHES = 16
