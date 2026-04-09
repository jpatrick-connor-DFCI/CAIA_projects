from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


PROFILE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_CSV = PROFILE_DIR / "OMOP_to_DFCI_lab_ids.csv"
DEFAULT_UNIQUE_LABS_CSV = PROFILE_DIR / "unique_lab_ids_w_units.csv"
STALE_UNMAPPED_NOTE_FRAGMENTS = (
    "No confident match found",
    "No supported TEST_NAME/unit combinations found",
)
COMBINED_BP_NAME_PATTERNS = (
    r"systolic/diastolic",
    r"blood pressure",
)


def merge_unit_rules(*rule_sets: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for rule_set in rule_sets:
        merged.update(rule_set)
    return merged


COUNT_1E3_PER_UL = {
    "k/ul": 1.0,
    "10*3/ul": 1.0,
    "10^3/ul": 1.0,
    "x103/ul": 1.0,
    "k/mm3": 1.0,
    "th/cmm": 1.0,
}

RBC_1E6_PER_UL = {
    "m/ul": 1.0,
    "10*6/ul": 1.0,
    "10^6/ul": 1.0,
    "x106/ul": 1.0,
    "m/mm3": 1.0,
}

TIME_SECONDS = {
    "sec": 1.0,
    "secs": 1.0,
    "seconds": 1.0,
    "s": 1.0,
}

MASS_MG_DL = {
    "mg/dl": 1.0,
}

MASS_G_DL = {
    "g/dl": 1.0,
    "gm/dl": 1.0,
}

ACTIVITY_UL = {
    "u/l": 1.0,
    "iu/l": 1.0,
}

COUNT_RULES_WITH_PER_UL = merge_unit_rules(
    COUNT_1E3_PER_UL,
    {
        "/ul": 0.001,
        "cell/mm3": 0.001,
    },
)

RULES_BY_MEASUREMENT = {
    "WBC": {
        "canonical_unit": "thousand per microliter",
        "unit_factors": COUNT_1E3_PER_UL,
    },
    "Hemoglobin": {
        "canonical_unit": "gram per deciliter",
        "unit_factors": merge_unit_rules(
            MASS_G_DL,
            {
                "gm/l": 0.1,
            },
        ),
    },
    "CEA": {
        "canonical_unit": "nanogram per milliliter",
        "unit_factors": {
            "ng/ml": 1.0,
        },
    },
    "Lymphocytes absolute": {
        "canonical_unit": "thousand per microliter",
        "unit_factors": COUNT_RULES_WITH_PER_UL,
    },
    "Glucose": {
        "canonical_unit": "milligram per deciliter",
        "unit_factors": MASS_MG_DL,
    },
    "Calcium": {
        "canonical_unit": "milligram per deciliter",
        "unit_factors": MASS_MG_DL,
    },
    "ALT": {
        "canonical_unit": "international unit per liter",
        "unit_factors": ACTIVITY_UL,
        "extra_prefixes": {"ALT"},
    },
    "Creatine kinase": {
        "canonical_unit": "international unit per liter",
        "unit_factors": ACTIVITY_UL,
    },
    "Free T4": {
        "canonical_unit": "nanogram per deciliter",
        "unit_factors": {
            "ng/dl": 1.0,
            "pmol/l": 1.0 / 12.871,
            "pg/ml": 0.1,
        },
        "extra_prefixes": {"FT"},
    },
    "Testosterone": {
        "canonical_unit": "nanogram per deciliter",
        "unit_factors": {
            "ng/dl": 1.0,
        },
    },
    "TSH": {
        "canonical_unit": "milli-international unit per liter",
        "unit_factors": {
            "uiu/ml": 1.0,
            "miu/l": 1.0,
            "iu/ml": 1000000.0,
            "mu/ml": 1000.0,
            "uu/ml": 1.0,
            "miu/ml": 1000.0,
            "mu/l": 1.0,
            "uiu/l": 0.001,
        },
    },
    "AFP": {
        "canonical_unit": "nanogram per milliliter",
        "unit_factors": {
            "ng/ml": 1.0,
        },
    },
    "MCHC": {
        "canonical_unit": "gram per deciliter",
        # MCHC is historically reported as either g/dL or percent with the same numeric scale.
        "unit_factors": merge_unit_rules(
            MASS_G_DL,
            {
                "%": 1.0,
            },
        ),
    },
    "MCH": {
        "canonical_unit": "picogram",
        "unit_factors": {
            "pg": 1.0,
        },
    },
    "Basophils absolute": {
        "canonical_unit": "thousand per microliter",
        "unit_factors": COUNT_1E3_PER_UL,
    },
    "aPTT": {
        "canonical_unit": "second",
        "unit_factors": TIME_SECONDS,
    },
    "PSA": {
        "canonical_unit": "nanogram per milliliter",
        "unit_factors": {
            "ng/ml": 1.0,
        },
    },
    "Neutrophils absolute": {
        "canonical_unit": "thousand per microliter",
        "unit_factors": COUNT_RULES_WITH_PER_UL,
    },
    "BUN": {
        "canonical_unit": "milligram per deciliter",
        "unit_factors": MASS_MG_DL,
    },
    "ESR": {
        "canonical_unit": "millimeter per hour",
        "unit_factors": {
            "mm/hr": 1.0,
            "mm/h": 1.0,
        },
    },
    "AST": {
        "canonical_unit": "international unit per liter",
        "unit_factors": ACTIVITY_UL,
    },
    "Chloride": {
        "canonical_unit": "milliequivalent per liter",
        "unit_factors": {
            "mmol/l": 1.0,
            "meq/l": 1.0,
        },
        "extra_prefixes": {"PCL"},
    },
    "CO2": {
        "canonical_unit": "milliequivalent per liter",
        "unit_factors": {
            "mmol/l": 1.0,
            "meq/l": 1.0,
            "nmol/l": 0.000001,
        },
        "extra_prefixes": {"PLCO2"},
    },
    "Fibrinogen": {
        "canonical_unit": "milligram per deciliter",
        "unit_factors": MASS_MG_DL,
    },
    "LDH": {
        "canonical_unit": "international unit per liter",
        "unit_factors": ACTIVITY_UL,
    },
    "Creatinine": {
        "canonical_unit": "milligram per deciliter",
        "unit_factors": MASS_MG_DL,
    },
    "Sodium": {
        "canonical_unit": "milliequivalent per liter",
        "unit_factors": {
            "mmol/l": 1.0,
            "meq/l": 1.0,
        },
        "extra_test_names": {
            "SODIUM",
            "SODIUM | CDR EHS LAB LRR",
            "NA,SODIUM",
            "SODIUM LEVEL | CDR MRMC LAB LRR",
            "SODIUM | CDR SSHCC LAB LRR",
        },
    },
    "RDW": {
        "canonical_unit": "percent",
        "unit_factors": {
            "%": 1.0,
        },
    },
    "RBC": {
        "canonical_unit": "million per microliter",
        "unit_factors": RBC_1E6_PER_UL,
    },
    "CRP": {
        "canonical_unit": "milligram per deciliter",
        "unit_factors": {
            "mg/dl": 1.0,
            "mg/l": 0.1,
        },
    },
    "Albumin/globulin ratio": {
        "canonical_unit": "ratio",
        "unit_factors": {
            "ratio": 1.0,
            "calc": 1.0,
        },
    },
    "Total protein": {
        "canonical_unit": "gram per deciliter",
        "unit_factors": MASS_G_DL,
    },
    "INR": {
        "canonical_unit": "ratio",
        "unit_factors": {
            "ratio": 1.0,
        },
    },
    "CA19-9": {
        "canonical_unit": "unit per milliliter",
        "unit_factors": {
            "u/ml": 1.0,
            "iu/ml": 1.0,
        },
    },
    "Potassium": {
        "canonical_unit": "milliequivalent per liter",
        "unit_factors": {
            "mmol/l": 1.0,
            "meq/l": 1.0,
            "nmol/l": 0.000001,
        },
    },
    "Hematocrit": {
        "canonical_unit": "percent",
        "unit_factors": {
            "%": 1.0,
        },
    },
    "MCV": {
        "canonical_unit": "femtoliter",
        "unit_factors": {
            "fl": 1.0,
        },
    },
    "Total bilirubin": {
        "canonical_unit": "milligram per deciliter",
        "unit_factors": MASS_MG_DL,
    },
    "Respiratory rate": {
        "canonical_unit": "breaths per minute",
        "unit_factors": {
            "rr": 1.0,
            "breaths/minute": 1.0,
            "breathsperminute": 1.0,
        },
        "extra_test_names": {"Respiratory rate"},
    },
    "Albumin": {
        "canonical_unit": "gram per deciliter",
        "unit_factors": MASS_G_DL,
    },
    "Platelets": {
        "canonical_unit": "thousand per microliter",
        "unit_factors": COUNT_1E3_PER_UL,
    },
    "Heart rate": {
        "canonical_unit": "beats per minute",
        "unit_factors": {
            "bpm": 1.0,
            "beats/minute": 1.0,
            "beatsperminute": 1.0,
        },
        "extra_test_names": {"Pulse"},
    },
    "Direct bilirubin": {
        "canonical_unit": "milligram per deciliter",
        "unit_factors": MASS_MG_DL,
    },
    "Globulin": {
        "canonical_unit": "gram per deciliter",
        "unit_factors": merge_unit_rules(
            MASS_G_DL,
            {
                "g/dlcalc": 1.0,
            },
        ),
    },
    "Eosinophils absolute": {
        "canonical_unit": "thousand per microliter",
        "unit_factors": COUNT_1E3_PER_UL,
    },
    "Monocytes absolute": {
        "canonical_unit": "thousand per microliter",
        "unit_factors": merge_unit_rules(
            COUNT_1E3_PER_UL,
            {
                "10*3ul": 1.0,
            },
        ),
    },
    "PT": {
        "canonical_unit": "second",
        "unit_factors": TIME_SECONDS,
    },
    "Alkaline phosphatase": {
        "canonical_unit": "international unit per liter",
        "unit_factors": ACTIVITY_UL,
    },
    "CA125": {
        "canonical_unit": "unit per milliliter",
        "unit_factors": {
            "u/ml": 1.0,
        },
    },
    "Systolic blood pressure": {
        "canonical_unit": "millimeter of mercury",
        "unit_factors": {
            "millimeterofmercury": 1.0,
            "mmhg": 1.0,
        },
        "extra_test_names": {"Systolic-Epic"},
    },
    "Diastolic blood pressure": {
        "canonical_unit": "millimeter of mercury",
        "unit_factors": {
            "millimeterofmercury": 1.0,
            "mmhg": 1.0,
        },
        "extra_test_names": {"Diastolic-Epic"},
    },
    "Body temperature": {
        "canonical_unit": "degree Celsius",
        "unit_factors": {
            "degreefahrenheit": 1.0,
            "fahrenheit": 1.0,
            "degf": 1.0,
            "degreecelsius": 1.0,
            "celsius": 1.0,
            "degc": 1.0,
        },
        "extra_test_names": {"Temperature"},
    },
    "Body weight": {
        "canonical_unit": "kilogram",
        "unit_factors": {
            "kilogram": 1.0,
            "kilograms": 1.0,
            "kg": 1.0,
            "pound": 0.45359237,
            "pounds": 0.45359237,
            "lb": 0.45359237,
            "lbs": 0.45359237,
            "oz": 0.028349523,
        },
        "extra_test_names": {"Weight"},
    },
    "Body height": {
        "canonical_unit": "centimeter",
        "unit_factors": {
            "centimeter": 1.0,
            "centimeters": 1.0,
            "cm": 1.0,
            "inch": 2.54,
            "inches": 2.54,
            "feet": 30.48,
            "foot": 30.48,
        },
        "extra_test_names": {"Height"},
    },
}


def extract_prefix(test_name: object) -> str:
    if pd.isna(test_name):
        return ""
    return re.sub(r" \(.*\)$", "", str(test_name)).strip()


def is_combined_blood_pressure_name(test_name: object) -> bool:
    if pd.isna(test_name):
        return False

    text = str(test_name).strip().lower()
    return any(re.search(pattern, text) for pattern in COMBINED_BP_NAME_PATTERNS)


def parse_combined_bp_value(value: object) -> tuple[float, float] | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None

    return float(match.group(1)), float(match.group(2))


def normalize_unit(unit_value: object) -> str:
    if pd.isna(unit_value):
        return ""

    unit = str(unit_value).strip()
    if not unit:
        return ""

    unit = unit.replace("μ", "u").replace("µ", "u")
    unit = unit.replace("[", "").replace("]", "")
    unit = unit.replace("(", "").replace(")", "")
    unit = unit.replace(" ", "")
    return unit.lower()


def safe_existing_note(note: object) -> str:
    if pd.isna(note):
        return ""
    return str(note).strip()


def merge_notes(existing_note: object, new_notes: list[str], *, drop_stale_unmapped: bool = False) -> str:
    note_parts = []
    existing = safe_existing_note(existing_note)
    if existing:
        note_parts.extend(
            part.strip()
            for part in re.split(r"(?<=\.)\s+", existing)
            if part.strip()
        )

    if drop_stale_unmapped:
        note_parts = [
            part
            for part in note_parts
            if not any(fragment in part for fragment in STALE_UNMAPPED_NOTE_FRAGMENTS)
        ]

    for note in new_notes:
        clean_note = note.strip()
        if clean_note and clean_note not in note_parts:
            note_parts.append(clean_note)

    return " ".join(note_parts)


def prepare_unique_labs_df(unique_labs_df: pd.DataFrame) -> pd.DataFrame:
    working_df = unique_labs_df.copy()

    if {"LAB_NAME", "LAB_UNIT", "count"}.issubset(working_df.columns):
        working_df = working_df.rename(
            columns={
                "LAB_NAME": "TEST_NAME",
                "LAB_UNIT": "RESULT_UOM_NM",
            }
        )

    required_columns = {"TEST_NAME", "RESULT_UOM_NM", "count"}
    missing = required_columns - set(working_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            "unique_lab_ids_w_units.csv must contain either "
            f"TEST_NAME/RESULT_UOM_NM/count or LAB_NAME/LAB_UNIT/count. Missing: {missing_list}"
        )

    return working_df[list(required_columns)]


def prepare_longitudinal_labs_df(labs_df: pd.DataFrame) -> pd.DataFrame:
    working_df = labs_df.copy()

    if {"LAB_NAME", "LAB_UNIT", "LAB_VALUE"}.issubset(working_df.columns):
        working_df = working_df.rename(
            columns={
                "LAB_NAME": "TEST_NAME",
                "LAB_UNIT": "RESULT_UOM_NM",
                "LAB_VALUE": "NUMERIC_RESULT",
            }
        )

    required_columns = {"TEST_NAME", "RESULT_UOM_NM", "NUMERIC_RESULT"}
    missing = required_columns - set(working_df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            "Longitudinal lab input must contain either "
            f"TEST_NAME/RESULT_UOM_NM/NUMERIC_RESULT or LAB_NAME/LAB_UNIT/LAB_VALUE. Missing: {missing_list}"
        )

    return working_df


def load_json_list(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []

    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return [str(item) for item in parsed]


def get_allowed_prefixes(mapping_row: pd.Series, measurement_rule: dict) -> set[str]:
    raw_prefixes = safe_existing_note(mapping_row.get("mapped_test_type_cds", ""))
    allowed_prefixes = {
        prefix.strip()
        for prefix in raw_prefixes.split("|")
        if prefix.strip()
    }
    allowed_prefixes.update(measurement_rule.get("extra_prefixes", set()))
    return allowed_prefixes


def select_supported_test_names(
    mapping_row: pd.Series,
    unique_labs_df: pd.DataFrame,
) -> tuple[list[str], list[str], int, int]:
    collapsed_measurement = mapping_row["collapsed_measurement"]
    measurement_rule = RULES_BY_MEASUREMENT.get(collapsed_measurement)
    if not measurement_rule:
        return [], [], 0, 0

    allowed_prefixes = get_allowed_prefixes(mapping_row, measurement_rule)
    extra_test_names = set(measurement_rule.get("extra_test_names", set()))
    unit_factors = measurement_rule["unit_factors"]

    candidates = unique_labs_df[
        unique_labs_df["prefix"].isin(allowed_prefixes)
        | unique_labs_df["TEST_NAME"].isin(extra_test_names)
    ].copy()

    if candidates.empty:
        return [], sorted(allowed_prefixes | extra_test_names), 0, 0

    candidates["normalized_unit"] = candidates["RESULT_UOM_NM"].map(normalize_unit)
    supported = candidates[candidates["normalized_unit"].isin(unit_factors)].copy()

    counts_by_name = (
        supported.groupby("TEST_NAME", as_index=False)["count"]
        .sum()
        .sort_values(["count", "TEST_NAME"], ascending=[False, True])
    )
    exact_names = counts_by_name["TEST_NAME"].tolist()

    prefixes_by_name = [
        extract_prefix(test_name)
        for test_name in exact_names
    ]
    unique_prefixes = list(dict.fromkeys(prefixes_by_name))

    supported_count = int(supported["count"].sum())
    excluded_count = int(candidates["count"].sum() - supported_count)
    return exact_names, unique_prefixes, supported_count, excluded_count


def refresh_mapping_dataframe(
    mapping_df: pd.DataFrame,
    unique_labs_df: pd.DataFrame,
) -> pd.DataFrame:
    working_unique = prepare_unique_labs_df(unique_labs_df)
    working_unique["prefix"] = working_unique["TEST_NAME"].map(extract_prefix)

    refreshed_rows = []
    for _, row in mapping_df.iterrows():
        refreshed = row.copy()
        collapsed_measurement = row["collapsed_measurement"]
        measurement_rule = RULES_BY_MEASUREMENT.get(collapsed_measurement)

        if not measurement_rule:
            refreshed["n_mapped_test_names"] = 0
            refreshed["mapped_test_name_prefixes_json"] = json.dumps([])
            refreshed["mapped_test_names_json"] = json.dumps([])
            refreshed["canonical_unit"] = ""
            refreshed_rows.append(refreshed)
            continue

        exact_names, name_prefixes, supported_count, excluded_count = select_supported_test_names(
            row,
            working_unique,
        )
        generated_notes: list[str] = []

        if not exact_names:
            refreshed["mapping_status"] = "unmapped"
            refreshed["n_mapped_test_names"] = 0
            refreshed["mapped_test_name_prefixes_json"] = json.dumps([])
            refreshed["mapped_test_names_json"] = json.dumps([])
            refreshed["canonical_unit"] = ""
            refreshed["summed_internal_count"] = 0
            generated_notes.append("No supported TEST_NAME/unit combinations found in unique_lab_ids_w_units.csv.")
            refreshed["notes"] = merge_notes(row.get("notes", ""), generated_notes)
            refreshed_rows.append(refreshed)
            continue

        refreshed["mapping_status"] = "mapped"
        if measurement_rule.get("extra_prefixes") or measurement_rule.get("extra_test_names"):
            generated_notes.append("Expanded beyond the legacy code list using new TEST_NAME aliases from unique_lab_ids_w_units.csv.")
        if excluded_count:
            generated_notes.append("Rows with unsupported or semantically incompatible units are excluded by consolidate_dfci_labs.py.")

        refreshed["n_mapped_test_names"] = len(exact_names)
        refreshed["mapped_test_name_prefixes_json"] = json.dumps(name_prefixes)
        refreshed["mapped_test_names_json"] = json.dumps(exact_names)
        refreshed["canonical_unit"] = measurement_rule["canonical_unit"]
        refreshed["summed_internal_count"] = supported_count
        refreshed["notes"] = merge_notes(
            row.get("notes", ""),
            generated_notes,
            drop_stale_unmapped=True,
        )
        refreshed_rows.append(refreshed)

    refreshed_df = pd.DataFrame(refreshed_rows)
    ordered_columns = [
        "measurement_concept_id",
        "omop_measurement_name",
        "collapsed_measurement",
        "mapping_status",
        "n_mapped_test_type_cds",
        "mapped_test_type_cds",
        "n_mapped_test_names",
        "mapped_test_name_prefixes_json",
        "mapped_test_names_json",
        "canonical_unit",
        "summed_internal_count",
        "notes",
    ]

    for numeric_column in [
        "measurement_concept_id",
        "n_mapped_test_type_cds",
        "n_mapped_test_names",
        "summed_internal_count",
    ]:
        refreshed_df[numeric_column] = pd.to_numeric(
            refreshed_df[numeric_column],
            errors="coerce",
        ).astype("Int64")

    return refreshed_df[ordered_columns]


def write_refreshed_mapping(
    mapping_csv: Path,
    unique_labs_csv: Path,
    output_csv: Path | None = None,
) -> Path:
    mapping_df = pd.read_csv(mapping_csv)
    unique_labs_df = prepare_unique_labs_df(pd.read_csv(unique_labs_csv))
    refreshed_df = refresh_mapping_dataframe(mapping_df, unique_labs_df)

    output_path = output_csv or mapping_csv
    refreshed_df.to_csv(output_path, index=False)
    return output_path


def build_mapping_lookup(mapping_df: pd.DataFrame) -> tuple[dict[str, dict], dict[str, dict]]:
    exact_lookup: dict[str, dict] = {}
    prefix_lookup: dict[str, dict] = {}

    for _, row in mapping_df[mapping_df["mapping_status"] == "mapped"].iterrows():
        collapsed_measurement = row["collapsed_measurement"]
        measurement_rule = RULES_BY_MEASUREMENT.get(collapsed_measurement)
        if not measurement_rule:
            continue

        metadata = {
            "measurement_concept_id": row["measurement_concept_id"],
            "omop_measurement_name": row["omop_measurement_name"],
            "collapsed_measurement": collapsed_measurement,
            "canonical_unit": measurement_rule["canonical_unit"],
            "unit_factors": measurement_rule["unit_factors"],
        }

        exact_names = load_json_list(row.get("mapped_test_names_json"))
        if not exact_names:
            continue

        for test_name in exact_names:
            exact_lookup[test_name] = metadata

        prefixes = load_json_list(row.get("mapped_test_name_prefixes_json"))
        if not prefixes:
            prefixes = [extract_prefix(test_name) for test_name in exact_names]

        for prefix in prefixes:
            prefix_lookup[prefix] = metadata

    return exact_lookup, prefix_lookup


def build_measurement_lookup(mapping_df: pd.DataFrame) -> dict[str, dict]:
    measurement_lookup: dict[str, dict] = {}

    for _, row in mapping_df[mapping_df["mapping_status"] == "mapped"].iterrows():
        collapsed_measurement = row["collapsed_measurement"]
        measurement_rule = RULES_BY_MEASUREMENT.get(collapsed_measurement)
        if not measurement_rule:
            continue

        measurement_lookup[collapsed_measurement] = {
            "measurement_concept_id": row["measurement_concept_id"],
            "omop_measurement_name": row["omop_measurement_name"],
            "collapsed_measurement": collapsed_measurement,
            "canonical_unit": measurement_rule["canonical_unit"],
            "unit_factors": measurement_rule["unit_factors"],
        }

    return measurement_lookup


def convert_value(value: float, factor: float) -> float:
    if pd.isna(value):
        return math.nan
    return value * factor


def convert_measurement_value(
    collapsed_measurement: str,
    normalized_unit: str,
    numeric_value: float,
    canonical_unit: str,
    unit_factors: dict[str, float],
) -> tuple[float | None, float | None, str]:
    if collapsed_measurement == "Body temperature":
        if normalized_unit in {"degreefahrenheit", "fahrenheit", "degf"}:
            converted_value = (numeric_value - 32.0) * 5.0 / 9.0
            return converted_value, None, "mapped_value_converted"

        if normalized_unit in {"degreecelsius", "celsius", "degc"}:
            canonical_normalized = normalize_unit(canonical_unit)
            if normalized_unit != canonical_normalized:
                return numeric_value, 1.0, "mapped_unit_normalized"
            return numeric_value, 1.0, "mapped_no_change"

    factor = unit_factors.get(normalized_unit)
    if factor is None:
        return None, None, "unsupported_unit"

    standardized_value = convert_value(numeric_value, factor)
    canonical_normalized = normalize_unit(canonical_unit)
    if factor != 1.0:
        return standardized_value, factor, "mapped_value_converted"
    if normalized_unit != canonical_normalized:
        return standardized_value, factor, "mapped_unit_normalized"
    return standardized_value, factor, "mapped_no_change"


def consolidate_dfci_labs(
    labs_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    exact_lookup, prefix_lookup = build_mapping_lookup(mapping_df)
    measurement_lookup = build_measurement_lookup(mapping_df)

    output_df = prepare_longitudinal_labs_df(labs_df)
    output_df["TEST_NAME_PREFIX"] = output_df["TEST_NAME"].map(extract_prefix)
    output_df["normalized_result_uom_nm"] = output_df["RESULT_UOM_NM"].map(normalize_unit)
    output_df["numeric_result_as_float"] = pd.to_numeric(output_df["NUMERIC_RESULT"], errors="coerce")
    output_rows: list[dict] = []

    for _, row in output_df.iterrows():
        base_row = row.to_dict()
        base_row["bp_component"] = pd.NA
        base_row["split_from_combined_bp"] = False

        metadata = exact_lookup.get(row["TEST_NAME"])
        mapping_source = "exact"

        if metadata is None:
            metadata = prefix_lookup.get(row["TEST_NAME_PREFIX"])
            mapping_source = "prefix" if metadata is not None else "unmapped"

        if metadata is None and is_combined_blood_pressure_name(row["TEST_NAME"]):
            bp_pair = parse_combined_bp_value(row["NUMERIC_RESULT"])
            if bp_pair is not None:
                split_specs = [
                    ("systolic", "Systolic blood pressure", bp_pair[0]),
                    ("diastolic", "Diastolic blood pressure", bp_pair[1]),
                ]
                for bp_component, collapsed_measurement, bp_value in split_specs:
                    bp_metadata = measurement_lookup.get(collapsed_measurement)
                    split_row = base_row.copy()
                    split_row["bp_component"] = bp_component
                    split_row["split_from_combined_bp"] = True

                    if bp_metadata is None:
                        split_row["measurement_concept_id"] = pd.NA
                        split_row["omop_measurement_name"] = pd.NA
                        split_row["collapsed_measurement"] = collapsed_measurement
                        split_row["canonical_result_uom_nm"] = "mmHg"
                        split_row["conversion_factor"] = pd.NA
                        split_row["numeric_result_standardized"] = bp_value
                        split_row["mapping_source"] = "special_case"
                        split_row["conversion_status"] = "requires_bp_split"
                    else:
                        split_row["measurement_concept_id"] = bp_metadata["measurement_concept_id"]
                        split_row["omop_measurement_name"] = bp_metadata["omop_measurement_name"]
                        split_row["collapsed_measurement"] = bp_metadata["collapsed_measurement"]
                        split_row["canonical_result_uom_nm"] = bp_metadata["canonical_unit"]
                        split_row["conversion_factor"] = 1.0
                        split_row["numeric_result_standardized"] = bp_value
                        split_row["mapping_source"] = "special_case"
                        split_row["conversion_status"] = "bp_split_mapped"

                    output_rows.append(split_row)
                continue

            base_row["measurement_concept_id"] = pd.NA
            base_row["omop_measurement_name"] = pd.NA
            base_row["collapsed_measurement"] = pd.NA
            base_row["canonical_result_uom_nm"] = pd.NA
            base_row["conversion_factor"] = pd.NA
            base_row["numeric_result_standardized"] = pd.NA
            base_row["mapping_source"] = "special_case"
            base_row["conversion_status"] = "requires_bp_split"
            output_rows.append(base_row)
            continue

        if metadata is None:
            base_row["measurement_concept_id"] = pd.NA
            base_row["omop_measurement_name"] = pd.NA
            base_row["collapsed_measurement"] = pd.NA
            base_row["canonical_result_uom_nm"] = pd.NA
            base_row["conversion_factor"] = pd.NA
            base_row["numeric_result_standardized"] = pd.NA
            base_row["mapping_source"] = mapping_source
            base_row["conversion_status"] = "unmapped_test_name"
            output_rows.append(base_row)
            continue

        base_row["measurement_concept_id"] = metadata["measurement_concept_id"]
        base_row["omop_measurement_name"] = metadata["omop_measurement_name"]
        base_row["collapsed_measurement"] = metadata["collapsed_measurement"]
        base_row["canonical_result_uom_nm"] = metadata["canonical_unit"]
        base_row["mapping_source"] = mapping_source

        numeric_value = row["numeric_result_as_float"]
        if pd.isna(numeric_value):
            base_row["conversion_factor"] = pd.NA
            base_row["numeric_result_standardized"] = pd.NA
            base_row["conversion_status"] = "non_numeric_result"
            output_rows.append(base_row)
            continue

        normalized_unit = row["normalized_result_uom_nm"]
        unit_factors = metadata["unit_factors"]
        standardized_value, factor, status = convert_measurement_value(
            collapsed_measurement=metadata["collapsed_measurement"],
            normalized_unit=normalized_unit,
            numeric_value=numeric_value,
            canonical_unit=metadata["canonical_unit"],
            unit_factors=unit_factors,
        )

        if status == "unsupported_unit":
            base_row["conversion_factor"] = pd.NA
            base_row["numeric_result_standardized"] = pd.NA
            base_row["conversion_status"] = status
            output_rows.append(base_row)
            continue

        base_row["conversion_factor"] = factor if factor is not None else pd.NA
        base_row["numeric_result_standardized"] = standardized_value
        base_row["conversion_status"] = status

        output_rows.append(base_row)

    return pd.DataFrame(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh OMOP-to-DFCI lab mappings and standardize raw DFCI lab pulls.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    refresh_parser = subparsers.add_parser(
        "refresh-mapping",
        help="Refresh OMOP_to_DFCI_lab_ids.csv from unique_lab_ids_w_units.csv.",
    )
    refresh_parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
    )
    refresh_parser.add_argument(
        "--unique-labs-csv",
        type=Path,
        default=DEFAULT_UNIQUE_LABS_CSV,
    )
    refresh_parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )

    consolidate_parser = subparsers.add_parser(
        "consolidate",
        help="Standardize raw DFCI lab data using the refreshed mapping.",
    )
    consolidate_parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
    )
    consolidate_parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
    )
    consolidate_parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
    )
    consolidate_parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Keep only rows whose TEST_NAME mapped and whose units were handled successfully.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "refresh-mapping":
        output_path = write_refreshed_mapping(
            mapping_csv=args.mapping_csv,
            unique_labs_csv=args.unique_labs_csv,
            output_csv=args.output_csv,
        )
        print(f"Wrote refreshed mapping to {output_path}")
        return

    if args.command == "consolidate":
        mapping_df = pd.read_csv(args.mapping_csv)
        input_df = pd.read_csv(args.input_csv)
        consolidated_df = consolidate_dfci_labs(input_df, mapping_df)

        if args.successful_only:
            consolidated_df = consolidated_df[
                consolidated_df["conversion_status"].isin(
                    {
                        "mapped_no_change",
                        "mapped_unit_normalized",
                        "mapped_value_converted",
                        "bp_split_mapped",
                    }
                )
            ].copy()

        consolidated_df.to_csv(args.output_csv, index=False)
        print(f"Wrote consolidated labs to {args.output_csv}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
