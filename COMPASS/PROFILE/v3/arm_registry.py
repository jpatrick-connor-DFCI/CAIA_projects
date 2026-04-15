import importlib


def load_arm_module(arm_name):
    normalized = str(arm_name).strip().lower()
    if normalized not in {"nepc", "avpc", "biomarker"}:
        raise ValueError(f"Unsupported arm: {arm_name}")
    return importlib.import_module(f"arms.{normalized}")
