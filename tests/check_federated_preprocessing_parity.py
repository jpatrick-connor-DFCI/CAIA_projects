"""Cross-repo parity: federated_scoring vs the real rhino_scripts preprocessing.

``survival_common/federated_scoring.py`` reimplements the feature derivation
that lives in ``caia-project-compass/rhino_scripts/*/preprocessing.py``.  Two
copies of a rule drift.  ``tests/test_federated_scoring.py`` pins the rules by
restating their expected values, which catches a change on *this* side; this
script catches a change on the *other* side, by importing the federated modules
and diffing the wide frames they build.

It is not part of the pytest suite because it needs the sibling repo checked out
next to this one.  Run it after pulling caia-project-compass, and whenever
``preprocessing.py`` there changes::

    python tests/check_federated_preprocessing_parity.py
    python tests/check_federated_preprocessing_parity.py --rhino-scripts /path/to/rhino_scripts

Exits non-zero on any difference, so it can gate a sync.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RHINO = (
    REPO_ROOT.parent / "caia-project-compass" / "rhino_scripts"
)

HGB = "Hemoglobin [Mass/volume] in Blood"
TESTO = "Testosterone [Mass/volume] in Serum or Plasma"
GLU = "Glucose [Mass/volume] in Serum or Plasma"
LANDMARKS = (0, 90, 180)


def make_long_frame(n_pat: int = 150, seed: int = 7) -> pd.DataFrame:
    """A long frame exercising the cases where the two copies could differ.

    Deliberately includes: patients with no labs at all, measurements exactly ON
    each landmark day, single-observation labs (so ``delta`` is NaN), pre-anchor
    castrate and non-castrate testosterone, and events past the 3650-day horizon.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_pat):
        dx = pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.integers(0, 800)))
        adt = dx + pd.Timedelta(days=int(rng.integers(10, 400)))
        has_platinum = rng.random() < 0.3
        platinum = (
            adt + pd.Timedelta(days=int(rng.integers(30, 4200)))
            if has_platinum else pd.NaT
        )
        last_fu = adt + pd.Timedelta(days=int(rng.integers(100, 4000)))
        base = {
            "person_id": pid,
            "age_at_diagnosis": 60 + float(rng.normal(0, 8)),
            "diagnosis_date": dx,
            "adt_start_date_post_diagnosis": adt,
            "platinum_start_date": platinum,
            "last_followup_date": last_fu,
        }
        n_meas = int(rng.integers(0, 9))
        if n_meas == 0:
            rows.append({
                **base, "lab_name": None, "lab_value": np.nan,
                "measurement_date": pd.NaT,
            })
            continue
        for _ in range(n_meas):
            lab = str(rng.choice([HGB, TESTO, GLU]))
            offset = int(rng.choice([-200, -30, -1, 0, 1, 89, 90, 91, 179, 180, 300]))
            value = (
                float(rng.choice([20.0, 80.0])) if lab == TESTO
                else float(rng.normal(10, 3))
            )
            rows.append({
                **base, "lab_name": lab, "lab_value": value,
                "measurement_date": adt + pd.Timedelta(days=offset),
            })
    return pd.DataFrame(rows)


def federated_wide(fed_prep, fed_io, long_df: pd.DataFrame, landmark: int):
    """The federated pipeline's own chain, in its own order."""
    derived = fed_io.derive_analysis_columns(long_df)
    static = fed_prep.extract_static_frame(derived)
    outcome, stats = fed_prep.make_outcome(static, landmark)
    pre = fed_prep.filter_pre_landmark(derived, landmark)
    pre = pre[pre["person_id"].isin(outcome["person_id"])]
    agg = fed_prep.aggregate_lab_features(pre)
    return fed_prep.build_wide_frame(agg, outcome), stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rhino-scripts", default=str(DEFAULT_RHINO),
                    help="path to caia-project-compass/rhino_scripts")
    ap.add_argument("--app", default="federated_cox_multivariate",
                    help="which app's preprocessing copy to diff against")
    args = ap.parse_args(argv)

    app_dir = Path(args.rhino_scripts) / args.app
    if not app_dir.is_dir():
        print(
            f"ERROR: {app_dir} not found.\n"
            "Check out caia-project-compass next to this repo, or pass "
            "--rhino-scripts.",
            file=sys.stderr,
        )
        return 2

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(app_dir))
    import io_utils as fed_io
    import preprocessing as fed_prep
    from survival_common import federated_scoring as fs

    long_df = make_long_frame()
    print(
        f"comparing against {app_dir}\n"
        f"{len(long_df):,} rows, {long_df['person_id'].nunique():,} patients\n"
    )

    failures = 0
    for exclusion in ("none", "pre_anchor_castrate"):
        fed_long, fed_stats = fed_prep.apply_exclusion(long_df, exclusion)
        my_long, my_stats = fs.apply_exclusion(long_df, exclusion)

        if fed_stats != my_stats:
            print(f"FAIL exclusion={exclusion}: stats differ")
            print(f"  federated: {fed_stats}")
            print(f"  local:     {my_stats}")
            failures += 1
        else:
            print(f"ok   exclusion={exclusion}: {my_stats}")

        for landmark in LANDMARKS:
            fed_frame, fed_ostats = federated_wide(
                fed_prep, fed_io, fed_long, landmark
            )
            my_frame, my_ostats = fs.build_landmark_frame(my_long, landmark)

            if list(fed_frame.columns) != list(my_frame.columns):
                only_fed = set(fed_frame.columns) - set(my_frame.columns)
                only_mine = set(my_frame.columns) - set(fed_frame.columns)
                print(
                    f"FAIL lm={landmark} exclusion={exclusion}: columns differ\n"
                    f"  federated only: {sorted(only_fed)}\n"
                    f"  local only:     {sorted(only_mine)}"
                )
                failures += 1
                continue

            left = fed_frame.sort_values("person_id").reset_index(drop=True)
            right = my_frame.sort_values("person_id").reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(
                    left, right, check_dtype=False, rtol=0, atol=0
                )
                print(
                    f"ok   lm={landmark} exclusion={exclusion}: identical "
                    f"({len(right):,} patients, {len(right.columns)} cols)"
                )
            except AssertionError as exc:
                print(
                    f"FAIL lm={landmark} exclusion={exclusion}: values differ\n"
                    f"  {str(exc)[:400]}"
                )
                failures += 1

            for key in (
                "n_patients", "n_events",
                "n_dropped_non_positive_duration", "n_admin_censored",
            ):
                if fed_ostats[key] != my_ostats[key]:
                    print(
                        f"FAIL lm={landmark} exclusion={exclusion}: {key} "
                        f"federated={fed_ostats[key]} local={my_ostats[key]}"
                    )
                    failures += 1

    print()
    if failures:
        print(
            f"{failures} difference(s) found. The two preprocessing copies have "
            "drifted -- reconcile survival_common/federated_scoring.py with "
            f"{app_dir / 'preprocessing.py'} before scoring anything."
        )
        return 1
    print("PARITY OK: the local feature chain reproduces the federated one exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
