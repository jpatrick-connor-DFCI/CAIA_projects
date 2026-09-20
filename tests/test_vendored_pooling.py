"""The landmark contract of the vendored note-pooling helpers.

``COMPASS/data_preprocessing/vendor/note_pooling.py`` is a verbatim fork of the
clinical text embedding project's ``survival/preprocessing.py``. Because it is a
fork, upstream fixes do not arrive on their own -- so the behavior COMPASS
actually depends on is pinned here rather than assumed.

What matters is the landmark contract, which IS the leakage control for the 03c
text arms: only notes strictly before the landmark may be pooled, and note times
must come back re-centered on it. If a re-sync from upstream ever breaks that,
these fail instead of silently training on post-landmark notes.

``continuous_window=False`` is the mode the builder uses (see
``pool_embeddings_for_landmark``); the ``True`` branch adds a contiguity
requirement COMPASS does not want.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("tqdm")

_PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "COMPASS" / "data_preprocessing"
if str(_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING_DIR))

from vendor.note_pooling import generate_survival_embedding_df  # noqa: E402

NOTE_TIME_COL = "NOTE_TIME_REL_ADT"
EMB_DIM = 4


def _notes(times_by_patient: dict[str, list[int]], note_type: str = "Clinician"):
    """Synthetic note metadata plus a matching embedding matrix."""
    rows, vectors = [], []
    for mrn, times in times_by_patient.items():
        for offset in times:
            rows.append(
                {
                    "DFCI_MRN": mrn,
                    NOTE_TIME_COL: offset,
                    "NOTE_TYPE": note_type,
                    # Only used for a pre-2015 year adjustment COMPASS does not
                    # request, but the pooler reads it unconditionally.
                    "NOTE_DATETIME": "2020-01-01",
                    "EMBEDDING_INDEX": len(rows),
                }
            )
            # Encode the note's own time so pooled output is traceable to inputs.
            vectors.append(np.full(EMB_DIM, float(offset), dtype=np.float32))
    return pl.DataFrame(rows), np.vstack(vectors)


def _pool(notes_meta, embeddings, landmark_day: int):
    return generate_survival_embedding_df(
        notes_meta,
        None,
        embeddings,
        note_types=["Clinician"],
        note_timing_col=NOTE_TIME_COL,
        max_note_window=landmark_day,
        pool_fx={"Clinician": "time_decay_mean"},
        decay_param=0.01,
        continuous_window=False,
    )


def test_notes_after_the_landmark_are_excluded():
    """The leakage guard: a post-landmark note must not reach the pooled vector."""
    notes, embeddings = _notes({"A": [-30, -10]})
    clean = _pool(notes, embeddings, landmark_day=0)

    leaky_notes, leaky_embeddings = _notes({"A": [-30, -10, 45]})
    with_future = _pool(leaky_notes, leaky_embeddings, landmark_day=0)

    col = next(c for c in clean.columns if c.endswith("_EMBEDDING_0"))
    assert clean[col].to_list() == pytest.approx(with_future[col].to_list())


def test_landmark_shifts_the_window_forward():
    """At landmark +90, notes up to day 90 become eligible."""
    notes, embeddings = _notes({"A": [-30, 45]})

    at_zero = _pool(notes, embeddings, landmark_day=0)
    at_ninety = _pool(notes, embeddings, landmark_day=90)

    col = next(c for c in at_zero.columns if c.endswith("_EMBEDDING_0"))
    # The day-45 note is excluded at landmark 0 and included at landmark 90,
    # so the pooled value must differ.
    assert at_zero[col].to_list() != pytest.approx(at_ninety[col].to_list())


def test_note_exactly_at_the_landmark_is_excluded():
    """The filter is strict (<), not inclusive -- a same-day note is future."""
    notes, embeddings = _notes({"A": [-30, 0]})
    pooled = _pool(notes, embeddings, landmark_day=0)

    only_past, only_past_emb = _notes({"A": [-30]})
    expected = _pool(only_past, only_past_emb, landmark_day=0)

    col = next(c for c in pooled.columns if c.endswith("_EMBEDDING_0"))
    assert pooled[col].to_list() == pytest.approx(expected[col].to_list())


def test_patient_with_no_pre_landmark_notes_is_dropped_not_zero_filled():
    """Such a patient must be absent, never carry a bogus all-zero vector.

    The builder left-joins this onto the COMPASS frame, so an absent row becomes
    NaN and is handled as missing downstream. A zero row would instead look like
    a real, confidently-central embedding.
    """
    notes, embeddings = _notes({"A": [-30], "B": [60]})
    pooled = _pool(notes, embeddings, landmark_day=0)

    assert pooled["DFCI_MRN"].to_list() == ["A"]


def test_time_decay_weights_recent_notes_more_heavily():
    """time_decay_mean must not collapse to an unweighted mean."""
    notes, embeddings = _notes({"A": [-400, -1]})
    pooled = _pool(notes, embeddings, landmark_day=0)

    col = next(c for c in pooled.columns if c.endswith("_EMBEDDING_0"))
    value = pooled[col].to_list()[0]
    # Unweighted would be -200.5; decay must pull it toward the recent note.
    assert value > -200.5
