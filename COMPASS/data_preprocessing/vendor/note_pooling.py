"""Vendored note-pooling helpers from the clinical text embedding project.

Copied verbatim -- do not edit the logic -- from that project's
``survival/preprocessing.py`` at commit 52a499f. Only the three functions
COMPASS needs are included:

* ``find_continuous_records_to_analyze``
* ``pool_embedding_series_vectorized``
* ``generate_survival_embedding_df``

WHY A COPY RATHER THAN AN IMPORT. The embedding project is a separate repo with
its own cluster checkout and SLURM deployment. Importing across the two coupled
their deploy cycles: PROFILE-testing could be updated while the other checkout
was not, and the mismatch only surfaced at runtime, deep inside a build. A
vendored copy makes this repo self-contained -- ``build_text_embedding_inputs``
needs nothing outside PROFILE-testing but its input data.

KEEPING IT IN SYNC. This is a fork, so upstream changes do not arrive
automatically. The landmark contract these functions implement (filter to notes
strictly before the landmark, re-center note times on it, assert
``max(note_time) <= 0``) is the part that must not drift: it is COMPASS's
leakage control. If the upstream pooling changes materially, re-copy the three
functions and re-run tests/test_vendored_pooling.py, which pins that contract.

Upstream requires ``polars``, ``numpy`` and ``tqdm``; so does this copy.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from tqdm.auto import tqdm


def find_continuous_records_to_analyze(notes_meta: pl.DataFrame, note_timing_col: str = 'NOTE_TIME_REL_FIRST_TREATMENT_START',
                                       gap_threshold: int = 2 * 365, max_note_window: int = 0) -> pl.DataFrame:
    """
    Identify continuous sequences of patient notes within a specified time window. Assume that time 0 is the beginning of the prediction window.

    A continuous segment is defined as consecutive notes where gaps do not exceed `gap_threshold`.

    Args:
        notes_meta (pl.DataFrame): Metadata for notes, must include 'DFCI_MRN' and note timing column.
        note_timing_col (str, optional): Column representing time relative to treatment start. Defaults to 'NOTE_TIME_REL_FIRST_TREATMENT_START'.
        gap_threshold (int, optional): Maximum allowed gap (in days) between consecutive notes in a segment. Defaults to 2*365.
        max_note_window (int, optional): Landmark time. Notes at or after this
            time are dropped and the remaining times are re-centred so the
            landmark becomes the new origin. Defaults to 0.

    Returns:
        pl.DataFrame: Subset of notes_meta containing only continuous records within allowed windows.
    """

    notes_within_window = notes_meta.filter(pl.col(note_timing_col) < max_note_window)
    notes_within_window = notes_within_window.with_columns(
        (pl.col(note_timing_col) - max_note_window).alias(note_timing_col)
    )
    notes_within_window = notes_within_window.with_row_index('_row_id')
    rows_to_include = []

    # Times above were re-centred on `max_note_window`, so the landmark now sits
    # at 0 in this frame.  Both the window test and the guard below compare
    # against this name rather than a bare literal, so the two cannot drift
    # apart if the re-centring above ever changes.
    recentred_origin = 0

    # Process each patient separately
    for mrn_val in notes_within_window['DFCI_MRN'].unique().to_list():
        mrn_notes = (
            notes_within_window.filter(pl.col('DFCI_MRN') == mrn_val)
            .sort(note_timing_col)
        )
        note_times = mrn_notes[note_timing_col].to_numpy()
        row_ids = mrn_notes['_row_id'].to_numpy()
        if len(note_times) == 0:
            continue

        # Build continuous segments
        idx_series = []
        lbound = note_times[0]
        for i in range(1, len(note_times)):
            if (note_times[i] - note_times[i - 1]) >= gap_threshold:
                idx_series.append((lbound, note_times[i - 1]))
                lbound = note_times[i]
        idx_series.append((lbound, note_times[-1]))

        # Choose the last valid window within bounds
        chosen_window = None
        for lbound, ubound in idx_series:
            if ubound <= recentred_origin:
                chosen_window = (lbound, ubound)

        if chosen_window is not None:
            lbound, ubound = chosen_window
            # Include all notes within the chosen segment
            mask = (note_times >= lbound) & (note_times <= ubound)
            rows_to_include.extend(row_ids[mask].tolist())

    notes_within_window = notes_within_window.filter(pl.col('_row_id').is_in(rows_to_include)).drop('_row_id')

    # Sanity check to ensure no notes exceed allowed window
    if not notes_within_window.is_empty():
        max_time = notes_within_window[note_timing_col].max()
        if max_time > recentred_origin:
            raise ValueError(
                f'Found note at re-centred time {max_time} > {recentred_origin} '
                f'(landmark max_note_window={max_note_window}). '
                f'Check your {note_timing_col} values.'
            )

    return notes_within_window


def pool_embedding_series_vectorized(meta_df: pl.DataFrame, embedding_array: np.ndarray, note_types: list[str],
                                     note_timing_col: str = 'NOTE_TIME_REL_FIRST_TREATMENT_START',
                                     pool_fx: dict[str, str] | None = None, decay_param: float | None = None,
                                     year_adj_cols: list[str] = ['Imaging', 'Pathology'],
                                     show_progress: bool = False,
                                     progress_desc: str | None = None) -> pl.DataFrame:
    """
    Pool embeddings for each patient and note type using specified strategies.

    Supports pooling strategies:
        - 'recent': most recent note
        - 'mean': unweighted mean
        - 'time_decay_mean': weighted mean by time proximity

    Also optionally computes year-based adjustments for specified note types.

    Args:
        meta_df (pl.DataFrame): Metadata for notes, including embedding indices.
        embedding_array (np.ndarray): Array of embeddings (rows correspond to embeddings in meta_df).
        note_types (list[str]): Note types to include.
        note_timing_col (str, optional): Column representing note timing. Defaults to 'NOTE_TIME_REL_FIRST_TREATMENT_START'.
        pool_fx (dict[str, str] | None, optional): Pooling strategy per note type. Defaults to None (mean).
        decay_param (float | None, optional): Decay parameter for time-decayed pooling. Required if using decay strategies. Defaults to None.
        year_adj_cols (list[str], optional): Note types for which year-based adjustments are computed. Defaults to ['Imaging', 'Pathology'].
        show_progress (bool, optional): Show patient/note-type pooling progress. Defaults to False.
        progress_desc (str | None, optional): Label for the progress bar.

    Returns:
        pl.DataFrame: DataFrame with one row per patient and pooled embeddings per note type.
    """
    unique_mrns = meta_df['DFCI_MRN'].unique(maintain_order=True).to_list()
    embed_dim = embedding_array.shape[1]

    # Add year column for year-based adjustments
    meta_df = meta_df.with_columns(
        pl.col('NOTE_DATETIME').cast(pl.Utf8).str.split('-').list.get(0).cast(pl.Int64).alias('NOTE_YEAR')
    )

    # Initialize pooled embeddings and year adjustments
    pooled_embeddings = {nt: np.full((len(unique_mrns), embed_dim), np.nan) for nt in note_types}
    year_adjustments = {nt: np.full((len(unique_mrns), 1), np.nan) for nt in note_types if nt in year_adj_cols}

    # Map MRN to row index for efficient assignment
    mrn_to_idx = {mrn: i for i, mrn in enumerate(unique_mrns)}

    # Group by patient and note type
    grouped_meta = meta_df.filter(pl.col('NOTE_TYPE').is_in(note_types))
    grouped = grouped_meta.group_by(['DFCI_MRN', 'NOTE_TYPE'])
    grouped_iter = grouped
    if show_progress:
        n_groups = grouped_meta.select(['DFCI_MRN', 'NOTE_TYPE']).unique().height
        grouped_iter = tqdm(
            grouped,
            total=n_groups,
            desc=progress_desc or "Pooling embeddings",
            unit="group",
            mininterval=1.0,
        )
    for (mrn, note_type), group in grouped_iter:
        if note_type not in note_types:
            continue
        idx = mrn_to_idx[mrn]
        embed_indices = group['EMBEDDING_INDEX'].to_numpy()
        note_times = group[note_timing_col].to_numpy()
        embeddings = embedding_array[embed_indices, :]

        strategy = pool_fx.get(note_type, 'mean') if pool_fx else 'mean'

        # Validate pooling strategy
        valid_strategies = {'recent', 'time_decay_mean', 'mean'}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}.")

        # Apply pooling strategy
        if strategy == 'recent':
            recent_idx = np.argmax(note_times)
            pooled_embeddings[note_type][idx, :] = embeddings[recent_idx, :]
        elif strategy == 'time_decay_mean':
            if decay_param is None:
                raise ValueError("decay_param must be provided for 'time_decay_mean'")
            weights = np.exp(-decay_param * np.abs(note_times)).reshape(-1, 1)
            weights /= np.sum(weights) if np.sum(weights) > 0 else 1
            pooled_embeddings[note_type][idx, :] = np.nansum(weights * embeddings, axis=0)
        elif strategy == 'mean':
            pooled_embeddings[note_type][idx, :] = np.nanmean(embeddings, axis=0)

        # Year-based adjustment
        if note_type in year_adj_cols:
            year_adjustments[note_type][idx] = (group['NOTE_YEAR'] < 2015).to_numpy().mean()

    # Build the frame column-by-column so MRNs do not force the numeric arrays
    # through a single NumPy object matrix.  The old concatenate path inferred
    # every column as Polars Object when DFCI_MRN was object-typed, which then
    # made otherwise-numeric embedding columns impossible to cast or validate.
    output_columns: dict[str, object] = {'DFCI_MRN': unique_mrns}
    for nt in year_adj_cols:
        if nt in year_adjustments:
            output_columns[f'PERCENT_{nt.upper()}_NOTES_PRE_2015'] = year_adjustments[nt][:, 0]
    for nt in note_types:
        for i in range(embed_dim):
            output_columns[f'{nt.upper()}_EMBEDDING_{i}'] = pooled_embeddings[nt][:, i]

    return pl.DataFrame(output_columns)


def generate_survival_embedding_df(notes_meta: pl.DataFrame, survival_df: pl.DataFrame | None, embedding_array: np.ndarray,
                                   note_types: list[str], note_timing_col: str = 'NOTE_TIME_REL_FIRST_TREATMENT_START',
                                   max_note_window: int = 0, pool_fx: dict[str, str] | None = None, decay_param: float | None = None,
                                   continuous_window: bool = True) -> pl.DataFrame:
    """
    Generate a DataFrame of pooled embeddings optionally merged with survival outcomes.

    Args:
        notes_meta (pl.DataFrame): Note metadata including embedding indices and timing.
        survival_df (pl.DataFrame | None): Survival outcome data. Must include 'DFCI_MRN' or 'PATIENT_ID'. Can be None.
        embedding_array (np.ndarray): Array of note embeddings.
        note_types (list[str]): Note types to include.
        note_timing_col (str, optional): Column for note timing relative to treatment. Defaults to 'NOTE_TIME_REL_FIRST_TREATMENT_START'.
        pool_fx (dict[str, str] | None, optional): Pooling strategy per note type. Defaults to None.
        decay_param (float | None, optional): Decay parameter for time-decayed pooling. Defaults to None.
        max_note_window (int, optional): Maximum note timing. Defaults to 0.
        continuous_window (bool, optional): If True, include only continuous note sequences. Defaults to True.

    Returns:
        pl.DataFrame: DataFrame with pooled embeddings, optionally merged with survival data.
    """
    # Select notes based on window strategy
    if continuous_window:
        notes_to_include = find_continuous_records_to_analyze(
            notes_meta, note_timing_col=note_timing_col, max_note_window=max_note_window)
    else:
        notes_to_include = notes_meta.filter(pl.col(note_timing_col) < max_note_window)
        notes_to_include = notes_to_include.with_columns(
            (pl.col(note_timing_col) - max_note_window).alias(note_timing_col)
        )

    assert(notes_to_include[note_timing_col].max() <= 0)

    # Pool embeddings for selected notes
    pooled_embedding_df = pool_embedding_series_vectorized(notes_to_include, embedding_array, note_types,
                                                           note_timing_col=note_timing_col, pool_fx=pool_fx,
                                                           decay_param=decay_param)

    if survival_df is not None:
        # Standardize patient ID column
        if 'PATIENT_ID' in survival_df.columns:
            survival_df = survival_df.rename({'PATIENT_ID': 'DFCI_MRN'})

        if max_note_window != 0:
            tt_event_cols = [col for col in survival_df.columns if col.startswith('tt_')]
            survival_df = survival_df.with_columns(
                [(pl.col(c) - max_note_window).alias(c) for c in tt_event_cols]
            )

        # Merge pooled embeddings with survival outcomes
        merged_df = survival_df.join(pooled_embedding_df, on='DFCI_MRN', how='left')

        return merged_df
    else:
        return pooled_embedding_df
