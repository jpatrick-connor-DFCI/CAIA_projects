Reviewed `/Users/connorpa/Desktop/ADT`: 383 files, including 342 PNGs (275 distinct
images), 14 baseline-characteristics CSVs, 14 corresponding Markdown tables,
and 13 Finder metadata files. The image review covered the current tree, the
older `by_figure_v0` tree, and supplement/checkpoint versions. All PNGs decoded.

| Finding | Generator change |
| --- | --- |
| Diagnosis-to-ADT and Figure 2 titles/captions run off the canvas | Wrap outside-panel text, increase margins and Figure 2 widths, shorten titles |
| Volcano legends cover observations and repeat category keys | Move one category legend below the panel |
| Many PSA/testosterone labels collide; faint background points are difficult to see | Label mean and strongest additional androgen statistic; retain all plotted observations, use darker points, allow more label separation and space |
| Fixed volcano x-limits hide extreme estimates | Expand axes to include every retained coefficient |
| Performance bars below 0.45 disappear | Use a zero-based axis with room above perfect scores for labels |
| Coverage counts sit on confidence intervals | Place counts above the upper interval bound |
| Long genomic subtitles and row labels are crowded | Wrap subtitles; increase forest row spacing, font size, and minimum panel height |
| Supplemental forests repeat both endpoints separately for each analyte | Two endpoint figures, each with PSA/testosterone columns; remove delta and observation-count rows |
| Notebook overlap/event-incidence figures are assembled panels | Export each component as its own figure |

New KM panels use existing landmark or indexed survival inputs, retain missing
calls as missing, and show stratum/event counts. Tertile boundaries preserve ties;
mutation features require Cox q < 0.05; Gleason scores are ≤6, 7, 8, 9 and 10.

The local download contains rendered figures and baseline summary tables, not
the patient-level outcomes, mutation calls, lab features, or Cox result tables
needed to regenerate the analysis. Validation therefore uses synthetic figures
and regression tests. Re-knit with `COMPASS_RENDER_OVERWRITE=true` to replace
existing graphics with the revised versions. Original downloaded figures are
left intact.
