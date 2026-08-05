# GAM pipeline performance: parallelism, I/O reduction, and grid tuning

Implementation plan. All file paths below are relative to the repository root. Companion to `GAM_SURVIVAL_PLAN.md`, which specifies the two GAM stages this plan speeds up.

## Context

The two GAM stages are the slow part of the survival pipeline:

- **Stage A** — `gam_trajectory_features.R`, one trajectory smooth per canonical lab per landmark (3 landmarks × tens of labs).
- **Stage B** — `gam_cox_nonlinearity.R`, **two** `gam(family=cox.ph(), method="REML")` fits per selected feature. `gam()` + `cox.ph()` + REML is much slower per fit than Stage A's `bam(discrete=TRUE)` Gaussian.

Stage B gets materially worse before it gets better: `GAM_SURVIVAL_PLAN.md:114` notes the merge adds 5 stats × N canonical labs to the tested feature set, so its feature list grows roughly 5×.

The largest algorithmic win is **already implemented** (uncommitted, working tree): the `two_stage_ridge` path at `gam_trajectory_features.R:180-225` replaced the `bs="fs"` scalability blowup flagged as the open risk at `GAM_SURVIVAL_PLAN.md:179` with one population `bam()` plus a vectorized closed-form ridge per patient. Above `--max-fs-patients` (default 500) every lab takes it, so at production scale the `fs` path is effectively dead code.

What remains is not modeling work. It is:

1. **No parallelism anywhere in the repo.** Verified: zero matches for `mclapply|parLapply|foreach|future|furrr|doParallel` across every `.R`, `.py`, and `.ipynb`. `--nthreads 1` is hardcoded in `COMPASS_run_GAMs.ipynb`. Local machine has 18 cores. Every `(landmark, lab)` and `(landmark, feature)` unit is fully independent.
2. **A grossly oversized curve CSV** (see below).
3. **Routine I/O waste** — an unindexed full table scan per lab, a quadratic `Reduce(merge, ...)`, redundant `copy()` calls, no checkpointing.

Two findings from exploration constrain the aggressive options:

**The curve CSV is not dead output.** Its own header (`gam_trajectory_features.R:21-23`) calls it "used only for GAM-specific figures", which reads like a deletion candidate. It is not — `COMPASS_generate_figures_pipeline.R:2605-2680` reads it and builds the stratified trajectory panels. That consumer does `group_by(t_lab, stratum)` (line 2540), so **every patient must share one common `t_lab` grid**; a per-patient or unsorted grid breaks it. It does, however, already skip gracefully when the file is absent (line 2610).

**Those figures use two labs.** `LAB_FIGURE_LABS` is `intersect(ALL_LABS, ANDROGEN)` = `c("PSA", "Testosterone")` unless `plot_non_androgen_lab_figures = TRUE` (`COMPASS_generate_figures_pipeline.R:276, 2005`). We currently write curves for **every** canonical lab at 25 grid points to render two labs. This is the single largest piece of pure waste in the pipeline and costs nothing scientifically to fix.

Intended outcome: a Stage A that scales with cores instead of labs, a Stage B that runs three landmarks concurrently, and a curve file roughly 70× smaller — with the feature columns that feed the Cox models either bit-identical or provably within r > 0.999.

**Non-negotiable throughout:** the leakage invariant at `gam_trajectory_features.R:34-38`. Every evaluation point stays `<= t_star = landmark_day - epsilon_days`. Changing grid *density* is permitted; changing a grid *endpoint* is not.

---

## Step 0 — Measure before optimizing

The pipeline is currently unmeasured, so "fastest" is not yet a defined target. Stage A already records `fit_seconds` per lab into `gam_fit_diagnostics_landmark{D}.csv` (`gam_trajectory_features.R:402`), timed tightly around the fit (lines 208-210) and **excluding** `extract_lab_features`. So fit time vs. everything else is one subtraction away.

```r
d <- rbindlist(lapply(c(0,90,180), function(L)
  fread(file.path(INPUTS_DIR, sprintf("gam_fit_diagnostics_landmark%d.csv", L)))))
d[, .(n_labs=.N, fit_min=sum(fit_seconds,na.rm=TRUE)/60,
      med=median(fit_seconds,na.rm=TRUE)), by=landmark_days]
d[, .N, by=basis_used]                          # expect ~100% two_stage_ridge
d[, .(median_edf=median(edf_pop, na.rm=TRUE))]  # decides Step 5.3
```

Wrap the Stage A `run_r_script(...)` call in cell 3 of `COMPASS_run_GAMs.ipynb` in `system.time()`. Then `stage_a_wall - sum(fit_seconds)` **is** the predict + `CJ` + merge + `fwrite` overhead.

Stage B writes no timing. Wrap each `run_r_script("gam_cox_nonlinearity.R", ...)` in cell 6 in `system.time()` and record `nrow(fread(selection_path))` — that is the fit count, at 2 `gam()` fits plus an `anova()` each.

**Gate — this decides the order of everything below:**

| Observation | Consequence |
| --- | --- |
| `sum(fit_seconds)/wall > 0.7` | Fit-bound. Step 2 parallelism is the whole game; Steps 4/5.1/5.2 are minor. |
| `sum(fit_seconds)/wall < 0.4` | Predict/IO-bound. Steps 5.1/5.2 and 4 matter more than parallelism. |
| Stage B wall / n_features > ~0.3s | Stage B dominates after the 5× growth. Prioritize Steps 1 and 2.5. |
| `basis_used` not ~100% `two_stage_ridge` | Do **not** rely on Step 5.4; the `fs` path is still live at scale. |

Deliverable: one table — `stage_a_wall`, `stage_a_fit_sum`, `stage_b_wall_per_landmark`, `n_features_per_landmark`, `median_edf_pop`.

---

## Step 1 — Stage B landmark parallelism (notebook only, best speedup per unit effort)

`COMPASS_run_GAMs.ipynb` cell 6 already spawns **one subprocess per landmark**, serially. Each writes its own output file with no shared state, so concurrency needs no R changes at all.

- In `run_r_script()` (cell 1), add a `wait = TRUE` argument passed through to `system2(..., wait = wait)`.
- In cell 6, **hoist the entire validation loop** — the two `file.exists` checks and the `selection_mtime < trajectory_mtime` staleness guard — above the spawn loop. Otherwise a stale landmark fails *after* the others have already launched.
- Spawn all three with `wait = FALSE`, then wait on the handles and check exit status before the closing `stopifnot`.

~10 lines for ~3× on Stage B.

---

## Step 2 — `mclapply` inside the scripts

### 2.1 Mechanism

Use **`parallel::mclapply`**, not `future.apply`. Both scripts are deliberately dependency-minimal — `GAM_SURVIVAL_PLAN.md:32` constrains them to "base R + `mgcv` + `data.table` only", and they `stop()` on missing packages at `gam_trajectory_features.R:40-47`. `parallel` ships with R. Fork also gives copy-on-write sharing of the big table; `future.apply` with `multisession` would serialize it once per worker, which is the memory failure we are avoiding.

Add to both scripts in the `DEFAULT_*` block, matching the existing `--kebab-case` CLI convention:

```r
DEFAULT_N_WORKERS <- 1L   # 1 = serial
```

Do **not** default to `detectCores()` — the cluster may allocate fewer cores than the node reports, and the existing `DEFAULT_NTHREADS <- 1` sets a conservative precedent. The notebook passes `--n-workers 8`.

### 2.2 Restructure the accumulators (Stage A)

Extract the inner loop body (`gam_trajectory_features.R:372-424`) into a top-level `process_one_lab(lab, dt_lab, landmark_day, ...)` that **returns** `list(features=, curves=, diagnostics=, log=)` instead of mutating `feature_frames` / `curve_frames` / `diagnostic_rows` (lines 367-369):

```r
per_lab <- parallel::mclapply(landmark_labs, process_one_lab, ...,
                              mc.cores = n_workers, mc.preschedule = FALSE)
names(per_lab) <- landmark_labs
```

`mclapply` preserves input order, so reassembly is `lapply(per_lab, `[[`, "diagnostics")` with `Filter(Negate(is.null), ...)` for the optional slots. Naming the list preserves the `feature_frames[[lab]]` semantics at line 417.

Three traps, all of which produce silent or confusing failures:

- **`mc.preschedule = FALSE` is required.** Lab fit cost varies widely (200k obs vs 2k). Prescheduling round-robins labs into fixed chunks and strands one worker with all the slow ones. One fork per lab is negligible at tens of labs.
- **`mclapply` returns `try-error` objects rather than aborting.** Check explicitly after collection and `stop()` naming the offending labs, or a failure surfaces as a cryptic `rbindlist` error instead.
- **`cat()` output interleaves unreadably across workers.** Buffer the per-lab progress lines (385-389, 395-399) into the returned `log` slot and `cat` them in the parent after collection. Keeps the batch log deterministic and diffable, which matters for this pipeline's provenance discipline.

### 2.3 `setTimeLimit` under fork

Keep `setTimeLimit` **strictly inside** `process_one_lab` (currently lines 232/234). Never hoist it to the parent — there its elapsed clock would cover the whole parallel batch rather than a single fit. Inside a forked child it still raises into the existing `tryCatch` at line 233, so the `re_fallback` logic survives the refactor unchanged.

The header comment at lines 146-149 already notes the limit is advisory (it polls at bytecode boundaries). That remains true and is unaffected. Note also that after Step 5.4 this branch is unreachable in production, since `n_patients > max_fs_patients` returns at line 224 first.

### 2.4 Memory and thread oversubscription

**Memory.** Fork is copy-on-write only until something touches the pages. The per-lab filter at line 372 runs *inside* the child, so materialize chunks in the parent instead — this doubles as Step 4.1:

```r
lab_long <- lab_long[is.finite(LAB_VALUE) & is.finite(t_lab)]
setkey(lab_long, LAB_NAME)
lab_chunks <- split(lab_long[, .(DFCI_MRN = get(id_col), LAB_VALUE, t_lab, LAB_NAME)],
                    by = "LAB_NAME", keep.by = FALSE)
rm(lab_long); gc()
```

`rm` + `gc()` before forking drops the full table so children never inherit it. This converts an "N × full table" risk into "1 × full table, split disjointly". Document in the header that peak RSS ≈ `n_workers × (largest lab chunk + its bam() working set)`; recommend `--n-workers 8`, not 18.

**Oversubscription.** `nthreads` (line 127) feeds `bam(..., nthreads=)` at lines 163/170/183, driving mgcv's OpenMP. With W workers × T threads you get W×T. Keep `--nthreads 1` and parallelize across labs instead: `bam(discrete=TRUE)` on a single-smooth Gaussian scales poorly with OpenMP, while lab-level parallelism is embarrassingly parallel. W=8/T=1 beats W=1/T=8 substantially.

- Warn near line 137 (with the other validations) if `n_workers * nthreads > parallel::detectCores()`.
- **Set `OMP_NUM_THREADS=1` in the child**, at the top of `process_one_lab`. BLAS inside `bam()`'s linear algebra spawns its own threads per worker even when mgcv's `nthreads` is 1. This is the most common cause of "8 workers isn't 8× faster."
- Ordering matters: `fread` in the parent at full `setDTthreads()`, then `setDTthreads(1)`, then fork.

### 2.5 Stage B feature parallelism

Only if Step 1 proves insufficient — likely, after the 5× feature growth. Apply the same treatment to the feature loop at `gam_cox_nonlinearity.R:266`. It is a simpler refactor than Stage A: the body already builds one self-contained `data.table` row per feature, so return it rather than appending to `rows`. `rbindlist` (line 314) and `p.adjust` (line 315) stay as-is, and **because `mclapply` preserves order, `q_lrt` remains bit-identical.**

Do **not** stack Steps 1 and 2.5 naively — that is 3 × W processes. Prefer landmarks serial with `--n-workers 12`: better load balance, since landmark 0 has more patients than 180.

---

## Step 3 — Cheap wins, bit-identical output

**3.1 Kill the per-lab full scan** (line 372) using the parent-side `split()` from 2.4. Removes N_labs full vector scans over a multi-million-row unindexed table.

> **Correctness trap.** Labs present in `landmark_labs` but with zero surviving rows vanish from `split()`'s output. Iterate over `landmark_labs`, **not** `names(lab_chunks)`, treating a missing chunk as a zero-row table — otherwise the `skipped_too_few_obs` diagnostic rows (376-380) silently disappear from the sidecar. Apply the `is.finite` filter *before* splitting so per-lab data matches today's exactly and the `min_obs_per_lab` check at line 374 sees the same count.

**3.2 One-shot merge instead of `Reduce`** (line 428). `Reduce(merge, ...)` does N_labs passes over a table widening by 5 columns each pass. Build the ID union once, key it, assign by reference:

```r
all_ids <- unique(unlist(lapply(feature_frames, `[[`, id_col)))
combined <- setNames(data.table(all_ids), id_col); setkeyv(combined, id_col)
for (fr in feature_frames) {
  setkeyv(fr, id_col); cols <- setdiff(names(fr), id_col)
  combined[fr, (cols) := mget(paste0("i.", cols))]
}
```

Identical modulo row order (both sort by key); column order preserved by iteration order.

**3.3 Drop redundant `copy()`** at lines 188, 413, 419. Line 188 exists only to protect the caller's `dt_lab` from the `:=` at 189-192; under 3.1 that is a private per-lab chunk read afterward only via `levels()` (line 409), which two extra columns do not disturb. Lines 413/419 wrap objects `extract_lab_features` already constructed fresh (326, 328).

**3.4 `fread(select=)`** at line 349 — read only `id_col`, `LAB_NAME`, `LAB_VALUE`, `t_lab`.

> The three `pre_treatment_lab_long_landmark{D}.csv` files are genuinely different files covering different pre-landmark windows. The per-landmark re-read is not redundant and cannot be eliminated.

**3.5 Single `predict()` call** (lines 290-291). Derive the response from the `type="terms"` matrix via `rowSums(term_mat) + attr(term_mat, "constant")`. Exact for this Gaussian identity-link model only — not if the family ever changes. **Low priority**, since Step 5.4 makes this branch unreached in production.

**3.6 Checkpointing.** Write the diagnostics CSV immediately after `mclapply` returns, before feature assembly, so a crash in the merge still preserves the timing data. Add `--resume` to skip a landmark whose feature CSV already exists — the notebook's `FORCE_RERUN` expresses this intent but is all-or-nothing across landmarks. Landmark granularity is the right unit; per-lab checkpoints are not resumable under `mclapply` anyway.

---

## Step 4 — Curve output reduction (largest I/O win, zero feature change)

New flags after line 71, following the existing `DEFAULT_*` convention:

```r
DEFAULT_WRITE_CURVES <- "true"        # --write-curves {true,false}
DEFAULT_CURVE_GRID_POINTS <- 25L      # --curve-grid-points
DEFAULT_CURVE_LABS <- ""              # --curve-labs "PSA,Testosterone"  ("" = all)
```

Recommended production settings: `--curve-labs "PSA,Testosterone"`, `--curve-grid-points 9`. From ~50 labs × 25 points to 2 labs × 9 points is roughly a **70× smaller file**.

**No feature column changes.** These flags gate only the `curves` element returned at line 328 and the writer at 441-453. `gam_trajectory_features_landmark{D}.csv` is untouched.

Downstream: `summarize_gam_curves` groups by `t_lab`, so 9 shared grid points still yield valid panels — 9 vertices instead of 25, with `n_patients` and `sem` per point unchanged (same patients, same fitted function, fewer evaluation points).

Keep `--write-curves true` as the default. `false` is safe (the figure loop hits its existing `!file.exists` guard at line 2610 and `next`s with a message) but loses panels silently, which is a trap. Set `--curve-labs` to match `LAB_FIGURE_LABS`; if `plot_non_androgen_lab_figures` is ever enabled, widen it.

---

## Step 5 — Grid and basis tuning (changes feature values)

### 5.1 Decouple the AUC grid from the curve grid

Lines 276-280 build one grid serving two masters. Add `DEFAULT_AUC_GRID_POINTS <- 25L` (`--auc-grid-points`, recommend **9**) driving `length.out` at line 277, independent of `--curve-grid-points`.

Exact per-column impact:

| Column | Effect |
| --- | --- |
| `<LAB>__gam_level` | **Bit-identical** — reads only `t_star` (line 308) |
| `<LAB>__gam_slope` | **Bit-identical** — reads `t_star`, `t_star-h` (308-309) |
| `<LAB>__gam_curvature` | **Bit-identical** — reads `t_star`, `t_star-h`, `t_star-2h` (308-310) |
| `<LAB>__gam_auc` | Changes. Rectangle-rule mean (line 323) of a smooth with `k_pop<=10`; expect r > 0.999 |
| `<LAB>__gam_dev` | Changes. On `two_stage_ridge`, `dev` is **exactly linear** in `t_lab` (286-287), so its RMS converges very fast |

Do not go below ~7 points: `gam_dev` degenerates toward a 3-point quadrature.

**Leakage.** Changing `length.out` alters density only; `seq()`'s endpoints remain `t_star - W` and `t_star`. Make the invariant structural rather than a comment — add after line 278:

```r
stopifnot(max(eval_points) <= t_star)
```

### 5.2 Reduce `k_pop` — decide from data, do not guess

`DEFAULT_K_POP <- 10` (line 59); `bam` cost scales roughly with `k²`. But `k_pop` drives the population smooth that is the backbone of every fitted curve (285/288), so it moves **all five** feature columns for every lab, plus `edf_pop` in the diagnostics.

Let Step 0's `median(edf_pop)` decide. If it sits well below `k_pop - 1`, the extra basis functions are already being penalized away and `k_pop = 8` is nearly free. If `edf_pop` presses against the ceiling, reducing `k` genuinely changes the estimated trajectory shape — trading science for speed, which needs the Verification §3 report before acceptance. Do this **last, in its own commit**, since it destroys the bit-identity signal that makes §3 a sharp test.

Note `k_pop_use` is already clamped to `n_unique_t - 1` at line 156, so sparse labs sit below 10 regardless.

### 5.3 Flip `--max-fs-patients 0`, but keep the code

Change the validation at line 137 from `< 1L` to `< 0L` so `0` means "always two-stage", document it, and pass `--max-fs-patients 0` from the notebook.

**Keep `fit_fs`/`fit_re` and the `else` branch of `extract_lab_features`.** Deleting ~60 lines and the whole `setTimeLimit` concern is tempting, but: the smoke test deliberately drives the two-stage path via `--max-fs-patients 100`; the `fs` path is the hierarchical method `GAM_SURVIVAL_PLAN.md:68` documents; and it is what makes `--fit-split train_val` meaningful at small scale. The branch at line 207 costs one integer comparison.

---

## Files touched

| File | Change |
| --- | --- |
| `COMPASS/survival_analysis/gam_trajectory_features.R` | `process_one_lab` extraction + `mclapply`; `--n-workers`, `--write-curves`, `--curve-grid-points`, `--curve-labs`, `--auc-grid-points`, `--resume`; parent-side `split()`; one-shot merge; `stopifnot` leakage assertion |
| `COMPASS/survival_analysis/gam_cox_nonlinearity.R` | `--n-workers` on the feature loop (Step 2.5, conditional on Step 0) |
| `COMPASS/survival_analysis/COMPASS_run_GAMs.ipynb` | `wait=` in `run_r_script`; hoisted validation + concurrent landmark spawn; `system.time()` instrumentation; new Stage A flags |
| `COMPASS/survival_analysis/tests/test_gam_trajectory_features_smoke.R` | Pin grid flags explicitly; add `fs`-path coverage; add worker-identity check |
| `COMPASS/survival_analysis/tests/test_gam_cox_nonlinearity_smoke.R` | Worker-identity check (only if 2.5 is done) |

Deliberately unchanged: `COMPASS_generate_figures_pipeline.R` (it already handles a missing or coarser curve file), the Python side entirely, and the IPIO arm.

## Verification

1. **Smoke tests** (~2s each; run on every change). `test_gam_trajectory_features_smoke.R` **breaks** on grid changes — lines 116, 118, 123 hardcode `25`. Fix by passing `--curve-grid-points` / `--auc-grid-points` explicitly in the `system2` args (lines 76-81) and pinning an `EXPECTED_CURVE_POINTS` constant, the same pattern the test already uses for `--max-fs-patients 100`. Line 110's `basis_used == "two_stage_ridge"` assertion still passes (200 patients > 100).

2. **New assertions worth adding.**
   - **Worker identity:** run with `--n-workers 1` and `--n-workers 2` into two temp dirs and `all.equal(fread(a), fread(b))`. This is the highest-value new check — it catches exactly the ordering/accumulator bugs Step 2.2 can introduce. Add the same to the Stage B test if 2.5 lands, since `q_lrt`'s BH adjustment is order-dependent.
   - **`fs`-path coverage:** a second invocation with `--max-fs-patients 1000` asserting `basis_used == "fs"`. Nothing currently tests the hierarchical path — the existing test forces two-stage with 200 > 100.

3. **Before/after feature comparison** (one-off interactive analysis; do not build a script):

   ```r
   old <- fread("<backup>/gam_trajectory_features_landmark90.csv")
   new <- fread("<new>/gam_trajectory_features_landmark90.csv")
   setkeyv(old,"DFCI_MRN"); setkeyv(new,"DFCI_MRN")
   stopifnot(identical(old$DFCI_MRN, new$DFCI_MRN))
   cols <- intersect(setdiff(names(old),"DFCI_MRN"), names(new))
   cmp <- rbindlist(lapply(cols, function(c) {
     a <- old[[c]]; b <- new[[c]]; ok <- is.finite(a) & is.finite(b)
     data.table(feature=c, r=if(sum(ok)>2) cor(a[ok],b[ok]) else NA_real_,
                identical=isTRUE(all.equal(a,b)))
   }))
   cmp[, stat := sub("^.*__","",feature)]
   cmp[, .(min_r=min(r,na.rm=TRUE), n_identical=sum(identical)), by=stat]
   ```

   **The shape of that table is itself the regression test.** After Steps 3 and 5.1, `gam_level` / `gam_slope` / `gam_curvature` must be **bit-identical** (`min_r == 1`), and `gam_auc` / `gam_dev` should show `min_r > 0.999`. If any of the first three moved, there is a bug — they read fixed evaluation points and cannot legitimately change. Run Step 5.2 separately, since it moves all five and destroys this signal.

4. **Timing.** Re-run the Step 0 query after each step; record Stage A wall, `sum(fit_seconds)`, their ratio, and Stage B wall per landmark in the commit message (this repo documents comparisons in commits — see `gam_trajectory_features.R:32`). Also capture peak RSS (`/usr/bin/time -l` on macOS, `-v` on Linux) for the parallel run — this is how fork memory multiplication is caught before the cluster OOM-kills the job.

5. **Figure regression.** Run `COMPASS_generate_figures_pipeline.R` after Step 4 and confirm the GAM trajectory panels still render for PSA and Testosterone with 9-vertex ribbons.

6. **Leakage re-check.** Confirm `t_star <- landmark_day - epsilon_days` still drives every `seq`/`c()` upper bound, and the `stopifnot` from 5.1 is present. No new flag may raise the upper endpoint — a wider curve window, if ever wanted, extends **backward** only.

## Execution order

1. **Step 0** — measure; gates everything below.
2. **Step 1** — Stage B landmark parallelism (notebook, ~10 lines, ~3×).
3. **Step 3** — cheap identical wins; verify all five stats bit-identical.
4. **Step 2** — `mclapply` in Stage A; verify with the worker-identity assertion.
5. **Step 2.5** — Stage B feature parallelism, only if Step 0 showed it dominating.
6. **Step 4** — curve lab/grid restriction; verify figures still render.
7. **Step 5.1** — AUC grid; verify level/slope/curvature identical, auc/dev r > 0.999.
8. **Step 5.3** — flip `--max-fs-patients 0` in the notebook.
9. **Step 5.2** — `k_pop`, only if `edf_pop` justifies it; own commit, own §3 report.

Expected: Stage A ≈ `n_workers`× on the fit portion minus Amdahl on read/merge/write, plus a large cut to the predict/IO tail from Step 4. Stage B ≈ 3× from Step 1 alone.

## Open risk to confirm during execution

Fork-based parallelism interacts badly with an already-warm OpenMP pool and a large uncollected heap. Step 2.4's `OMP_NUM_THREADS=1` in the child and `rm(lab_long); gc()` before the fork address both. If peak RSS in Verification §4 scales with `n_workers` rather than staying near one copy of the largest chunk, the copy-on-write assumption has broken — drop `--n-workers` and re-measure before pushing further.
