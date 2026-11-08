**COMPASS Figure 6: Gleason and TSG augmentation analysis plan**

Prepared November 7, 2026. Proposed analyses; no new performance results have been calculated. This plan fixes the comparison strategy before examining augmentation results. It does not make the already inspected DFCI cohort a new external validation set.

The objective is to determine whether the transferred three-item lab checklist adds predictive information to Gleason score, and then to Gleason plus tumor suppressor gene (TSG) status, while retaining a practical screening rule. Start with the largest Gleason-available cohort. Pursue a four- or five-item checklist only if the combined analyses support it. Improved performance is a hypothesis to test.

The current reference is the generalized day-90, five-year checklist: minimum ALT ≤15 U/L; maximum alkaline phosphatase ≥100 U/L; maximum testosterone ≤550 ng/dL; positive when at least two criteria are met. The manuscript reports AUCs of 0.73 versus 0.71 for checklist versus Gleason ≥8 for platinum, and 0.67 versus 0.65 for AVPC, in patients with both Gleason and sequencing. Those small point-estimate differences motivate evaluating incremental value and screening utility.

1. **Reproduce the current result and freeze the analysis definitions.**

   Recover the existing DFCI analysis code, frozen three-site checklist, feature definitions, and patient-level evaluation outputs. These are not present in this manuscript folder; implementation requires the authorized analysis environment. Reproduce Figure 6 before adding predictors, including its patient/event counts, item scores, flags, and AUCs. Resolve discrepancies rather than changing definitions to match a desired result.

   Preserve the original lab thresholds, lookback windows, aggregation rules, unit conversions, and treatment of missing labs. Record all of these in a configuration file. In particular, recover what the original code does when a lab is absent; do not silently interpret an unmeasured lab as normal. Report checklist evaluability and per-item missingness, with a complete-lab sensitivity analysis if needed.

   Use the 90-day landmark and five-year horizon for the primary analysis. Platinum is the primary endpoint; AVPC is the key secondary endpoint; NEPC is exploratory. Do not select a different landmark, horizon, or checklist because it gives a larger augmentation gain. Existing alternative landmark/horizon results remain supplementary context.

2. **Construct separate, temporally valid cohorts.**

   | Cohort | Eligibility and prediction time | Purpose |
   |---|---|---|
   | Full checklist cohort | Original eligible, endpoint-free patients at ADT +90 days | Preserve frozen checklist transfer performance and report coverage |
   | Gleason cohort: primary | Checklist-evaluable patients with Gleason information available by ADT +90 days; sequencing is not required | Test Gleason + checklist in the largest appropriate cohort |
   | Gleason + TSG cohort: extension | Checklist-evaluable patients with Gleason and an interpretable sequencing report available by the genomic prediction landmark | Test addition of the checklist to Gleason + TSG |

   For the genomic extension, first audit the existing sequencing-relative implementation. Retain sequencing report date +90 days to match the manuscript's genomic analyses, provided this is after ADT initiation; explicitly exclude and count patients whose genomic landmark precedes ADT. Require survival, observation, and freedom from the endpoint at this landmark. Recompute all time-varying inputs using only information available then. Document the feature window used on this clock. Applying an ADT-trained checklist on a sequencing-relative clock is a distinct application and must be labeled accordingly.

   Add an ADT +90-day genomic sensitivity analysis using only patients whose sequencing reports were already available by that date, if sample size permits. Later sequencing must not be used as a baseline predictor. Do not choose between clocks based on performance.

   Use Gleason results documented by the landmark, following the manuscript's closest-available-score rule; a historical specimen date alone does not establish that its result was available. Require adequate sequencing coverage for a negative gene call. Missing Gleason, absent sequencing, and unassessable genes are unavailable information, not negative findings.

   Build endpoint-specific risk sets and report patients, events within five years, total observed events, follow-up support, and exclusions separately. Current Figure 6 counts are reproduction targets, not guaranteed counts for the revised cohorts. Compare all models within a panel on exactly the same patients, clock, horizon, and outcome definition. Do not interpret AUC differences across availability cohorts as augmentation gains.

3. **Test incremental information using the checklist item count.**

   Define `L` as the number of lab criteria met (0–3) and retain the original binary flag `B = I(L ≥2)`. Evaluate both frozen scores directly. The count retains distinctions that the flag discards, but its AUC is not guaranteed to improve.

   Fit small penalized Cox models using the existing time-to-event framework. Include age identically in each fitted comparison, using the same coding. Use total Gleason score as a single ordered term for the initial analysis; retain Gleason ≥8 as the historical binary comparator. If primary and secondary Gleason patterns support reliable Grade Group derivation, evaluate Grade Group as a specified sensitivity analysis rather than substituting whichever representation performs best.

   | Cohort | Clinical baseline | Augmented model | Additional references |
   |---|---|---|---|
   | Gleason | Age + Gleason | Age + Gleason + L | Frozen L; original B; Gleason ≥8; locally fitted age + L |
   | Gleason + TSG | Age + Gleason + TP53 + RB1 + PTEN indicators | Same baseline + L | Frozen L; original B; locally fitted age + L; baseline without TSG to describe genomic contribution |

   Start without interactions or nonlinear terms. Freeze gene-alteration definitions before fitting. Reproduce the manuscript's alteration coding for continuity, but audit amplifications, variants of uncertain significance, and loss/inactivation annotations. A sensitivity analysis using biologically appropriate pathogenic alterations should be defined from annotation rules, not observed associations.

   Fit endpoint-specific augmentations for platinum and AVPC and label them as local development. NEPC has only ten events in the current paired comparison; use frozen scores and descriptive estimates initially, without a separately tuned multivariable NEPC model. If maintaining cross-endpoint transfer is a desired claim, additionally fit the augmentation on platinum in each training fold and apply it unchanged to held-out AVPC/NEPC outcomes. Do not describe an AVPC-trained augmentation as transfer without task-specific fitting.

4. **Generate paired out-of-sample estimates.**

   Use five patient-level outer folds, shared across competing models within each cohort. Stratify for platinum and AVPC events where feasible, fix the seed, and record fold assignments. A patient must never contribute to both training and validation within a fold. Any repeated records remain grouped by patient.

   Fit preprocessing, any imputation of eligible routine covariates, scaling, penalty selection, baseline hazards, and calibration within training data. Choose ridge strength with inner cross-validation. Do not impute unavailable Gleason or sequencing to enlarge an availability cohort. Generate one outer-fold prediction per patient. When evaluating survival-model predictions across folds, use a consistent horizon-specific risk scale derived from each training fold, rather than pooling arbitrary Cox linear-predictor offsets.

   Compute five-year IPCW AUC with the manuscript's estimator and clearly stated censoring convention. The primary contrast is the Gleason-cohort platinum AUC difference between the augmented model and its clinical baseline. The corresponding AVPC contrast is key secondary; genomic extension contrasts are secondary/exploratory. Also report augmentation versus the locally fitted age + L model to assess whether clinical information improves on the lab score.

   For each paired contrast, report both AUCs, their difference, and a 95% patient-bootstrap interval. For final inference, use 1,000 resamples that repeat the fitting and tuning pipeline; keep repeated copies of a resampled patient together to avoid train/test leakage. A bootstrap of fixed out-of-fold predictions may be used for an initial diagnostic, but label its intervals as conditional on the fitted predictions. If sparse events cause invalid folds or resamples, report that instability and simplify the analysis rather than silently dropping failures.

   Report five-year follow-up support and censoring-weight diagnostics. If the horizon is poorly supported, state that limitation; do not silently replace it with the horizon that looks strongest. Treat death as in the original analysis for the primary reproduction and provide a competing-risk sensitivity analysis when reliable death dates are available. Specify whether a reported probability is net event risk or cumulative incidence in the presence of death.

5. **Assess whether the gain translates into a useful screening rule.**

   Report flag rate, sensitivity, specificity, PPV, and NPV at five years with censoring handled consistently. Raw event fractions over unequal follow-up are not five-year PPVs. Compare fitted models at a screening workload close to the original checklist's flag rate within the same cohort. Derive thresholds from training-fold score distributions and report the actual held-out flag rates. Discrete scores may not permit an exact match; show the attainable neighboring operating points without selecting thresholds on held-out outcomes.

   In the Gleason cohort, cross-classify patients by Gleason <8 versus ≥8 and original checklist negative versus positive. Show all four groups with patient/event counts and five-year risk estimates with intervals. Estimate risks with death handled explicitly; if using cumulative incidence, use an appropriate competing-risk estimator. Test whether the checklist separates risk within each Gleason stratum, reporting uncertainty and avoiding an interaction claim based only on subgroup point estimates.

   This analysis answers two practical questions: whether the lab flag identifies additional risk among lower-Gleason patients, and whether it separates risk within high-Gleason patients. Do not split the genomic cohort into all eight Gleason/checklist/TSG combinations unless counts support interpretable estimates.

   Decision-curve analysis is optional after the intended action and its plausible threshold range are defined. Any model-derived absolute risks require training-fold calibration and held-out assessment. Do not infer clinical benefit from AUC alone.

6. **Evaluate simple augmentation candidates after the information test.**

   | Candidate | Score | Intended evaluation |
   |---|---|---|
   | Original | L, with flag L ≥2 | Frozen transfer reference |
   | Gleason addition | L + I(Gleason ≥8), range 0–4 | Evaluate the full score and prespecified flag thresholds ≥2 and ≥3 |
   | Gleason + TSG addition | L + I(Gleason ≥8) + I(any qualifying TSG alteration), range 0–5 | Evaluate the full score; select a screening threshold only within training folds |
   | Genomic sensitivity | Replace the any-TSG point with I(≥2 qualifying genes altered) | Test a biologically motivated combined-defect definition |

   The ≥2-of-4 rule expands the original flagged group; ≥3-of-4 narrows it. Evaluate their sensitivity/enrichment tradeoff directly. Equal points are an interpretability choice, not evidence that predictors have equal effects. Compare each simple score with the cross-validated fitted augmentation to quantify any performance cost of simplification.

   The ≥2-gene sensitivity analysis is motivated by the published AVPC molecular profile, but reported alteration count must not be equated automatically with functional loss. Keep both definitions visible rather than choosing the stronger-looking result after inspection.

   If selecting among scores or thresholds, selection belongs inside the outer validation loop. Report selection stability and the performance of that selection procedure. A final full-data rule is a development result requiring subsequent validation; its apparent full-data performance is not the validation estimate.

   Use the four-item rule only when Gleason is available, and the five-item rule only when both pathology and sequencing are available. Otherwise retain the appropriate simpler rule. Validate each version separately. Do not claim a validated whole-cohort routing strategy by merely combining metrics from different populations or prediction clocks.

7. **Build the revised figure around the prespecified comparisons.**

   | Panel | Content | Main message tested |
   |---|---|---|
   | a | Original three lab criteria, units, lookback window, and any supported optional Gleason/TSG extension | Explain exactly how the candidate screen works |
   | b | Gleason-cohort platinum and AVPC AUCs for age + Gleason, age + L, and age + Gleason + L; paired augmentation differences and intervals | Added value in the larger available cohort |
   | c | Sequencing-cohort AUCs for age + Gleason + TSG versus the same model + L, with a lab reference | Added value beyond pathology and genomics |
   | d | Five-year risks in the four Gleason/checklist groups, with counts and intervals | Show which patients receive additional risk information |

   Use point-and-interval plots with readable model labels. Distinguish fitted models from frozen checklist scores. Give each panel its own cohort size, event count, prediction clock, and horizon. If useful, add a compact screening-metrics table below the figure.

   Preserve the original full-cohort transfer and survival results in the supplement, together with binary comparators, NEPC estimates, calibration, missingness, genomic definitions, and sensitivity analyses. If the augmented rules do not show useful performance, panel a should retain the original checklist and the remaining panels should report the measured limits of augmentation. Do not present an unsupported optional extension as an established clinical tool.

8. **Use explicit decision points and deliverables.**

   First deliver the reproduction audit, cohort flow/counts, input definitions, and missingness summary. Then deliver the Gleason comparison, four-group risk estimates, and screening metrics. Complete the genomic extension next. Evaluate bedside simplification after these analyses establish whether complementary signal exists.

   Favor the augmented approach when paired out-of-sample estimates show a reasonably precise, useful gain, screening metrics support that gain, and the simple rule retains acceptable performance. Establish any practical minimum improvement with the study team before viewing the new results; this plan does not invent a universal clinically meaningful AUC threshold. If discrimination gains are small but within-Gleason stratification is informative, emphasize that finding. If intervals are wide or gains are absent, report uncertainty or lack of demonstrated added value rather than asserting equivalence or superiority.

   Save a versioned analysis configuration, cohort/exclusion tables, patient-level out-of-fold predictions within the authorized environment, aggregate metric tables, bootstrap diagnostics, figure source data, and revised figure files. Draft replacement Results, Methods, and caption text only after results are available. Final wording must distinguish frozen external transfer from locally developed, internally validated augmentation and keep NEPC exploratory.

**Sources and existing project material**

- [Current manuscript](v2%20COMPASS%20Manuscript%20Draft.docx): checklist results and Methods on transfer, clinical covariates, and genomic landmarks.
- [Current Figure 6](figures/main/06_federated_checklist_transfer_to_DFCI.pdf) and [Supplementary Figure 6](figures/supplement/S6_full_transfer_to_dfci.pdf): frozen checklist comparisons and existing combined-model template.
- [Existing reviewer-analysis notes](COMPASS_reviewer_preemption.html): screening workload, calibration, sparse NEPC events, and time-origin consistency.
- [Aparicio et al., combined tumor suppressor defects](https://pubmed.ncbi.nlm.nih.gov/26546618/) and [AVPC molecular-profile study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9265062/): rationale for a combined-defect genomic sensitivity analysis.
- [Vickers and Elkin, decision-curve analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC2577036/): assessing clinical consequences beyond discrimination.
