# TRIPOD+AI checklist

Paper: *Published Diagnostic Statistics as Bayesian Priors for Coronary Artery Disease: Prior-Data Conflict and a Small-Sample Window* (IEEE EMBC 2027 submission).

Checklist items are from Collins et al., "TRIPOD+AI statement," *BMJ* 385:e078378 (2024), Table 2 (extracted text: `sources/prior_art/tripod_ai_checklist_extracted.txt`). "D" means development and "E" means evaluation.

This is a methods study that compares prior specifications. It does not develop a model for clinical deployment, so several items about deployment are marked not applicable. Items marked **AUTHOR** need information only the author can give. Complete them before the code is published.

| Item | Topic | Where reported, or why not |
|---|---|---|
| 1 | Title | Title: prediction of coronary artery disease (CAD), with prior specifications evaluated. The target population (patients referred for angiography) is stated in the Abstract |
| 2 | Abstract | Abstract |
| 3a | Healthcare context and rationale | Sec. I. Diagnostic, not prognostic; existing pretest models are cited [2]–[4] |
| 3b | Target population and intended use | Sec. II-A (patients referred for coronary angiography). Intended use: a methodological comparison, not a model for clinical deployment |
| 3c | Known health inequalities | Sec. III-B: calibration checked separately for women and men, because pretest tables overestimate CAD in women [4] |
| 4 | Objectives | Sec. I, last paragraph |
| 5a | Data sources | Sec. II-A (Cleveland Clinic file, UCI Heart Disease [14]); Sec. II-E (Hungarian and Long Beach VA files; the Z-Alizadeh Sani dataset, UCI 2013, doi:10.24432/C5Q31T). All four cohorts underwent angiography |
| 5b | Dates of data collection | Not in the paper. The UCI documentation (`data/heart-disease.names`) gives only the donation date (July 1988), not accrual dates |
| 6a | Setting and centers | Cleveland Clinic Foundation; Hungarian Institute of Cardiology, Budapest; V.A. Medical Center, Long Beach, CA (`data/heart-disease.names`); Z-Alizadeh Sani: Tehran according to secondary sources (see `sources/uci_z_alizadeh_sani_page_2026-11-15.md`), not stated on the UCI page. Sec. II-A and II-E |
| 6b | Eligibility | Referral for coronary angiography (Sec. II-A). New centers: complete cases for the five reduced-model predictors (Sec. II-E) |
| 6c | Treatments | Not applicable (diagnostic, cross-sectional) |
| 7 | Data preparation | Sec. II-A: coding into 18 columns, mode imputation of 4 calcification and 2 thallium values, centering on training means. Code: `code/analysis.py` (`encode`) and `code/transfer.py` (`encode_red`) |
| 8a | Outcome | Sec. II-A: angiographic CAD, >50% diameter narrowing in any major vessel. Cross-sectional, no time horizon |
| 8b | Outcome assessors | Not reported in the source data |
| 8c | Blinding of outcome assessment | Not reported in the source data |
| 9a | Choice of predictors | Sec. II-A and II-B: all 13 recorded predictors (Cleveland); predictors with published statistics at the new centers |
| 9b | Predictor definitions | Sec. II-A; UCI documentation |
| 9c | Predictor assessors | Not reported in the source data |
| 10 | Sample size | Fixed by the available data; no formal calculation. The learning curves (Sec. II-E, III-C, III-D) examine performance as a function of sample size |
| 11 | Missing data | Sec. II-A (mode imputation at Cleveland, recomputed within each cross-validation fold); Sec. II-E (complete cases at the new centers: 293 of 294 Hungarian, 144 of 200 VA) |
| 12a | Use of data | Sec. II-A (stratified 80/20 split); Sec. II-D (LOO, test set, 5-fold cross-validation); Sec. II-E (subsets, scoring on the remaining patients) |
| 12b | Predictor handling | Sec. II-A (clinical units, indicators, centering); ridge uses standardized predictors (Sec. II-D) |
| 12c | Model type and building steps | Sec. II-B to II-E. No predictor selection; hyperparameters: prior SDs as stated, ridge penalty by inner 5-fold cross-validated log loss |
| 12d | Heterogeneity across clusters | Centers analyzed separately (Sec. II-E, III-D); no pooled model |
| 12e | Performance measures | Sec. II-D: log-likelihood (log loss), LOO ELPD, calibration slope, AUC, net benefit (decision curves) |
| 12f | Model updating | Sec. II-E: at the new centers, models are refitted on local subsets with priors fixed in advance |
| 12g | How predictions were calculated | Posterior predictive means. Code: `code/analysis.py` (`Model.predict_proba`), `code/transfer.py` (`proba`), `code/extend.py` |
| 13 | Class imbalance | None used |
| 14 | Fairness | Calibration by sex (Sec. III-B); no other fairness methods |
| 15 | Model output | Predicted probabilities. No classification threshold, except the 5%–50% range in the decision curves |
| 16 | Differences between development and evaluation data | Sec. II-E and III-D: reduced predictors (no fluoroscopy or thallium); different prevalence (36%, 79% and 71% vs. 46%); Z-Alizadeh Sani has only age, sex and chest-pain type; see `results/characteristics.csv` |
| 17 | Ethical approval | **AUTHOR**: the IRB determination is pending (see `../NEXT-STEPS.md`). The paper states that the data are public and de-identified (Sec. II-A) |
| 18a | Funding | None. This work received no funding |
| 18b | Conflicts of interest | None declared |
| 18c | Protocol | No protocol was prepared |
| 18d | Registration | Not registered |
| 18e | Data availability | Public: UCI Machine Learning Repository, doi:10.24432/C52P4X (CC BY 4.0); copies in `data/` |
| 18f | Code availability | Public repository https://github.com/tpattabhi/literature-priors-cad, cited in Sec. II-E (last sentence) |
| 19 | Patient and public involvement | None |
| 20a | Participant flow | Sec. II-A (303; 242 training, 61 test; 139 with CAD); Sec. II-E (complete cases at the new centers) |
| 20b | Characteristics | Partly in the text (prevalence, sex counts); full table in `results/characteristics.csv` (`code/characteristics.py`); summarized below |
| 20c | Development vs. evaluation distributions | `results/characteristics.csv` (below) |
| 21 | Participants and events per analysis | Sec. II-A, II-E; Table III note |
| 22 | Full model specification | Posterior means in Table I (literature coefficients); full posteriors reproducible with `code/analysis.py`. Not intended for deployment |
| 23a | Performance with uncertainty | SEs for every contrast (Tables II–III, text); calibration-slope bootstrap intervals in `results/cv_metrics.csv` |
| 23b | Heterogeneity across clusters | Sec. III-D, Table III |
| 24 | Model updating results | Sec. III-D, Table III |
| 25 | Interpretation | Sec. IV |
| 26 | Limitations | Sec. IV, second paragraph |
| 27a | Handling poor-quality input data in use | Not applicable (no deployment) |
| 27b | User interaction and expertise | Not applicable (no deployment) |
| 27c | Next steps | Sec. IV (future work) |

## Patient characteristics by center (items 20b and 20c)

From `results/characteristics.csv` (`code/characteristics.py`). Hungarian and Long Beach VA are complete cases for the reduced model.

| | Cleveland, all | Cleveland, training | Cleveland, test | Hungarian (complete cases) | Long Beach VA (complete cases) | Z-Alizadeh Sani |
|---|---|---|---|---|---|---|
| patients | 303 | 242 | 61 | 293 | 144 | 303 |
| CAD, n (%) | 139 (46) | 111 (46) | 28 (46) | 106 (36) | 114 (79) | 216 (71) |
| age, mean (SD) | 54.4 (9.0) | 54.1 (9.1) | 55.7 (8.9) | 47.8 (7.8) | 59.4 (7.9) | 58.9 (10.4) |
| male, n (%) | 206 (68) | 165 (68) | 41 (67) | 213 (73) | 139 (97) | 176 (58) |
| typical angina, n (%) | 23 (8) | 22 (9) | 1 (2) | 11 (4) | 3 (2) | 164 (54) |
| atypical angina, n (%) | 50 (17) | 39 (16) | 11 (18) | 105 (36) | 7 (5) | 93 (31) |
| nonanginal pain, n (%) | 86 (28) | 61 (25) | 25 (41) | 54 (18) | 28 (19) | 16 (5) |
| asymptomatic, n (%) | 144 (48) | 120 (50) | 24 (39) | 123 (42) | 106 (74) | 30 (10) |
| exercise angina, n (%) | 99 (33) | 80 (33) | 19 (31) | 89 (30) | 94 (65) | not recorded |
| ST depression mm, mean (SD) | 1.04 (1.16) | 1.04 (1.15) | 1.05 (1.21) | 0.58 (0.91) | 1.32 (1.11) | not recorded |
| fluoroscopy recorded, n | 299 | 238 | 61 | 3 | 2 | 0 |
| thallium recorded, n | 301 | 241 | 60 | 28 | 32 | 0 |
