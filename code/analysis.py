"""Reproduce every number in the EMBC 2027 paper on literature-informed priors for CAD prediction.

Usage (from this folder):
    python analysis.py            # full run (about 40 min on a 2020 Intel laptop CPU)
    SMOKE=1 python analysis.py    # tiny settings, to check that the script runs

Data: ../data/processed.cleveland.data (UCI Heart Disease, Cleveland Clinic file; Janosi et al., 1989,
CC BY 4.0). Outputs go to ../results/ (Fig. 1 is drawn by make_figure.py).

The model code follows the author's earlier course analysis (Statistics for AI, Assignment 3), with
the same split, encoding, priors and seeds, so the full-data results reproduce that analysis up to
Monte Carlo error. New here: the learning-curve experiment uses more subsets (25 per size), four
subset sizes and also includes the face-value literature model M1, and calibration is reported by sex.

Pinned versions: see requirements.txt (PyMC 6.3.2, ArviZ 1.3.0, scikit-learn 1.9.1, Python 3.13).
"""

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import optimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "processed.cleveland.data"
RES = HERE.parent / "results"
RES.mkdir(exist_ok=True)

SMOKE = os.environ.get("SMOKE") == "1"
SEED = 2026
DRAWS, TUNE, CHAINS = (200, 200, 2) if SMOKE else (1000, 1000, 4)
LC_DRAWS, LC_TUNE = (200, 200) if SMOKE else (500, 500)
LC_SIZES = [25, 50, 100, 200]
LC_REPS = 2 if SMOKE else 25
LC_MODELS = ["M0", "M0b", "M0c", "M1", "M2"]
CV_SPLITS = 5
START = time.time()

# ----------------------------------------------------------------------------------------------
# Data and preprocessing (identical to the course analysis)
# ----------------------------------------------------------------------------------------------
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
# The UCI processed files store chest-pain type in the reverse of the documented order: their code 4 is
# typical angina and code 1 is no pain. Detrano et al., Circulation 1986;73:970-977 (the same 303 patients)
# reports Table 1 and Table 8 that reproduce exactly only with cp = 5 - code. Set UCI_CP_AS_DOCUMENTED=1 to
# use the documented order instead (the analysis before this correction).
CP_AS_DOCUMENTED = os.environ.get("UCI_CP_AS_DOCUMENTED") == "1"


def read_uci(path):
    """Read a UCI processed heart-disease file, with chest-pain type in clinical order (1 = typical angina)."""
    d = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    if not CP_AS_DOCUMENTED:
        d["cp"] = 5 - d["cp"]
    return d


raw = read_uci(DATA)
raw["disease"] = (raw["num"] > 0).astype(int)

train_raw, test_raw = train_test_split(raw, test_size=0.2, stratify=raw["disease"], random_state=SEED)
IMPUTE = {c: train_raw[c].mode()[0] for c in ["ca", "thal"]}


def encode(df, impute):
    d = df.fillna(impute)
    return pd.DataFrame(
        {
            "age (per 10 y)": d.age / 10,
            "male": d.sex,
            "typical angina": (d.cp == 1),
            "atypical angina": (d.cp == 2),
            "non-anginal pain": (d.cp == 3),
            "resting BP (per 10 mm Hg)": d.trestbps / 10,
            "cholesterol (per 50 mg/dL)": d.chol / 50,
            "fasting sugar > 120": d.fbs,
            "abnormal resting ECG": (d.restecg > 0),
            "max heart rate (per 10 bpm)": d.thalach / 10,
            "exercise angina": d.exang,
            "ST depression (per mm)": d.oldpeak,
            "flat ST slope": (d.slope == 2),
            "down sloping ST slope": (d.slope == 3),
            "any calcified vessel": (d.ca >= 1),
            "extra calcified vessels": (d.ca - 1).clip(lower=0),
            "fixed thallium defect": (d.thal == 6),
            "reversible thallium defect": (d.thal == 7),
        },
        index=df.index,
    ).astype(float)


FEATURES = list(encode(train_raw, IMPUTE).columns)
CENTER = encode(train_raw, IMPUTE).mean()
X_train = (encode(train_raw, IMPUTE) - CENTER).values
X_test = (encode(test_raw, IMPUTE) - CENTER).values
y_train = train_raw["disease"].values
y_test = test_raw["disease"].values

# ----------------------------------------------------------------------------------------------
# Priors from Diamond et al. (1980), Tables I and II, and Patel et al. (2010)
# ----------------------------------------------------------------------------------------------
DIAMOND_TABLE_I = pd.DataFrame(
    [
        ("M", 65, 0.943, 0.671, 0.281, 0.123),
        ("M", 55, 0.920, 0.589, 0.215, 0.097),
        ("M", 45, 0.873, 0.461, 0.141, 0.055),
        ("M", 35, 0.697, 0.218, 0.052, 0.019),
        ("F", 65, 0.906, 0.544, 0.186, 0.075),
        ("F", 55, 0.794, 0.324, 0.084, 0.032),
        ("F", 45, 0.552, 0.133, 0.028, 0.010),
        ("F", 35, 0.258, 0.042, 0.008, 0.003),
    ],
    columns=["sex", "age_mid", "TA", "AA", "NA", "AS"],
)
cells = DIAMOND_TABLE_I.melt(id_vars=["sex", "age_mid"], var_name="symptom", value_name="p")
D = np.column_stack(
    [
        np.ones(len(cells)),
        (cells.age_mid - 55) / 10,
        cells.sex.eq("M"),
        cells.symptom.eq("TA"),
        cells.symptom.eq("AA"),
        cells.symptom.eq("NA"),
    ]
).astype(float)
coef_table_I, *_ = np.linalg.lstsq(D, logit(cells.p.values), rcond=None)
resid = logit(cells.p.values) - D @ coef_table_I

st_mid = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75])
st_lr = np.array([0.143, 0.208, 0.233, 0.088, 0.133, 0.195]) / np.array([0.625, 0.227, 0.110, 0.021, 0.012, 0.005])
calc_cad = np.array([0.420, 0.235, 0.201, 0.145])
calc_no = np.array([0.961, 0.023, 0.016, 0.000])
thal_lr = np.array([0.147, 0.146, 0.707]) / np.array([0.836, 0.103, 0.061])
calc_lr = calc_cad[:3] / calc_no[:3]

LIT = {
    "age (per 10 y)": coef_table_I[1],
    "male": coef_table_I[2],
    "typical angina": coef_table_I[3],
    "atypical angina": coef_table_I[4],
    "non-anginal pain": coef_table_I[5],
    "ST depression (per mm)": np.polyfit(st_mid, np.log(st_lr), 1)[0],
    "any calcified vessel": np.log(calc_lr[1] / calc_lr[0]),
    "extra calcified vessels": np.log(calc_lr[2] / calc_lr[1]),
    "fixed thallium defect": np.log(thal_lr[1] / thal_lr[0]),
    "reversible thallium defect": np.log(thal_lr[2] / thal_lr[0]),
}
PREVALENCE_PRIOR = 0.376
CHEST_PAIN = ["typical angina", "atypical angina", "non-anginal pain"]


@dataclass
class PriorSpec:
    name: str
    alpha_mu: float
    alpha_sd: float
    beta_mu: np.ndarray
    beta_sd: np.ndarray
    discount: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES), bool))


lit_mean = np.array([LIT.get(f, 0.0) for f in FEATURES])
is_lit = np.array([f in LIT for f in FEATURES])
is_cp = np.isin(FEATURES, CHEST_PAIN)
refined_mean = np.where(is_cp, 0.0, lit_mean)
refined_sd = np.where(is_lit & ~is_cp, 0.5, 1.0)
discount = is_lit & ~is_cp
a0 = logit(PREVALENCE_PRIOR)
K = len(FEATURES)
SPECS = {
    "M0": PriorSpec("M0", 0.0, 2.5, np.zeros(K), np.full(K, 2.5)),
    "M0b": PriorSpec("M0b", a0, 0.5, np.zeros(K), np.ones(K)),
    "M0c": PriorSpec("M0c", a0, 0.5, np.zeros(K), refined_sd),
    "M1": PriorSpec("M1", a0, 0.5, lit_mean, np.where(is_lit, 0.5, 1.0)),
    "M2": PriorSpec("M2", a0, 0.5, refined_mean, refined_sd, discount),
}
ORDER = ["M0", "M0b", "M0c", "M1", "M2"]


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class Model:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, X, y, seed=SEED, draws=DRAWS, tune=TUNE):
        s = self.spec
        with pm.Model(coords={"feature": FEATURES}):
            alpha = pm.Normal("alpha", s.alpha_mu, s.alpha_sd)
            if s.discount.any():
                c = pm.Beta("c", 2, 2)
                mu = pm.math.where(s.discount, c * s.beta_mu, s.beta_mu)
            else:
                mu = s.beta_mu
            beta = pm.Normal("beta", mu, s.beta_sd, dims="feature")
            pm.Bernoulli("y", logit_p=alpha + pm.math.dot(X, beta), observed=y)
            self.idata = pm.sample(
                draws,
                tune=tune,
                chains=CHAINS,
                target_accept=0.9,
                random_seed=seed,
                progressbar=False,
                idata_kwargs={"log_likelihood": True},
                mp_ctx="fork",
            )
        return self

    def draws(self, name):
        v = self.idata.posterior[name].values
        return v.reshape(-1, *v.shape[2:])

    def predict_proba(self, X):
        return expit(self.draws("alpha")[:, None] + self.draws("beta") @ X.T).mean(axis=0)

    def diagnostics(self):
        names = [v for v in ["alpha", "beta", "c"] if v in self.idata.posterior]
        s = az.summary(self.idata, var_names=names)
        return {
            "max_rhat": float(s["r_hat"].max()),
            "min_bulk_ess": float(s["ess_bulk"].min()),
            "min_tail_ess": float(s["ess_tail"].min()),
            "divergences": int(self.idata.sample_stats["diverging"].values.sum()),
        }


def pointwise_ll(y, p):
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def calibration(y, p):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    slope = LogisticRegression(C=np.inf).fit(lp[:, None], y).coef_[0, 0]
    citl = optimize.minimize_scalar(lambda a: -np.sum(pointwise_ll(y, expit(a + lp)))).x
    return slope, citl


key = {
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_test_cad": int(y_test.sum()),
    "prev_all": float(raw.disease.mean()),
    "n_all": int(len(raw)),
    "n_female": int((raw.sex == 0).sum()),
    "n_male": int((raw.sex == 1).sum()),
    "tableI_resid_sd": float(np.std(resid)),
    "tableI_max_abs_resid": float(np.abs(resid).max()),
    "lit": {k: float(v) for k, v in LIT.items()},
    "alpha_prior_mean": float(a0),
    "missing_ca": int(raw.ca.isna().sum()),
    "missing_thal": int(raw.thal.isna().sum()),
}

# chest-pain disease rates in the training set (descriptive)
tf = train_raw.fillna(IMPUTE)
key["train_cad_rate_by_cp"] = {
    int(k): [float(v), int(n)] for k, (v, n) in tf.groupby("cp").disease.agg(["mean", "size"]).iterrows()
}

# ----------------------------------------------------------------------------------------------
# 1. Full training-set fits, convergence, prior-data conflict, LOO
# ----------------------------------------------------------------------------------------------
models, diag, loo = {}, {}, {}
for k in ORDER:
    models[k] = Model(SPECS[k]).fit(X_train, y_train)
    diag[k] = models[k].diagnostics()
    loo[k] = az.loo(models[k].idata)
    print(k, diag[k], f"elpd_loo {loo[k].elpd:.1f}", flush=True)
pd.DataFrame(diag).T.to_csv(RES / "diagnostics.csv")

post_mean = {k: models[k].draws("beta").mean(0) for k in ORDER}
conflict = pd.DataFrame(
    {
        "literature": lit_mean,
        "M1_post": post_mean["M1"],
        "conflict_z": (post_mean["M1"] - SPECS["M1"].beta_mu) / SPECS["M1"].beta_sd,
        "M0_post": post_mean["M0"],
        "M0b_post": post_mean["M0b"],
        "M2_post": post_mean["M2"],
        "M2_over_lit": post_mean["M2"] / np.where(lit_mean == 0, np.nan, lit_mean),
    },
    index=FEATURES,
).loc[is_lit]
conflict.to_csv(RES / "prior_conflict.csv")
c_draws = models["M2"].draws("c")
key["c_mean"], key["c_lo"], key["c_hi"] = float(c_draws.mean()), *map(float, np.quantile(c_draws, [0.025, 0.975]))

elpd_i = {k: np.asarray(loo[k].elpd_i).ravel() for k in ORDER}
best = max(ORDER, key=lambda k: loo[k].elpd)
rows = []
for k in ORDER:
    d = elpd_i[k] - elpd_i[best]
    rows.append(
        {
            "model": k,
            "elpd_loo": loo[k].elpd,
            "se": loo[k].se,
            "p_loo": loo[k].p,
            "diff": d.sum(),
            "diff_se": np.sqrt(len(d)) * d.std(ddof=1) if k != best else 0.0,
            "max_k": float(np.max(np.asarray(loo[k].pareto_k))),
        }
    )
pd.DataFrame(rows).set_index("model").to_csv(RES / "loo.csv")
key["loo_best"] = best

# ----------------------------------------------------------------------------------------------
# 2. Held-out test set (n = 61): never used for any design decision
# ----------------------------------------------------------------------------------------------
test_prob = {k: models[k].predict_proba(X_test) for k in ORDER}
test_prob["MLE"] = LogisticRegression(C=np.inf, max_iter=10_000).fit(X_train, y_train).predict_proba(X_test)[:, 1]
ll_ref = pointwise_ll(y_test, test_prob["M2"])
rows = []
for k, p in test_prob.items():
    ll = pointwise_ll(y_test, p)
    d = ll - ll_ref
    rows.append(
        {
            "model": k,
            "accuracy": accuracy_score(y_test, p >= 0.5),
            "auc": roc_auc_score(y_test, p),
            "loglik": ll.sum(),
            "diff_vs_M2": d.sum(),
            "diff_se": np.sqrt(len(d)) * d.std(ddof=1),
            "brier": brier_score_loss(y_test, p),
        }
    )
pd.DataFrame(rows).set_index("model").to_csv(RES / "test_metrics.csv")

# ----------------------------------------------------------------------------------------------
# 3. Five-fold cross-validation on all 303 patients (sensitivity analysis), calibration overall and by sex
# ----------------------------------------------------------------------------------------------
cv = {k: np.zeros(len(raw)) for k in ORDER + ["MLE"]}
for f, (tr, te) in enumerate(StratifiedKFold(CV_SPLITS, shuffle=True, random_state=SEED).split(raw, raw.disease)):
    imp = {c: raw.iloc[tr][c].mode()[0] for c in ["ca", "thal"]}
    Xtr_raw, Xte_raw = encode(raw.iloc[tr], imp), encode(raw.iloc[te], imp)
    ctr = Xtr_raw.mean()
    Xtr, Xte, ytr = (Xtr_raw - ctr).values, (Xte_raw - ctr).values, raw.disease.values[tr]
    for k in ORDER:
        cv[k][te] = Model(SPECS[k]).fit(Xtr, ytr, seed=SEED + f).predict_proba(Xte)
    cv["MLE"][te] = LogisticRegression(C=np.inf, max_iter=10_000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    print("cv fold", f + 1, "done", flush=True)
y_all = raw.disease.values
male = raw.sex.values == 1
pd.DataFrame({**cv, "disease": y_all, "male": male.astype(int)}, index=raw.index).to_csv(RES / "cv_predictions.csv")
ll_ref = pointwise_ll(y_all, cv["M2"])
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(y_all), len(y_all)) for _ in range(200 if SMOKE else 2000)]
rows = []
for k, p in cv.items():
    ll = pointwise_ll(y_all, p)
    d = ll - ll_ref
    slope, citl = calibration(y_all, p)
    bs = np.array([calibration(y_all[i], p[i])[0] for i in boot_idx])
    row = {
        "model": k,
        "accuracy": accuracy_score(y_all, p >= 0.5),
        "auc": roc_auc_score(y_all, p),
        "loglik": ll.sum(),
        "diff_vs_M2": d.sum(),
        "diff_se": np.sqrt(len(d)) * d.std(ddof=1),
        "brier": brier_score_loss(y_all, p),
        "cal_slope": slope,
        "slope_lo": np.quantile(bs, 0.025),
        "slope_hi": np.quantile(bs, 0.975),
        "citl": citl,
    }
    for sex_name, mask in [("female", ~male), ("male", male)]:
        row[f"obs_{sex_name}"] = y_all[mask].mean()
        row[f"pred_{sex_name}"] = p[mask].mean()
        row[f"citl_{sex_name}"] = calibration(y_all[mask], p[mask])[1]
        row[f"auc_{sex_name}"] = roc_auc_score(y_all[mask], p[mask])
    rows.append(row)
pd.DataFrame(rows).set_index("model").to_csv(RES / "cv_metrics.csv")

# ----------------------------------------------------------------------------------------------
# 4. Learning curve: fit every model on the same random subsets of the training set,
#    score on the 61 held-out test patients (clean) and on the training patients outside the subset
# ----------------------------------------------------------------------------------------------
rng = np.random.default_rng(SEED + 1)
lc_rows, lc_diag = [], []
for n_sub in LC_SIZES:
    for r in range(LC_REPS):
        idx = rng.choice(len(y_train), n_sub, replace=False)
        rest = np.setdiff1d(np.arange(len(y_train)), idx)
        for k in LC_MODELS:
            m = Model(SPECS[k]).fit(
                X_train[idx], y_train[idx], seed=SEED + 100 * n_sub + r, draws=LC_DRAWS, tune=LC_TUNE
            )
            lc_diag.append({"n": n_sub, "rep": r, "model": k, **m.diagnostics()})
            for where, Xe, ye, ids in [
                ("test", X_test, y_test, np.arange(len(y_test))),
                ("other_train", X_train[rest], y_train[rest], rest),
            ]:
                ll = pointwise_ll(ye, m.predict_proba(Xe))
                lc_rows.append(
                    pd.DataFrame({"n": n_sub, "rep": r, "model": k, "scored_on": where, "patient": ids, "loglik": ll})
                )
        print(f"subset size {n_sub} rep {r} done ({(time.time() - START) / 60:.1f} min)", flush=True)
    pd.concat(lc_rows).to_csv(RES / "learning_curve_points.csv", index=False)
    pd.DataFrame(lc_diag).to_csv(RES / "learning_curve_diagnostics.csv", index=False)
lc = pd.concat(lc_rows, ignore_index=True)

# Summaries. Per-subset mean log-likelihood gives one paired difference per subset; its SE across
# subsets is the uncertainty used for the test set (fixed 61 patients scored by every fit).
summ = []
for (n_sub, where), g in lc.groupby(["n", "scored_on"]):
    per_rep = g.groupby(["rep", "model"]).loglik.mean().unstack()
    for k in LC_MODELS:
        dd = per_rep["M2"] - per_rep[k]
        # patient-level SE: average each patient over the subsets in which they were scored
        pa = g.groupby(["model", "patient"]).loglik.mean().unstack(0)
        dp = (pa["M2"] - pa[k]).dropna()
        summ.append(
            {
                "n": n_sub,
                "scored_on": where,
                "model": k,
                "mean_log_loss": -per_rep[k].mean(),
                "M2_gain": dd.mean(),
                "M2_gain_se_subsets": dd.std(ddof=1) / np.sqrt(len(dd)),
                "M2_gain_se_patients": dp.std(ddof=1) / np.sqrt(len(dp)),
                "M2_better_subsets": int((dd > 0).sum()),
                "reps": len(dd),
            }
        )
curve = pd.DataFrame(summ)
curve["M2_gain_se"] = np.hypot(curve["M2_gain_se_subsets"], curve["M2_gain_se_patients"])
curve.to_csv(RES / "learning_curve.csv", index=False)
d = pd.DataFrame(lc_diag)
key["lc_max_rhat"] = float(d.max_rhat.max())
key["lc_total_divergences"] = int(d.divergences.sum())
key["lc_fits_with_divergences"] = int((d.divergences > 0).sum())
key["lc_min_bulk_ess"] = float(d.min_bulk_ess.min())
key["lc_n_fits"] = int(len(d))

# Fig. 1 of the paper is drawn from results/learning_curve.csv by make_figure.py.

key["runtime_min"] = (time.time() - START) / 60
(RES / "key_numbers.json").write_text(json.dumps(key, indent=2))
print(f"done in {key['runtime_min']:.1f} min")
