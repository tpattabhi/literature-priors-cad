"""Supporting and sensitivity analyses requested in review (run after analysis.py).

Usage (from this folder): python sensitivity.py  -> ../results/sensitivity.json, ../results/lc_extra.csv

1. Univariable estimates on the training set for every literature-informed coefficient (unpenalized
   logistic regression with only that predictor group), to separate population differences from
   multivariable adjustment.
2. Descriptive counts behind two caveats: exercise angina by chest-pain code, and calcified vessels on
   fluoroscopy among patients without CAD.
3. A penalized frequentist baseline: L2 (ridge) logistic regression on standardized predictors with the
   penalty chosen by inner 5-fold cross-validated log loss, evaluated on the test set and in the same
   5-fold cross-validation as analysis.py.
4. The learning-curve subsets of analysis.py regenerated exactly (same random generator and order). On
   each subset: the M2 posterior of the attenuation factor c, M2's test log-likelihood (to confirm the
   regenerated fits match the saved ones), and the ridge baseline scored like the Bayesian models.
5. Full-data variants of M2: (a) age prior dropped (N(0, 1), like the chest-pain priors), (b) Beta(1, 1)
   prior on c, (c) the fluoroscopy columns removed (for M0b and M2), each compared by PSIS-LOO and test
   log-likelihood.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_src = (Path(__file__).resolve().parent / "analysis.py").read_text()
_ns = {"__file__": str(Path(__file__).resolve().parent / "analysis.py"), "__name__": "analysis_core"}
exec(compile(_src.split("# 1. Full training-set fits")[0], "analysis.py", "exec"), _ns)
ac = SimpleNamespace(**_ns)

import arviz as az
import pymc as pm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

out = {}
F = ac.FEATURES
Xtr, ytr, Xte, yte = ac.X_train, ac.y_train, ac.X_test, ac.y_test
col = {f: i for i, f in enumerate(F)}

# ---------------------------------------------------------------- 1. univariable estimates
groups = [
    ["age (per 10 y)"],
    ["male"],
    ["typical angina", "atypical angina", "non-anginal pain"],
    ["ST depression (per mm)"],
    ["any calcified vessel", "extra calcified vessels"],
    ["fixed thallium defect", "reversible thallium defect"],
]
uni = {}
for g in groups:
    idx = [col[f] for f in g]
    m = LogisticRegression(C=np.inf, max_iter=10_000).fit(Xtr[:, idx], ytr)
    uni.update(dict(zip(g, m.coef_[0].tolist())))
out["univariable_train"] = uni

# ---------------------------------------------------------------- 2. descriptive counts (training set)
tr = ac.train_raw.fillna(ac.IMPUTE)
out["exang_by_cp_train"] = {
    int(k): [float(v), int(n)] for k, (v, n) in tr.groupby("cp").exang.agg(["mean", "size"]).iterrows()
}
no_cad = tr[tr.disease == 0]
out["ca_ge1_among_no_cad_train"] = [float((no_cad.ca >= 1).mean()), int((no_cad.ca >= 1).sum()), int(len(no_cad))]
out["diamond_fp_ca_ge1"] = 0.023 + 0.016 + 0.000  # Table II of Diamond et al. (1980), P(1, 2, 3 vessels | no CAD)


# ---------------------------------------------------------------- 3. ridge baseline
def ridge():
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 25),
            cv=StratifiedKFold(5, shuffle=True, random_state=ac.SEED),
            scoring="neg_log_loss",
            max_iter=10_000,
        ),
    )


p_te = ridge().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
ll_te = ac.pointwise_ll(yte, p_te)
saved_test = pd.read_csv(ac.RES / "test_metrics.csv", index_col=0)
out["ridge_test"] = {
    "loglik": float(ll_te.sum()),
    "accuracy": float(accuracy_score(yte, p_te >= 0.5)),
    "auc": float(roc_auc_score(yte, p_te)),
    "diff_vs_M2": float(ll_te.sum() - saved_test.loc["M2", "loglik"]),
}
cvp = np.zeros(len(ac.raw))
for f, (a, b) in enumerate(
    StratifiedKFold(ac.CV_SPLITS, shuffle=True, random_state=ac.SEED).split(ac.raw, ac.raw.disease)
):
    imp = {c: ac.raw.iloc[a][c].mode()[0] for c in ["ca", "thal"]}
    A, B = ac.encode(ac.raw.iloc[a], imp), ac.encode(ac.raw.iloc[b], imp)
    ctr = A.mean()
    cvp[b] = ridge().fit((A - ctr).values, ac.raw.disease.values[a]).predict_proba((B - ctr).values)[:, 1]
y_all = ac.raw.disease.values
cvsaved = pd.read_csv(ac.RES / "cv_predictions.csv", index_col=0)
ll_cv = ac.pointwise_ll(y_all, cvp)
d = ll_cv - ac.pointwise_ll(y_all, cvsaved["M2"].values)
slope, citl = ac.calibration(y_all, cvp)
out["ridge_cv"] = {
    "loglik": float(ll_cv.sum()),
    "diff_vs_M2": float(d.sum()),
    "diff_se": float(np.sqrt(len(d)) * d.std(ddof=1)),
    "auc": float(roc_auc_score(y_all, cvp)),
    "accuracy": float(accuracy_score(y_all, cvp >= 0.5)),
    "cal_slope": float(slope),
}

# ---------------------------------------------------------------- 4. learning-curve subsets regenerated
saved = pd.read_csv(ac.RES / "learning_curve_points.csv")
rng = np.random.default_rng(ac.SEED + 1)
rows, pts = [], []
for n_sub in ac.LC_SIZES:
    for r in range(ac.LC_REPS):
        idx = rng.choice(len(ytr), n_sub, replace=False)
        rest = np.setdiff1d(np.arange(len(ytr)), idx)
        m = ac.Model(ac.SPECS["M2"]).fit(
            Xtr[idx], ytr[idx], seed=ac.SEED + 100 * n_sub + r, draws=ac.LC_DRAWS, tune=ac.LC_TUNE
        )
        c = m.draws("c")
        m2_test = ac.pointwise_ll(yte, m.predict_proba(Xte)).mean()
        saved_m2 = saved.query("n == @n_sub and rep == @r and model == 'M2' and scored_on == 'test'").loglik.mean()
        try:
            rg = ridge().fit(Xtr[idx], ytr[idx])
            ll_t = ac.pointwise_ll(yte, rg.predict_proba(Xte)[:, 1])
            ll_o = ac.pointwise_ll(ytr[rest], rg.predict_proba(Xtr[rest])[:, 1])
            rg_test, rg_other = ll_t.mean(), ll_o.mean()
            pts.append(
                pd.DataFrame(
                    {"n": n_sub, "rep": r, "scored_on": "test", "patient": np.arange(len(yte)), "ridge_ll": ll_t}
                )
            )
            pts.append(
                pd.DataFrame({"n": n_sub, "rep": r, "scored_on": "other_train", "patient": rest, "ridge_ll": ll_o})
            )
        except ValueError:  # a fold without both classes
            rg_test = rg_other = np.nan
        saved_m2_other = saved.query(
            "n == @n_sub and rep == @r and model == 'M2' and scored_on == 'other_train'"
        ).loglik.mean()
        rows.append(
            {
                "n": n_sub,
                "rep": r,
                "c_mean": c.mean(),
                "c_lo": np.quantile(c, 0.025),
                "c_hi": np.quantile(c, 0.975),
                "M2_test_ll_regen": m2_test,
                "M2_test_ll_saved": saved_m2,
                "ridge_test_ll": rg_test,
                "ridge_other_ll": rg_other,
                "M2_other_ll_saved": saved_m2_other,
            }
        )
    print("subset size", n_sub, "done", flush=True)
lcx = pd.DataFrame(rows)
lcx.to_csv(ac.RES / "lc_extra.csv", index=False)
# patient-level differences M2 - ridge, averaged over the subsets in which each patient was scored
rp = pd.concat(pts, ignore_index=True)
m2p = saved.query("model == 'M2'").rename(columns={"loglik": "m2_ll"})[["n", "rep", "scored_on", "patient", "m2_ll"]]
rp = rp.merge(m2p, on=["n", "rep", "scored_on", "patient"], how="left")
assert rp.m2_ll.notna().all()
rp["d"] = rp.m2_ll - rp.ridge_ll


def se_patients(n_sub, where):
    dp = rp.query("n == @n_sub and scored_on == @where").groupby("patient").d.mean()
    return float(dp.std(ddof=1) / np.sqrt(len(dp)))


summ = {}
for n_sub, g in lcx.groupby("n"):
    dt = g.M2_test_ll_saved - g.ridge_test_ll
    do = g.M2_other_ll_saved - g.ridge_other_ll
    summ[int(n_sub)] = {
        "c_mean": float(g.c_mean.mean()),
        "c_mean_min": float(g.c_mean.min()),
        "c_mean_max": float(g.c_mean.max()),
        "regen_max_abs_diff": float((g.M2_test_ll_regen - g.M2_test_ll_saved).abs().max()),
        "ridge_test_logloss": float(-g.ridge_test_ll.mean()),
        "M2_gain_vs_ridge_test": float(dt.mean()),
        "se": float(dt.std(ddof=1) / np.sqrt(dt.notna().sum())),
        "M2_better_test": int((dt > 0).sum()),
        "M2_gain_vs_ridge_other": float(do.mean()),
        "se_other": float(do.std(ddof=1) / np.sqrt(do.notna().sum())),
        "ridge_failed": int(g.ridge_test_ll.isna().sum()),
    }
    se_sub_t, se_sub_o = summ[int(n_sub)]["se"], summ[int(n_sub)]["se_other"]
    summ[int(n_sub)]["se_combined"] = float(np.hypot(se_sub_t, se_patients(n_sub, "test")))
    summ[int(n_sub)]["se_other_combined"] = float(np.hypot(se_sub_o, se_patients(n_sub, "other_train")))
out["lc_extra"] = summ


# ---------------------------------------------------------------- 5. full-data variants of M2
def fit_eval(spec, keep=None, beta_c=(2, 2)):
    keep = np.arange(len(F)) if keep is None else keep
    s = replace(spec, beta_mu=spec.beta_mu[keep], beta_sd=spec.beta_sd[keep], discount=spec.discount[keep])
    with pm.Model():
        alpha = pm.Normal("alpha", s.alpha_mu, s.alpha_sd)
        if s.discount.any():
            c = pm.Beta("c", *beta_c)
            mu = pm.math.where(s.discount, c * s.beta_mu, s.beta_mu)
        else:
            mu = s.beta_mu
        beta = pm.Normal("beta", mu, s.beta_sd, shape=len(keep))
        pm.Bernoulli("y", logit_p=alpha + pm.math.dot(Xtr[:, keep], beta), observed=ytr)
        idata = pm.sample(
            ac.DRAWS,
            tune=ac.TUNE,
            chains=ac.CHAINS,
            target_accept=0.9,
            random_seed=ac.SEED,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
            mp_ctx="fork",
        )
    post = idata.posterior
    a = post["alpha"].values.reshape(-1)
    b = post["beta"].values.reshape(-1, len(keep))
    p = ac.expit(a[:, None] + b @ Xte[:, keep].T).mean(0)
    loo = az.loo(idata)
    res = {
        "elpd_loo": float(loo.elpd),
        "loo_se": float(loo.se),
        "test_loglik": float(ac.pointwise_ll(yte, p).sum()),
        "_test_p": p,
        "_elpd_i": np.asarray(loo.elpd_i).ravel(),
        "max_rhat": float(az.summary(idata)["r_hat"].max()),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
    }
    if s.discount.any():
        cc = post["c"].values.reshape(-1)
        res.update(c_mean=float(cc.mean()), c_lo=float(np.quantile(cc, 0.025)), c_hi=float(np.quantile(cc, 0.975)))
    return res


M2 = ac.SPECS["M2"]
age = col["age (per 10 y)"]
no_age = replace(
    M2,
    beta_mu=np.where(np.arange(len(F)) == age, 0.0, M2.beta_mu),
    beta_sd=np.where(np.arange(len(F)) == age, 1.0, M2.beta_sd),
    discount=M2.discount & (np.arange(len(F)) != age),
)
keep_no_ca = np.array([i for i, f in enumerate(F) if f not in ("any calcified vessel", "extra calcified vessels")])
variants = {
    "M2_base": fit_eval(M2),
    "M2_age_prior_dropped": fit_eval(no_age),
    "M2_beta11": fit_eval(M2, beta_c=(1, 1)),
    "M2_no_ca": fit_eval(M2, keep_no_ca),
    "M0b_no_ca": fit_eval(ac.SPECS["M0b"], keep_no_ca),
    "M0_no_ca": fit_eval(ac.SPECS["M0"], keep_no_ca),
}
base = variants["M2_base"]
for k, v in variants.items():  # paired differences from the refitted base M2 (same run)
    if k != "M2_base" and not k.endswith("no_ca"):
        dl = v["_elpd_i"] - base["_elpd_i"]
        dt = ac.pointwise_ll(yte, v["_test_p"]) - ac.pointwise_ll(yte, base["_test_p"])
        v.update(
            d_elpd=float(dl.sum()),
            d_elpd_se=float(np.sqrt(len(dl)) * dl.std(ddof=1)),
            d_test=float(dt.sum()),
            d_test_se=float(np.sqrt(len(dt)) * dt.std(ddof=1)),
        )
nb, n2 = variants["M0b_no_ca"], variants["M2_no_ca"]
dl = n2["_elpd_i"] - nb["_elpd_i"]
out["no_ca_M2_minus_M0b"] = {"d_elpd": float(dl.sum()), "d_elpd_se": float(np.sqrt(len(dl)) * dl.std(ddof=1))}
dt = ll_te - ac.pointwise_ll(yte, base["_test_p"])
out["ridge_test"].update(
    diff_vs_M2_refit=float(dt.sum()),
    diff_se=float(np.sqrt(len(dt)) * dt.std(ddof=1)),
    M2_refit_test_loglik=base["test_loglik"],
)
for v in variants.values():
    v.pop("_test_p")
    v.pop("_elpd_i")
out["variants"] = variants
(ac.RES / "sensitivity.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
