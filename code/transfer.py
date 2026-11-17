"""Analyses added after the prior-art review (run after analysis.py and sensitivity.py).

Usage (from this folder; the parts are independent and can run in parallel):
    python transfer.py auto     # 1. pre-specified refinement rule inside every Cleveland learning-curve subset
    python transfer.py sites    # 2. new-clinic experiment at the Hungarian and Long Beach VA sites
    python transfer.py dca      # 3. literature used directly as a calculator, and decision-curve analysis
    SMOKE=1 python transfer.py auto   # tiny settings, to check that the script runs

1. M2 was designed by inspecting M1 on all 242 training patients, so the learning curve of analysis.py
   inherits that design. Here a fixed rule replaces the design step and is applied inside each subset,
   using only the patients in that subset:
       a literature prior is dropped (replaced by N(0, 1)) if the M0b posterior mean has the opposite sign
       to the literature mean, or if the M1 conflict score |z| exceeds 2; the means of the remaining
       literature priors are scaled by a shared factor kappa ~ Beta(2, 2), as in M2.
   The subsets, seeds and sampler settings are those of analysis.py, so M0b and M1 are refitted exactly.
2. The other UCI sites lack fluoroscopy and thallium for nearly all patients, so a reduced model with
   age, sex, chest-pain type, exercise angina and ST depression is used. Everything is fixed in advance
   from Cleveland and the literature: the prior specifications (M2 drops the chest-pain priors and
   discounts the age, sex and ST-depression priors), the automatic rule of part 1, and the intercept prior.
   Each model is fitted on random subsets of n local patients and scored on the site's other patients.
   Two fixed references need no local data: the literature used directly as a calculator, and a model
   fitted by maximum likelihood on all 303 Cleveland patients and transported unchanged.
3. The literature calculator (Diamond et al. 1980 Table I pretest probability, multiplied by the ST-depression
   likelihood ratio of Table II) on the 61 Cleveland test patients, and net benefit of the cross-validated
   predictions of analysis.py at risk thresholds from 5% to 50%.
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_src = (HERE / "analysis.py").read_text()
_ns = {"__file__": str(HERE / "analysis.py"), "__name__": "analysis_core"}
exec(compile(_src.split("# 1. Full training-set fits")[0], "analysis.py", "exec"), _ns)
ac = SimpleNamespace(**_ns)

import arviz as az
import pymc as pm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RES = ac.RES
START = time.time()
A0, ASD = ac.a0, 0.5


def fit(X, y, mu, sd, disc, a_mu, a_sd, seed, draws, tune):
    """Logistic regression with independent normal priors; literature means flagged in `disc` are scaled
    by kappa ~ Beta(2, 2). Same sampler settings as analysis.Model."""
    with pm.Model():
        alpha = pm.Normal("alpha", a_mu, a_sd)
        if disc.any():
            kappa = pm.Beta("c", 2, 2)
            m = pm.math.where(disc, kappa * mu, mu)
        else:
            m = mu
        beta = pm.Normal("beta", m, sd, shape=X.shape[1])
        pm.Bernoulli("y", logit_p=alpha + pm.math.dot(X, beta), observed=y)
        idata = pm.sample(
            draws,
            tune=tune,
            chains=ac.CHAINS,
            target_accept=0.9,
            random_seed=seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
            mp_ctx="fork",
        )
    return idata


def post(idata, name):
    v = idata.posterior[name].values
    return v.reshape(-1, *v.shape[2:])


def proba(idata, X):
    return expit(post(idata, "alpha")[:, None] + post(idata, "beta") @ X.T).mean(axis=0)


def diagnostics(idata):
    names = [v for v in ["alpha", "beta", "c"] if v in idata.posterior]
    s = az.summary(idata, var_names=names)
    return {
        "max_rhat": float(s["r_hat"].max()),
        "min_bulk_ess": float(s["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
    }


def auto_rule(lit_mean, is_lit, m0b_post, m1_post):
    """Indices of literature priors kept by the pre-specified rule (sign agreement and |z| <= 2)."""
    z = (m1_post - lit_mean) / 0.5
    return is_lit & (np.sign(m0b_post) == np.sign(lit_mean)) & (np.abs(z) <= 2)


def ridge(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-3, 3, 25),
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            scoring="neg_log_loss",
            max_iter=10_000,
        ),
    )


def summarize(points, ref, others, by):
    """Mean log loss per model, and the gain of `ref` over each other model, with the SE used in the
    paper: root sum of squares of the SE across subsets and the SE across patients."""
    rows = []
    for keys, g in points.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        per_rep = g.groupby(["rep", "model"]).loglik.mean().unstack()
        pa = g.groupby(["model", "patient"]).loglik.mean().unstack(0)
        for k in per_rep.columns:
            row = dict(zip(by, keys), model=k, mean_log_loss=-per_rep[k].mean(), reps=int(per_rep[k].notna().sum()))
            if k != ref and k in others:
                dd = (per_rep[ref] - per_rep[k]).dropna()
                dp = (pa[ref] - pa[k]).dropna()
                se_s, se_p = dd.std(ddof=1) / np.sqrt(len(dd)), dp.std(ddof=1) / np.sqrt(len(dp))
                row.update(
                    {
                        f"{ref}_gain": dd.mean(),
                        f"{ref}_gain_se": float(np.hypot(se_s, se_p)),
                        f"{ref}_better_subsets": int((dd > 0).sum()),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================================
# 1. Automatic refinement inside the Cleveland learning-curve subsets
# ==============================================================================================
def part_auto():
    S = ac.SPECS
    lit, is_lit = ac.lit_mean, ac.is_lit
    Xtr, ytr, Xte, yte = ac.X_train, ac.y_train, ac.X_test, ac.y_test
    out = {}

    def auto_spec(m0b_post, m1_post):
        keep = auto_rule(lit, is_lit, m0b_post, m1_post)
        return keep, np.where(keep, lit, 0.0), np.where(keep, 0.5, 1.0)

    # full training set
    i0 = fit(Xtr, ytr, S["M0b"].beta_mu, S["M0b"].beta_sd, np.zeros(ac.K, bool), A0, ASD, ac.SEED, ac.DRAWS, ac.TUNE)
    i1 = fit(Xtr, ytr, S["M1"].beta_mu, S["M1"].beta_sd, np.zeros(ac.K, bool), A0, ASD, ac.SEED, ac.DRAWS, ac.TUNE)
    keep, mu, sd = auto_spec(post(i0, "beta").mean(0), post(i1, "beta").mean(0))
    ia = fit(Xtr, ytr, mu, sd, keep, A0, ASD, ac.SEED, ac.DRAWS, ac.TUNE)
    loo_a = az.loo(ia)
    saved_loo = pd.read_csv(RES / "loo.csv", index_col=0)
    test_saved = pd.read_csv(RES / "test_metrics.csv", index_col=0)
    ll_a = ac.pointwise_ll(yte, proba(ia, Xte))
    out["full"] = {
        "kept": [f for f, k in zip(ac.FEATURES, keep) if k],
        "dropped": [f for f, k in zip(ac.FEATURES, is_lit & ~keep) if k],
        "c_mean": float(post(ia, "c").mean()) if keep.any() else None,
        "elpd_loo": float(loo_a.elpd),
        "elpd_loo_M2": float(saved_loo.loc["M2", "elpd_loo"]),
        "test_loglik": float(ll_a.sum()),
        "test_loglik_M2": float(test_saved.loc["M2", "loglik"]),
        **diagnostics(ia),
    }
    print("full:", out["full"], flush=True)

    saved = pd.read_csv(RES / "learning_curve_points.csv")
    rng = np.random.default_rng(ac.SEED + 1)  # the generator of analysis.py, same order
    pts, rows, diag = [], [], []
    for n_sub in ac.LC_SIZES:
        for r in range(ac.LC_REPS):
            idx = rng.choice(len(ytr), n_sub, replace=False)
            rest = np.setdiff1d(np.arange(len(ytr)), idx)
            seed = ac.SEED + 100 * n_sub + r
            X, y = Xtr[idx], ytr[idx]
            j0 = fit(
                X, y, S["M0b"].beta_mu, S["M0b"].beta_sd, np.zeros(ac.K, bool), A0, ASD, seed, ac.LC_DRAWS, ac.LC_TUNE
            )
            j1 = fit(
                X, y, S["M1"].beta_mu, S["M1"].beta_sd, np.zeros(ac.K, bool), A0, ASD, seed, ac.LC_DRAWS, ac.LC_TUNE
            )
            keep, mu, sd = auto_spec(post(j0, "beta").mean(0), post(j1, "beta").mean(0))
            ja = fit(X, y, mu, sd, keep, A0, ASD, seed, ac.LC_DRAWS, ac.LC_TUNE)
            diag.append({"n": n_sub, "rep": r, **diagnostics(ja)})
            # check that the refitted M0b reproduces the saved learning-curve fit
            m0b_regen = ac.pointwise_ll(yte, proba(j0, Xte)).mean()
            m0b_saved = saved.query(
                "n == @n_sub and rep == @r and model == 'M0b' and scored_on == 'test'"
            ).loglik.mean()
            rows.append(
                {
                    "n": n_sub,
                    "rep": r,
                    "n_kept": int(keep.sum()),
                    **{f"keep_{f}": bool(k) for f, k in zip(ac.FEATURES, keep) if f in ac.LIT},
                    "c_mean": float(post(ja, "c").mean()) if keep.any() else np.nan,
                    "M0b_regen_minus_saved": m0b_regen - m0b_saved,
                }
            )
            for where, Xe, ye, ids in [
                ("test", Xte, yte, np.arange(len(yte))),
                ("other_train", Xtr[rest], ytr[rest], rest),
            ]:
                pts.append(
                    pd.DataFrame(
                        {
                            "n": n_sub,
                            "rep": r,
                            "model": "M2auto",
                            "scored_on": where,
                            "patient": ids,
                            "loglik": ac.pointwise_ll(ye, proba(ja, Xe)),
                        }
                    )
                )
            print(f"auto n={n_sub} rep={r} kept={int(keep.sum())} ({(time.time() - START) / 60:.1f} min)", flush=True)
    auto_pts = pd.concat(pts, ignore_index=True)
    auto_pts.to_csv(RES / "auto_rule_points.csv", index=False)
    pd.DataFrame(rows).to_csv(RES / "auto_rule_subsets.csv", index=False)
    pd.DataFrame(diag).to_csv(RES / "auto_rule_diagnostics.csv", index=False)
    allp = pd.concat([saved, auto_pts], ignore_index=True)
    summ = summarize(allp, "M2auto", ["M0", "M0b", "M0c", "M1", "M2"], ["n", "scored_on"])
    summ.to_csv(RES / "auto_rule_curve.csv", index=False)
    sub = pd.DataFrame(rows)
    out["subsets"] = {
        int(n): {
            "mean_kept": float(g.n_kept.mean()),
            "c_mean": float(g.c_mean.mean()),
            "keep_rate": {f: float(g[f"keep_{f}"].mean()) for f in ac.LIT},
            "max_abs_M0b_regen_diff": float(g.M0b_regen_minus_saved.abs().max()),
        }
        for n, g in sub.groupby("n")
    }
    d = pd.DataFrame(diag)
    out["diag"] = {
        "max_rhat": float(d.max_rhat.max()),
        "divergences": int(d.divergences.sum()),
        "n_fits": int(len(d)) + 1,
    }
    out["runtime_min"] = (time.time() - START) / 60
    (RES / "auto_rule.json").write_text(json.dumps(out, indent=2))
    print("auto done", out["runtime_min"], flush=True)


# ==============================================================================================
# 2. New-clinic experiment at the other UCI sites
# ==============================================================================================
RED = [
    "age (per 10 y)",
    "male",
    "typical angina",
    "atypical angina",
    "non-anginal pain",
    "exercise angina",
    "ST depression (per mm)",
]
SITES = {"hungarian": "processed.hungarian.data", "va": "processed.va.data"}
SITE_SIZES = [25, 50, 100]


def load_site(fname):
    d = ac.read_uci(ac.HERE.parent / "data" / fname)
    d = d.dropna(subset=["age", "sex", "cp", "exang", "oldpeak"]).reset_index(drop=True)
    d["disease"] = (d["num"] > 0).astype(int)
    return d


def encode_red(d):
    return pd.DataFrame(
        {
            "age (per 10 y)": d.age / 10,
            "male": d.sex,
            "typical angina": d.cp == 1,
            "atypical angina": d.cp == 2,
            "non-anginal pain": d.cp == 3,
            "exercise angina": d.exang,
            "ST depression (per mm)": d.oldpeak,
        }
    ).astype(float)


def literature_calculator(d):
    """Diamond et al. (1980): Table I pretest probability (logistic surface fitted in analysis.py),
    multiplied in odds by the ST-depression likelihood ratio of Table II (log-linear fit)."""
    b = ac.coef_table_I
    st_slope, st_int = np.polyfit(ac.st_mid, np.log(ac.st_lr), 1)
    lp = (
        b[0]
        + b[1] * (d.age - 55) / 10
        + b[2] * d.sex
        + b[3] * (d.cp == 1)
        + b[4] * (d.cp == 2)
        + b[5] * (d.cp == 3)
        + st_int
        + st_slope * d.oldpeak
    )
    return expit(lp.values.astype(float))


def part_sites():
    lit = np.array([ac.LIT.get(f, 0.0) for f in RED])
    is_lit = np.array([f in ac.LIT for f in RED])
    is_cp = np.isin(RED, ac.CHEST_PAIN)
    K = len(RED)
    disc = is_lit & ~is_cp
    specs = {  # (alpha_mu, alpha_sd, beta_mu, beta_sd, discount): fixed from the Cleveland design
        "M0": (0.0, 2.5, np.zeros(K), np.full(K, 2.5), np.zeros(K, bool)),
        "M0b": (A0, ASD, np.zeros(K), np.ones(K), np.zeros(K, bool)),
        "M0c": (A0, ASD, np.zeros(K), np.where(disc, 0.5, 1.0), np.zeros(K, bool)),
        "M1": (A0, ASD, lit, np.where(is_lit, 0.5, 1.0), np.zeros(K, bool)),
        "M2": (A0, ASD, np.where(disc, lit, 0.0), np.where(disc, 0.5, 1.0), disc),
    }
    # Cleveland model transported unchanged (all 303 patients, reduced predictors, maximum likelihood)
    cle = ac.raw.dropna(subset=["exang", "oldpeak"])
    transported = LogisticRegression(C=np.inf, max_iter=10_000).fit(encode_red(cle).values, cle.disease.values)

    out, pts, diag, info = {}, [], [], []
    for site, fname in SITES.items():
        d = load_site(fname)
        Xraw, yall = encode_red(d).values, d.disease.values
        p_lit, p_tr = literature_calculator(d), transported.predict_proba(Xraw)[:, 1]
        out[site] = {
            "n": int(len(d)),
            "prevalence": float(yall.mean()),
            "cad_rate_by_cp": {
                int(k): [float(v), int(n)] for k, (v, n) in d.groupby("cp").disease.agg(["mean", "size"]).iterrows()
            },
            "literature_calculator_logloss_all": float(-ac.pointwise_ll(yall, p_lit).mean()),
            "transported_logloss_all": float(-ac.pointwise_ll(yall, p_tr).mean()),
        }
        rng = np.random.default_rng(ac.SEED + 7)
        for n_sub in SITE_SIZES:
            for r in range(ac.LC_REPS):
                idx = rng.choice(len(yall), n_sub, replace=False)
                rest = np.setdiff1d(np.arange(len(yall)), idx)
                ctr = Xraw[idx].mean(0)
                X, y, Xe, ye = Xraw[idx] - ctr, yall[idx], Xraw[rest] - ctr, yall[rest]
                seed = ac.SEED + 1000 * (1 + list(SITES).index(site)) + 100 * n_sub + r
                probs, fits = {}, {}
                for k, (am, asd, mu, sd, dc) in specs.items():
                    fits[k] = fit(X, y, mu, sd, dc, am, asd, seed, ac.LC_DRAWS, ac.LC_TUNE)
                    probs[k] = proba(fits[k], Xe)
                    diag.append({"site": site, "n": n_sub, "rep": r, "model": k, **diagnostics(fits[k])})
                keep = auto_rule(lit, is_lit, post(fits["M0b"], "beta").mean(0), post(fits["M1"], "beta").mean(0))
                fa = fit(
                    X,
                    y,
                    np.where(keep, lit, 0.0),
                    np.where(keep, 0.5, 1.0),
                    keep,
                    A0,
                    ASD,
                    seed,
                    ac.LC_DRAWS,
                    ac.LC_TUNE,
                )
                probs["M2auto"] = proba(fa, Xe)
                diag.append({"site": site, "n": n_sub, "rep": r, "model": "M2auto", **diagnostics(fa)})
                try:
                    probs["ridge"] = ridge(ac.SEED).fit(X, y).predict_proba(Xe)[:, 1]
                except ValueError:
                    probs["ridge"] = np.full(len(ye), np.nan)
                probs["literature"], probs["transported"] = p_lit[rest], p_tr[rest]
                info.append(
                    {
                        "site": site,
                        "n": n_sub,
                        "rep": r,
                        "cases_in_subset": int(y.sum()),
                        "n_kept_auto": int(keep.sum()),
                        **{f"keep_{f}": bool(k) for f, k in zip(RED, keep) if f in ac.LIT},
                        "c_mean_M2": float(post(fits["M2"], "c").mean()),
                        **{
                            f"auc_{k}": (
                                roc_auc_score(ye, p) if len(set(ye)) == 2 and not np.isnan(p).any() else np.nan
                            )
                            for k, p in probs.items()
                        },
                    }
                )
                for k, p in probs.items():
                    pts.append(
                        pd.DataFrame(
                            {
                                "site": site,
                                "n": n_sub,
                                "rep": r,
                                "model": k,
                                "patient": rest,
                                "loglik": ac.pointwise_ll(ye, p),
                            }
                        )
                    )
                print(f"{site} n={n_sub} rep={r} ({(time.time() - START) / 60:.1f} min)", flush=True)
            pd.concat(pts).to_csv(RES / "sites_points.csv", index=False)
    P = pd.concat(pts, ignore_index=True)
    P.to_csv(RES / "sites_points.csv", index=False)
    pd.DataFrame(diag).to_csv(RES / "sites_diagnostics.csv", index=False)
    I = pd.DataFrame(info)
    I.to_csv(RES / "sites_subsets.csv", index=False)
    summ = summarize(P, "M2", ["M0", "M0b", "M0c", "M1", "M2auto", "ridge", "literature", "transported"], ["site", "n"])
    auc = I.groupby(["site", "n"])[[c for c in I.columns if c.startswith("auc_")]].mean()
    summ = summ.merge(
        auc.stack().rename("mean_auc").reset_index().assign(model=lambda t: t.level_2.str[4:]).drop(columns="level_2"),
        on=["site", "n", "model"],
        how="left",
    )
    summ.to_csv(RES / "sites_curve.csv", index=False)
    dg = pd.DataFrame(diag)
    out["diag"] = {
        "max_rhat": float(dg.max_rhat.max()),
        "divergences": int(dg.divergences.sum()),
        "fits_with_divergences": int((dg.divergences > 0).sum()),
        "n_fits": int(len(dg)),
    }
    out["auto_keep_rate"] = {
        f"{s}_{n}": {f: float(g[f"keep_{f}"].mean()) for f in RED if f in ac.LIT}
        for (s, n), g in I.groupby(["site", "n"])
    }
    out["c_mean_M2"] = {f"{s}_{n}": float(g.c_mean_M2.mean()) for (s, n), g in I.groupby(["site", "n"])}
    out["ridge_failed"] = int(
        P.query("model == 'ridge'").groupby(["site", "n", "rep"]).loglik.apply(lambda v: v.isna().all()).sum()
    )
    sw = ac.read_uci(ac.HERE.parent / "data" / "processed.switzerland.data")
    out["switzerland"] = {
        "n": int(len(sw)),
        "prevalence": float((sw.num > 0).mean()),
        "chol_zero": int((sw.chol == 0).sum()),
    }
    out["runtime_min"] = (time.time() - START) / 60
    (RES / "sites.json").write_text(json.dumps(out, indent=2))
    print("sites done", out["runtime_min"], flush=True)


# ==============================================================================================
# 3. Literature calculator on the Cleveland test set, and decision-curve analysis
# ==============================================================================================
def net_benefit(y, p, t):
    pos = p >= t
    return (np.sum(pos & (y == 1)) - np.sum(pos & (y == 0)) * t / (1 - t)) / len(y)


def part_dca():
    out = {}
    te = ac.test_raw
    p_lit = literature_calculator(te)
    yte = te.disease.values
    test = pd.read_csv(RES / "test_metrics.csv", index_col=0)
    ll = ac.pointwise_ll(yte, p_lit)
    out["literature_calculator_test"] = {
        "loglik": float(ll.sum()),
        "auc": float(roc_auc_score(yte, p_lit)),
        "diff_vs_M2": float(ll.sum() - test.loc["M2", "loglik"]),
        "cal": list(map(float, ac.calibration(yte, p_lit))),
    }
    cvp = pd.read_csv(RES / "cv_predictions.csv", index_col=0)
    y = cvp.disease.values
    thresholds = np.round(np.arange(0.05, 0.501, 0.05), 2)
    models = ["M0", "M0b", "M0c", "M1", "M2", "MLE"]
    rows = []
    for t in thresholds:
        row = {"threshold": t, "treat_all": net_benefit(y, np.ones_like(y, float), t), "treat_none": 0.0}
        for k in models:
            row[k] = net_benefit(y, cvp[k].values, t)
        rows.append(row)
    dca = pd.DataFrame(rows)
    dca.to_csv(RES / "dca.csv", index=False)
    out["dca"] = {
        "thresholds": thresholds.tolist(),
        "M2_minus_best_other": (dca["M2"] - dca[[m for m in models if m != "M2"]].max(axis=1)).tolist(),
        "M2_minus_treat_all": (dca["M2"] - dca["treat_all"]).tolist(),
        "M1_minus_M2": (dca["M1"] - dca["M2"]).tolist(),
    }
    (RES / "dca.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# ==============================================================================================
# 4. Review round 1 additions (python transfer.py extra | beta11 | wideint)
# ==============================================================================================
def full_calculator(d):
    """Detrano-style literature calculator for Cleveland: Table I pretest odds multiplied by the Table II
    likelihood ratios of ST depression, thallium (normal, fixed, reversible) and calcified vessels (0, 1, 2;
    three vessels, whose likelihood ratio is infinite, treated as two)."""
    lp = np.log(literature_calculator(d) / (1 - literature_calculator(d)))
    thal = d.thal.map({3.0: 0, 6.0: 1, 7.0: 2}).fillna(0).astype(int).values
    ca = d.ca.fillna(0).clip(upper=2).astype(int).values
    return expit(lp + np.log(ac.thal_lr)[thal] + np.log(ac.calc_lr)[ca])


def recalibrate(p_fit, y_fit, p_new, slope=True):
    """Logistic recalibration of fixed predictions: logit p' = a + b logit p (b = 1 if slope is False)."""
    from scipy import optimize

    lp_fit, lp_new = (np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))) for q in (p_fit, p_new))
    # If the fitting subset is perfectly separated, the unpenalized slope diverges and predictions saturate
    # at 0 or 1. Outputs are clipped like the inputs, so a saturated wrong prediction costs log(1e6) = 13.8 nats.
    if slope:
        m = LogisticRegression(C=np.inf, max_iter=10_000).fit(lp_fit[:, None], y_fit)
        return np.clip(m.predict_proba(lp_new[:, None])[:, 1], 1e-6, 1 - 1e-6)
    a = optimize.minimize_scalar(lambda a: -np.sum(ac.pointwise_ll(y_fit, expit(a + lp_fit)))).x
    return np.clip(expit(a + lp_new), 1e-6, 1 - 1e-6)


def part_extra():
    out = {}
    te, tr = ac.test_raw, ac.train_raw
    yte, ytr = te.disease.values, tr.disease.values
    # (a) Cleveland: full literature calculator, as is and recalibrated on each learning-curve subset
    p_te, p_tr = full_calculator(te), full_calculator(tr)
    ll = ac.pointwise_ll(yte, p_te)
    test = pd.read_csv(RES / "test_metrics.csv", index_col=0)
    out["full_calculator_test"] = {
        "loglik": float(ll.sum()),
        "auc": float(roc_auc_score(yte, p_te)),
        "cal": list(map(float, ac.calibration(yte, p_te))),
        "diff_vs_M2": float(ll.sum() - test.loc["M2", "loglik"]),
    }
    saved = pd.read_csv(RES / "learning_curve_points.csv")
    autop = pd.read_csv(RES / "auto_rule_points.csv")
    rng = np.random.default_rng(ac.SEED + 1)
    pts = []
    for n_sub in ac.LC_SIZES:
        for r in range(ac.LC_REPS):
            idx = rng.choice(len(ytr), n_sub, replace=False)
            rest = np.setdiff1d(np.arange(len(ytr)), idx)
            for k, sl in [("lit_recal", True), ("lit_recal_int", False)]:
                for where, pe, ye, ids in [
                    ("test", p_te, yte, np.arange(len(yte))),
                    ("other_train", p_tr[rest], ytr[rest], rest),
                ]:
                    q = recalibrate(p_tr[idx], ytr[idx], pe, sl)
                    pts.append(
                        pd.DataFrame(
                            {
                                "n": n_sub,
                                "rep": r,
                                "model": k,
                                "scored_on": where,
                                "patient": ids,
                                "loglik": ac.pointwise_ll(ye, q),
                            }
                        )
                    )
    lcx = pd.concat([saved, autop] + pts, ignore_index=True)
    lcx.query("model in ['lit_recal', 'lit_recal_int']").to_csv(RES / "lc_recal_points.csv", index=False)
    summ = summarize(lcx, "M2", ["M0", "M0b", "M0c", "M1", "M2auto", "lit_recal", "lit_recal_int"], ["n", "scored_on"])
    summ.to_csv(RES / "lc_recal_curve.csv", index=False)
    # (b) test set reweighted to the training mix of chest-pain types
    w_cp = tr.cp.value_counts(normalize=True) / te.cp.value_counts(normalize=True)
    w = te.cp.map(w_cp).values
    w = w / w.mean()
    tp = lcx.query("scored_on == 'test'").copy()
    tp["loglik"] = tp.loglik * w[tp.patient.values]
    rw = summarize(tp, "M2", ["M0", "M0b", "M0c", "M1", "M2auto", "lit_recal"], ["n"])
    rw2 = summarize(tp, "M2auto", ["M0b"], ["n"])
    rw.to_csv(RES / "lc_test_reweighted.csv", index=False)
    out["test_cp_counts"] = {int(k): int(v) for k, v in te.cp.value_counts().sort_index().items()}
    out["test_cp_cad"] = {int(k): int(v) for k, v in te.groupby("cp").disease.sum().sort_index().items()}
    out["reweighted"] = {
        int(n): {
            "M2_minus_M0b": float(g.query("model == 'M0b'").M2_gain.iloc[0]),
            "M2auto_minus_M0b": float(rw2.query("n == @n and model == 'M0b'").M2auto_gain.iloc[0]),
            "logloss": {m: float(v) for m, v in zip(g.model, g.mean_log_loss)},
        }
        for n, g in rw.groupby("n")
    }
    # (c) new centers: recalibrated literature calculator and recalibrated transported model
    cle = ac.raw.dropna(subset=["exang", "oldpeak"])
    transported = LogisticRegression(C=np.inf, max_iter=10_000).fit(encode_red(cle).values, cle.disease.values)
    sp = pd.read_csv(RES / "sites_points.csv")
    new = []
    for site, fname in SITES.items():
        d = load_site(fname)
        Xraw, yall = encode_red(d).values, d.disease.values
        p_lit, p_tr2 = literature_calculator(d), transported.predict_proba(Xraw)[:, 1]
        rng = np.random.default_rng(ac.SEED + 7)  # the generator of part_sites, same order
        for n_sub in SITE_SIZES:
            for r in range(ac.LC_REPS):
                idx = rng.choice(len(yall), n_sub, replace=False)
                rest = np.setdiff1d(np.arange(len(yall)), idx)
                ye = yall[rest]
                # the transported model is updated in its intercept only: an intercept-and-slope update was
                # tried first and is numerically unstable at n = 25 (separation in the recalibration fit)
                for k, base, sl in [("lit_recal", p_lit, True), ("transported_recal_int", p_tr2, False)]:
                    q = recalibrate(base[idx], yall[idx], base[rest], sl)
                    new.append(
                        pd.DataFrame(
                            {
                                "site": site,
                                "n": n_sub,
                                "rep": r,
                                "model": k,
                                "patient": rest,
                                "loglik": ac.pointwise_ll(ye, q),
                            }
                        )
                    )
    newp = pd.concat(new, ignore_index=True)
    newp.to_csv(RES / "sites_recal_points.csv", index=False)
    allp = pd.concat([sp, newp], ignore_index=True)
    s2 = summarize(
        allp,
        "M2",
        [
            "M0",
            "M0b",
            "M0c",
            "M1",
            "M2auto",
            "ridge",
            "literature",
            "transported",
            "lit_recal",
            "transported_recal_int",
        ],
        ["site", "n"],
    )
    s3 = summarize(allp, "M2auto", ["M0b"], ["site", "n"]).query("model == 'M0b'")[
        ["site", "n", "M2auto_gain", "M2auto_gain_se"]
    ]
    s2 = s2.merge(s3.assign(model="M0b"), on=["site", "n", "model"], how="left")
    s2.to_csv(RES / "sites_curve2.csv", index=False)
    # (d) the fluoroscopy variable cannot be an angiographic count of diseased vessels
    raw = ac.raw.dropna(subset=["ca"])
    out["ca_vs_outcome"] = {
        "no_cad_ca_ge1": [int(((raw.disease == 0) & (raw.ca >= 1)).sum()), int((raw.disease == 0).sum())],
        "cad_ca_0": [int(((raw.disease == 1) & (raw.ca == 0)).sum()), int((raw.disease == 1).sum())],
    }
    # (e) share of the full-data gain of M2 over M0 recovered by the regularized baselines (CV log-likelihood)
    cvm = pd.read_csv(RES / "cv_metrics.csv", index_col=0)
    sen = json.loads((RES / "sensitivity.json").read_text())
    gap = cvm.loc["M2", "loglik"] - cvm.loc["M0", "loglik"]
    out["cv_share_recovered"] = {
        k: float((cvm.loc[k, "loglik"] - cvm.loc["M0", "loglik"]) / gap) for k in ["M0b", "M0c"]
    }
    out["cv_share_recovered"]["ridge"] = float((sen["ridge_cv"]["loglik"] - cvm.loc["M0", "loglik"]) / gap)
    (RES / "extra.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def part_beta11():
    """M2 with a uniform Beta(1, 1) prior on kappa, on the Cleveland learning-curve subsets with n = 25 and 50."""
    S = ac.SPECS["M2"]
    Xtr, ytr, Xte, yte = ac.X_train, ac.y_train, ac.X_test, ac.y_test
    rng = np.random.default_rng(ac.SEED + 1)
    pts, diag = [], []
    for n_sub in ac.LC_SIZES:
        for r in range(ac.LC_REPS):
            idx = rng.choice(len(ytr), n_sub, replace=False)
            if n_sub not in (25, 50):
                continue
            rest = np.setdiff1d(np.arange(len(ytr)), idx)
            with pm.Model():
                alpha = pm.Normal("alpha", A0, ASD)
                kappa = pm.Beta("c", 1, 1)
                beta = pm.Normal("beta", pm.math.where(S.discount, kappa * S.beta_mu, S.beta_mu), S.beta_sd, shape=ac.K)
                pm.Bernoulli("y", logit_p=alpha + pm.math.dot(Xtr[idx], beta), observed=ytr[idx])
                idata = pm.sample(
                    ac.LC_DRAWS,
                    tune=ac.LC_TUNE,
                    chains=ac.CHAINS,
                    target_accept=0.9,
                    random_seed=ac.SEED + 100 * n_sub + r,
                    progressbar=False,
                    mp_ctx="fork",
                )
            diag.append({"n": n_sub, "rep": r, "c_mean": float(post(idata, "c").mean()), **diagnostics(idata)})
            for where, Xe, ye, ids in [
                ("test", Xte, yte, np.arange(len(yte))),
                ("other_train", Xtr[rest], ytr[rest], rest),
            ]:
                pts.append(
                    pd.DataFrame(
                        {
                            "n": n_sub,
                            "rep": r,
                            "model": "M2beta11",
                            "scored_on": where,
                            "patient": ids,
                            "loglik": ac.pointwise_ll(ye, proba(idata, Xe)),
                        }
                    )
                )
            print(f"beta11 n={n_sub} rep={r}", flush=True)
    P = pd.concat([pd.read_csv(RES / "learning_curve_points.csv").query("n in [25, 50]")] + pts, ignore_index=True)
    summ = summarize(P, "M2beta11", ["M0b", "M2"], ["n", "scored_on"])
    summ.to_csv(RES / "beta11_curve.csv", index=False)
    d = pd.DataFrame(diag)
    d.to_csv(RES / "beta11_diagnostics.csv", index=False)
    print(summ.to_string(), d.groupby("n").c_mean.mean(), d.max_rhat.max(), d.divergences.sum())


def part_wideint():
    """M0b and M2 at the new centers with a weakly informative intercept prior N(0, 2.5^2) instead of the
    registry-based N(logit 0.376, 0.5^2): does the intercept prior explain the result at Long Beach?"""
    lit = np.array([ac.LIT.get(f, 0.0) for f in RED])
    is_lit = np.array([f in ac.LIT for f in RED])
    disc = is_lit & ~np.isin(RED, ac.CHEST_PAIN)
    K = len(RED)
    specs = {
        "M0b_w": (np.zeros(K), np.ones(K), np.zeros(K, bool)),
        "M2_w": (np.where(disc, lit, 0.0), np.where(disc, 0.5, 1.0), disc),
    }
    pts, diag = [], []
    for site, fname in SITES.items():
        d = load_site(fname)
        Xraw, yall = encode_red(d).values, d.disease.values
        rng = np.random.default_rng(ac.SEED + 7)
        for n_sub in SITE_SIZES:
            for r in range(ac.LC_REPS):
                idx = rng.choice(len(yall), n_sub, replace=False)
                rest = np.setdiff1d(np.arange(len(yall)), idx)
                ctr = Xraw[idx].mean(0)
                seed = ac.SEED + 1000 * (1 + list(SITES).index(site)) + 100 * n_sub + r
                for k, (mu, sd, dc) in specs.items():
                    idata = fit(Xraw[idx] - ctr, yall[idx], mu, sd, dc, 0.0, 2.5, seed, ac.LC_DRAWS, ac.LC_TUNE)
                    diag.append({"site": site, "n": n_sub, "rep": r, "model": k, **diagnostics(idata)})
                    pts.append(
                        pd.DataFrame(
                            {
                                "site": site,
                                "n": n_sub,
                                "rep": r,
                                "model": k,
                                "patient": rest,
                                "loglik": ac.pointwise_ll(yall[rest], proba(idata, Xraw[rest] - ctr)),
                            }
                        )
                    )
                print(f"wideint {site} n={n_sub} rep={r} ({(time.time() - START) / 60:.1f} min)", flush=True)
    P = pd.concat(pts, ignore_index=True)
    P.to_csv(RES / "wideint_points.csv", index=False)
    allp = pd.concat([pd.read_csv(RES / "sites_points.csv"), P], ignore_index=True)
    summ = summarize(allp, "M2_w", ["M0b_w", "M0", "ridge", "M2"], ["site", "n"])
    summ.to_csv(RES / "wideint_curve.csv", index=False)
    dg = pd.DataFrame(diag)
    dg.to_csv(RES / "wideint_diagnostics.csv", index=False)
    print(summ.to_string(), dg.max_rhat.max(), dg.divergences.sum())


if __name__ == "__main__":
    {
        "auto": part_auto,
        "sites": part_sites,
        "dca": part_dca,
        "extra": part_extra,
        "beta11": part_beta11,
        "wideint": part_wideint,
    }[sys.argv[1]]()
