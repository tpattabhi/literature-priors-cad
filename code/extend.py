"""Extensions for the submission-strength revision (run after analysis.py, sensitivity.py and transfer.py).

Usage (from this folder; the two parts are independent and can run in parallel):
    python extend.py cle      # Cleveland: full data and learning-curve subsets
    python extend.py sites    # Budapest, Long Beach and Tehran: learning curves on local subsets
    SMOKE=1 python extend.py cle

New models (all with the intercept prior of M0b-M2):
  M2-check  M2 with the design chosen inside each sample by a formal prior-predictive conflict check
            (Box 1980; Evans and Moshonov 2006): with b_j and v_j the posterior mean and variance of
            coefficient j under the weakly informative M0, the literature prior N(m_j, 0.5^2) implies
            b_j ~ N(m_j, 0.5^2 + v_j) approximately, so a prior is dropped (replaced by N(0, 1)) when
            |b_j - m_j| / sqrt(0.25 + v_j) > 1.96; the remaining means are scaled by kappa ~ Beta(2, 2).
  M2-h      M2 with a separate attenuation factor for each literature coefficient,
            logit kappa_j = mu + tau * e_j, mu ~ N(0, 1), tau ~ HalfNormal(1), e_j ~ N(0, 1).
  M1-CC     Face-value priors from the CAD Consortium basic model (Genders et al., BMJ 2012, Table 2):
            per 10 years of age OR 1.89, male 3.89, atypical 1.93 and typical 7.21 versus non-specific
            chest pain. Chest pain is therefore coded relative to nonanginal pain, and the asymptomatic
            indicator, which that model does not cover, gets N(0, 1). Test priors are those of M1.
  M2-CC     The same means, all scaled by kappa ~ Beta(2, 2).
Tehran (Z-Alizadeh Sani dataset, UCI, 2013): a third center with age, sex and chest-pain type only.
Decision curves: test-set net benefit of the models fitted on the 25 subsets of size 25.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transfer as T  # noqa: E402  (runs the definitions part of analysis.py)

ac = T.ac
import pymc as pm  # noqa: E402
import pytensor.tensor as pt  # noqa: E402
import arviz as az  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RES = ac.RES


# macOS occasionally kills a forked sampler process ("crashed on child side of fork"). Retry the same call
# (same random seed, hence the same draws) instead of losing the run.
_sample = pm.sample


def _sample_retry(*args, **kwargs):
    for attempt in range(3):
        try:
            return _sample(*args, **kwargs)
        except Exception as e:  # BrokenPipeError, EOFError, or PyMC's chain-failure errors
            if attempt == 2:
                raise
            print(f"sampling failed ({e!r}); retrying with the same seed", flush=True)


pm.sample = _sample_retry
START = time.time()
A0, ASD = T.A0, T.ASD
CC = {
    "age (per 10 y)": np.log(1.89),
    "male": np.log(3.89),
    "atypical angina": np.log(1.93),
    "typical angina": np.log(7.21),
}  # Genders et al. 2012, basic model, vs. non-specific chest pain


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def fit_h(X, y, mu, sd, disc, seed, draws, tune):
    """Coefficient-specific attenuation for the literature means flagged in `disc`."""
    k = int(disc.sum())
    with pm.Model():
        alpha = pm.Normal("alpha", A0, ASD)
        m = pm.Normal("mu_k", 0.0, 1.0)
        t = pm.HalfNormal("tau_k", 1.0)
        e = pm.Normal("e_k", 0.0, 1.0, shape=k)
        kap = pm.Deterministic("kappa", pm.math.sigmoid(m + t * e))
        idx = np.flatnonzero(disc)
        mult = pt.set_subtensor(pt.ones(len(mu))[idx], kap)
        beta = pm.Normal("beta", mult * pt.as_tensor(np.asarray(mu, float)), sd, shape=X.shape[1])
        pm.Bernoulli("y", logit_p=alpha + pm.math.dot(X, beta), observed=y)
        idata = pm.sample(
            draws,
            tune=tune,
            chains=ac.CHAINS,
            target_accept=0.95,
            random_seed=seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
            mp_ctx="fork",
        )
    return idata


def diag(idata):
    names = [v for v in ["alpha", "beta", "c", "mu_k", "tau_k", "e_k"] if v in idata.posterior]
    s = az.summary(idata, var_names=names)
    return {
        "max_rhat": float(s["r_hat"].max()),
        "min_bulk_ess": float(s["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
    }


def conflict_t(b_mean, b_var, lit, prior_sd=0.5):
    """Prior-predictive conflict statistic: under the prior N(m_j, s^2) and an approximately normal
    likelihood with variance v_j, the estimate b_j ~ N(m_j, s^2 + v_j)."""
    return (b_mean - lit) / np.sqrt(prior_sd**2 + b_var)


def check_keep(lit, is_lit, m0_idata):
    """Approximate prior-predictive conflict check against the M0 posterior (approximate likelihood)."""
    b = T.post(m0_idata, "beta")
    t = conflict_t(b.mean(0), b.var(0), lit)
    return is_lit & (np.abs(t) <= 1.96), t


def cc_coding(X, feats):
    """Recode the three chest-pain indicators (vs. asymptomatic) as typical, atypical and asymptomatic
    versus nonanginal pain. X holds raw (uncentered) indicator columns."""
    X = X.copy()
    i_t, i_a, i_n = (feats.index(f) for f in ["typical angina", "atypical angina", "non-anginal pain"])
    asym = 1.0 - X[:, i_t] - X[:, i_a] - X[:, i_n]
    X[:, i_n] = asym  # the nonanginal column now holds the asymptomatic indicator
    return X


def cc_specs(feats, base_mu, base_sd, base_lit):
    """M1-CC and M2-CC priors: CAD Consortium means for age, sex and chest pain; M1's test priors."""
    mu = np.array(
        [
            CC.get(f, base_mu[i] if base_lit[i] and f not in ac.CHEST_PAIN + ["age (per 10 y)", "male"] else 0.0)
            for i, f in enumerate(feats)
        ]
    )
    lit = np.array(
        [
            (f in CC) or (base_lit[i] and f not in ac.CHEST_PAIN + ["age (per 10 y)", "male"])
            for i, f in enumerate(feats)
        ]
    )
    sd = np.where(lit, 0.5, 1.0)
    return mu, sd, lit


def summarize(points, ref, others, by):
    return T.summarize(points, ref, others, by)


# ----------------------------------------------------------------------------------------------
# Cleveland
# ----------------------------------------------------------------------------------------------
def part_cle():
    F = ac.FEATURES
    S = ac.SPECS
    lit, is_lit = ac.lit_mean, ac.is_lit
    disc_m2 = S["M2"].discount
    # raw (uncentered) design for the CAD Consortium coding, centered with the same training means
    enc_tr, enc_te = ac.encode(ac.train_raw, ac.IMPUTE), ac.encode(ac.test_raw, ac.IMPUTE)
    Xtr_cc_raw, Xte_cc_raw = cc_coding(enc_tr.values, F), cc_coding(enc_te.values, F)
    ctr_cc = Xtr_cc_raw.mean(0)
    Xtr_cc, Xte_cc = Xtr_cc_raw - ctr_cc, Xte_cc_raw - ctr_cc
    mu_cc, sd_cc, lit_cc = cc_specs(F, lit, S["M1"].beta_sd, is_lit)
    Xtr, ytr, Xte, yte = ac.X_train, ac.y_train, ac.X_test, ac.y_test
    out, diags = {"cc_means": dict(zip(F, mu_cc.tolist())), "cc_lit": [f for f, l in zip(F, lit_cc) if l]}, []
    saved_loo = pd.read_csv(RES / "loo.csv", index_col=0)

    def fit_models(X, y, Xc, seed, draws, tune, with_probs=None):
        res = {}
        i0 = T.fit(X, y, np.zeros(ac.K), np.full(ac.K, 2.5), np.zeros(ac.K, bool), 0.0, 2.5, seed, draws, tune)
        keep, tstat = check_keep(lit, is_lit, i0)
        res["M2check"] = T.fit(
            X, y, np.where(keep, lit, 0.0), np.where(keep, 0.5, 1.0), keep, A0, ASD, seed, draws, tune
        )
        res["M2h"] = fit_h(X, y, S["M2"].beta_mu, S["M2"].beta_sd, disc_m2, seed, draws, tune)
        res["M1CC"] = T.fit(Xc, y, mu_cc, sd_cc, np.zeros(ac.K, bool), A0, ASD, seed, draws, tune)
        res["M2CC"] = T.fit(Xc, y, mu_cc, sd_cc, lit_cc, A0, ASD, seed, draws, tune)
        if with_probs:
            res["M0b"] = T.fit(
                X, y, S["M0b"].beta_mu, S["M0b"].beta_sd, np.zeros(ac.K, bool), A0, ASD, seed, draws, tune
            )
            res["M2"] = T.fit(X, y, S["M2"].beta_mu, S["M2"].beta_sd, disc_m2, A0, ASD, seed, draws, tune)
        return res, keep, tstat

    # full training set
    res, keep, tstat = fit_models(Xtr, ytr, Xtr_cc, ac.SEED, ac.DRAWS, ac.TUNE)
    test_saved = pd.read_csv(RES / "test_metrics.csv", index_col=0)
    full = {
        "check_kept": [f for f, k in zip(F, keep) if k],
        "check_t": dict(zip([f for f, l in zip(F, is_lit) if l], tstat[is_lit].tolist())),
        "M2_test_loglik": float(test_saved.loc["M2", "loglik"]),
        "M2_elpd": float(saved_loo.loc["M2", "elpd_loo"]),
    }
    for k, idata in res.items():
        Xe = Xte_cc if k in ("M1CC", "M2CC") else Xte
        p = T.proba(idata, Xe)
        loo = az.loo(idata)
        full[k] = {
            "elpd_loo": float(loo.elpd),
            "elpd_se": float(loo.se),
            "test_loglik": float(ac.pointwise_ll(yte, p).sum()),
            "test_auc": float(roc_auc_score(yte, p)),
            **diag(idata),
        }
        if "c" in idata.posterior:
            full[k]["kappa_mean"] = float(T.post(idata, "c").mean())
        if "kappa" in idata.posterior:
            full[k]["kappa_j"] = dict(
                zip([f for f, d in zip(F, disc_m2) if d], T.post(idata, "kappa").mean(0).tolist())
            )
        if k in ("M1CC", "M2CC"):
            full[k]["post_mean"] = dict(zip(F, T.post(idata, "beta").mean(0).tolist()))
            full[k]["z"] = dict(
                zip(
                    [f for f, l in zip(F, lit_cc) if l],
                    ((T.post(idata, "beta").mean(0) - mu_cc) / 0.5)[lit_cc].tolist(),
                )
            )
        print("full", k, {kk: v for kk, v in full[k].items() if not isinstance(v, dict)}, flush=True)
    # elpd differences vs. saved M2 (pointwise)
    out["full"] = full

    # learning-curve subsets (same generator and order as analysis.py)
    saved = pd.read_csv(RES / "learning_curve_points.csv")
    rng = np.random.default_rng(ac.SEED + 1)
    pts, probs, keeps = [], [], []
    for n_sub in ac.LC_SIZES:
        for r in range(ac.LC_REPS):
            idx = rng.choice(len(ytr), n_sub, replace=False)
            rest = np.setdiff1d(np.arange(len(ytr)), idx)
            seed = ac.SEED + 100 * n_sub + r
            res, keep, _ = fit_models(
                Xtr[idx], ytr[idx], Xtr_cc[idx], seed, ac.LC_DRAWS, ac.LC_TUNE, with_probs=(n_sub == 25)
            )
            keeps.append({"n": n_sub, "rep": r, **{f"keep_{f}": bool(k) for f, k in zip(F, keep) if f in ac.LIT}})
            for k, idata in res.items():
                diags.append({"n": n_sub, "rep": r, "model": k, **diag(idata)})
                cc = k in ("M1CC", "M2CC")
                for where, Xe, ye, ids in [
                    ("test", Xte_cc if cc else Xte, yte, np.arange(len(yte))),
                    ("other_train", (Xtr_cc if cc else Xtr)[rest], ytr[rest], rest),
                ]:
                    p = T.proba(idata, Xe)
                    if k in ("M0b", "M2"):
                        if where == "test":
                            probs.append(pd.DataFrame({"n": n_sub, "rep": r, "model": k, "patient": ids, "p": p}))
                        continue  # learning-curve log-likelihoods already saved
                    pts.append(
                        pd.DataFrame(
                            {
                                "n": n_sub,
                                "rep": r,
                                "model": k,
                                "scored_on": where,
                                "patient": ids,
                                "loglik": ac.pointwise_ll(ye, p),
                            }
                        )
                    )
                    if where == "test" and n_sub == 25:
                        probs.append(pd.DataFrame({"n": n_sub, "rep": r, "model": k, "patient": ids, "p": p}))
            print(f"cle n={n_sub} rep={r} ({(time.time() - START) / 60:.1f} min)", flush=True)
            pd.concat(pts).to_csv(RES / "ext_cle_points.csv", index=False)
            pd.concat(probs).to_csv(RES / "ext_cle_probs_n25.csv", index=False)
            pd.DataFrame(keeps).to_csv(RES / "ext_cle_check_keep.csv", index=False)
            pd.DataFrame(diags).to_csv(RES / "ext_cle_diagnostics.csv", index=False)
    P = pd.concat(pts, ignore_index=True)
    P.to_csv(RES / "ext_cle_points.csv", index=False)
    pd.concat(probs).to_csv(RES / "ext_cle_probs_n25.csv", index=False)
    pd.DataFrame(keeps).to_csv(RES / "ext_cle_check_keep.csv", index=False)
    pd.DataFrame(diags).to_csv(RES / "ext_cle_diagnostics.csv", index=False)
    allp = pd.concat([saved, pd.read_csv(RES / "auto_rule_points.csv"), P], ignore_index=True)
    summarize(
        allp, "M2", ["M0", "M0b", "M0c", "M1", "M2auto", "M2check", "M2h", "M1CC", "M2CC"], ["n", "scored_on"]
    ).to_csv(RES / "ext_cle_curve_vsM2.csv", index=False)
    summarize(allp, "M2check", ["M0b", "M2auto"], ["n", "scored_on"]).to_csv(
        RES / "ext_cle_curve_check.csv", index=False
    )
    summarize(allp, "M2CC", ["M0b", "M1CC", "M2"], ["n", "scored_on"]).to_csv(RES / "ext_cle_curve_cc.csv", index=False)
    summarize(allp, "M2h", ["M0b", "M2"], ["n", "scored_on"]).to_csv(RES / "ext_cle_curve_h.csv", index=False)
    kp = pd.DataFrame(keeps)
    out["check_keep_rate"] = {
        int(n): {c[5:]: float(g[c].mean()) for c in g.columns if c.startswith("keep_")} for n, g in kp.groupby("n")
    }
    d = pd.DataFrame(diags)
    out["diag"] = {
        "n_fits": int(len(d)),
        "max_rhat": float(d.max_rhat.max()),
        "divergences": int(d.divergences.sum()),
        "fits_with_divergences": int((d.divergences > 0).sum()),
        "min_bulk_ess": float(d.min_bulk_ess.min()),
    }
    # decision curves at n = 25 (test set): mean net benefit over the 25 subsets
    pr = pd.concat(probs, ignore_index=True)
    thr = np.round(np.arange(0.05, 0.501, 0.05), 2)
    rows = []
    for (m, r), g in pr.groupby(["model", "rep"]):
        p = g.sort_values("patient").p.values
        for t in thr:
            rows.append({"model": m, "rep": r, "threshold": t, "nb": T.net_benefit(yte, p, t)})
    nb = pd.DataFrame(rows)
    nb_all = [
        {"model": "treat_all", "rep": -1, "threshold": t, "nb": T.net_benefit(yte, np.ones(len(yte)), t)} for t in thr
    ]
    nb = pd.concat([nb, pd.DataFrame(nb_all)], ignore_index=True)
    nb.to_csv(RES / "ext_dca_n25.csv", index=False)
    out["runtime_min"] = (time.time() - START) / 60
    (RES / "ext_cle.json").write_text(json.dumps(out, indent=2))
    print("cle done", out["runtime_min"], flush=True)


# ----------------------------------------------------------------------------------------------
# new centers, including Tehran
# ----------------------------------------------------------------------------------------------
RED3 = ["age (per 10 y)", "male", "typical angina", "atypical angina", "non-anginal pain"]


def load_tehran():
    d = pd.read_csv(ac.HERE.parent / "data" / "z-alizadeh-sani.csv")
    out = pd.DataFrame({"age": d.Age.astype(float), "sex": (d.Sex.str.strip() == "Male").astype(float)})
    cp = np.full(len(d), 4.0)
    cp[d.Nonanginal.str.strip() == "Y"] = 3
    cp[d.Atypical.str.strip() == "Y"] = 2
    cp[d["Typical Chest Pain"].astype(float) == 1] = 1
    out["cp"] = cp
    out["disease"] = (d.Cath.str.strip() == "Cad").astype(int)
    return out


def enc3(d):
    return pd.DataFrame(
        {
            "age (per 10 y)": d.age / 10,
            "male": d.sex,
            "typical angina": d.cp == 1,
            "atypical angina": d.cp == 2,
            "non-anginal pain": d.cp == 3,
        }
    ).astype(float)


def pretest_calculator(d):
    b = ac.coef_table_I
    lp = b[0] + b[1] * (d.age - 55) / 10 + b[2] * d.sex + b[3] * (d.cp == 1) + b[4] * (d.cp == 2) + b[5] * (d.cp == 3)
    return expit(lp.values.astype(float))


def summarize_sites(P):
    """Learning-curve summaries for the new centers from the saved per-patient points (no refitting)."""
    allp = pd.concat(
        [pd.read_csv(RES / "sites_points.csv"), pd.read_csv(RES / "sites_recal_points.csv"), P], ignore_index=True
    )
    models = [
        "M0",
        "M0b",
        "M0c",
        "M1",
        "M2auto",
        "M2check",
        "M2h",
        "M1CC",
        "M2CC",
        "ridge",
        "literature",
        "lit_recal",
        "transported",
        "transported_recal_int",
    ]
    summarize(allp, "M2", models, ["site", "n"]).to_csv(RES / "ext_sites_curve_vsM2.csv", index=False)
    summarize(allp, "M2check", ["M0b", "M2auto", "M2"], ["site", "n"]).to_csv(
        RES / "ext_sites_curve_check.csv", index=False
    )
    summarize(allp, "M2CC", ["M0b", "M1CC", "M2"], ["site", "n"]).to_csv(RES / "ext_sites_curve_cc.csv", index=False)
    summarize(allp, "M2h", ["M0b", "M2"], ["site", "n"]).to_csv(RES / "ext_sites_curve_h.csv", index=False)


def part_sites():
    out, pts, diags, keeps = {}, [], [], []
    # Cleveland models transported to Tehran: age, sex and chest pain only
    cle = ac.raw
    tr3 = LogisticRegression(C=np.inf, max_iter=10_000).fit(enc3(cle).values, cle.disease.values)
    centers = {"hungarian": ("hungarian", T.RED), "va": ("va", T.RED), "tehran": ("tehran", RED3)}
    for site, (key, feats) in centers.items():
        if site == "tehran":
            d = load_tehran()
            Xraw, yall = enc3(d).values, d.disease.values
            p_lit, p_tr = pretest_calculator(d), tr3.predict_proba(Xraw)[:, 1]
            cp_rate = {
                int(k): [float(v), int(n)] for k, (v, n) in d.groupby("cp").disease.agg(["mean", "size"]).iterrows()
            }
            out["tehran"] = {
                "n": int(len(d)),
                "prevalence": float(yall.mean()),
                "cad_rate_by_cp": cp_rate,
                "male": float(d.sex.mean()),
                "age_mean": float(d.age.mean()),
            }
        else:
            d = T.load_site(T.SITES[site])
            Xraw, yall = T.encode_red(d).values, d.disease.values
        K = len(feats)
        lit = np.array([ac.LIT.get(f, 0.0) for f in feats])
        is_lit = np.array([f in ac.LIT for f in feats])
        is_cp = np.isin(feats, ac.CHEST_PAIN)
        disc = is_lit & ~is_cp
        mu_cc, sd_cc, lit_cc = cc_specs(feats, lit, None, is_lit)
        rng = np.random.default_rng(ac.SEED + 7 if site != "tehran" else ac.SEED + 9)
        for n_sub in T.SITE_SIZES:
            for r in range(ac.LC_REPS):
                idx = rng.choice(len(yall), n_sub, replace=False)
                rest = np.setdiff1d(np.arange(len(yall)), idx)
                ctr = Xraw[idx].mean(0)
                X, y, Xe, ye = Xraw[idx] - ctr, yall[idx], Xraw[rest] - ctr, yall[rest]
                Xcc_raw = cc_coding(Xraw, feats)
                ctr_cc = Xcc_raw[idx].mean(0)
                Xc, Xce = Xcc_raw[idx] - ctr_cc, Xcc_raw[rest] - ctr_cc
                seed = ac.SEED + 1000 * (1 + ["hungarian", "va", "tehran"].index(site)) + 100 * n_sub + r
                probs, fits = {}, {}
                i0 = T.fit(
                    X, y, np.zeros(K), np.full(K, 2.5), np.zeros(K, bool), 0.0, 2.5, seed, ac.LC_DRAWS, ac.LC_TUNE
                )
                keep, _ = check_keep(lit, is_lit, i0)
                keeps.append(
                    {
                        "site": site,
                        "n": n_sub,
                        "rep": r,
                        **{f"keep_{f}": bool(k) for f, k in zip(feats, keep) if f in ac.LIT},
                    }
                )
                fits["M2check"] = T.fit(
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
                fits["M2h"] = fit_h(
                    X, y, np.where(disc, lit, 0.0), np.where(disc, 0.5, 1.0), disc, seed, ac.LC_DRAWS, ac.LC_TUNE
                )
                fits["M1CC"] = T.fit(Xc, y, mu_cc, sd_cc, np.zeros(K, bool), A0, ASD, seed, ac.LC_DRAWS, ac.LC_TUNE)
                fits["M2CC"] = T.fit(Xc, y, mu_cc, sd_cc, lit_cc, A0, ASD, seed, ac.LC_DRAWS, ac.LC_TUNE)
                if site == "tehran":  # Tehran is new: fit every model of transfer.py as well
                    fits["M0"] = i0
                    for k, (am, asd, mu, sd, dc) in {
                        "M0b": (A0, ASD, np.zeros(K), np.ones(K), np.zeros(K, bool)),
                        "M0c": (A0, ASD, np.zeros(K), np.where(disc, 0.5, 1.0), np.zeros(K, bool)),
                        "M1": (A0, ASD, lit, np.where(is_lit, 0.5, 1.0), np.zeros(K, bool)),
                        "M2": (A0, ASD, np.where(disc, lit, 0.0), np.where(disc, 0.5, 1.0), disc),
                    }.items():
                        fits[k] = T.fit(X, y, mu, sd, dc, am, asd, seed, ac.LC_DRAWS, ac.LC_TUNE)
                    ka = T.auto_rule(
                        lit, is_lit, T.post(fits["M0b"], "beta").mean(0), T.post(fits["M1"], "beta").mean(0)
                    )
                    fits["M2auto"] = T.fit(
                        X, y, np.where(ka, lit, 0.0), np.where(ka, 0.5, 1.0), ka, A0, ASD, seed, ac.LC_DRAWS, ac.LC_TUNE
                    )
                    try:
                        probs["ridge"] = T.ridge(ac.SEED).fit(X, y).predict_proba(Xe)[:, 1]
                    except ValueError:
                        probs["ridge"] = np.full(len(ye), np.nan)
                    probs["literature"], probs["transported"] = p_lit[rest], p_tr[rest]
                    probs["lit_recal"] = T.recalibrate(p_lit[idx], y, p_lit[rest], True)
                    probs["transported_recal_int"] = T.recalibrate(p_tr[idx], y, p_tr[rest], False)
                for k, idata in fits.items():
                    diags.append({"site": site, "n": n_sub, "rep": r, "model": k, **diag(idata)})
                    probs[k] = T.proba(idata, Xce if k in ("M1CC", "M2CC") else Xe)
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
            pd.concat(pts).to_csv(RES / "ext_sites_points.csv", index=False)
    P = pd.concat(pts, ignore_index=True)
    P.to_csv(RES / "ext_sites_points.csv", index=False)
    pd.DataFrame(diags).to_csv(RES / "ext_sites_diagnostics.csv", index=False)
    pd.DataFrame(keeps).to_csv(RES / "ext_sites_check_keep.csv", index=False)
    summarize_sites(P)
    kp = pd.DataFrame(keeps)
    out["check_keep_rate"] = {
        f"{s}_{n}": {c[5:]: float(g[c].mean()) for c in g.columns if c.startswith("keep_") and g[c].notna().any()}
        for (s, n), g in kp.groupby(["site", "n"])
    }
    dg = pd.DataFrame(diags)
    out["diag"] = {
        "n_fits": int(len(dg)),
        "max_rhat": float(dg.max_rhat.max()),
        "divergences": int(dg.divergences.sum()),
        "fits_with_divergences": int((dg.divergences > 0).sum()),
        "min_bulk_ess": float(dg.min_bulk_ess.min()),
    }
    out["runtime_min"] = (time.time() - START) / 60
    (RES / "ext_sites.json").write_text(json.dumps(out, indent=2))
    print("sites done", out["runtime_min"], flush=True)


def part_tehw():
    """Z-Alizadeh Sani with the weakly informative intercept prior N(0, 2.5^2) of M0 for M0b, M2 and M2-CC."""
    d = load_tehran()
    feats = RED3
    Xraw, yall = enc3(d).values, d.disease.values
    K = len(feats)
    lit = np.array([ac.LIT.get(f, 0.0) for f in feats])
    is_lit = np.array([f in ac.LIT for f in feats])
    disc = is_lit & ~np.isin(feats, ac.CHEST_PAIN)
    mu_cc, sd_cc, lit_cc = cc_specs(feats, lit, None, is_lit)
    rng = np.random.default_rng(ac.SEED + 9)  # the generator of part_sites for this center
    pts, diags = [], []
    for n_sub in T.SITE_SIZES:
        for r in range(ac.LC_REPS):
            idx = rng.choice(len(yall), n_sub, replace=False)
            rest = np.setdiff1d(np.arange(len(yall)), idx)
            ctr = Xraw[idx].mean(0)
            X, y, Xe, ye = Xraw[idx] - ctr, yall[idx], Xraw[rest] - ctr, yall[rest]
            Xcc_raw = cc_coding(Xraw, feats)
            ctr_cc = Xcc_raw[idx].mean(0)
            seed = ac.SEED + 3000 + 100 * n_sub + r
            fits = {
                "M0b_w": (X, Xe, np.zeros(K), np.ones(K), np.zeros(K, bool)),
                "M2_w": (X, Xe, np.where(disc, lit, 0.0), np.where(disc, 0.5, 1.0), disc),
                "M2CC_w": (Xcc_raw[idx] - ctr_cc, Xcc_raw[rest] - ctr_cc, mu_cc, sd_cc, lit_cc),
            }
            for k, (Xf, Xs, mu, sd, dc) in fits.items():
                idata = T.fit(Xf, y, mu, sd, dc, 0.0, 2.5, seed, ac.LC_DRAWS, ac.LC_TUNE)
                diags.append({"n": n_sub, "rep": r, "model": k, **diag(idata)})
                pts.append(
                    pd.DataFrame(
                        {
                            "site": "tehran",
                            "n": n_sub,
                            "rep": r,
                            "model": k,
                            "patient": rest,
                            "loglik": ac.pointwise_ll(ye, T.proba(idata, Xs)),
                        }
                    )
                )
            print(f"tehw n={n_sub} rep={r} ({(time.time() - START) / 60:.1f} min)", flush=True)
    pd.concat(pts).to_csv(RES / "ext_tehw_points.csv", index=False)
    pd.DataFrame(diags).to_csv(RES / "ext_tehw_diagnostics.csv", index=False)
    print("tehw done", (time.time() - START) / 60, flush=True)


def part_kfix():
    """M2 with the attenuation fixed at kappa = 0.25 or 1 (no learning), Cleveland learning-curve subsets with n = 25:
    does the small-sample gain depend on the learned factor sitting near its prior mean of 0.5?"""
    S = ac.SPECS["M2"]
    Xtr, ytr, Xte, yte = ac.X_train, ac.y_train, ac.X_test, ac.y_test
    rng = np.random.default_rng(ac.SEED + 1)
    pts = []
    for r in range(ac.LC_REPS):
        idx = rng.choice(len(ytr), 25, replace=False)  # the first 25 draws are the n = 25 subsets of analysis.py
        rest = np.setdiff1d(np.arange(len(ytr)), idx)
        for kap in (0.25, 1.0):
            mu = np.where(S.discount, kap * S.beta_mu, S.beta_mu)
            idata = T.fit(
                Xtr[idx],
                ytr[idx],
                mu,
                S.beta_sd,
                np.zeros(ac.K, bool),
                A0,
                ASD,
                ac.SEED + 2500 + r,
                ac.LC_DRAWS,
                ac.LC_TUNE,
            )
            for where, Xe, ye, ids in [
                ("test", Xte, yte, np.arange(len(yte))),
                ("other_train", Xtr[rest], ytr[rest], rest),
            ]:
                pts.append(
                    pd.DataFrame(
                        {
                            "n": 25,
                            "rep": r,
                            "model": f"M2k{kap}",
                            "scored_on": where,
                            "patient": ids,
                            "loglik": ac.pointwise_ll(ye, T.proba(idata, Xe)),
                        }
                    )
                )
        print(f"kfix rep={r}", flush=True)
    pd.concat(pts).to_csv(RES / "ext_kfix_points.csv", index=False)


if __name__ == "__main__":
    {
        "cle": part_cle,
        "sites": part_sites,
        "sites-summary": lambda: summarize_sites(pd.read_csv(RES / "ext_sites_points.csv")),
        "tehw": part_tehw,
        "kfix": part_kfix,
    }[sys.argv[1]]()
