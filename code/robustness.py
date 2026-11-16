"""Uncertainty checks requested in review (no model fitting; run after analysis.py, transfer.py and extend.py).

Usage (from this folder): python robustness.py  -> ../results/robustness.json

1. Two-way bootstrap of the learning-curve contrasts: resample the 25 subsets and the scored patients with
   replacement (2,000 draws) and recompute the mean per-patient difference in log-likelihood. Its standard
   deviation is compared with the analytic SE used in the paper (root sum of squares of the SE across subsets
   and the SE across patients of subset-averaged differences).
2. Decision curves at n = 25 (test set): the net-benefit difference between two models, converted to net
   false-positive referrals avoided per 100 patients, (dNB) (1 - t) / t * 100, with a two-way bootstrap
   95% interval, at each threshold t from 5% to 50%.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
RNG = np.random.default_rng(2026)
B = 2000


def matrix(points, model):
    """Subsets x patients matrix of log-likelihoods (NaN where a patient was not scored in a subset)."""
    return points[points.model == model].pivot_table(index="rep", columns="patient", values="loglik")


def contrast(points, model, ref):
    a, b = matrix(points, model), matrix(points, ref)
    d = (a - b).loc[:, a.columns.intersection(b.columns)]
    reps, pats = d.index.to_numpy(), d.columns.to_numpy()
    est = float(np.nanmean(d.to_numpy().mean(axis=1) if not d.isna().any().any() else d.mean(axis=1)))
    per_rep = d.mean(axis=1)
    per_pat = d.mean(axis=0)
    se_analytic = float(np.hypot(per_rep.std(ddof=1) / np.sqrt(len(per_rep)), per_pat.std(ddof=1) / np.sqrt(len(per_pat))))
    D = d.to_numpy()
    boots = np.empty(B)
    for i in range(B):
        r = RNG.integers(0, len(reps), len(reps))
        p = RNG.integers(0, len(pats), len(pats))
        boots[i] = np.nanmean(D[np.ix_(r, p)])
    return {"estimate": float(per_rep.mean()), "se_analytic": se_analytic, "se_bootstrap": float(boots.std(ddof=1)),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def main():
    out = {}
    cle = pd.concat([pd.read_csv(RES / f) for f in ["learning_curve_points.csv", "auto_rule_points.csv",
                                                    "ext_cle_points.csv"]], ignore_index=True)
    sites = pd.concat([pd.read_csv(RES / f) for f in ["sites_points.csv", "ext_sites_points.csv"]], ignore_index=True)
    cases = [("cle", 25, "test", "M2", "M0b"), ("cle", 25, "other_train", "M2", "M0b"),
             ("cle", 25, "test", "M2CC", "M0b"), ("cle", 25, "other_train", "M2CC", "M0b"),
             ("cle", 25, "test", "M2auto", "M0b"), ("cle", 25, "test", "M2check", "M0b"),
             ("hungarian", 25, None, "M2CC", "M0b"), ("hungarian", 25, None, "M2", "M0b")]
    for site, n, where, m, ref in cases:
        if site == "cle":
            pts = cle[(cle.n == n) & (cle.scored_on == where)]
        else:
            pts = sites[(sites.site == site) & (sites.n == n)]
        out[f"{site}_{n}_{where or 'local'}_{m}_vs_{ref}"] = contrast(pts, m, ref)
    out["se_ratio_bootstrap_over_analytic"] = [v["se_bootstrap"] / v["se_analytic"] for v in out.values()]

    # decision curves at n = 25
    pr = pd.read_csv(RES / "ext_cle_probs_n25.csv")
    y = pd.read_csv(RES / "cv_predictions.csv", index_col=0)  # only for the file check below
    import sys

    sys.path.insert(0, str(HERE))
    test_y = None
    tm = pd.read_csv(RES / "test_metrics.csv", index_col=0)
    assert len(tm) > 0 and len(y) == 303
    # the test outcomes, in the order used by analysis.py (patients 0..60)
    from sklearn.model_selection import train_test_split

    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
            "thal", "num"]
    raw = pd.read_csv(HERE.parent / "data" / "processed.cleveland.data", header=None, names=cols, na_values="?")
    raw["disease"] = (raw["num"] > 0).astype(int)
    _, test_raw = train_test_split(raw, test_size=0.2, stratify=raw["disease"], random_state=2026)
    test_y = test_raw.disease.to_numpy()
    P = {m: g.pivot_table(index="rep", columns="patient", values="p").to_numpy() for m, g in pr.groupby("model")}
    thr = np.round(np.arange(0.05, 0.501, 0.05), 2)

    def nb(p, yy, t):
        pos = p >= t
        return (np.sum(pos & (yy == 1), axis=-1) - np.sum(pos & (yy == 0), axis=-1) * t / (1 - t)) / yy.shape[-1]

    dca = {}
    for m in ["M2CC", "M2"]:
        rows = []
        for t in thr:
            d_obs = float(np.mean(nb(P[m], test_y, t) - nb(P["M0b"], test_y, t)))
            boots = np.empty(B)
            for i in range(B):
                r = RNG.integers(0, P[m].shape[0], P[m].shape[0])
                pidx = RNG.integers(0, len(test_y), len(test_y))
                yy = test_y[pidx]
                boots[i] = np.mean(nb(P[m][np.ix_(r, pidx)], yy, t) - nb(P["M0b"][np.ix_(r, pidx)], yy, t))
            f = (1 - t) / t * 100
            rows.append({"threshold": float(t), "dNB": d_obs, "avoided_per_100": d_obs * f,
                         "ci95": [float(np.quantile(boots, 0.025) * f), float(np.quantile(boots, 0.975) * f)]})
        dca[m] = rows
    out["dca_vs_M0b_n25"] = dca
    (RES / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "dca_vs_M0b_n25"}, indent=1))
    for m in dca:
        print(m, [(r["threshold"], round(r["avoided_per_100"], 1), [round(x, 1) for x in r["ci95"]]) for r in dca[m]])


if __name__ == "__main__":
    main()
