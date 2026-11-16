"""Fast tests of the analysis logic (no MCMC). Run from the repository root: uv run pytest"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import transfer as T  # noqa: E402  (runs the definitions part of analysis.py)
import extend as E  # noqa: E402

ac = T.ac


# ---------------------------------------------------------------- data
@pytest.mark.parametrize(
    "fname, sha1",
    [
        ("processed.cleveland.data", "91916bcbf1ddc8517f496d00f029f11865f35243"),
        ("processed.hungarian.data", "06c0f18dcff3d2236bdfad4e708cac2766b298a9"),
        ("processed.va.data", "371f1bb8b7fa93809cbdea7811f639738898617b"),
        ("processed.switzerland.data", "717920d5ce1380ca9dd6238e8d13fb2af5a96622"),
    ],
)
def test_uci_files_unchanged(fname, sha1):
    assert hashlib.sha1((ROOT / "data" / fname).read_bytes()).hexdigest() == sha1


def test_cohort_sizes_and_split():
    assert len(ac.raw) == 303 and int(ac.raw.disease.sum()) == 139
    assert len(ac.y_train) == 242 and len(ac.y_test) == 61 and int(ac.y_test.sum()) == 28
    assert len(T.load_site(T.SITES["hungarian"])) == 293 and len(T.load_site(T.SITES["va"])) == 144
    teh = E.load_tehran()
    assert len(teh) == 303 and int(teh.disease.sum()) == 216
    assert set(teh.cp.unique()) == {1.0, 2.0, 3.0, 4.0}


def test_training_columns_are_centered():
    assert np.allclose(ac.X_train.mean(0), 0, atol=1e-12)


# ---------------------------------------------------------------- priors from the published tables
def test_table_I_surface_fits_the_32_cells():
    assert ac.D.shape == (32, 6)
    assert np.std(ac.resid) < 0.35 and np.abs(ac.resid).max() < 0.8


def test_likelihood_ratio_prior_means():
    # Diamond et al. (1980) Table II: P(result | CAD) / P(result | no CAD)
    assert ac.LIT["any calcified vessel"] == pytest.approx(np.log((0.235 / 0.023) / (0.420 / 0.961)))
    assert ac.LIT["reversible thallium defect"] == pytest.approx(np.log((0.707 / 0.061) / (0.147 / 0.836)))
    assert ac.LIT["ST depression (per mm)"] == pytest.approx(1.934, abs=1e-3)
    assert set(ac.LIT) == set(ac.FEATURES[:5]) | {
        "ST depression (per mm)",
        "any calcified vessel",
        "extra calcified vessels",
        "fixed thallium defect",
        "reversible thallium defect",
    }


def test_m2_design():
    s = ac.SPECS["M2"]
    for f in ac.CHEST_PAIN:  # chest-pain priors dropped
        i = ac.FEATURES.index(f)
        assert s.beta_mu[i] == 0 and s.beta_sd[i] == 1 and not s.discount[i]
    assert s.discount.sum() == 7 and ac.SPECS["M0c"].beta_sd.tolist() == s.beta_sd.tolist()


def test_consortium_priors():
    mu, sd, lit = E.cc_specs(ac.FEATURES, ac.lit_mean, None, ac.is_lit)
    F = ac.FEATURES
    assert mu[F.index("typical angina")] == pytest.approx(np.log(7.21))
    assert mu[F.index("male")] == pytest.approx(np.log(3.89))
    assert mu[F.index("non-anginal pain")] == 0 and not lit[F.index("non-anginal pain")]  # asymptomatic slot
    assert mu[F.index("reversible thallium defect")] == pytest.approx(ac.LIT["reversible thallium defect"])
    assert set(np.unique(sd)) == {0.5, 1.0}


def test_consortium_coding():
    F = E.RED3
    X = np.array([[5.0, 1, 1, 0, 0], [5.0, 0, 0, 0, 0], [6.0, 1, 0, 0, 1]])  # typical, asymptomatic, nonanginal
    Y = E.cc_coding(X, F)
    assert Y[:, 2].tolist() == [1, 0, 0] and Y[:, 3].tolist() == [0, 0, 0]
    assert Y[:, 4].tolist() == [0, 1, 0]  # the nonanginal column now holds the asymptomatic indicator


# ---------------------------------------------------------------- design rules
def test_auto_rule():
    lit = np.array([1.0, 2.0, -1.0, 0.0])
    is_lit = np.array([True, True, True, False])
    m0b = np.array([0.5, -0.1, -0.5, 0.3])  # second has the opposite sign
    m1 = np.array([0.9, 1.9, -2.2, 0.0])  # third has z = -2.4
    assert T.auto_rule(lit, is_lit, m0b, m1).tolist() == [True, False, False, False]


def test_conflict_t():
    t = E.conflict_t(np.array([0.0, 1.0]), np.array([0.75, 0.0]), np.array([2.0, 1.0]))
    assert t.tolist() == pytest.approx([-2.0, 0.0])


# ---------------------------------------------------------------- comparators and metrics
def test_recalibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    y = rng.binomial(1, p)
    assert np.allclose(T.recalibrate(p, y, p, slope=True), p, atol=0.03)  # calibrated input is kept
    q = expit(logit(p) + 1.0)  # shifted predictions
    q_int = T.recalibrate(q, y, q, slope=False)
    assert abs(q_int.mean() - y.mean()) < 0.01


def test_literature_calculator_is_pretest_times_st_lr():
    d = pd.DataFrame({"age": [55.0], "sex": [0.0], "cp": [4.0], "oldpeak": [0.0]})
    b = ac.coef_table_I
    slope, intercept = np.polyfit(ac.st_mid, np.log(ac.st_lr), 1)
    assert T.literature_calculator(d)[0] == pytest.approx(expit(b[0] + intercept))
    d2 = d.assign(oldpeak=[2.0])
    assert T.literature_calculator(d2)[0] > T.literature_calculator(d)[0]


def test_net_benefit():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1])
    # threshold 0.5: TP = 1, FP = 1 -> (1 - 1 * 1) / 4 = 0
    assert T.net_benefit(y, p, 0.5) == pytest.approx(0.0)
    assert T.net_benefit(y, np.ones(4), 0.2) == pytest.approx((2 - 2 * 0.25) / 4)


def test_summarize_standard_error():
    rows = []
    for rep in range(3):
        for patient in range(4):
            rows.append({"n": 25, "rep": rep, "model": "A", "patient": patient, "loglik": -0.5})
            rows.append({"n": 25, "rep": rep, "model": "B", "patient": patient, "loglik": -0.5 - 0.1 * (rep + patient)})
    s = T.summarize(pd.DataFrame(rows), "A", ["B"], ["n"]).set_index("model")
    per_rep = np.array([0.1 * (r + 1.5) for r in range(3)])
    per_pat = np.array([0.1 * (1 + p) for p in range(4)])
    se = np.hypot(per_rep.std(ddof=1) / np.sqrt(3), per_pat.std(ddof=1) / np.sqrt(4))
    assert s.loc["B", "A_gain"] == pytest.approx(per_rep.mean())
    assert s.loc["B", "A_gain_se"] == pytest.approx(se)
