"""The chest-pain recode in analysis.read_uci reproduces Detrano et al., Circulation 1986;73:970-977.

Table 8, subset 1 (n = 303): logistic regression of CAD on age (years), sex (male = -1, female = +1)
and pain (1 = typical angina ... 4 = no pain): constant -2.29, age 0.07, sex -0.85, pain -1.10.
Table 2: 55, 36, 35 and 13 patients with 1, 2, 3 vessels or left-main disease.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import transfer as T  # noqa: E402

ac = T.ac


def _logit_fit(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        b += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
    return b


def test_cleveland_chest_pain_matches_detrano_1986_table8():
    d = ac.read_uci(ac.DATA)
    y = (d.num > 0).to_numpy(float)
    sex = np.where(d.sex == 1, -1.0, 1.0)
    b = _logit_fit(np.column_stack([d.age, sex, d.cp]), y)
    assert np.allclose(b, [-2.29, 0.07, -0.85, -1.10], atol=0.006)


def test_cleveland_typical_angina_has_highest_cad_rate():
    d = ac.read_uci(ac.DATA)
    rate = (d.num > 0).groupby(d.cp).mean()
    assert rate.idxmax() == 1 and rate.idxmin() != 1
    assert [int((d.num == k).sum()) for k in (1, 2, 3, 4)] == [55, 36, 35, 13]
