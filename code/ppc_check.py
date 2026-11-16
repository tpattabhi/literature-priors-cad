"""Posterior predictive check by chest-pain group (Section III-A of the paper).

Refits M1 and M2 on the 242 training patients (same split, priors, sampler settings and seed as
analysis.py), simulates new outcomes for the training patients from each posterior, and compares the
simulated share with CAD in each chest-pain group with the observed share.

Usage (from this folder): python ppc_check.py  -> ../results/group_ppc.csv
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Run the definitions part of analysis.py (data, split, encoding, priors, model class) unchanged, so
# this check uses exactly the same set-up as the paper's main analysis.
_src = (Path(__file__).resolve().parent / "analysis.py").read_text()
_ns = {"__file__": str(Path(__file__).resolve().parent / "analysis.py"), "__name__": "analysis_core"}
exec(compile(_src.split("# 1. Full training-set fits")[0], "analysis.py", "exec"), _ns)
ac = SimpleNamespace(**_ns)

rng = np.random.default_rng(ac.SEED)
groups = {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"}
cp = ac.train_raw["cp"].values
rows = []
for k in ["M1", "M2"]:
    m = ac.Model(ac.SPECS[k]).fit(ac.X_train, ac.y_train)
    p = ac.expit(m.draws("alpha")[:, None] + m.draws("beta") @ ac.X_train.T)  # draws x patients
    y_rep = rng.binomial(1, p)
    for code, name in groups.items():
        mask = cp == code
        share = y_rep[:, mask].mean(axis=1)
        lo, hi = np.quantile(share, [0.025, 0.975])
        obs = ac.y_train[mask].mean()
        rows.append(
            {
                "model": k,
                "group": name,
                "patients": int(mask.sum()),
                "observed": obs,
                "predicted_mean": share.mean(),
                "lo": lo,
                "hi": hi,
                "observed_inside": bool(lo <= obs <= hi),
            }
        )
out = pd.DataFrame(rows)
out.to_csv(ac.RES / "group_ppc.csv", index=False)
print(out.round(3).to_string(index=False))
