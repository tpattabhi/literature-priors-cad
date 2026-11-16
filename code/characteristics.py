"""Patient characteristics by center (supplementary table for TRIPOD+AI items 20b and 20c).

Usage (from this folder): python characteristics.py  -> ../results/characteristics.csv
Cleveland: all 303 patients and the 242/61 training/test split of analysis.py. Hungarian and Long Beach VA:
the complete cases used in transfer.py (age, sex, chest-pain type, exercise angina, ST depression present).
Z-Alizadeh Sani: all 303 patients (age, sex and chest-pain type only; extend.py).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
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


def load(fname, complete=False):
    d = pd.read_csv(DATA / fname, header=None, names=COLUMNS, na_values="?")
    if complete:
        d = d.dropna(subset=["age", "sex", "cp", "exang", "oldpeak"])
    d["disease"] = (d["num"] > 0).astype(int)
    return d


def describe(d):
    return {
        "patients": len(d),
        "CAD, n (%)": f"{d.disease.sum()} ({100 * d.disease.mean():.0f})",
        "age, mean (SD)": f"{d.age.mean():.1f} ({d.age.std():.1f})",
        "male, n (%)": f"{int(d.sex.sum())} ({100 * d.sex.mean():.0f})",
        "typical angina, n (%)": f"{(d.cp == 1).sum()} ({100 * (d.cp == 1).mean():.0f})",
        "atypical angina, n (%)": f"{(d.cp == 2).sum()} ({100 * (d.cp == 2).mean():.0f})",
        "nonanginal pain, n (%)": f"{(d.cp == 3).sum()} ({100 * (d.cp == 3).mean():.0f})",
        "asymptomatic, n (%)": f"{(d.cp == 4).sum()} ({100 * (d.cp == 4).mean():.0f})",
        "exercise angina, n (%)": f"{int(d.exang.sum())} ({100 * d.exang.mean():.0f})",
        "ST depression mm, mean (SD)": f"{d.oldpeak.mean():.2f} ({d.oldpeak.std():.2f})",
        "fluoroscopy recorded, n": int(d.ca.notna().sum()),
        "thallium recorded, n": int(d.thal.notna().sum()),
    }


cle = load("processed.cleveland.data")
tr, te = train_test_split(cle, test_size=0.2, stratify=cle["disease"], random_state=2026)  # as in analysis.py
cols = {
    "Cleveland, all": cle,
    "Cleveland, training": tr,
    "Cleveland, test": te,
    "Hungarian (complete cases)": load("processed.hungarian.data", True),
    "Long Beach VA (complete cases)": load("processed.va.data", True),
}
zas = pd.read_csv(DATA / "z-alizadeh-sani.csv")
cp = pd.Series(4.0, index=zas.index)
cp[zas.Nonanginal.str.strip() == "Y"] = 3
cp[zas.Atypical.str.strip() == "Y"] = 2
cp[zas["Typical Chest Pain"].astype(float) == 1] = 1
z = pd.DataFrame(
    {
        "age": zas.Age.astype(float),
        "sex": (zas.Sex.str.strip() == "Male").astype(float),
        "cp": cp,
        "disease": (zas.Cath.str.strip() == "Cad").astype(int),
    }
)
out = pd.DataFrame({k: describe(v) for k, v in cols.items()})
zrow = describe(z.assign(exang=float("nan"), oldpeak=float("nan"), ca=float("nan"), thal=float("nan")))
zrow.update(
    {
        "exercise angina, n (%)": "not recorded",
        "ST depression mm, mean (SD)": "not recorded",
        "fluoroscopy recorded, n": 0,
        "thallium recorded, n": 0,
    }
)
out["Z-Alizadeh Sani"] = pd.Series(zrow)
out.to_csv(HERE.parent / "results" / "characteristics.csv")
print(out.to_string())
