"""
Missing Values: Potenzial und Gefahren der Imputation
=======================================================
Datensatz: Palmer Penguins (Gorman, Williams & Fraser 2014)
source: https://allisonhorst.github.io/palmerpenguins/

Vorgehen: Originalwerte löschen und mit verschiedenen Ansätzen "ersetzen" (Imputation):
  1. Zeilenweise Loeschung (Baseline, kein Imputationswert)
  2. Mean Replacement
  3. Regression-Imputation (linear, auf uebrigen numerischen +
     kategorialen Praediktoren)
  4. Nearest-Neighbor-Imputation (KNNImputer, sklearn)

Die Regression-Imputation ud die Nearest-Neighbor-Imputation wurden mit Claude-AI erstellt.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# patch fuer scikit-learn >= 1.7 Kompatibilitaet --> meine Version ist zu neu, mit diesem Fix läuft das script
# todo: fix.. oder downgrade..
try:
    from sklearn.utils import check_array as _check_array_orig

    def check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _check_array_orig(*args, **kwargs)

    import sklearn.utils

    sklearn.utils.check_array = check_array
except Exception:
    pass

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. Daten laden und "Wahrheit" definieren (Complete Cases)
# ---------------------------------------------------------------
raw = sns.load_dataset("penguins")
print(f"Rohdaten: {raw.shape[0]} Beobachtungen, "
      f"{raw.isna().any(axis=1).sum()} davon mit mind. einer echten Luecke.")

truth = raw.dropna().reset_index(drop=True)
print(f"Complete Cases (= 'Wahrheit' fuer die Simulation): {truth.shape[0]}")

TARGET = "body_mass_g"
true_values = truth[TARGET].copy()
n = len(truth)

# ---------------------------------------------------------------
# 2. Fehlende Werte simulieren: MCAR und MAR
# ---------------------------------------------------------------

# 20% der 333 vollständigen Beobachtungen sollen im Feld body_mass_g auf NaN gesetzt werden fuer die Simulation..
SHARE_MISSING = 0.20

# MCAR: rein zufaellige Auswahl
mcar_mask = RNG.random(n) < SHARE_MISSING

# MAR: Wahrscheinlichkeit fuer Missing steigt mit flipper_length_mm
# (z.B. grosse Tiere waren schwieriger zu wiegen/fangen/vermessen -> haengt von einer
# BEOBACHTETEN Variable ab, nicht vom Gewicht selbst)
flipper = truth["flipper_length_mm"]
prob = (flipper - flipper.min()) / (flipper.max() - flipper.min())
prob = prob * SHARE_MISSING * 2.2  # skaliert auf im Schnitt ~SHARE_MISSING
mar_mask = RNG.random(n) < prob

print(f"MCAR: {mcar_mask.sum()} Luecken ({mcar_mask.mean():.1%})")
print(f"MAR:  {mar_mask.sum()} Luecken ({mar_mask.mean():.1%})")


def make_missing(mask):
    df = truth.copy()
    df.loc[mask, TARGET] = np.nan
    return df


data_mcar = make_missing(mcar_mask)
data_mar = make_missing(mar_mask)

# ---------------------------------------------------------------
# 3. Imputationsmethoden
# ---------------------------------------------------------------
NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
CATEGORICAL = ["species", "island", "sex"]


def impute_mean(df):
    out = df.copy()
    out[TARGET] = out[TARGET].fillna(out[TARGET].mean())
    return out


def impute_regression(df):
    out = df.copy()
    known = out[out[TARGET].notna()]
    unknown = out[out[TARGET].isna()]
    if unknown.empty:
        return out

    enc = OneHotEncoder(drop="first", sparse_output=False)
    cat_known = enc.fit_transform(known[CATEGORICAL])
    cat_unknown = enc.transform(unknown[CATEGORICAL])

    X_known = np.hstack([known[NUMERIC].values, cat_known])
    X_unknown = np.hstack([unknown[NUMERIC].values, cat_unknown])

    model = LinearRegression().fit(X_known, known[TARGET])
    out.loc[out[TARGET].isna(), TARGET] = model.predict(X_unknown)
    return out


def impute_knn(df, k=5):
    out = df.copy()
    enc = OneHotEncoder(sparse_output=False)
    cat_encoded = enc.fit_transform(out[CATEGORICAL])
    feature_matrix = np.hstack([out[NUMERIC].values, cat_encoded,
                                 out[[TARGET]].values])
    imputer = KNNImputer(n_neighbors=k)
    imputed = imputer.fit_transform(feature_matrix)
    out[TARGET] = imputed[:, -1]
    return out


def listwise_deletion_rmse(df):
    # Baseline: Zeilen mit Luecke werden verworfen -> es gibt fuer
    # diese Beobachtungen gar keinen Schaetzwert. RMSE ist daher hier
    # nicht direkt vergleichbar; ich zeige stattdessen den
    # Informationsverlust (Anteil verlorener Zeilen) getrennt auf.
    remaining = df[TARGET].notna().sum()
    lost = df[TARGET].isna().sum()
    return remaining, lost


# ---------------------------------------------------------------
# 4. Auswertung: RMSE gegen wahre Werte
# ---------------------------------------------------------------
def rmse(df, mask):
    pred = df.loc[mask, TARGET].values
    true = true_values.loc[mask].values
    return float(np.sqrt(np.mean((pred - true) ** 2)))


results = []
for mechanism, data, mask in [("MCAR", data_mcar, mcar_mask),
                               ("MAR", data_mar, mar_mask)]:
    mean_df = impute_mean(data)
    reg_df = impute_regression(data)
    knn_df = impute_knn(data)

    results.append({"Mechanismus": mechanism, "Methode": "Mean Replacement",
                     "RMSE": rmse(mean_df, mask),
                     "Bias_Mittelwert": mean_df[TARGET].mean() - true_values.mean()})
    results.append({"Mechanismus": mechanism, "Methode": "Regression",
                     "RMSE": rmse(reg_df, mask),
                     "Bias_Mittelwert": reg_df[TARGET].mean() - true_values.mean()})
    results.append({"Mechanismus": mechanism, "Methode": "KNN (k=5)",
                     "RMSE": rmse(knn_df, mask),
                     "Bias_Mittelwert": knn_df[TARGET].mean() - true_values.mean()})

    remaining, lost = listwise_deletion_rmse(data)
    results.append({"Mechanismus": mechanism, "Methode": "Listwise Deletion",
                     "RMSE": np.nan,
                     "Bias_Mittelwert": data[TARGET].mean() - true_values.mean()})
    print(f"[{mechanism}] Listwise Deletion verliert {lost} von {n} Zeilen "
          f"({lost / n:.1%}).")

results_df = pd.DataFrame(results)
print("\nErgebnistabelle:")
print(results_df.round(2).to_string(index=False))

# ---------------------------------------------------------------
# 5. Grafik 1: RMSE pro Methode (MCAR vs. MAR)
# ---------------------------------------------------------------
sns.set_style("whitegrid")
plot_df = results_df.dropna(subset=["RMSE"])  # Listwise hat keinen RMSE

fig, ax = plt.subplots(figsize=(7, 4.2))
sns.barplot(data=plot_df, x="Methode", y="RMSE", hue="Mechanismus",
            ax=ax, palette=["#4C72B0", "#C44E52"])
ax.set_title("Fehler der Imputation je Methode (Palmer Penguins, body_mass_g)")
ax.set_ylabel("RMSE zum wahren Wert [g]")
ax.set_xlabel("")
ax.legend(title="Missing-Mechanismus")
fig.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "m06_03_penguins_mv_rmse.png")
fig.savefig(fig1_path, dpi=150)
plt.close(fig)
print(f"\nGrafik 1 gespeichert: {fig1_path}")

# ---------------------------------------------------------------
# 6. Grafik 2: Bias des Mittelwerts pro Methode (MCAR vs. MAR)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
order = ["Listwise Deletion", "Mean Replacement", "Regression", "KNN (k=5)"]
sns.barplot(data=results_df, x="Methode", y="Bias_Mittelwert", hue="Mechanismus",
            order=order, ax=ax, palette=["#4C72B0", "#C44E52"])
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Verzerrung des Mittelwerts nach Imputation")
ax.set_ylabel("Bias [g] (Imputiert - Wahrer Mittelwert)")
ax.set_xlabel("")
ax.legend(title="Missing-Mechanismus")
fig.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "m06_03_penguins_mv_bias.png")
fig.savefig(fig2_path, dpi=150)
plt.close(fig)
print(f"Grafik 2 gespeichert: {fig2_path}")

print(f"\nWahrer Mittelwert body_mass_g: {true_values.mean():.1f} g")
