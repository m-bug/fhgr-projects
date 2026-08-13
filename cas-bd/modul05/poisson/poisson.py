"""
Poisson-Regression auf ALLBUS 2023 (ZA8831): Anzahl Kinder einer Person.
Basiert auf pyreadstat.read_sav() zum Einlesen der SPSS .sav Datei.

Installation (falls noch nicht vorhanden):
    pip install pyreadstat pandas numpy scikit-learn matplotlib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyreadstat
from sklearn.linear_model import LinearRegression, PoissonRegressor

# ---------------------------------------------------------------
# 1. Daten einlesen
# ---------------------------------------------------------------
SAV_PATH = "ZA8831_v1-3-0.sav"

df, meta = pyreadstat.read_sav(SAV_PATH)

print(f"Eingelesen: {df.shape[0]} Fälle, {df.shape[1]} Variablen")

# Variablen nach Stichwort suchen (Name + Label), z.B. um Variablen zu finden
def find_vars(meta, keyword):
    keyword = keyword.lower()
    hits = []
    for name, label in meta.column_names_to_labels.items():
        label = label or ""
        if keyword in name.lower() or keyword in label.lower():
            hits.append((name, label))
    return hits

# print(find_vars(meta, "kind"))   # zum Explorieren, falls gewünscht

# ---------------------------------------------------------------
# 2. Relevante Variablen auswählen und aufbereiten
# ---------------------------------------------------------------
# DK11 = Anzahl Kinder im Haushalt, DK06 = Anzahl Kinder ausser Haus
# DK06 == -10 bedeutet "trifft nicht zu" (Filterfrage DK05), d.h. faktisch 0 Kinder ausser Haus,
# und ist daher KEIN echtes Missing.
#
# Hinweis: pyreadstat gibt die LANGEN Variablennamen als Spaltennamen zurück (bei ALLBUS
# kleingeschrieben, z.B. "dk05" statt "DK05"), nicht die alten 8-Zeichen-SPSS-Kurznamen.
# Deshalb wählen wir hier case-insensitiv aus, unabhängig davon wie die Datei benannt ist.
wanted = ["DK05", "DK11", "DK06", "AGE", "SEX", "EDUC", "EASTWEST"]
name_lookup = {c.upper(): c for c in df.columns}
missing = [w for w in wanted if w not in name_lookup]
if missing:
    raise KeyError(f"Variablen nicht gefunden: {missing}. Verfügbare Spalten (Auszug): {list(df.columns)[:20]}")
cols = [name_lookup[w] for w in wanted]
sub = df[cols].copy()
sub.columns = wanted  # auf einheitliche Grossschreibung zurückbenennen fürs restliche Skript

sub.loc[sub["DK06"] == -10, "DK06"] = 0

# echte ALLBUS-Missings sind negative Codes -> als NaN markieren
for c in ["DK05", "DK11", "DK06", "AGE", "SEX", "EDUC"]:
    sub[c] = sub[c].apply(lambda x: np.nan if (pd.notna(x) and x < 0) else x)

sub["n_kinder"] = sub["DK11"] + sub["DK06"]

sex_map = {1.0: "Mann", 2.0: "Frau", 3.0: "Divers"}
educ_map = {1.0: "kein_Abschluss", 2.0: "Volks_Hauptschule", 3.0: "Mittlere_Reife",
            4.0: "Fachhochschulreife", 5.0: "Hochschulreife", 6.0: "anderer_Abschluss",
            7.0: "noch_Schueler"}
east_map = {1.0: "West", 2.0: "Ost"}

sub["sex"] = sub["SEX"].map(sex_map)
sub["educ"] = sub["EDUC"].map(educ_map)
sub["region"] = sub["EASTWEST"].map(east_map)
sub["age"] = sub["AGE"]

dat = sub[["n_kinder", "age", "sex", "educ", "region"]].dropna()
dat = dat[dat["educ"] != "noch_Schueler"].copy()   # ohne abgeschlossene Bildung nicht aussagekräftig

print(f"Nach Bereinigung: n = {len(dat)}")

# ---------------------------------------------------------------
# 3. Prädiktoren kodieren (Referenzkategorien: Mann, Volks_Hauptschule, West)
# ---------------------------------------------------------------
dat["sex"] = pd.Categorical(dat["sex"], categories=["Mann", "Frau", "Divers"])
dat["region"] = pd.Categorical(dat["region"], categories=["West", "Ost"])
dat["educ"] = pd.Categorical(dat["educ"], categories=[
    "Volks_Hauptschule", "kein_Abschluss", "Mittlere_Reife",
    "Fachhochschulreife", "Hochschulreife", "anderer_Abschluss"])

X = pd.get_dummies(dat[["age", "sex", "educ", "region"]], drop_first=True).astype(float)
y = dat["n_kinder"].values

# ---------------------------------------------------------------
# 4. Modelle fitten: Lineare Regression vs. Poisson Regression
# ---------------------------------------------------------------
lm = LinearRegression().fit(X, y)
pm = PoissonRegressor(alpha=0, max_iter=500).fit(X, y)

pred_lm = lm.predict(X)
pred_pm = pm.predict(X)

mse_lm = np.mean((pred_lm - y) ** 2)
mse_pm = np.mean((pred_pm - y) ** 2)

print("\n--- Lineare Regression ---")
for name, coef in zip(X.columns, lm.coef_):
    print(f"  {name:25s} {coef:8.4f}")
print(f"  Intercept: {lm.intercept_:.4f}")
print(f"  MSE_LM: {mse_lm:.4f}")

print("\n--- Poisson Regression ---")
for name, coef in zip(X.columns, pm.coef_):
    print(f"  {name:25s} {coef:8.4f}   IRR = exp(coef) = {np.exp(coef):.4f}")
print(f"  Intercept: {pm.intercept_:.4f}")
print(f"  MSE_PM: {mse_pm:.4f}")

besser = "Poisson" if mse_pm < mse_lm else "Linear"
print(f"\nVergleich: {besser} besser  ({mse_pm:.4f} vs {mse_lm:.4f})")

# ---------------------------------------------------------------
# 5. Vorhersage für zwei fiktive Personen
# ---------------------------------------------------------------
neu = pd.DataFrame({
    "age": [30, 55],
    "sex": pd.Categorical(["Frau", "Mann"], categories=["Mann", "Frau", "Divers"]),
    "educ": pd.Categorical(["Hochschulreife", "Volks_Hauptschule"],
                            categories=["Volks_Hauptschule", "kein_Abschluss", "Mittlere_Reife",
                                        "Fachhochschulreife", "Hochschulreife", "anderer_Abschluss"]),
    "region": pd.Categorical(["West", "Ost"], categories=["West", "Ost"])
})
neuX = pd.get_dummies(neu, drop_first=True).astype(float).reindex(columns=X.columns, fill_value=0.0)

print("\nVorhersage für zwei fiktive Personen:")
print("Poisson:", np.round(pm.predict(neuX), 2))
print("Linear :", np.round(lm.predict(neuX), 2))

# ---------------------------------------------------------------
# 6. Grafiken
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].hist(y, bins=range(0, 12), color="lightgray", edgecolor="black", align="left")
axes[0].set_title("Histogram: Anzahl Kinder (ALLBUS 2023)")
axes[0].set_xlabel("Anzahl Kinder")
axes[0].set_ylabel("Häufigkeit")

axes[1].scatter(y, pred_lm, alpha=0.3, s=15, label="Linear")
axes[1].scatter(y, pred_pm, alpha=0.3, s=15, marker="x", label="Poisson")
lims = [0, max(y)]
axes[1].plot(lims, lims, "r--", linewidth=1)
axes[1].set_xlabel("Tatsächliche Anzahl Kinder")
axes[1].set_ylabel("Vorhergesagt")
axes[1].set_title("Vorhersage vs. tatsächlich")
axes[1].legend()

plt.tight_layout()
plt.savefig("m03_02_allbus_poisson.png", dpi=130)
print("\nPlot gespeichert: allbus_poisson.png")