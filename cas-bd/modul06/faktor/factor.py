"""
Faktorenanalyse: Populismus, politisches Vertrauen, Links-Rechts, EU-Skepsis
Datensatz: GLES Querschnitt 2025, Nachwahl (ZA10100, Version 4.0.0)
Quelle:    GESIS - Leibniz-Institut fuer Sozialwissenschaften

Voraussetzungen (lokal installieren):
    pip install pyreadstat pandas numpy matplotlib seaborn factor_analyzer scikit-learn

Hinweis: Falls factor_analyzer mit neueren scikit-learn-Versionen (>=1.7)
einen TypeError bzgl. 'force_all_finite' wirft, folgendes installieren:
    pip install "scikit-learn<1.7"

Datei ZA10100_v4-0-0.sav muss im selben Verzeichnis liegen (oder Pfad unten anpassen).
"""

import inspect

import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------
# Kompatibilitaets-Patch: factor_analyzer 0.5.1 ruft check_array() noch mit
# dem Parameter 'force_all_finite' auf. scikit-learn >= 1.7 hat diesen
# Parameter in 'ensure_all_finite' umbenannt, was sonst zu
#   TypeError: check_array() got an unexpected keyword argument
#     'force_all_finite'
# fuehrt. Der Patch mappt den alten auf den neuen Parameternamen, nur
# falls das lokal installierte scikit-learn das auch tatsaechlich braucht.
# -----------------------------------------------------------------------
import sklearn.utils.validation as _skl_val

_check_array_params = inspect.signature(_skl_val.check_array).parameters
if "force_all_finite" not in _check_array_params and "ensure_all_finite" in _check_array_params:
    _orig_check_array = _skl_val.check_array

    def _check_array_compat(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _orig_check_array(*args, **kwargs)

    _skl_val.check_array = _check_array_compat
    # factor_analyzer importiert check_array direkt in seinen Namespace,
    # daher muss die Referenz auch dort ersetzt werden.
    import factor_analyzer.factor_analyzer as _fa_mod
    _fa_mod.check_array = _check_array_compat

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# -----------------------------------------------------------------------
# 1) Konfiguration
# -----------------------------------------------------------------------
SAV_PATH = "ZA10100_v4-0-0.sav"   # Pfad ggf. anpassen
N_FACTORS = 4                      # aus Kaiser-Kriterium (Eigenwert > 1)

# Ausgewaehlte Items:
#   Populismus (9 Items, GLES-Batterie q51a-q51i)
#   Politisches Vertrauen (4 Items, q79a-q79d)
#   Links-Rechts-Selbsteinstufung (q37)
#   Umverteilung / EU-Einigung (q27d, q27f)
ITEM_COLS = [
    "q51a", "q51b", "q51c", "q51d", "q51e", "q51f", "q51g", "q51h", "q51i",
    "q79a", "q79b", "q79c", "q79d",
    "q37", "q27d", "q27f",
]

# Sprechende Labels fuer Plots
LABELS = {
    "q51a": "Pop: Kompromiss=Verrat",
    "q51b": "Pop: Volk soll entscheiden",
    "q51c": "Pop: Abg. setzen Volkswillen um",
    "q51d": "Pop: Kluft Eliten/Volk",
    "q51e": "Pop: Buerger bessere Vertretung",
    "q51f": "Pop: Politiker reden zu viel",
    "q51g": "Pop: starke:r Fuehrer:in gut",
    "q51h": "Pop: Gerichte stoppen Reg.",
    "q51i": "Pop: Demokratie vorzuziehen",
    "q79a": "Vertr: Bundesregierung",
    "q79b": "Vertr: Bundestag",
    "q79c": "Vertr: Parteien",
    "q79d": "Vertr: Politiker:innen",
    "q37": "Links-Rechts (Ego)",
    "q27d": "Einkommen verringern",
    "q27f": "EU-Einigung vorantreiben",
}

# case-insensitive lookup pattern (pyreadstat liefert Spaltennamen in Kleinbuchstaben)
# hier bereits konsistent kleingeschrieben, Pattern trotzdem als Sicherheitsnetz:
def name_lookup_pattern(df):
    return {c.upper(): c for c in df.columns}


# -----------------------------------------------------------------------
# 2) Daten einlesen
# -----------------------------------------------------------------------
# WINDOWS-1252, da die .sav-Datei eine defekte/nicht-UTF8-Stringspalte enthaelt;
# usecols umgeht das Problem, da die betroffene Spalte nicht mitgelesen wird.
df_raw, meta = pyreadstat.read_sav(
    SAV_PATH, encoding="WINDOWS-1252", usecols=ITEM_COLS
)
print(f"Eingelesen: {df_raw.shape[0]} Faelle, {df_raw.shape[1]} Variablen")

name_lookup = name_lookup_pattern(df_raw)

# -----------------------------------------------------------------------
# 3) Missings bereinigen
# -----------------------------------------------------------------------
# GLES kodiert alle Missing-Typen als negative Werte (-99 .. -71)
df_clean = df_raw.copy()
for col in ITEM_COLS:
    df_clean.loc[df_clean[col] < 0, col] = np.nan

print("\nMissings pro Variable:")
print(df_clean[ITEM_COLS].isna().sum())

df_fa = df_clean[ITEM_COLS].dropna()
print(f"\nN nach Listwise Deletion: {df_fa.shape[0]}")

# -----------------------------------------------------------------------
# 4) Korrelationsmatrix + Heatmap
# -----------------------------------------------------------------------
corr = df_fa.corr()
corr_labeled = corr.rename(index=LABELS, columns=LABELS)

plt.figure(figsize=(11, 9))
sns.heatmap(
    corr_labeled, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
    square=True, cbar_kws={"label": "Korrelation"}, annot_kws={"size": 7},
)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.title("Korrelationsmatrix: Populismus, Vertrauen, Links-Rechts, EU")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()
print("\nGrafik gespeichert: heatmap.png")

# -----------------------------------------------------------------------
# 5) Faktorabilitaet pruefen: Bartlett-Test + KMO
# -----------------------------------------------------------------------
chi2, p_value = calculate_bartlett_sphericity(df_fa)
print(f"\nBartlett-Test: chi2 = {chi2:.1f}, p = {p_value:.3g}")

kmo_per_var, kmo_total = calculate_kmo(df_fa)
print(f"KMO gesamt: {kmo_total:.3f}")
kmo_series = pd.Series(kmo_per_var, index=df_fa.columns).sort_values()
print("KMO je Variable:")
print(kmo_series.round(3))

# -----------------------------------------------------------------------
# 6) Eigenwerte + Scree-Plot (Kaiser-Kriterium)
# -----------------------------------------------------------------------
eigvals, _ = np.linalg.eigh(corr.values)
eigvals = eigvals[::-1]  # absteigend sortieren

print("\nEigenwerte (absteigend):")
for i, ev in enumerate(eigvals, 1):
    print(f"  Faktor {i}: {ev:.3f}")

n_factors_kaiser = int((eigvals > 1).sum())
print(f"Anzahl Faktoren nach Kaiser-Kriterium (EW>1): {n_factors_kaiser}")

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(eigvals) + 1), eigvals, "o-", color="#1f4e79")
plt.axhline(1, color="red", linestyle="--", linewidth=1, label="Kaiser-Kriterium (EW=1)")
plt.xlabel("Faktor")
plt.ylabel("Eigenwert")
plt.title("Scree-Plot")
plt.xticks(range(1, len(eigvals) + 1))
plt.legend()
plt.tight_layout()
plt.savefig("screeplot.png", dpi=150)
plt.close()
print("Grafik gespeichert: screeplot.png")

# -----------------------------------------------------------------------
# 7) Faktorenanalyse mit Varimax-Rotation
# -----------------------------------------------------------------------
fa = FactorAnalyzer(n_factors=N_FACTORS, rotation="varimax", method="principal")
fa.fit(df_fa)

loadings = pd.DataFrame(
    fa.loadings_,
    index=[LABELS.get(c, c) for c in df_fa.columns],
    columns=[f"Faktor{i+1}" for i in range(N_FACTORS)],
)
print("\nFaktorladungen (Varimax-rotiert):")
print(loadings.round(2))

variance, prop_var, cum_var = fa.get_factor_variance()
print("\nErklaerte Varianz je Faktor:", [round(v, 3) for v in variance])
print("Anteil erklaerte Varianz je Faktor:", [round(v, 3) for v in prop_var])
print("Kumulierte erklaerte Varianz:", [round(v, 3) for v in cum_var])

loadings.to_csv("faktorladungen.csv")
print("\nLadungen gespeichert: faktorladungen.csv")

# -----------------------------------------------------------------------
# 8) Kurzinterpretation (automatisch: staerkste Ladung je Item)
# -----------------------------------------------------------------------
print("\nZuordnung Item -> staerkster Faktor:")
assign = loadings.abs().idxmax(axis=1)
for item, factor in assign.items():
    val = loadings.loc[item, factor]
    print(f"  {item:35s} -> {factor} (Ladung {val:+.2f})")