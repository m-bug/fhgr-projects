"""
E-Commerce Analyse: Einfluss der Verweildauer auf den Umsatz
Datenquelle: https://www.kaggle.com/datasets/kzmontage/e-commerce-website-logs

**Disclaimer**: Dieser Code wurde mit der Hilfe von claude.ai erstellt. 

Struktur:
1. Daten einlesen & Qualitätscheck
2. Deskriptive Statistik
3. Naive lineare Regression (Baseline, wie im Ursprungs-Script)
4. Zweistufiges Modell: Konversion (Logit) + Ausgabehöhe bei Käufern (OLS)
5. Log-transformierte Regression (schiefe Verteilung)
6. Nichtlinearität: Spearman-Korrelation + polynomialer Term
7. Residuen-Diagnostik (Q-Q Plot, Residuen vs. Fitted)
8. Visualisierung für die Präsentations-Folie
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
DATA_PATH = 'e-commerce-website-logs.csv'
DPI = 300
DURATION_COL = 'duration_(secs)'
SALES_COL = 'sales'

sns.set_style('whitegrid')

# ---------------------------------------------------------------------------
# 1. Daten einlesen & Qualitätscheck
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

print('=' * 80)
print('1. DATENQUALITÄT')
print('=' * 80)

# Spalte mit gemischten Typen identifizieren (verursacht die DtypeWarning)
mixed_col = df.columns[6]
print(f"\nTypen in Spalte '{mixed_col}':")
print(df[mixed_col].apply(type).value_counts())

print(f"\nFehlende Werte in relevanten Spalten:")
print(df[[DURATION_COL, SALES_COL]].isna().sum())

print(f"\nDeskriptive Statistik:")
print(df[[DURATION_COL, SALES_COL]].describe())

anteil_nullkaeufe = (df[SALES_COL] == 0).mean()
print(f"\nAnteil Zeilen mit sales == 0: {anteil_nullkaeufe:.1%}")

# Fehlende Werte in den relevanten Spalten entfernen
df = df.dropna(subset=[DURATION_COL, SALES_COL]).copy()

# ---------------------------------------------------------------------------
# 2. Naive lineare Regression (Baseline — Ursprungs-Ansatz)
# ---------------------------------------------------------------------------
print('\n' + '=' * 80)
print('2. BASELINE: Lineare Regression über ALLE Zeilen (inkl. Nullkäufe)')
print('=' * 80)

X = sm.add_constant(df[DURATION_COL])
model_baseline = sm.OLS(df[SALES_COL], X).fit()
print(model_baseline.summary())

# Robuste Standardfehler zum Vergleich (bei schiefen Daten oft aussagekräftiger)
model_baseline_robust = sm.OLS(df[SALES_COL], X).fit(cov_type='HC3')
print("\n--- Vergleich mit robusten Standardfehlern (HC3) ---")
print(model_baseline_robust.summary().tables[1])

# ---------------------------------------------------------------------------
# 3. Zweistufiges Modell: Konversion + Ausgabehöhe
# ---------------------------------------------------------------------------
print('\n' + '=' * 80)
print('3. ZWEISTUFIGES MODELL')
print('=' * 80)

df['converted'] = (df[SALES_COL] > 0).astype(int)

print("\n--- Stufe 1: Logistische Regression (Kauf ja/nein) ---")
logit_model = smf.logit(f'converted ~ Q("{DURATION_COL}")', data=df).fit(disp=0)
print(logit_model.summary())

print("\n--- Stufe 2: OLS nur auf Käufer (Ausgabehöhe) ---")
df_buyers = df[df[SALES_COL] > 0].copy()
X_buyers = sm.add_constant(df_buyers[DURATION_COL])
model_buyers = sm.OLS(df_buyers[SALES_COL], X_buyers).fit()
print(model_buyers.summary())

# ---------------------------------------------------------------------------
# 4. Log-transformierte Regression (schiefe Verteilung)
# ---------------------------------------------------------------------------
print('\n' + '=' * 80)
print('4. LOG-TRANSFORMIERTE REGRESSION (nur Käufer)')
print('=' * 80)

df_buyers['log_sales'] = np.log1p(df_buyers[SALES_COL])
model_log = sm.OLS(df_buyers['log_sales'], X_buyers).fit()
print(model_log.summary())

# ---------------------------------------------------------------------------
# 5. Nichtlinearität prüfen
# ---------------------------------------------------------------------------
print('\n' + '=' * 80)
print('5. NICHTLINEARITÄT')
print('=' * 80)

corr, p_value = spearmanr(df[DURATION_COL], df[SALES_COL])
print(f"\nSpearman-Korrelation (alle Zeilen): r={corr:.3f}, p={p_value:.3f}")

corr_b, p_value_b = spearmanr(df_buyers[DURATION_COL], df_buyers[SALES_COL])
print(f"Spearman-Korrelation (nur Käufer):  r={corr_b:.3f}, p={p_value_b:.3f}")

df['duration_sq'] = df[DURATION_COL] ** 2
X_poly = sm.add_constant(df[[DURATION_COL, 'duration_sq']])
model_poly = sm.OLS(df[SALES_COL], X_poly).fit()
print("\n--- Polynomiales Modell (quadratischer Term) ---")
print(model_poly.summary().tables[1])

# ---------------------------------------------------------------------------
# 6. Residuen-Diagnostik für das Baseline-Modell
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sm.qqplot(model_baseline.resid, line='45', ax=axes[0])
axes[0].set_title('Q-Q Plot der Residuen (Baseline-Modell)')

axes[1].scatter(model_baseline.fittedvalues, model_baseline.resid, alpha=0.3, s=10)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Residuen')
axes[1].set_title('Residuen vs. Fitted Values')

plt.tight_layout()
plt.savefig('residual_diagnostics.png', dpi=DPI, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# 7. Visualisierung für die Präsentations-Folie
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Links: alle Daten (Baseline)
sns.regplot(
    data=df, x=DURATION_COL, y=SALES_COL, ax=axes[0],
    scatter_kws={'s': 30, 'alpha': 0.4, 'color': '#1f77b4'},
    line_kws={'color': 'red', 'linewidth': 2},
)
axes[0].set_title('Alle Sessions (inkl. Nullkäufe)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Verweildauer (Sekunden)')
axes[0].set_ylabel('Umsatz')

# Rechts: nur Käufer, log-skalierter Umsatz
sns.regplot(
    data=df_buyers, x=DURATION_COL, y='log_sales', ax=axes[1],
    scatter_kws={'s': 30, 'alpha': 0.4, 'color': '#2ca02c'},
    line_kws={'color': 'red', 'linewidth': 2},
)
axes[1].set_title('Nur Käufer, log(Umsatz)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Verweildauer (Sekunden)')
axes[1].set_ylabel('log(1 + Umsatz)')

fig.suptitle('Einfluss der Verweildauer auf den Umsatz', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_verweildauer_sales.png', dpi=DPI, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# 8. Kurzes Fazit in der Konsole
# ---------------------------------------------------------------------------
print('\n' + '=' * 80)
print('ZUSAMMENFASSUNG')
print('=' * 80)
print(f"""
Baseline (alle Zeilen):        R² = {model_baseline.rsquared:.4f}, p = {model_baseline.pvalues[DURATION_COL]:.4f}
Logit (Konversion):             p = {logit_model.pvalues[f'Q("{DURATION_COL}")']:.4f}
OLS nur Käufer:                 R² = {model_buyers.rsquared:.4f}, p = {model_buyers.pvalues[DURATION_COL]:.4f}
OLS log(sales), nur Käufer:     R² = {model_log.rsquared:.4f}, p = {model_log.pvalues[DURATION_COL]:.4f}
Spearman (alle Zeilen):         r = {corr:.4f}, p = {p_value:.4f}
Spearman (nur Käufer):          r = {corr_b:.4f}, p = {p_value_b:.4f}

Interpretationshilfe: Ist p < 0.05 in mehreren Modellen konsistent nicht
signifikant, spricht das gegen die Hypothese "längere Verweildauer -> mehr
Umsatz" -- zumindest als einfacher linearer Zusammenhang in diesem Datensatz.
""")