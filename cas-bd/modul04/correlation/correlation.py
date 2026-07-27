import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat

# 1. SPSS-Datei (.sav) einlesen
# Source: https://search.gesis.org/research_data/ZA10132?doi=10.4232/5.ZA10132.2.0.0
file_path = "ZA10132_v2-0-0.sav"
df, meta = pyreadstat.read_sav(file_path)

# 2. VARIABLEN AUSWÄHLEN (kommen aus dem script corelation_prep.py)
selected_vars = [
    't1086',  # Wahlkampf: Interessantheit
    't1067',  # Wahlergebnis: Zufriedenheit
    't1071',  # Wahlentscheidung: Schwierigkeit
    't1072',  # Wahlentscheidung: Zufriedenheit
    't232c',  # Koalitions-Skalometer: CDU/CSU + SPD
    't232d'   # Koalitions-Skalometer: CDU/CSU + GRÜNE
]

print("--- VARIABLEN-ÜBERSICHT ---")
for var in selected_vars:
    label = meta.column_names_to_labels.get(var, "Keine Beschreibung")
    print(f"Variable: {var:<10} | Label: {label}")

# 3. Datenbereinigung
df_sub = df[selected_vars].copy()

# GLES-Fehlwerte (negative Zahlen wie -99, -98, -97) durch NaN ersetzen
df_sub[df_sub < 0] = np.nan

# Nur Vollständige Fälle behalten
df_clean = df_sub.dropna()
print(f"\nVerbleibende Fälle nach Bereinigung: {len(df_clean)} von {len(df)}")

# 4. Spearman-Korrelation berechnen (ideal für Ordinalskalen/Umfragedaten)
corr_matrix = df_clean.corr(method='spearman')

print("\n--- SPEARMAN KORRELATIONSMATRIX ---")
print(corr_matrix.round(3))

# 5. Visualisierung
plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
plt.title("Korrelations-Heatmap (GLES T60W - BTW 2025)", fontsize=14)
plt.tight_layout()
plt.show()

# Paarweise Streudiagramme
sns.pairplot(df_clean, diag_kind='kde')
plt.suptitle("Paarweise Verteilungen & Zusammenhänge", y=1.02)
plt.show()