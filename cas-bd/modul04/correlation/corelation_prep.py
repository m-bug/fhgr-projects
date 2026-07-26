import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pyreadstat

# 1. SPSS-Datei (.sav) einlesen
# Source: https://search.gesis.org/research_data/ZA10132?doi=10.4232/5.ZA10132.2.0.0
file_path = "ZA10132_v2-0-0.sav"

df, meta = pyreadstat.read_sav(file_path)
print(f"Datensatz erfolgreich geladen: {df.shape[0]} Zeilen und {df.shape[1]} Variablen.\n")

# 2. ECHTE VARIABLEN-NAMEN IN DER DATEI ANZEIGEN
# Alle Variablen von Index 20 bis 70 zu sehen (weil vorher nur Metadaten wie Studiennummer usw.) --> SKIPPEN
for var_name in list(df.columns)[20:70]:
    label = meta.column_names_to_labels.get(var_name, "Kein Label")
    print(f"Variable: {var_name:<15} | Label: {label}")


# 3. VARIABLEN FÜR DIE KORRELATION AUSWÄHLEN
selected_vars = []

if not selected_vars:
    # Automatische Fallback-Auswahl: Die ersten 5 numerischen Spalten
    selected_vars = df.select_dtypes(include=[np.number]).columns[:5].tolist()
    print(f"\n[Hinweis] Keine Variablen manuell definiert. Nutze automatisch: {selected_vars}")

available_vars = [var for var in selected_vars if var in df.columns]

# 4. Datenbereinigung (Fehlwerte / Misseingaben behandeln)
df_sub = df[available_vars].copy()

# GLES-Spezifisch: Werte < 0 (wie -99, -98, -97) stellen Fehlwerte dar
df_sub[df_sub < 0] = np.nan

# Fälle mit fehlenden Werten entfernen
df_clean = df_sub.dropna()
print(f"Verbleibende Fälle nach Bereinigung: {len(df_clean)} von {len(df)}")

# 5. Korrelationsanalyse & Visualisierung
if not df_clean.empty and df_clean.shape[1] > 0:
    corr_matrix_spearman = df_clean.corr(method='spearman')

    print("\n--- SPEARMAN KORRELATIONSMATRIX ---")
    print(corr_matrix_spearman.round(3))

    # Heatmap zeichnen
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix_spearman, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title("Korrelations-Heatmap (GLES T60W)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Pairplot zeichnen
    sns.pairplot(df_clean, diag_kind='kde')
    plt.suptitle("Paarweise Verteilung und Zusammenhänge", y=1.02)
    plt.show()
else:
    print("\nFehler: Keine gültigen Daten/Spalten für die Berechnung vorhanden.")