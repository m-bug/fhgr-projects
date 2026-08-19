"""
Outlier-Erkennung mit Hauptkomponentenanalyse (HKA / PCA)
===========================================================
Datensatz: World Bank Indikatoren (Wirtschaft, Preise, Arbeitsmarkt,
           Staatshaushalt, Aussenhandel) für alle Länder eines Jahres.

Methoden:
  1) Hotelling's T2  -> Ausreisser INNERHALB des vom Modell erklärten Raums
                        (ungewöhnliche Kombination der Hauptkomponenten)
  2) SPE / DmodX      -> Ausreisser AUSSERHALB des Modellraums
                        (Beobachtung, die das PCA-Modell schlecht erklärt)

Install:
    pip install wbgapi pca scikit-learn pandas matplotlib --break-system-packages


Das Thema HKA war etwas neu für mich. Deshalb wurde dieses Skript mit Hilfe von Claude Code erstellt.

wichtig: Die benötigten Files werden via API Calls heruntergeladen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pca import pca

# Alle Output-Dateien landen im selben Ordner wie dieses Script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------
# 1) KONFIGURATION
# -----------------------------------------------------------------------

# World-Bank-Indikatorcodes -> sprechende Spaltennamen
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP_growth",         # BIP-Wachstum (%)
    "NY.GDP.PCAP.CD":    "GDP_per_capita",     # BIP pro Kopf (US$)
    "FP.CPI.TOTL.ZG":    "Inflation",          # Inflation (%)
    "SL.UEM.TOTL.ZS":    "Unemployment",       # Arbeitslosenquote (%)
    "GC.NLD.TOTL.GD.ZS": "Fiscal_balance",     # Net lending/borrowing (% BIP)
    "BN.CAB.XOKA.GD.ZS": "Current_account",    # Leistungsbilanz (% BIP)
}

YEAR = 2022          # Betrachtetes Jahr (je nach Verfügbarkeit anpassen)
USE_LIVE_DATA = True  # False -> synthetischer Demo-Datensatz (z.B. ohne Internet)


# -----------------------------------------------------------------------
# 2) DATEN LADEN
# -----------------------------------------------------------------------

def load_worldbank_data(year: int) -> pd.DataFrame:
    """Lädt die Indikatoren für alle Länder (kein Aggregat wie 'World') von der World Bank API."""
    import wbgapi as wb

    df = wb.data.DataFrame(
        list(INDICATORS.keys()),
        time=year,
        labels=True,
        skipAggs=True,   # keine Regions-/Einkommensgruppen-Aggregate, nur echte Länder
    )
    df = df.rename(columns=INDICATORS)
    df = df.rename(columns={"Country": "country"})
    df.index.name = "country_code"
    df = df.reset_index()
    return df


def load_demo_data(n_countries: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Synthetischer Ersatzdatensatz mit plausiblen Verteilungen, falls kein
    Internetzugriff besteht. Enthält bewusst ein paar extreme Länder
    (z.B. Hyperinflation, Schuldenkrise), damit die Ausreisser-Erkennung
    etwas zu finden hat.
    """
    rng = np.random.default_rng(seed)
    codes = [f"C{i:03d}" for i in range(n_countries)]

    df = pd.DataFrame({
        "country_code": codes,
        "country": [f"Land_{i}" for i in range(n_countries)],
        "GDP_growth": rng.normal(2.5, 2.5, n_countries),
        "GDP_per_capita": rng.lognormal(9.3, 1.1, n_countries),
        "Inflation": rng.normal(4.0, 3.0, n_countries),
        "Unemployment": rng.normal(7.0, 4.0, n_countries).clip(min=0.5),
        "Fiscal_balance": rng.normal(-3.0, 3.0, n_countries),
        "Current_account": rng.normal(-1.0, 4.0, n_countries),
    })

    # Ein paar realistische "Ausreisser" künstlich einbauen
    outlier_idx = rng.choice(n_countries, size=4, replace=False)
    df.loc[outlier_idx[0], ["Inflation", "GDP_growth"]] = [180.0, -8.0]   # Hyperinflation/Krise
    df.loc[outlier_idx[1], ["Fiscal_balance", "Current_account"]] = [-18.0, -15.0]  # Schuldenkrise
    df.loc[outlier_idx[2], "GDP_per_capita"] = 130000  # sehr reiches Land
    df.loc[outlier_idx[3], "Unemployment"] = 32.0       # extreme Arbeitslosigkeit

    return df


if USE_LIVE_DATA:
    try:
        raw = load_worldbank_data(YEAR)
        if raw[list(INDICATORS.values())].dropna(how="all").empty:
            raise ValueError("Keine Daten von der World Bank API erhalten.")
    except Exception as e:
        print(f"[Hinweis] Live-Daten nicht verfügbar ({e}). Nutze Demo-Datensatz.")
        raw = load_demo_data()
else:
    raw = load_demo_data()

print(f"Rohdaten: {raw.shape[0]} Länder, {raw.shape[1]} Spalten")


# -----------------------------------------------------------------------
# 3) DATENAUFBEREITUNG
# -----------------------------------------------------------------------

feature_cols = list(INDICATORS.values())

# Länder mit zu vielen fehlenden Werten ausschliessen (z.B. > 1 fehlender Indikator)
df = raw.dropna(subset=feature_cols, thresh=len(feature_cols) - 1).copy()

# Verbleibende einzelne Lücken mit dem Spaltenmedian auffüllen
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

print(f"Nach Bereinigung: {df.shape[0]} Länder verbleiben "
      f"({raw.shape[0] - df.shape[0]} entfernt wegen fehlender Werte)")

# Standardisierung (Mittelwert 0, Varianz 1) ist für PCA zwingend nötig,
# da die Indikatoren stark unterschiedliche Skalen haben
# (z.B. GDP_per_capita in US$ vs. Inflation in %).
X_scaled = StandardScaler().fit_transform(df[feature_cols])
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df["country"])


# -----------------------------------------------------------------------
# 4) HKA MIT AUSREISSERERKENNUNG
# -----------------------------------------------------------------------

model = pca(
    n_components=0.95,        # so viele Komponenten, dass 95% Varianz erklärt werden
    alpha=0.05,                # Signifikanzniveau für T2 / SPE Grenzwerte
    detect_outliers=["ht2", "spe"],
    normalize=False,            # bereits oben manuell standardisiert
)

results = model.fit_transform(X_scaled)

# Ergebnis-Tabelle mit beiden Ausreisser-Kennzahlen
outlier_table = results["outliers"].copy()
outlier_table["country"] = X_scaled.index
outlier_table = outlier_table[
    ["country", "y_bool", "y_bool_spe", "y_score", "y_score_spe"]
].rename(columns={
    "y_bool": "Ausreisser_T2",
    "y_bool_spe": "Ausreisser_SPE",
    "y_score": "T2_Wert",
    "y_score_spe": "SPE_Wert",
})
outlier_table["Ausreisser_gesamt"] = (
    outlier_table["Ausreisser_T2"] | outlier_table["Ausreisser_SPE"]
)

print("\nErklärte Varianz je Hauptkomponente:")
print(results["variance_ratio"].round(3))

print(f"\nAls Ausreisser klassifizierte Länder "
      f"({outlier_table['Ausreisser_gesamt'].sum()} von {len(outlier_table)}):")
print(
    outlier_table[outlier_table["Ausreisser_gesamt"]]
    .sort_values("T2_Wert", ascending=False)
    .to_string(index=False)
)

outlier_table.to_csv(os.path.join(OUTPUT_DIR, "pca_outlier_ergebnisse.csv"), index=False)


# -----------------------------------------------------------------------
# 5) VISUALISIERUNG
# -----------------------------------------------------------------------

# (a) Biplot: Länder in PC1/PC2-Raum, Ausreisser hervorgehoben, plus Loadings der Variablen
fig, ax = model.biplot(
    legend=True,
    label=False,
    title="HKA-Biplot: Ausreissererkennung (Hotelling T2 / SPE)",
    figsize=(9, 7),
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_biplot.png"), dpi=150)
plt.close()

# (b) Scatterplot mit hervorgehobenen Ausreissern (T2 und SPE)
fig, ax = model.scatter(
    SPE=True,
    HT2=True,
    title="Ausreisser nach Hotelling T2 und SPE/DmodX",
    figsize=(9, 7),
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_outlier_scores.png"), dpi=150)
plt.close()

# (c) Scree-Plot: erklärte Varianz je Komponente (für Methodik-Slide nützlich)
fig, ax = model.plot(figsize=(7, 5))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_scree.png"), dpi=150)
plt.close()

print("\nGespeicherte Dateien:")
print("  - pca_outlier_ergebnisse.csv  (Tabelle aller Länder mit T2/SPE-Werten)")
print("  - pca_biplot.png              (Länder + Variablen-Loadings)")
print("  - pca_outlier_scores.png      (Scatterplot mit hervorgehobenen T2-/SPE-Ausreissern)")
print("  - pca_scree.png               (erklärte Varianz je Hauptkomponente)")
