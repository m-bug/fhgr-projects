import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------
# 1. DATEN EINLESEN
# source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
# ---------------------------------------------------------
df = pd.read_csv("spotify_dataset.csv")

# Songdauer von Millisekunden in Minuten umrechnen (bessere Lesbarkeit)
df["duration_min"] = df["duration_ms"] / 60000

# Zu untersuchende Variablen
vars_to_analyze = ["tempo", "duration_min", "danceability", "popularity"]

# ---------------------------------------------------------
# 2. DESKRIPTIVE KENNZAHLEN BEWERTEN
# ---------------------------------------------------------
print("=" * 60)
print("DESKRIPTIVE STATISTIK DER VERTEILUNGEN")
print("=" * 60)

for var in vars_to_analyze:
    data = df[var].dropna()

    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    skew_val = data.skew()  # Schiefe (>0: rechtsschief, <0: linksschief)
    kurt_val = data.kurt()  # Wölbung / Steilheit

    # Kolmogorov-Smirnov-Test auf Normalverteilung
    # (Bei sehr grossen Datensätzen N > 5000 ist K-S robuster als Shapiro-Wilk)
    stat, p_value = stats.kstest(
        data, "norm", args=(mean_val, std_val)
    )

    print(f"\nVariabel: {var.upper()}")
    print(f"  - Mittelwert (Mean): {mean_val:.2f}")
    print(f"  - Median:            {median_val:.2f}")
    print(f"  - Standardabw. (Std):{std_val:.2f}")
    print(
        f"  - Schiefe (Skewness):{skew_val:.2f} ({'Rechtsschief' if skew_val > 0.5 else 'Linksschief' if skew_val < -0.5 else 'Nahezu symmetrisch'})"
    )
    print(f"  - K-S Test p-Wert:   {p_value:.4e}")
    if p_value < 0.05:
        print("    -> H0 verworfen: Die Variable ist NICHT normalverteilt.")
    else:
        print("    -> H0 nicht verworfen: Die Variable ist näherungsweise normalverteilt.")

# ---------------------------------------------------------
# 3. VISUALISIERUNG IM 4-RASTER-DIAGRAMM (Beispiel für 'tempo')
# ---------------------------------------------------------
target_var = "tempo"
data = df[target_var].dropna()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    f"Verteilungsanalyse von '{target_var}' (Spotify Tracks)",
    fontsize=16,
    fontweight="bold",
)

# A) Histogramm mit Dichtekurve (KDE)
sns.histplot(
    data, kde=True, ax=axes[0, 0], color="skyblue", bins=30, stat="density"
)
axes[0, 0].axvline(
    data.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mittelwert ({data.mean():.1f})",
)
axes[0, 0].axvline(
    data.median(),
    color="green",
    linestyle="-",
    linewidth=2,
    label=f"Median ({data.median():.1f})",
)
axes[0, 0].set_title("1. Histogramm & Dichteschätzung (KDE)")
axes[0, 0].legend()

# B) Boxplot (Ausreisser-Identifikation)
sns.boxplot(x=data, ax=axes[0, 1], color="lightgreen")
axes[0, 1].set_title("2. Boxplot (Quantile & Ausreisser)")

# C) Q-Q Plot (Quantile-Quantile Plot gegen Normalverteilung)
stats.probplot(data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("3. Q-Q Plot (Normalverteilungs-Abgleich)")

# D) Empirische kumulative Verteilungsfunktion (ECDF)
sns.ecdfplot(data, ax=axes[1, 1], color="purple")
axes[1, 1].set_title("4. Kumulierte Verteilung (ECDF)")
axes[1, 1].set_ylabel("Kumulierter Anteil")

plt.tight_layout()
plt.savefig("04_03_verteilung.png", bbox_inches="tight")
plt.show()