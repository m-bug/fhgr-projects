import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

###
# Wieso ein zweites Script? Ich kenne hier aus dem ersten Skript bereits die interessanten Variblen für
# die Folien. Deshalb dient dieses Script der Auswertung der Variablen: Tempo und Duration.
###


# 1. Daten laden
# source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
df = pd.read_csv("spotify_dataset.csv")
df["duration_min"] = df["duration_ms"] / 60000

# Styling-Einstellungen  die folien..
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "figure.autolayout": True})

# =========================================================
# DIAGRAMM 1: TEMPO - Histogramm & KDE
# =========================================================
plt.figure(figsize=(7, 5))
ax1 = sns.histplot(
    df["tempo"],
    kde=True,
    color="skyblue",
    bins=30,
    stat="density",
    edgecolor="none",
)
plt.axvline(
    df["tempo"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mittelwert ({df['tempo'].mean():.2f})",
)
plt.axvline(
    df["tempo"].median(),
    color="green",
    linestyle="-",
    linewidth=2,
    label=f"Median ({df['tempo'].median():.2f})",
)

plt.title(
    "TEMPO: Symmetrische Verteilung\nMittelwert ≈ Median (Skewness: 0.23)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Tempo (BPM)")
plt.ylabel("Dichte")
plt.legend(frameon=True, facecolor="white")
plt.savefig("m04_03_01_tempo_histogramm.png", dpi=300)
plt.show()

# =========================================================
# DIAGRAMM 2: TEMPO - Q-Q Plot
# =========================================================
fig, ax2 = plt.subplots(figsize=(7, 5))
stats.probplot(df["tempo"].dropna(), dist="norm", plot=ax2)

plt.title(
    "TEMPO: Q-Q Plot\n(Punkte folgen gut der roten Ideallinie)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Theoretische Quantile (Normalverteilung)")
plt.ylabel("Empirische Quantile (BPM)")
plt.savefig("m04_03_02_tempo_qqplot.png", dpi=300)
plt.show()

# =========================================================
# DIAGRAMM 3: DURATION - Histogramm & KDE
# =========================================================
plt.figure(figsize=(7, 5))
ax3 = sns.histplot(
    df["duration_min"],
    kde=True,
    color="salmon",
    bins=50,
    stat="density",
    edgecolor="none",
)
plt.axvline(
    df["duration_min"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mittelwert ({df['duration_min'].mean():.2f} min)",
)
plt.axvline(
    df["duration_min"].median(),
    color="green",
    linestyle="-",
    linewidth=2,
    label=f"Median ({df['duration_min'].median():.2f} min)",
)

plt.xlim(0, 12)  # Achrung: Hier mache ich einen Shrink auf 0-12min. Der Rest ist nicht relevant..
plt.title(
    "DURATION: Extrem Rechtsschief\nAusreisser ziehen Mittelwert nach rechts (Skewness: 11.20)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Dauer (Minuten)")
plt.ylabel("Dichte")
plt.legend(frameon=True, facecolor="white")
plt.savefig("m04_03_03_duration_histogramm.png", dpi=300)
plt.show()

# =========================================================
# DIAGRAMM 4: DURATION - Q-Q Plot
# =========================================================
fig, ax4 = plt.subplots(figsize=(7, 5))
stats.probplot(df["duration_min"].dropna(), dist="norm", plot=ax4)

plt.title(
    "DURATION: Q-Q Plot\n(Starke Abweichungen an den Rändern / Heavy Tails)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Theoretische Quantile (Normalverteilung)")
plt.ylabel("Empirische Quantile (Minuten)")
plt.savefig("m04_03_04_duration_qqplot.png", dpi=300)
plt.show()