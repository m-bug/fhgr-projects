import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Getting started:
# pip install pandas numpy matplotlib seaborn statsmodels


# 1. Lokale CSV-Datei einlesen
# source: https://www.kaggle.com/datasets/swaptr/fifa-wc-2026-players
csv_dateipfad = "players.csv"

try:
    df = pd.read_csv(csv_dateipfad)
    print(f"Erfolgreich geladen: {len(df)} Zeilen aus '{csv_dateipfad}' eingelesen.\n")
except FileNotFoundError:
    print(f"Fehler: Die Datei '{csv_dateipfad}' wurde nicht gefunden. Generiere synthetische Testdaten...")
    np.random.seed(42)
    n_samples = 400
    # Simulation mit englischen Original-Kürzeln für den Fallback
    positionen_roh = np.random.choice(["GK", "DF", "MF", "FW"], size=n_samples, p=[0.1, 0.35, 0.35, 0.2])
    alter = [int(np.random.normal(29.5, 3.5)) if p == "GK" else int(np.random.normal(26.8, 3.0)) for p in positionen_roh]
    df = pd.DataFrame({"position": positionen_roh, "age": alter, "player": [f"Spieler_{i}" for i in range(n_samples)]})

if not df.empty:
    # 2. Datenbereinigung & Vorbereitung
    df = df.dropna(subset=["age", "position"])
    
    # Alter in Ganzzahl konvertieren (falls Format wie "27-023" vorliegt)
    if df["age"].dtype == object:
        df["Alter"] = df["age"].astype(str).str.split("-").str[0].astype(int)
    else:
        df["Alter"] = df["age"].astype(int)
        
    # --- Mapping der Abkürzungen auf lesbare deutsche Begriffe für die grafik für die slides ---
    # alle Varianten (Gross-/Kleinschreibung) abfangen
    positions_mapping = {
        "GK": "Torhüter",
        "gk": "Torhüter",
        "DF": "Verteidiger",
        "df": "Verteidiger",
        "MF": "Mittelfeld",
        "mf": "Mittelfeld",
        "FW": "Angriff",
        "fw": "Angriff"
    }
    
    # Zuweisung der neuen Spalte und Bereinigung von Leerzeichen
    df["Position_Roh"] = df["position"].astype(str).str.strip()
    df["Position"] = df["Position_Roh"].map(positions_mapping)
    
    # Falls es Positionen gibt, die nicht im Mapping waren wibehalten =>Originalwert
    df["Position"] = df["Position"].fillna(df["Position_Roh"])
    
    # Duplikate löschen
    zeilen_vorher = len(df)
    df = df.drop_duplicates()
    if len(df) < zeilen_vorher:
        print(f"Bereinigung: {zeilen_vorher - len(df)} Duplikate wurden gelöscht.\n")

    # 3. Statistik mit den neuen Begriffen
    print("--- Statistik: Durchschnittsalter nach Position ---")
    summary_stats = df.groupby("Position")["Alter"].agg(["count", "mean", "std", "min", "max"])
    # Sortierung für eine logische Reihenfolge im Output (von hinten nach vorne im Spielfeld)
    reihenfolge = ["Torhüter", "Verteidiger", "Mittelfeld", "Angriff"]
    summary_stats = summary_stats.reindex([p for p in reihenfolge if p in summary_stats.index])
    print(summary_stats.round(2).to_string())
    print("\n")

    # 4. ANOVA (Varianzanalyse)
    print("--- Statistische Testung: ANOVA (Varianzanalyse) ---")
    modell = ols("Alter ~ C(Position)", data=df).fit()
    anova_tabelle = sm.stats.anova_lm(modell, typ=2)
    print(anova_tabelle)
    
    p_wert_anova = anova_tabelle["PR(>F)"].iloc[0]
    print(f"\nErgebnis ANOVA p-Wert: {p_wert_anova:.5f}")
    
    if p_wert_anova < 0.05:
        print("Das Ergebnis ist STATISTISCH SIGNIFICANT (p < 5%).")
        print("Wir führen einen Tukey-HSD Post-Hoc-Test durch.\n")
        
        # 5. Post-Hoc-Test (Tukey HSD) - jetzt ebenfalls mit lesbaren Gruppen
        tukey = pairwise_tukeyhsd(endog=df["Alter"], groups=df["Position"], alpha=0.05)
        print("--- Paarweiser Vergleich (Tukey HSD) ---")
        print(tukey.summary())
    else:
        print("Das Ergebnis ist NICHT SIGNIFICANT (p >= 5%).\n")

    # 6. Visualisierung Boxplot & Jitter
    plt.figure(figsize=(11, 7))
    
    # Definierte Reihenfolge für die x-Achse festlegen
    vorhandene_gruppen = [p for p in reihenfolge if p in df["Position"].unique()]
    
    sns.boxplot(
        x="Position", 
        y="Alter", 
        data=df, 
        order=vorhandene_gruppen,
        palette="Set2", 
        fliersize=0
    )
    
    sns.stripplot(
        x="Position", 
        y="Alter", 
        data=df, 
        order=vorhandene_gruppen,
        color="black", 
        alpha=0.25, 
        jitter=0.18, 
        size=4
    )
    
    # Highlight prominenten Ausreisser (z. B. Ronaldo)
    ronaldo = df[df["player"].astype(str).str.contains("Ronaldo", case=False, na=False)]
    if not ronaldo.empty:
        plt.scatter(
            ronaldo["Position"], 
            ronaldo["Alter"], 
            color="gold", 
            edgecolor="black", 
            s=180, 
            zorder=5, 
            label="Ausreisser (z. B. CR7)"
        )
        for idx, row in ronaldo.iterrows():
            plt.text(
                row["Position"], 
                row["Alter"] + 0.5, 
                f"{row['player']} ({row['Alter']})", 
                fontsize=10, 
                fontweight="bold", 
                ha="center", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
            )
        plt.legend()

    # Beschriftungen komplett auf Deutsch und verständlich
    plt.title(
        "Vergleich der Altersstruktur nach Spielerpositionen bei der WM",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    plt.xlabel("Position auf dem Spielfeld (Kategorische Ursache)", fontsize=11)
    plt.ylabel("Alter des Spielers in Jahren (Numerische Wirkung)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig("m03_01_wm_player_age.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    
    plt.show()