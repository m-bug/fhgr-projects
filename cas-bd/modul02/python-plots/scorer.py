import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Datenset einlesen
# Link: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?select=shootouts.csv
df = pd.read_csv("goalscorers.csv")

# 2. Datum in echtes Datetime-Format umwandeln
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# 3. Filter für WM-Jahre (Historische Jahre aller bisherigen Weltmeisterschaften)
world_cup_years = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 
    1978, 1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022
]

df_wm = df[df["year"].isin(world_cup_years)].copy()

# Da im Juni/Juli (und Nov/Dez für 2022) die WMs stattfanden, filtern wir darauf,
# um eventuelle Qualifikations- oder Freundschaftsspiele im Frühjahr/Herbst auszuschließen.
df_wm = df_wm[df_wm["month"].isin([5, 6, 7, 11, 12])]


# 4. Jahrzehnte-Spalte für die X-Achse berechnen
def get_decade(year):
    if year < 1960:
        return "Vor 1960"
    elif year < 1980:
        return "1960er/70er"
    elif year < 1990:
        return "1980er"
    elif year < 2000:
        return "1990er"
    elif year < 2010:
        return "2000er"
    elif year < 2020:
        return "2010er"
    else:
        return "2020er"


df_wm["Jahrzehnt"] = df_wm["year"].apply(get_decade)

# Sortieren, damit die X-Achse zeitlich korrekt von links nach rechts läuft
df_wm = df_wm.sort_values("year")


# =========================================================================
# PLOT 1: VIOLINPLOT (Tor-Minuten im Wandel der Jahrzehnte)
# =========================================================================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(13, 7))

# Moderne Farbpalette
colors = sns.color_palette("plasma", n_colors=len(df_wm["Jahrzehnt"].unique()))

# Der Violinplot
ax = sns.violinplot(
    x="Jahrzehnt",
    y="minute",
    hue="Jahrzehnt",     # Behebt die FutureWarning
    legend=False,        # Verhindert eine doppelte, unschöne Legende
    data=df_wm,
    palette=colors,
    inner="quartile",    # Zeigt das 25%, 50% (Median) und 75% Quartil an
    cut=0,               # Beschränkt die Kurve strikt auf die echten Min/Max Werte
    bw_adjust=0.5,       # Macht den Plot sensibel genug für die "Nachspielzeit-Peaks"
)

# Taktische Orientierungslinien hinzufügen
plt.axhline(
    45,
    color="darkgray",
    linestyle=":",
    linewidth=1.5,
    label="Halbzeit (45. Minute)",
)
plt.axhline(
    90,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Reguläre Spielzeit (90. Minute)",
)

# Beschriftungen & Titel (inkl. suptitle-Fix gegen Überlappung)
plt.title(
    "In welcher Spielminute fallen die Tore bei Fussball-Weltmeisterschaften?",
    fontsize=16,
    fontweight="bold",
    pad=30,  
)

plt.xlabel("Ära / Jahrzehnt der Weltmeisterschaft", fontsize=12, labelpad=10)
plt.ylabel("Spielminute des Tores", fontsize=12, labelpad=10)

# Y-Achse begrenzen
plt.ylim(0, 125)
plt.legend(loc="upper left")

# Layout optimieren und speichern
plt.tight_layout()
plt.savefig("m02_02_wm_torminuten_violinplot.png", dpi=300, bbox_inches="tight")
print("Grafik 1 erfolgreich gespeichert: m02_02_wm_torminuten_violinplot.png")


# =========================================================================
# PLOT 2: TREND-PLOT (Statistisch bereinigte Tore pro Spiel)
# =========================================================================

# Statistische Bereinigung: Tore und Spiele pro Jahr zählen
# 1. Gesamttore pro Jahr (Jede Zeile im gefilterten df_wm entspricht einem Tor)
tore_pro_jahr = df_wm.groupby("year").size().reset_index(name="gesamt_tore")

# 2. Echte Spiele pro Jahr zählen (Eindeutige Kombination aus Datum, Heim- und Auswärtsteam)
# Behebt die Pandas FutureWarning durch Nutzung von .drop_duplicates() und .groupby()
spiele_pro_jahr = (
    df_wm[["year", "date", "home_team", "away_team"]]
    .drop_duplicates()
    .groupby("year")
    .size()
    .reset_index(name="anzahl_spiele")
)

# Beide Tabellen mergen und den Schnitt pro Spiel berechnen
stats_bereinigt = pd.merge(tore_pro_jahr, spiele_pro_jahr, on="year")
stats_bereinigt["tore_pro_spiel"] = stats_bereinigt["gesamt_tore"] / stats_bereinigt["anzahl_spiele"]

# Neue Figure für den zweiten Plot erstellen
plt.figure(figsize=(13, 6))

# Regressionslinie (Trendlinie) + Punkte zeichnen
sns.regplot(
    x="year", 
    y="tore_pro_spiel", 
    data=stats_bereinigt,
    scatter_kws={"s": 90, "color": "#1d3557", "alpha": 0.8}, # Stil der Punkte
    line_kws={"color": "#e63946", "linewidth": 2.5},          # Stil der Trendlinie
    order=2  # Erlaubt der Trendlinie eine leichte, realistische Kurve
)

# Beschriftungen & Titel für Plot 2
#plt.title("Führt moderner Fußball zu weniger Toren?", fontsize=16, fontweight="bold", pad=20)
plt.xlabel("Weltmeisterschaft (Jahr)", fontsize=12, labelpad=10)
plt.ylabel("Durchschnittliche Tore pro Spiel", fontsize=12, labelpad=10)

# Alle historischen WM-Jahre exakt auf der X-Achse als Ticks setzen
plt.xticks(world_cup_years, rotation=45)

# Layout optimieren und speichern
plt.tight_layout()
plt.savefig("m02_02_wm_tore_pro_spiel_trend.png", dpi=300, bbox_inches="tight")
print("Grafik 2 erfolgreich gespeichert: m02_03_wm_tore_pro_spiel_trend.png")

# Beide Plots final auf dem Bildschirm ausgeben
plt.show()