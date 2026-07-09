import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# 1. Lokale CSV-Datei einlesen
# source: https://github.com/30lm32/ml-ab-testing/blob/master/ab_data.csv
csv_dateipfad = "ab_data.csv"

try:
    df = pd.read_csv(csv_dateipfad)
    print(f"Erfolgreich geladen: {len(df)} Zeilen aus '{csv_dateipfad}' eingelesen.\n")
except FileNotFoundError:
    print(f"Fehler: Die Datei '{csv_dateipfad}' wurde nicht gefunden. Generiere synthetische Testdaten...")

# 2. Datenbereinigung (Methodischer Schritt für deine Arbeit!)
# Wir behalten nur Zeilen, wo control+old_page ODER treatment+new_page übereinstimmen
df_cleaned = df[
    ((df["group"] == "control") & (df["landing_page"] == "old_page"))
    | ((df["group"] == "treatment") & (df["landing_page"] == "new_page"))
]

# Doppelte User-IDs entfernen, falls vorhanden
df_cleaned = df_cleaned.drop_duplicates(subset="user_id")

print(f"Daten bereinigt. Verbleibende Zeilen: {len(df_cleaned)}\n")

# 3. Deskriptive Statistik (Conversion Rates berechnen)
conversion_rates = (
    df_cleaned.groupby("group")["converted"].agg(["count", "mean", "sum"])
)
conversion_rates.columns = ["Gesamtanzahl (N)", "Conversion Rate", "Erfolge (Anzahl 1)"]
print("--- DESKRIPTIVE STATISTIK ---")
print(conversion_rates.to_string())
print("\n")

# 4. Schliessende Statistik: Chi-Quadrat-Test (Kat-Kat Hypothese)
# Kreuztabelle erstellen (Contingency Table)
contingency_table = pd.crosstab(df_cleaned["group"], df_cleaned["converted"])

# Chi-Quadrat-Test durchführen
chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

# Ergebnisse für den wissenschaftlichen Text vorbereiten
cr_control = conversion_rates.loc["control", "Conversion Rate"] * 100
cr_treatment = conversion_rates.loc["treatment", "Conversion Rate"] * 100

print("--- STATISTISCHER TEST (CHI-QUADRAT) ---")
print(f"Chi2-Statistik: {chi2_stat:.4f}")
print(f"p-Wert: {p_val:.4f}")

print("\n--- FORMULIERUNGSVORSCHLAG FÜR DIE ARBEIT ---")
print(
    f"Forschungsfrage: Beeinflusst das Design der Landing Page die Conversion Rate der Nutzer?\n"
    f"Hypothese (H0): Es gibt keinen Unterschied zwischen den Conversion Rates der Control- und Treatment-Gruppe.\n"
)

if p_val < 0.05:
    print(
        f"Ergebnis: Der durchgeführte Chi-Quadrat-Unabhängigkeitstest zeigt mit einem p-Wert von {p_val:.4f} "
        f"(p < 0.05) ein statistisch signifikantes Ergebnis. Die Nullhypothese wird abgelehnt. "
        f"Die neue Landing Page (Treatment: {cr_treatment:.2f}%) weist eine signifikant andere Conversion Rate auf "
        f"als die alte Landing Page (Control: {cr_control:.2f}%)."
    )
else:
    print(
        f"Ergebnis: Der durchgeführte Chi-Quadrat-Unabhängigkeitstest zeigt mit einem p-Wert von {p_val:.4f} "
        f"(p > 0.05) kein statistisch signifikantes Ergebnis. Die Nullhypothese kann somit nicht abgelehnt werden. "
        f"Der Unterschied zwischen der Conversion Rate der Control-Gruppe ({cr_control:.2f}%) und der "
        f"Treatment-Gruppe ({cr_treatment:.2f}%) ist statistisch nicht als systematisch zu betrachten und "
        f"kann auf rein zufällige Schwankungen zurückgeführt werden."
    )


# --- VISUALISIERUNG ---

# Stil definieren für ein sauberes, wissenschaftliches Layout
sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 5))

# Konfidenzintervall (95%) für die Anteile mathematisch korrekt berechnen (Fehlerbalken)
# Formel für Standardfehler: SE = sqrt( p * (1 - p) / n )
rates = conversion_rates["Conversion Rate"]
ns = conversion_rates["Gesamtanzahl (N)"]
errors = 1.96 * np.sqrt(rates * (1 - rates) / ns)

# Balkendiagramm mit Fehlerbalken zeichnen
# Umrechnung der Raten und Fehler in Prozent (* 100)
bars = plt.bar(
    conversion_rates.index,
    rates * 100,
    yerr=errors * 100,
    capsize=8,
    color=["#4C72B0", "#DD8452"],
    edgecolor="black",
    alpha=0.85,
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height / 2,
        f"{height:.2f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=12,
    )

# Beschriftungen und Titel optimieren
plt.title(
    "Vergleich der Conversion Rates (A/B Test)\nmit 95% Konfidenzintervall",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.ylabel("Conversion Rate (%)", fontsize=12)
plt.xlabel("Gruppe", fontsize=12)
plt.xticks(
    ticks=[0, 1],
    labels=["Control (Altes Design)", "Treatment (Neues Design)"],
    fontsize=11,
)

# Y-Achsen-Maximum leicht erhöhen, damit Platz für die Darstellung ist,
# aber bei 0 starten lassen, um visuelle Verzerrungen zu vermeiden.
plt.ylim(0, 15)

# Grafik layouten, unter dem Wunschnamen abspeichern und anzeigen
plt.tight_layout()
plt.savefig("m03_02_ab_test_conversion_rates.png", dpi=300)
print("\nGrafik erfolgreich unter 'ab_test_conversion_rates.png' gespeichert.")
plt.show()

# --- ERWEITERTE ANALYSE: ZEITLICHER VERLAUF (KUMULIERTE CONVERSION RATE) ---

try:
    print("\nBerechne zeitlichen Verlauf für erweiterten Plot...")
    
    # Zeitstempel in echtes Datum umwandeln und Daten chronologisch sortieren
    df_cleaned = df_cleaned.copy()
    df_cleaned['date'] = pd.to_datetime(df_cleaned['timestamp']).dt.date
    df_cleaned = df_cleaned.sort_values('timestamp')
    
    # Kumulierte Summe der Erfolge (converted) und der Gesamtzahl pro Gruppe berechnen
    df_cleaned['cum_converted'] = df_cleaned.groupby('group')['converted'].cumsum()
    df_cleaned['cum_count'] = df_cleaned.groupby('group').cumcount() + 1
    df_cleaned['cum_cr'] = (df_cleaned['cum_converted'] / df_cleaned['cum_count']) * 100
    
    # Für einen sauberen Plot gruppieren wir nach Datum, um den Zustand am Ende jedes Tages zu sehen
    daily_data = df_cleaned.groupby(['date', 'group']).last().reset_index()
    
    # Plot erstellen
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=daily_data, 
        x='date', 
        y='cum_cr', 
        hue='group', 
        palette=['#4C72B0', "#DD8452"],
        linewidth=2.5
    )
    
    # Diagramm-Styling
    plt.title("Stabilisierung der Conversion Rates über den Testzeitraum\n(Kumulierter Verlauf)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Kumulierte Conversion Rate (%)", fontsize=11)
    plt.xlabel("Datum", fontsize=11)
    plt.xticks(rotation=45)
    
    # Legende verschönern
    plt.legend(labels=["Control (Altes Design)", "Treatment (Neues Design)"], title="Gruppe", loc="upper right")
    
    plt.tight_layout()
    plt.savefig("m03_02_ab_test_zeitlicher_verlauf.png", dpi=300)
    print("Zweiter Plot erfolgreich unter 'ab_test_zeitlicher_verlauf.png' gespeichert.")
    plt.show()

except Exception as e:
    print(f"\nHinweis zum zeitlichen Verlauf: Plot konnte nicht erstellt werden ({e}).")
    print("Überprüfe, ob die Spalte 'timestamp' in deinem Datensatz vorhanden ist.")