import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###
# Getting started:
#
# pip install matplotlib seaborn pandas
#
###

## download dataset and store it in local dir as "ecommerce_kaggle.csv"
## link: https://www.kaggle.com/code/mfaisalqureshi/pakistan-e-commerce-data-analysis/input
DATASET = "ecommerce_kaggle.csv"

# ====== 1. DATENBASIS LADEN & REINIGEN (Schnittstelle zu Fall 1) ======
print("="*60)
print("--- FALL 2: SIMULATION LIFECYCLE-MODELL & SMART DATA ---")
print("="*60)

# Datensatz einlesen
df = pd.read_csv(DATASET, low_memory=False)

# Relevante Spalten für die Zeitreihe bereinigen
df['Working Date'] = pd.to_datetime(df['Working Date'], errors='coerce')
df['grand_total'] = pd.to_numeric(df['grand_total'], errors='coerce')

# Fehlerhafte Zeilen eliminieren, um das Signal nicht zu verzerren
df_clean = df.dropna(subset=['Working Date', 'grand_total'])
df_clean = df_clean[df_clean['grand_total'] > 0] # Nur echte Umsätze

# ====== 2. ZEITLICHE AGGREGATION (Vom Rauschen zum Signal) ======
# Umsatz pro Tag aggregieren
daily_sales = df_clean.groupby('Working Date')['grand_total'].sum().sort_index()

# A: Das "Mittelwert-Rauschen" (Statischer, globaler Mittelwert)
global_mean = daily_sales.mean()

# B: Das "Smart Data Signal" (Rollierender 30-Tage-Durchschnitt)
rolling_smart_signal = daily_sales.rolling(window=30, center=True).mean()

# C: Kontext-Analyse (Umsatz nach Wochentag)
df_clean['Weekday'] = df_clean['Working Date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_sales = df_clean.groupby('Weekday')['grand_total'].mean().reindex(weekday_order)

print(f" -> Zeitreihe analysiert von {daily_sales.index.min().strftime('%Y-%m-%d')} bis {daily_sales.index.max().strftime('%Y-%m-%d')}")


# ==================================================================
# FOLIE 2 - GRAFIK A: Rauschen vs. Smart Data (Umsatzverlauf)
# ==================================================================
plt.figure(figsize=(10, 5))

# Die echten, wilden Tagesumsätze im Hintergrund (hellgrau)
plt.plot(daily_sales.index, daily_sales.values, color='#b2bec3', alpha=0.4, label='Tägliches Rauschen (Rohdaten)')

# Der statische Mittelwert (Rot, flach, gefährlich)
plt.axhline(global_mean, color='#c0392b', linestyle='--', linewidth=2.5, 
            label=f'Globaler Mittelwert: CHF {global_mean:,.0f}.- (Statische Big Data Sicht)'.replace(",", "'"))

# Das Smart Data Signal (Blau, gefiltert, zeigt den echten Geschäftsverlauf)
plt.plot(rolling_smart_signal.index, rolling_smart_signal.values, color='#2980b9', linewidth=3, 
         label='30-Tage Trend-Signal (Smart Data Sicht)')

plt.title("Umsatzverlauf: Statischer Mittelwert vs. Dynamisches Smart Data Signal", fontsize=13, weight='bold', pad=15)
plt.ylabel("Umsatz pro Tag (CHF)", fontsize=11)
plt.legend(loc="upper right", fontsize=10, frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Erste Grafik speichern
filename_2a = "m01_02a_rauschen_vs_signal.png"
plt.savefig(filename_2a, dpi=300, bbox_inches='tight')
print(f" -> Grafik 2A erfolgreich gespeichert als: '{filename_2a}'")
plt.close() # Schliesst die Grafik, um Speicher für die nächste freizugeben


# ==================================================================
# FOLIE 2 - GRAFIK B: Kontext durch Wochentags-Analyse
# ==================================================================
plt.figure(figsize=(8, 4.5))

# Schweizer Bezeichnungen für die Wochentage
ch_weekdays = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
sns.barplot(x=ch_weekdays, y=weekday_sales.values, palette='Blues_d')

plt.title("Operativer Kontext: Durchschnittlicher Umsatz nach Wochentag", fontsize=13, weight='bold', pad=15)
plt.ylabel("Ø Umsatz (CHF)", fontsize=11)
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Werte gut lesbar über den Balken platzieren (Schweizer Format)
for i, val in enumerate(weekday_sales.values):
    val_formatted = f"{val:,.0f}.-".replace(",", "'")
    plt.text(i, val + (val*0.02), val_formatted, ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()

# Zweite Grafik speichern
filename_2b = "m01_02b_kontext_wochentage.png"
plt.savefig(filename_2b, dpi=300, bbox_inches='tight')
print(f" -> Grafik 2B erfolgreich gespeichert als: '{filename_2b}'")
print("="*60)
plt.close()