import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

###
# Getting started:
#
# pip install numpy matplotlib seaborn pandas
#
###

## download dataset and store it in local dir as "ecommerce_kaggle.csv"
## link: https://www.kaggle.com/code/mfaisalqureshi/pakistan-e-commerce-data-analysis/input
DATASET = "ecommerce_kaggle.csv"

print("="*60)
print("--- KAGGLE DATA AUDIT: EVALUIERUNG DES 4V-MODELLS ---")
print("="*60)

# 0. DATEN EINLESEN
try:
    # low_memory=False, da Spalten wie BI Status oder sales_commission_code gemischte Typen haben
    df = pd.read_csv(DATASET, low_memory=False)
    print(f"-> Datei erfolgreich geladen. Datensatz enthält {len(df):,} Zeilen und {len(df.columns)} Spalten.\n")
except FileNotFoundError:
    print("Fehler: Die Datei 'ecommerce_kaggle.csv' wurde im aktuellen Ordner nicht gefunden.")
    print("Bitte platziere die CSV-Datei im selben Verzeichnis wie dieses Skript.")
    exit()

# ==========================================
# 1. VOLUME (Volumen)
# ==========================================
print("[1. VOLUME] Evaluierung des Speicherbedarfs:")
actual_memory_bytes = df.memory_usage(deep=True).sum()
actual_memory_mb = actual_memory_bytes / (1024 * 1024)

print(f" - Echter RAM-Verbrauch im System: {actual_memory_mb:.2f} MB")
print(f" - Zeilenanzahl: {len(df):,} Datensätze.")

# IT-Management-Kontext: Zeilengrenze von Excel aufzeigen
excel_limit = 1_048_576
if len(df) > excel_limit:
    print(f" -> IT-Kontext: Dieser Datensatz überschreitet die kritische Excel-Grenze von {excel_limit:,} Zeilen!")
    print("    Der Einsatz von Python/Pandas ist hier zwingend erforderlich (Big Data Infrastruktur).")
else:
    print(f" -> IT-Kontext: Aktuell noch unter der Excel-Grenze, aber bei linearem Wachstum kritisch für In-Memory-Analysen.")


# ==========================================
# 2. VELOCITY (Geschwindigkeit / Transaktionsdichte)
# ==========================================
print("\n[2. VELOCITY] Evaluierung der zeitlichen Dynamik:")

# Zeitstempel konvertieren für die Analyse (Fehlertolerant wegen unterschiedlicher Formate)
df['Working Date'] = pd.to_datetime(df['Working Date'], errors='coerce')
valid_dates = df['Working Date'].dropna()

if not valid_dates.empty:
    min_date = valid_dates.min().strftime('%Y-%m-%d')
    max_date = valid_dates.max().strftime('%Y-%m-%d')
    total_days = (valid_dates.max() - valid_dates.min()).days
    if total_days == 0: total_days = 1
    
    avg_orders_per_day = len(df) / total_days
    print(f" - Daten-Zeitraum: Von {min_date} bis {max_date} ({total_days} Tage)")
    print(f" - Durchschnittliche Verarbeitungsrate: {avg_orders_per_day:.2f} Bestellungen pro Tag")
    print(" -> IT-Kontext: Erlaubt die Dimensionierung von Event-Handlern und Queue-Systemen für Peak-Zeiten.")
else:
    print(" - Warnung: Spalte 'Working Date' konnte nicht in Datetime konvertiert werden.")


# ==========================================
# 3. VARIETY (Datenvielfalt & Polystrukturierung)
# ==========================================
print("\n[3. VARIETY] Evaluierung der Datenstruktur:")

# Zählen der unterschiedlichen Kategorien und Zahlungsmethoden
unique_categories = df['category_name_1'].nunique() if 'category_name_1' in df.columns else 0
unique_payments = df['payment_method'].nunique() if 'payment_method' in df.columns else 0

print(f" - Anzahl Produktkategorien: {unique_categories}")
print(f" - Anzahl genutzter Zahlungsmethoden: {unique_payments}")

# Fokus auf die unstrukturierte SKU-Spalte (Text-Parsing notwendig)
if 'sku' in df.columns:
    # Wie viele SKUs enthalten komplexe Marketingtexte statt sauberer IDs?
    marketing_skus = df['sku'].astype(str).str.contains('Buy|Get|Free|Off', case=False).sum()
    marketing_rate = (marketing_skus / len(df)) * 100
    print(f" - Polystrukturierungs-Audit: {marketing_skus:,} SKUs ({marketing_rate:.1f}%) enthalten unstrukturierten Promotion-Text.")
    print(" -> Erkenntnis: Daten sind nicht rein relational; erfordert Text-Parsing (Regex) zur Bereinigung.")


# ==========================================
# 4. VERACITY (Glaubwürdigkeit & Datenqualität)
# ==========================================
print("\n[4. VERACITY] Data Quality Cleansing Audit:")

# A) Identifikation von Excel-Exportfehlern (#REF!)
ref_errors = 0
for col in df.select_dtypes(include=['object']).columns:
    ref_errors += df[col].astype(str).str.contains('#REF!').sum()

# B) Identifikation von Datenbank-Null-Werten, die als Text '\N' exportiert wurden
db_null_errors = 0
for col in df.select_dtypes(include=['object']).columns:
    db_null_errors += df[col].astype(str).str.strip().eq(r'\N').sum()

# C) Fehlende Werte (NaN) in den Kernspalten (z.B. grand_total)
missing_totals = df['grand_total'].isna().sum() if 'grand_total' in df.columns else 0

# Berechnung des Data Quality Scores
total_records = len(df) * len(df.columns)
total_detected_errors = ref_errors + db_null_errors + missing_totals
quality_score = ((total_records - total_detected_errors) / total_records) * 100

print(f" - Gefundene Excel-Referenzfehler (#REF!): {ref_errors:,}")
print(f" - Gefundene Datenbank-Null-Strings (\\N): {db_null_errors:,}")
print(f" - Fehlende Beträge (NaN in grand_total): {missing_totals:,}")
print(f" >>> REINHEITSGRAD DER REALLIFE-DATEN (Data Quality Score): {quality_score:.2f}%")
print(" -> Management-Fazit: Unbereinigte Rohdaten verfälschen Finanzberichte (Gefahr von Fehlentscheidungen).")


# ==========================================
# 5. VISUALISIERUNG FÜR DIE FOLIEN (Separate Grafiken)
# ==========================================
print("\n" + "="*50)
print("--- GENERIERE SEPARATE GRAFIKEN FÜR DIE FOLIEN ---")
print("="*50)

# Stil für saubere Grafiken setzen
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ------------------------------------------------------------------
# GRAFIK 1: Veracity Audit (Donut Chart) - Optimierte Lesbarkeit
# ------------------------------------------------------------------
# Grösseres Figure-Fenster, um Platz für die Legende zu schaffen
plt.figure(figsize=(9, 7))

labels_veracity = ['Korrekte Datenpunkte', 'Systemfehler (\\N & #REF!)', 'Fehlende Werte (NaN)']
sizes_veracity = [total_records - total_detected_errors, ref_errors + db_null_errors, missing_totals]
colors_veracity = ['#27ae60', '#e67e22', '#c0392b']

# Funktion, die NUR die absoluten Zahlen gross im Kuchenstück anzeigt
def make_absolute_label(values):
    def my_labels(pct):
        if pct < 1.5:  # Falls ein Segment winzig ist (<1.5%), Text ausblenden gegen Überlappung
            return ''
        total = sum(values)
        val = int(round(pct * total / 100.0))
        # Schweizer Tausendertrennzeichen (')
        return f"{val:,}".replace(",", "'")
    return my_labels

# Diagramm zeichnen - labels=None blendet die äusseren Texte aus (wandern in die Legende)
wedges, texts, autotexts = plt.pie(
    sizes_veracity, 
    labels=None, 
    colors=colors_veracity,
    autopct=make_absolute_label(sizes_veracity), 
    startangle=140, 
    pctdistance=0.72
)

# Schriftgrösse der absoluten Zahlen im Kuchenstück deutlich vergrössern
plt.setp(autotexts, size=14, weight='bold', color='white')

# Weissen Kreis in die Mitte für den Donut-Effekt (Loch etwas kleiner für mehr Textplatz)
centre_circle = plt.Circle((0,0), 0.50, fc='white')
plt.gca().add_artist(centre_circle)

# Titel deutlich grösser formatieren
plt.title(
    f"Veracity: Echte Datenqualität\n(Data Quality Score: {quality_score:.2f}%)", 
    fontsize=16, 
    weight='bold', 
    pad=20
)

# Legende ausserhalb des Diagramms platzieren mit grosser Schrift
# Kombiniert das Label mit dem Prozentwert für die Legende
legend_labels = [f"{label} ({size/total_records*100:.2f}%)" for label, size in zip(labels_veracity, sizes_veracity)]

plt.legend(
    wedges, 
    legend_labels,
    title="Datenabdeckung / Kategorien",
    title_fontsize=13,
    loc="upper left",
    bbox_to_anchor=(0.85, 0.95),  # Schiebt die Legende rechts neben die Torte
    fontsize=12,
    frameon=True
)

# tight_layout anpassen, damit die Legende rechts nicht abgeschnitten wird
plt.tight_layout()

# Grafik als separates File speichern
filename_1 = "m01_01_veracity_quality_audit.png"
plt.savefig(filename_1, dpi=300, bbox_inches='tight')
print(f" -> Grafik 1 erfolgreich gespeichert als: '{filename_1}'")
plt.show()

# ------------------------------------------------------------------
# GRAFIK 2: Variety - Top Zahlungsmethoden (Balken) - Eigenständiges Bild
# ------------------------------------------------------------------
if 'payment_method' in df.columns:
    plt.figure(figsize=(8, 5))
    
    # Top 7 Zahlungsmethoden herausfiltern
    payment_counts = df['payment_method'].value_counts().head(7)
    
    # Horizontales Balkendiagramm zeichnen
    bars = plt.barh(payment_counts.index, payment_counts.values, color='#3498db', edgecolor='none')
    plt.gca().invert_yaxis()  # Höchste Werte nach oben
    
    plt.title("Variety: Heterogenität der Zahlungsmethoden (Datenvielfalt)", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Anzahl Transaktionen (Volumen pro Typ)", fontsize=11)
    plt.ylabel("Zahlungsmethode", fontsize=11)
    
    # Werte direkt an die Balken schreiben
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f' {width:,}', 
                 va='center', ha='left', fontsize=10, weight='bold')
                 
    plt.tight_layout()
    
    # Zweite Grafik als separates File speichern
    filename_2 = "m01_01_variety_payment_methods.png"
    plt.savefig(filename_2, dpi=300)
    print(f" -> Grafik 2 erfolgreich gespeichert als: '{filename_2}'")
    plt.show() # Zeigt Grafik 2 an
else:
    print(" - Grafik 2 konnte nicht erstellt werden: Spalte 'payment_method' fehlt.")