"""
Survival Analysis / Überlebenszeitanalyse
IBM HR Analytics Employee Attrition & Performance

Konzept:
- "Zeit" (Duration)  = YearsAtCompany  (wie lange war/ist die Person im Unternehmen)
- "Event"            = Attrition == 'Yes'  (1 = Kündigung beobachtet, 0 = zensiert,
                        d.h. Person ist zum Zeitpunkt der Datenerhebung noch im Unternehmen)

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

sns.set_theme(style="whitegrid")

# pip install lifelines pandas matplotlib seaborn

# source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# ---------------------------------------------------------------------------
# 1. Daten laden & vorbereiten
# ---------------------------------------------------------------------------

CSV_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"  # ggf. Pfad anpassen

df = pd.read_csv(CSV_PATH)

# Event-Variable: 1 = Kündigung (beobachtet), 0 = zensiert (noch im Unternehmen)
df["Event"] = (df["Attrition"] == "Yes").astype(int)

# Duration: mind. 0.5 statt 0 verwenden, da lifelines mit Duration=0 Probleme
# bei manchen Berechnungen (z.B. log-Transformationen in Cox) haben kann
df["Duration"] = df["YearsAtCompany"].clip(lower=0.5)

print(f"n = {len(df)}, davon Events (Attrition=Yes): {df['Event'].sum()} "
      f"({df['Event'].mean():.1%})")

# ---------------------------------------------------------------------------
# 2. Kaplan-Meier: Gesamtkurve
# ---------------------------------------------------------------------------

kmf = KaplanMeierFitter()
kmf.fit(durations=df["Duration"], event_observed=df["Event"], label="Gesamt")

fig, ax = plt.subplots(figsize=(8, 5))
kmf.plot_survival_function(ax=ax)
ax.set_title("Kaplan-Meier: Verbleibswahrscheinlichkeit im Unternehmen")
ax.set_xlabel("Jahre im Unternehmen")
ax.set_ylabel("Anteil verbleibender Mitarbeiter")
plt.tight_layout()
plt.savefig("m05_03_km_overall.png", dpi=150)
plt.close()

print(f"Median-Verweildauer (gesamt): {kmf.median_survival_time_}")

# ---------------------------------------------------------------------------
# 3. Kaplan-Meier: Gruppenvergleich nach OverTime
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for value, group in df.groupby("OverTime"):
    kmf_grp = KaplanMeierFitter()
    kmf_grp.fit(group["Duration"], group["Event"], label=f"OverTime = {value}")
    kmf_grp.plot_survival_function(ax=ax)

ax.set_title("Kaplan-Meier nach Überstunden (OverTime)")
ax.set_xlabel("Jahre im Unternehmen")
ax.set_ylabel("Anteil verbleibender Mitarbeiter")
plt.tight_layout()
plt.savefig("m05_03_km_overtime.png", dpi=150)
plt.close()

# Log-Rank-Test: unterscheiden sich die Kurven statistisch signifikant?
grp_yes = df[df["OverTime"] == "Yes"]
grp_no = df[df["OverTime"] == "No"]

lr_result = logrank_test(
    grp_yes["Duration"], grp_no["Duration"],
    event_observed_A=grp_yes["Event"], event_observed_B=grp_no["Event"],
)
print(f"\nLog-Rank-Test OverTime Yes vs. No: "
      f"p-Wert = {lr_result.p_value:.4f}")

# ---------------------------------------------------------------------------
# 4. Kaplan-Meier: Gruppenvergleich nach Department (>2 Gruppen)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for dept, group in df.groupby("Department"):
    kmf_grp = KaplanMeierFitter()
    kmf_grp.fit(group["Duration"], group["Event"], label=dept)
    kmf_grp.plot_survival_function(ax=ax)

ax.set_title("Kaplan-Meier nach Abteilung (Department)")
ax.set_xlabel("Jahre im Unternehmen")
ax.set_ylabel("Anteil verbleibender Mitarbeiter")
plt.tight_layout()
plt.savefig("m05_03_km_department.png", dpi=150)
plt.close()

# Bei >2 Gruppen: multivariater Log-Rank-Test statt paarweisem logrank_test
mv_result = multivariate_logrank_test(
    df["Duration"], df["Department"], df["Event"]
)
print(f"Multivariater Log-Rank-Test Department: p-Wert = {mv_result.p_value:.4f}")

# ---------------------------------------------------------------------------
# 5. Cox Proportional Hazards Modell
# ---------------------------------------------------------------------------

# Auswahl relevanter Kovariaten (Mischung aus numerisch & kategorial)
cox_vars = [
    "Duration", "Event",
    "Age", "MonthlyIncome", "DistanceFromHome",
    "JobSatisfaction", "WorkLifeBalance", "NumCompaniesWorked",
    "OverTime", "MaritalStatus", "JobLevel",
]

cox_df = df[cox_vars].copy()

# Kategoriale Variablen dummy-codieren (drop_first, um Referenzkategorie zu setzen)
cox_df = pd.get_dummies(
    cox_df, columns=["OverTime", "MaritalStatus"], drop_first=True
)

# get_dummies erzeugt bool-Spalten -> lifelines will numerisch
bool_cols = cox_df.select_dtypes(include="bool").columns
cox_df[bool_cols] = cox_df[bool_cols].astype(int)

cph = CoxPHFitter()
cph.fit(cox_df, duration_col="Duration", event_col="Event")

print("\n--- Cox Proportional Hazards Modell: Zusammenfassung ---")
cph.print_summary()  # HR, CI, p-Werte je Kovariate

# Hazard Ratios als Forest Plot
fig, ax = plt.subplots(figsize=(8, 6))
cph.plot(ax=ax)
ax.set_title("Cox-Modell: Hazard Ratios (log-Skala) mit 95%-KI")
plt.tight_layout()
plt.savefig("m05_03_cox_hazard_ratios.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Proportional-Hazards-Annahme prüfen
# ---------------------------------------------------------------------------
# WICHTIG für die Interpretation: Cox setzt voraus, dass sich die Hazard Ratios
# über die Zeit nicht verändern (proportionale Hazards). Das sollte man testen
# und in den Slides kurz erwähnen (auch wenn Verletzungen bei diesem Datensatz
# in der Praxis oft vorkommen, weil YearsAtCompany kein "echtes" Ereigniszeit-
# Design ist).

print("\n--- Test der Proportional-Hazards-Annahme ---")
cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)