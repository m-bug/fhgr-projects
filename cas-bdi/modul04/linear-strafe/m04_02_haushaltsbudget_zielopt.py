"""
Budgetplanung Student

Tech:
Solver: scipy.optimize.linprog (HiGHS)
mit Strafpunkten
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ---------------------------------------------------------------
# 1) Parameterblock
# ---------------------------------------------------------------
kategorien = ["Wohnen", "Verpflegung", "Versicherung/Steuern", "Fahrkosten",
              "Studium", "Persönliches", "Sparen"]

# Sollwerte in CHF pro Monat (Zusammenfassung der UZH-Richtwerte)
# source: https://www.studienfinanzierung.uzh.ch/dam/jcr:dd3e9d27-09d0-4b35-acf4-e9c9447b4de9/Budgetvorlage_UZH.pdf 
soll = np.array([
    740 + 170,        # Wohnen: Miete inkl. Heizkosten + Nebenkosten (Internet, Serafe)
    250 + 200,        # Verpflegung: zuhause + auswärts
    185 + 15 + 25,    # Krankenkasse (KVG) + AHV + Steuern
    95,               # Fahrkosten (ÖV)
    130 + 120,        # Studium: Studiengebühren + Lehrmittel
    80 + 20 + 150 + 20,  # Persönliches: Kleider, Hygiene, Freizeit, Mobiltelefon
    100,              # Sparen (Annahme, kein Richtwert vorhanden)
], dtype=float)

EINKOMMEN = 2100.0    # Annahme: Nebenjob und Unterstützung, CHF pro Monat
MINDESTANTEIL = 0.7   # Annahme: keine Kategorie unter 70 % des Sollwerts (Sparen: 0)

# Strafpunkte pro CHF Abweichung
strafe_unter = np.array([5, 3, 10, 4, 6, 1, 2], dtype=float)   # weniger als Soll
strafe_ueber = np.array([1, 1, 1, 1, 1, 1, 0], dtype=float)    # mehr als Soll
# Begründung: Je weniger sich eine Ausgabe senken lässt, desto höher die Strafe
# (Versicherung/Steuern 10, Studium 6, Wohnen 5, Fahrkosten 4, Verpflegung 3,
#  Sparen 2, Persönliches 1). Mehr Sparen als geplant ist nicht schlimm (0).

n = len(kategorien)


# ---------------------------------------------------------------
# 2) Modell
# Entscheidungsvariablen (3n = 21):
#   x_i = Ausgabe in Kategorie i (CHF)
#   m_i = Unterschreitung des Sollwerts ("minus")
#   p_i = Überschreitung des Sollwerts ("plus")
# Ziel:  Strafpunkte = sum(strafe_unter_i * m_i + strafe_ueber_i * p_i) -> minimieren
# NB:    x_i + m_i - p_i = soll_i       (je Kategorie)
#        sum(x_i) = Einkommen           (das ganze Geld wird verteilt)
#        x_i >= MINDESTANTEIL * soll_i  (ausser Sparen)
#
#
# ## source: https://ergo-code.github.io/HiGHS/dev/solvers/
# ---------------------------------------------------------------
def loesen(einkommen=EINKOMMEN):
    c = np.concatenate([np.zeros(n), strafe_unter, strafe_ueber])
    A_eq = np.zeros((n + 1, 3 * n))
    b_eq = np.zeros(n + 1)
    for i in range(n):
        A_eq[i, i] = 1        # x_i
        A_eq[i, n + i] = 1    # + m_i
        A_eq[i, 2 * n + i] = -1   # - p_i
        b_eq[i] = soll[i]
    A_eq[n, :n] = 1           # Summe der Ausgaben
    b_eq[n] = einkommen
    untergrenze = MINDESTANTEIL * soll
    untergrenze[-1] = 0.0     # Sparen darf bis 0 sinken
    bounds = [(untergrenze[i], None) for i in range(n)] + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    x = res.x[:n]
    m = res.x[n:2 * n]
    p = res.x[2 * n:]
    return res, x, m, p


def ausgeben(titel, einkommen):
    res, x, m, p = loesen(einkommen)
    print(f"\n=== {titel} (Einkommen {einkommen:.0f} CHF) ===")
    print(f"{'Kategorie':<22}{'Soll':>8}{'Plan':>9}{'Abw.':>8}{'Strafe':>9}")
    for i in range(n):
        strafe = strafe_unter[i] * m[i] + strafe_ueber[i] * p[i]
        print(f"{kategorien[i]:<22}{soll[i]:8.0f}{x[i]:9.1f}{x[i] - soll[i]:8.1f}{strafe:9.1f}")
    print(f"{'Total':<22}{soll.sum():8.0f}{x.sum():9.1f}{'':8}{res.fun:9.1f}")
    return res, x, m, p


# Basisfall
res, x, m, p = ausgeben("Basisfall", EINKOMMEN)

# Kontrolle: gleiches Ergebnis mit einer einfachen Rechnung von Hand
# (zuerst dort kürzen, wo ein Franken am wenigsten Strafe kostet)
fehlbetrag = soll.sum() - EINKOMMEN
kuerzbar = soll - np.where(np.arange(n) == n - 1, 0.0, MINDESTANTEIL * soll)
x_hand = soll.copy()
for i in np.argsort(strafe_unter):
    schnitt = min(kuerzbar[i], fehlbetrag)
    x_hand[i] -= schnitt
    fehlbetrag -= schnitt
assert np.allclose(x, x_hand, atol=1e-6), "LP und Handrechnung weichen ab"
print("\nKontrolle: Handrechnung (günstigste Kürzung zuerst) stimmt mit dem Solver überein.")

# Varianten
res_hoch, x_hoch, _, _ = ausgeben("Variante: mehr Einkommen", 2300.0)
res_tief, x_tief, _, _ = ausgeben("Variante: weniger Einkommen", 1900.0)

# ---------------------------------------------------------------
# 3) Grafik: Soll und Plan sowie Strafpunkte pro Kategorie
# ---------------------------------------------------------------
BLAU = "#2a78d6"
GRAU = "#c3c2b7"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
FLAECHE = "#fcfcfb"

strafen = strafe_unter * m + strafe_ueber * p
ypos = np.arange(n)[::-1]
h = 0.36

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.7), facecolor=FLAECHE,
                               gridspec_kw={"width_ratios": [1.25, 1]})

ax1.barh(ypos + h / 2, soll, height=h, color=GRAU, label="Sollwert")
ax1.barh(ypos - h / 2, x, height=h, color=BLAU, label="Plan")
for y, s, v in zip(ypos, soll, x):
    ax1.text(s + 15, y + h / 2, f"{s:.0f}", va="center", fontsize=8, color=TEXT2)
    ax1.text(v + 15, y - h / 2, f"{v:.0f}", va="center", fontsize=8, color=TEXT)
ax1.set_yticks(ypos)
ax1.set_yticklabels(kategorien, fontsize=8.5, color=TEXT)
ax1.set_xlim(0, soll.max() * 1.18)
ax1.set_title("Sollwert und Plan (CHF pro Monat)", fontsize=10, color=TEXT, loc="left")
ax1.legend(loc="lower right", fontsize=8, frameon=False)
ax1.set_xticks([])

ax2.barh(ypos, strafen, height=0.5, color=BLAU)
for y, s in zip(ypos, strafen):
    ax2.text(s + 4, y, f"{s:.0f}", va="center", fontsize=8.5, color=TEXT)
ax2.set_yticks(ypos)
ax2.set_yticklabels([""] * n)
ax2.set_xlim(0, max(strafen.max() * 1.2, 1))
ax2.set_title(f"Strafpunkte (total {res.fun:.0f})", fontsize=10, color=TEXT, loc="left")
ax2.set_xticks([])

for ax in (ax1, ax2):
    ax.set_facecolor(FLAECHE)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(TEXT2)
    ax.tick_params(axis="y", length=0)

fig.tight_layout()

fig.savefig("m04_02_haushaltsbudget_zielopt.png", dpi=200, facecolor=FLAECHE)