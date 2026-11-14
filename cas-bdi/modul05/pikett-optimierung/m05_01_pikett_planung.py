"""
Einsatz-Optimierung: Pikett-Planung für eine IT-Abteilung (Jahr 2027)

#GETTING STARTED
# pip install numpy pandas matplotlib scipy
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

# ------------------------------------------------------------
# 1. Eingaben
# ------------------------------------------------------------
JAHR = 2027
N = 10                                           # Anzahl Mitarbeitende
personen = [f"MA{i + 1}" for i in range(N)]

erster = dt.date(JAHR, 1, 1)
tage = [erster + dt.timedelta(days=i)
        for i in range((dt.date(JAHR + 1, 1, 1) - erster).days)]
D = len(tage)
werktag = np.array([t.weekday() < 5 for t in tage])   # Mo-Fr

n_primary = D                                    # jeden Tag ein Primary
n_secondary = int(werktag.sum())                 # nur Mo-Fr ein Secondary
n_total = n_primary + n_secondary

print(f"{JAHR}: {D} Tage, davon {n_secondary} Werktage und {D - n_secondary} Wochenendtage")
print(f"Primary-Tage: {n_primary}, Secondary-Tage: {n_secondary}, "
      f"Pikett-Tage total: {n_total}")
print(f"Durchschnitt pro Person: Primary {n_primary / N:.1f}, "
      f"Secondary {n_secondary / N:.1f}, total {n_total / N:.1f}\n")

# ------------------------------------------------------------
# 2. Modell: Variablen und Nebenbedingungen
# ------------------------------------------------------------
PRIMARY, SECONDARY = 0, 1
n_var = N * D * 2


def v(p, d, r):
    """Index der Entscheidungsvariable x[p,d,r]."""
    return (p * D + d) * 2 + r


zeilen, spalten, werte = [], [], []
lb, ub = [], []


def nebenbedingung(koeffizienten, untere, obere):
    """Fügt die Zeile  untere <= sum(koeff * x) <= obere  hinzu."""
    z = len(lb)
    for spalte, wert in koeffizienten:
        zeilen.append(z)
        spalten.append(spalte)
        werte.append(wert)
    lb.append(untere)
    ub.append(obere)


for d in range(D):
    # Abdeckung: jeden Tag genau ein Primary
    nebenbedingung([(v(p, d, PRIMARY), 1) for p in range(N)], 1, 1)
    # Abdeckung: Mo-Fr genau ein Secondary, am Wochenende keiner
    soll = 1 if werktag[d] else 0
    nebenbedingung([(v(p, d, SECONDARY), 1) for p in range(N)], soll, soll)

for p in range(N):
    for d in range(D):
        # eine Person hat pro Tag höchstens eine Rolle
        nebenbedingung([(v(p, d, PRIMARY), 1), (v(p, d, SECONDARY), 1)], 0, 1)
    for d in range(D - 1):
        # Zusatzregel: nicht an zwei aufeinanderfolgenden Tagen auf Pikett
        nebenbedingung([(v(p, d + dd, r), 1) for dd in (0, 1) for r in (0, 1)], 0, 1)

# Fairness Nebenbedingung: jede Person liegt zwischen abgerundetem und aufgerundetem Durchschnitt
for p in range(N):
    for rollen, anzahl in (((PRIMARY,), n_primary),
                           ((SECONDARY,), n_secondary),
                           ((PRIMARY, SECONDARY), n_total)):
        nebenbedingung([(v(p, d, r), 1) for d in range(D) for r in rollen],
                       int(np.floor(anzahl / N)), int(np.ceil(anzahl / N)))

A = coo_matrix((werte, (zeilen, spalten)), shape=(len(lb), n_var)).tocsr()
print(f"Modell: {n_var} binäre Variablen, {len(lb)} Nebenbedingungen")

# Reines Planungsproblem: Ziel 0,
# jede zulässige Lösung ist optimal.
res = milp(
    c=np.zeros(n_var),
    constraints=LinearConstraint(A, lb, ub),
    integrality=np.ones(n_var),
    bounds=Bounds(0, 1),
    options={"time_limit": 300},
)
assert res.x is not None, f"Keine Lösung gefunden: {res.message}"
print(f"Solver: {res.message}\n")

x = np.round(res.x).reshape(N, D, 2).astype(int)

# ------------------------------------------------------------
# 3. Kontrolle: Plan unabhängig vom Modell nachrechnen
# Nebenbedingungen Rule Set Checks
# ------------------------------------------------------------
assert (x[:, :, PRIMARY].sum(axis=0) == 1).all(), "Primary-Abdeckung verletzt"
assert (x[:, :, SECONDARY].sum(axis=0) == werktag.astype(int)).all(), "Secondary-Abdeckung verletzt"
im_dienst = x.sum(axis=2)                                   # Person x Tag
assert (im_dienst <= 1).all(), "Person doppelt eingeteilt"
assert ((im_dienst[:, :-1] + im_dienst[:, 1:]) <= 1).all(), "Pikett an zwei Tagen hintereinander"

p_tage = x[:, :, PRIMARY].sum(axis=1)
s_tage = x[:, :, SECONDARY].sum(axis=1)
t_tage = p_tage + s_tage
for name, werte_ in (("Primary", p_tage), ("Secondary", s_tage), ("Total", t_tage)):
    assert werte_.max() - werte_.min() <= 1, f"{name}: Unterschied grösser als 1"
assert t_tage.sum() == n_total
print("Kontrollen bestanden: Abdeckung, keine Doppelrollen, keine Folgetage, "
      "Unterschied je höchstens 1.\n")

we = ~werktag
uebersicht = pd.DataFrame({
    "Primary": p_tage,
    "Secondary": s_tage,
    "Total": t_tage,
    "davon Primary am Wochenende": x[:, we, PRIMARY].sum(axis=1),
}, index=personen)
print(uebersicht.to_string())

# Plan als Tabelle speichern
plan = pd.DataFrame({
    "Datum": tage,
    "Wochentag": [t.strftime("%a") for t in tage],
    "Primary": [personen[int(np.argmax(x[:, d, PRIMARY]))] for d in range(D)],
    "Secondary": [personen[int(np.argmax(x[:, d, SECONDARY]))] if werktag[d] else "-"
                  for d in range(D)],
})

#debug
#plan.to_csv("m05_02_pikett_plan_2027.csv", index=False)

# ------------------------------------------------------------
# 4. Grafik: Plan-Ausschnitt (8 Wochen) und Tage pro Person
# ------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
C_PRIMARY, C_SECONDARY = "#2a78d6", "#eb6834"      # Kategorien 1 und 2
C_FREI, C_FREI_WE = "#f0efec", "#e1e0d9"

plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED,
                     "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})

fig = plt.figure(figsize=(11, 7.2), facecolor=SURFACE)
gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.42,
                      left=0.07, right=0.97, top=0.93, bottom=0.07)

# --- oben: Ausschnitt aus dem Plan ---
ax1 = fig.add_subplot(gs[0], facecolor=SURFACE)
AUSSCHNITT = 56                                    # 8 Wochen ab 1. Januar
start = 0
for d in range(start, start + AUSSCHNITT):
    for p in range(N):
        if x[p, d, PRIMARY]:
            farbe = C_PRIMARY
        elif x[p, d, SECONDARY]:
            farbe = C_SECONDARY
        else:
            farbe = C_FREI if werktag[d] else C_FREI_WE
        ax1.add_patch(Rectangle((d - start, N - 1 - p), 1, 1, facecolor=farbe,
                                edgecolor=SURFACE, linewidth=1.8))
ax1.set_xlim(0, AUSSCHNITT)
ax1.set_ylim(0, N)
ax1.set_yticks([N - 1 - p + 0.5 for p in range(N)])
ax1.set_yticklabels(personen)
montage = [d for d in range(start, start + AUSSCHNITT) if tage[d].weekday() == 0]
ax1.set_xticks([d - start + 0.5 for d in montage])
ax1.set_xticklabels([f"Mo {tage[d].day}.{tage[d].month}." for d in montage], fontsize=9)
ax1.xaxis.tick_top()
ax1.tick_params(length=0)
for s in ax1.spines.values():
    s.set_visible(False)
ax1.set_title(f"Ausschnitt aus dem Plan {JAHR}: erste 8 Wochen (grau = frei, dunkler = Wochenende)",
              loc="left", fontsize=11, color=INK2, pad=26)
ax1.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=C_PRIMARY, label="Primary"),
                    Rectangle((0, 0), 1, 1, facecolor=C_SECONDARY, label="Secondary")],
           loc="upper right", bbox_to_anchor=(1.0, -0.02), ncol=2, frameon=False,
           fontsize=10)

# --- unten: Pikett-Tage pro Person im ganzen Jahr ---
ax2 = fig.add_subplot(gs[1], facecolor=SURFACE)
y = np.arange(N)[::-1]
ax2.barh(y, p_tage, color=C_PRIMARY, edgecolor=SURFACE, linewidth=2, height=0.7)
ax2.barh(y, s_tage, left=p_tage, color=C_SECONDARY, edgecolor=SURFACE, linewidth=2, height=0.7)
for i in range(N):
    ax2.text(p_tage[i] / 2, y[i], f"{p_tage[i]}", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")
    ax2.text(p_tage[i] + s_tage[i] / 2, y[i], f"{s_tage[i]}", ha="center", va="center",
             color="white", fontsize=10, fontweight="bold")
    ax2.text(t_tage[i] + 0.8, y[i], f"{t_tage[i]}", ha="left", va="center",
             color=INK, fontsize=10)
ax2.set_yticks(y)
ax2.set_yticklabels(personen)
ax2.set_xlim(0, t_tage.max() + 6)
ax2.set_xlabel("Pikett-Tage im Jahr")
ax2.tick_params(length=0)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.spines["bottom"].set_color("#c3c2b7")
ax2.xaxis.grid(True, color="#e1e0d9", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.set_title(f"Pikett-Tage pro Person {JAHR}: Primary (blau) + Secondary (orange) = Total",
              loc="left", fontsize=11, color=INK2)

plt.savefig("m05_01_pikett_planung.png", dpi=200, facecolor=SURFACE)
