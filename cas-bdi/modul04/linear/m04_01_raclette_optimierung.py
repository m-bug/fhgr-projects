"""
Raclette-Plausch eines Fussballvereins

Tech:
Solver: scipy.optimize.linprog (HiGHS)
Hinweis: linprog minimiert immer. Das Maximierungsproblem wird daher mit
negierter Zielfunktion gelöst.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ---------------------------------------------------------------
# 1) Parameterblock (Annahmen)
# ---------------------------------------------------------------
produkte = ["Raclette", "Dessert", "Wein/Bier", "Softgetränk"]

# preise
verkaufspreis = np.array([22.0, 5.0, 5.0, 3.0])   # CHF pro Portion
einkaufskosten = np.array([9.0, 1.8, 1.8, 0.8])   # CHF pro Portion
gewinn = verkaufspreis - einkaufskosten           # CHF pro Portion

helferminuten = np.array([5.0, 1.0, 1.5, 1.0])    # Minuten pro Portion
kuehlplatz = np.array([0.8, 0.3, 0.6, 0.5])       # Liter Kühlplatz pro Portion

GAESTE = 120            # Saalgrösse = maximale Zahl Raclette-Portionen
BUDGET = 1500           # Einkaufsbudget (Vorschuss aus der Vereinskasse), CHF
HELFER = 6              # Anzahl Helfer
STUNDEN = 3             # Einsatzdauer pro Helfer, Stunden
KUEHLPLATZ = 260        # verfügbarer Kühlplatz, Liter
DESSERT_MAX = GAESTE * 2 / 3    # höchstens 2 von 3 Gästen nehmen ein Dessert
GETRAENKE_MAX = GAESTE * 3      # höchstens 3 Getränke pro Gast

# ---------------------------------------------------------------
# 2) Modell
# Entscheidungsvariablen: x1..x4 = Anzahl Portionen (siehe produkte)
# Ziel:   Gewinn = 13*x1 + 3.2*x2 + 3.2*x3 + 2.2*x4  -> maximieren
# Saalgrösse: 0 <= x1 <= 120 (als Schranke der Variablen x1 eingebaut)
# ---------------------------------------------------------------
nebenbedingungen = [
    "Einkaufsbudget",
    "Helferzeit",
    "Kühlplatz",
    "Dessert-Nachfrage",
    "Getränke-Nachfrage",
    "Getränke ≥ Raclette",
]

## source: https://ergo-code.github.io/HiGHS/dev/solvers/
def loesen(budget=BUDGET, helfer=HELFER, kuehl=KUEHLPLATZ, ganzzahlig=False):
    """Löst das Modell und gibt Lösung, Verbrauch und Grenzen zurück."""
    A = np.array([
        einkaufskosten,        # Einkaufsbudget
        helferminuten,         # Helferzeit
        kuehlplatz,            # Kühlplatz
        [0, 1, 0, 0],          # Dessert <= 2/3 der Gäste
        [0, 0, 1, 1],          # Getränke <= 3 pro Gast
        [1, 0, -1, -1],        # Raclette <= Getränke (mind. 1 Getränk pro Raclette)
    ])
    b = np.array([budget, helfer * STUNDEN * 60, kuehl,
                  DESSERT_MAX, GETRAENKE_MAX, 0])
    res = linprog(-gewinn, A_ub=A, b_ub=b,
                  bounds=[(0, GAESTE), (0, None), (0, None), (0, None)],
                  integrality=np.ones(4) if ganzzahlig else None,
                  method="highs")
    return res, A @ res.x, b


def ausgeben(titel, res, verbrauch, grenze):
    print(f"\n=== {titel} ===")
    for name, x in zip(produkte, res.x):
        print(f"  {name:<12} {x:7.1f} Portionen")
    print(f"  Gewinn       {-res.fun:7.1f} CHF")
    print("  Auslastung der Grenzen:")
    for name, v, g in zip(nebenbedingungen[:5], verbrauch[:5], grenze[:5]):
        print(f"    {name:<20} {v:8.1f} von {g:8.1f}  ({v / g * 100:5.1f} %)")
    print(f"    {'Saalplätze':<20} {res.x[0]:8.1f} von {GAESTE:8.1f}  "
          f"({res.x[0] / GAESTE * 100:5.1f} %)")


# Basisfall (Simplex, Portionen dürfen Dezimalzahlen sein)
res, verbrauch, grenze = loesen()
ausgeben("Basisfall", res, verbrauch, grenze)

# Ganzzahlige Portionen (praktisch umsetzbar)
res_int, v_int, g_int = loesen(ganzzahlig=True)
ausgeben("Basisfall mit ganzen Portionen", res_int, v_int, g_int)

# Variante 1: weniger Einkaufsbudget
res_b, v_b, g_b = loesen(budget=1200)
ausgeben("Variante: Einkaufsbudget nur 1200 CHF", res_b, v_b, g_b)

# Variante 2: zwei Helfer mehr
res_h, v_h, g_h = loesen(helfer=HELFER + 2)
ausgeben("Variante: 8 statt 6 Helfer", res_h, v_h, g_h)

# Variante 3: mehr Kühlplatz
res_k, v_k, g_k = loesen(kuehl=KUEHLPLATZ + 100)
ausgeben("Variante: 100 Liter mehr Kühlplatz", res_k, v_k, g_k)

# ---------------------------------------------------------------
# 3) Grafik: Gewinnbeitrag pro Produkt und Auslastung der Grenzen
# ---------------------------------------------------------------
BLAU = "#2a78d6"        # Datenfarbe (eine Serie)
GRAU = "#c3c2b7"        # nicht ausgeschöpfte Grenzen
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
FLAECHE = "#fcfcfb"

portionen = res.x
beitrag = gewinn * portionen
grenzen_namen = nebenbedingungen[:5] + ["Saalplätze"]
auslastung = np.append(verbrauch[:5] / grenze[:5] * 100, res.x[0] / GAESTE * 100)
ausgeschoepft = auslastung >= 99.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3), facecolor=FLAECHE,
                               gridspec_kw={"width_ratios": [1, 1.15]})

# Links: Gewinnbeitrag pro Produkt
ypos = np.arange(len(produkte))[::-1]
ax1.barh(ypos, beitrag, color=BLAU, height=0.55)
for y, b, p in zip(ypos, beitrag, portionen):
    ax1.text(b + 25, y, f"{b:,.0f} CHF  ({p:.0f} Port.)".replace(",", "'"),
             va="center", fontsize=8.5, color=TEXT)
ax1.set_yticks(ypos)
ax1.set_yticklabels(produkte, fontsize=9, color=TEXT)
ax1.set_xlim(0, beitrag.max() * 2.05)
ax1.set_title(f"Gewinn pro Produkt (total {-res.fun:,.0f} CHF)".replace(",", "'"),
              fontsize=10, color=TEXT, loc="left")
ax1.set_xticks([])

# Rechts: Auslastung der Grenzen
ypos2 = np.arange(len(auslastung))[::-1]
farben = [BLAU if a else GRAU for a in ausgeschoepft]
ax2.barh(ypos2, auslastung, color=farben, height=0.55)
ax2.axvline(100, color=TEXT2, linewidth=0.8, linestyle="--")
for y, a, voll in zip(ypos2, auslastung, ausgeschoepft):
    if voll:
        ax2.text(a + 2, y, f"{a:.0f} %", va="center", fontsize=8.5, color=TEXT)
    else:   # Label im Balken, damit es die 100-%-Linie nicht schneidet
        ax2.text(a - 2, y, f"{a:.0f} %", va="center", ha="right",
                 fontsize=8.5, color=TEXT)
ax2.set_yticks(ypos2)
ax2.set_yticklabels(grenzen_namen, fontsize=9, color=TEXT)
ax2.set_xlim(0, 125)
ax2.set_title("Auslastung der Grenzen\n(blau = ausgeschöpft)",
              fontsize=10, color=TEXT, loc="left")
ax2.set_xticks([])

for ax in (ax1, ax2):
    ax.set_facecolor(FLAECHE)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(TEXT2)
    ax.tick_params(axis="y", length=0)

fig.tight_layout()
fig.savefig("m04_01_raclette_optimierung.png", dpi=200, facecolor=FLAECHE)
print("\nGrafik gespeichert: m04_01_raclette_optimierung.png")
