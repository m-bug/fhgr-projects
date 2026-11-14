"""

#GETTING STARTED
# pip install numpy matplotlib scipy

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds

# ------------------------------------------------------------
# 1. Netz: Städte (Länge, Breite) und Zugverbindungen
# ------------------------------------------------------------

# GPS Koordinaten
staedte = {
    "Zürich":      (8.54, 47.38),
    "Innsbruck":   (11.40, 47.27),
    "München":     (11.58, 48.14),
    "Salzburg":    (13.04, 47.81),
    "Wien":        (16.37, 48.21),
    "Budapest":    (19.04, 47.50),
    "Oradea":      (21.92, 47.05),
    "Cluj-Napoca": (23.59, 46.77),
}

# Pfeile (von, nach, Reisezeit in Stunden)
# nur in die Reiserichtung da keine Rückreise
verbindungen = [
    ("Zürich",    "Innsbruck",   3.5),
    ("Zürich",    "München",     3.5),
    ("Zürich",    "Wien",        8.0),
    ("Innsbruck", "Salzburg",    2.0),
    ("Innsbruck", "Wien",        4.5),
    ("München",   "Salzburg",    1.5),
    ("München",   "Wien",        4.0),
    ("Salzburg",  "Wien",        2.5),
    ("Wien",      "Budapest",    2.5),
    ("Budapest",  "Oradea",      4.0),
    ("Budapest",  "Cluj-Napoca", 9.5),
    ("Oradea",    "Cluj-Napoca", 4.0),
]

start, ziel = "Zürich", "Cluj-Napoca"
namen = list(staedte)
n, m = len(namen), len(verbindungen)

# ------------------------------------------------------------
# 2. Modell: Entscheidungsvariablen x_ij (binär), Ziel, Nebenbedingungen
# ------------------------------------------------------------
# Zielfunktion: Summe k_ij * x_ij  -> minimieren
zeiten = np.array([k for _, _, k in verbindungen])

# Flusserhaltung: (rein) - (raus) = rhs
#   Start: -1 (ein Pfeil geht raus), Ziel: +1 (ein Pfeil kommt rein),
#   alle Zwischenstädte: 0
A = np.zeros((n, m))
for j, (von, nach, _) in enumerate(verbindungen):
    A[namen.index(von), j] -= 1
    A[namen.index(nach), j] += 1

rhs = np.zeros(n)
rhs[namen.index(start)] = -1
rhs[namen.index(ziel)] = 1

# beispiel: https://medium.com/@jenny6449/optimizing-inventory-levels-with-milp-easy-walkthrough-in-python-a0ca1726c1f9
res = milp(
    c=zeiten,
    constraints=LinearConstraint(A, lb=rhs, ub=rhs),
    integrality=np.ones(m),          # alle x_ij ganzzahlig ...
    bounds=Bounds(0, 1),             # ... und zwischen 0 und 1, also binär
)

gewaehlt = [verbindungen[j] for j in range(m) if res.x[j] > 0.5]
zeit_modell = res.fun

print("Optimaler Weg laut Modell (binäres LP):")
for von, nach, k in gewaehlt:
    print(f"  {von} -> {nach}: {k} h")
print(f"Gesamtzeit: {zeit_modell} h\n")

# ------------------------------------------------------------
# 3. Kontrollen: Dijkstra und Durchprobieren aller Wege
# source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html
# Implementation: Google Gemini
# ------------------------------------------------------------
G = nx.DiGraph()
G.add_weighted_edges_from(verbindungen)

zeit_dijkstra = nx.dijkstra_path_length(G, start, ziel, weight="weight")
print(f"Kontrolle Dijkstra: {zeit_dijkstra} h")
assert abs(zeit_modell - zeit_dijkstra) < 1e-9, "Modell und Dijkstra weichen ab!"

alle = list(nx.all_simple_paths(G, start, ziel))
zeit_von = lambda p: sum(G[a][b]["weight"] for a, b in zip(p, p[1:]))
bestzeit = min(zeit_von(p) for p in alle)
print(f"Kontrolle Durchprobieren: {len(alle)} mögliche Wege, beste Zeit {bestzeit} h")
assert abs(zeit_modell - bestzeit) < 1e-9, "Modell und Durchprobieren weichen ab!"
print("Alle drei Methoden stimmen überein.\n")

optimale_wege = [p for p in alle if abs(zeit_von(p) - bestzeit) < 1e-9]
print(f"Anzahl gleich schneller optimaler Wege: {len(optimale_wege)}")
for p in optimale_wege:
    print("  ", " -> ".join(p))

# Alle Pfeile, die in mindestens einem optimalen Weg vorkommen
optimale_pfeile = {(a, b) for p in optimale_wege for a, b in zip(p, p[1:])}

# ------------------------------------------------------------
# 4. Grafik: Netz mit optimalen Wegen in rot
# ------------------------------------------------------------
# Schematische Karte: Länge unverändert, Breite 4-fach gestreckt,
# damit sich die Pfeile und Beschriftungen nicht überdecken.
STRECK = 4
pos = {name: (x, y * STRECK) for name, (x, y) in staedte.items()}

# Verbindungen, die gebogen gezeichnet werden (sonst läuft der Pfeil durch Salzburg)
bogen = {("Zürich", "Wien"): 0.38, ("Budapest", "Cluj-Napoca"): 0.3}

# Position der Zeitbeschriftung entlang des Pfeils (0 = Start, 1 = Ende)
label_pos = {("Zürich", "Innsbruck"): 0.5, ("Zürich", "München"): 0.5}

# Versatz der Städtenamen (Punkte)
name_versatz = {
    "Zürich": (0, -16), "Innsbruck": (0, -16), "München": (0, 10),
    "Salzburg": (0, -16), "Wien": (0, 10), "Budapest": (34, 12),
    "Oradea": (0, -16), "Cluj-Napoca": (0, -16),
}

fig, ax = plt.subplots(figsize=(11, 5))

for von, nach, k in verbindungen:
    x0, y0 = pos[von]
    x1, y1 = pos[nach]
    optimal = (von, nach) in optimale_pfeile
    rad = bogen.get((von, nach), 0.0)
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>", mutation_scale=14,
            connectionstyle=f"arc3,rad={rad}",
            color="red" if optimal else "#b0b0b0",
            lw=2.8 if optimal else 1.2,
            shrinkA=9, shrinkB=9,
        ),
        zorder=3 if optimal else 2,
    )
    # Mittelpunkt der Linie bzw. des Bogens für die Zeitbeschriftung
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if rad != 0:
        dx, dy = x1 - x0, y1 - y0
        cx, cy = mx + rad * dy, my - rad * dx            # Kontrollpunkt des Bogens
        mx, my = 0.25 * x0 + 0.5 * cx + 0.25 * x1, 0.25 * y0 + 0.5 * cy + 0.25 * y1
    ax.text(mx, my, f"{k:g} h", ha="center", va="center", fontsize=9,
            color="red" if optimal else "#606060",
            fontweight="bold" if optimal else "normal",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
            zorder=5)

for name, (x, y) in pos.items():
    ax.scatter(x, y, s=70, color="steelblue", zorder=4)
    ox, oy = name_versatz[name]
    ax.annotate(name, (x, y), xytext=(ox, oy), textcoords="offset points",
                ha="center", va="center", fontsize=10, zorder=6)

ax.set_title(f"Schnellster Weg Zürich - Cluj-Napoca: {zeit_modell:g} h "
             f"({len(optimale_wege)} gleich schnelle Wege in rot)")
ax.set_aspect("equal")
ax.margins(0.08)
ax.axis("off")
plt.tight_layout()
plt.savefig("m05_03_weg_optimierung.png", dpi=200)
