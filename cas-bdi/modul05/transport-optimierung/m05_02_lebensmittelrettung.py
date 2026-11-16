import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.optimize import milp, LinearConstraint, Bounds

# ------------------------------------------------------------
# 1. static date
# koordinaten kopiert von map.geo.admin.ch
# beispiel: https://s.geo.admin.ch/xbtswp24xsyi
# ------------------------------------------------------------
koordinaten = {
    "Oerlikon":       (47.4110, 8.5440),
    "Altstetten":     (47.3910, 8.4880),
    "Enge":           (47.3640, 8.5300),
    "Aussersihl":     (47.3770, 8.5240),
    "Schwamendingen": (47.4040, 8.5700),
    "Albisrieden":    (47.3790, 8.4850),
    "Seebach":        (47.4230, 8.5450),
}
angebot = {"Oerlikon": 14, "Altstetten": 20, "Enge": 18}                      # Total: 52 Paletten
bedarf = {"Aussersihl": 14, "Schwamendingen": 10, "Albisrieden": 9, "Seebach": 12}  # Mindestbedarf: 45

UMWEG = 1.3
MAX_KM = 7.5

S = list(angebot)
D = list(bedarf)


def luftlinie_km(a, b):
    """Distanz in km (Haversine) zwischen zwei Punkten (Breite, Länge)."""
    la1, lo1, la2, lo2 = map(np.radians, (*a, *b))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


kosten = {}
for i in S:
    for j in D:
        km = luftlinie_km(koordinaten[i], koordinaten[j]) * UMWEG
        if km <= MAX_KM:
            kosten[(i, j)] = int(round(km * 100))

verbindungen = list(kosten)
m = len(verbindungen)
print(f"Angebot: {sum(angebot.values())} Paletten, Mindestbedarf: {sum(bedarf.values())} Paletten")
print(f"Erlaubte Verbindungen (<= {MAX_KM} km): {m} von {len(S) * len(D)}\n")
print("Kosten k_ij (km pro Palette):")
for (i, j) in verbindungen:
    print(f"  {i:<11} -> {j:<15} {kosten[(i, j)] / 100:5.2f} km")
print()

# ------------------------------------------------------------
# 2. Modell: Alle 52 Paletten werden verteilt (Abgabestellen >= Mindestbedarf)
# ------------------------------------------------------------
c = np.array([kosten[v] for v in verbindungen], dtype=float)

zeilen, lb, ub = [], [], []
# Standorte: GENAU das gesamte Angebot abgeben (alle 52 Paletten sollen weg)
for i in S:
    zeilen.append([1.0 if v[0] == i else 0.0 for v in verbindungen])
    lb.append(angebot[i])
    ub.append(angebot[i])

# Abgabestellen: MINDESTENS den Bedarf decken (kein Deckel nach oben)
# lb = lower bound, ub = upper bound
for j in D:
    zeilen.append([1.0 if v[1] == j else 0.0 for v in verbindungen])
    lb.append(bedarf[j])
    ub.append(np.inf)

res = milp(
    c=c,
    constraints=LinearConstraint(np.array(zeilen), lb, ub),
    integrality=np.ones(m),
    bounds=Bounds(0, np.inf),
)
assert res.success, f"Keine Lösung: {res.message}"

x = {v: int(round(res.x[k])) for k, v in enumerate(verbindungen)}
total_modell = int(round(res.fun))

print("Optimale Verteilung (Paletten):")
for (i, j), n in x.items():
    if n > 0:
        print(f"  {i:<11} -> {j:<15} {n:>3} Pal. ({kosten[(i, j)] / 100:.2f} km je Palette)")
print(f"Total: {total_modell / 100:.1f} Palettenkilometer\n")

# ------------------------------------------------------------
# 3. Kontrollen
# source: https://towardsdatascience.com/introduction-to-minimum-cost-flow-optimization-in-python/
# Code korrigiert durch: Google Gemini
# ------------------------------------------------------------
# (a) Min-Cost-Flow: Ein Dummy-Knoten "Überschuss" nimmt die freien Kapazitäten der Abgabestellen auf
G = nx.DiGraph()
for i in S:
    G.add_node(i, demand=-angebot[i])
for j in D:
    G.add_node(j, demand=bedarf[j])
ueberschuss = sum(angebot.values()) - sum(bedarf.values())
G.add_node("Uebernahme", demand=ueberschuss)

for (i, j), k in kosten.items():
    G.add_edge(i, j, weight=k)
for j in D:
    # Abgabestellen können beliebig viele Zusatzpaletten an "Uebernahme" weiterleiten
    G.add_edge(j, "Uebernahme", weight=0)

fluss = nx.min_cost_flow(G)
total_fluss = nx.cost_of_flow(G, fluss)
print(f"Kontrolle Min-Cost-Flow: {total_fluss / 100:.1f} Palettenkilometer")
assert total_fluss == total_modell, "Modell und Min-Cost-Flow weichen ab!"

# (b) Nebenbedingungen am Resultat nachrechnen
for i in S:
    assert sum(n for (a, _), n in x.items() if a == i) == angebot[i]
for j in D:
    assert sum(n for (_, b), n in x.items() if b == j) >= bedarf[j]
print("Kontrolle Nebenbedingungen: Gesamtes Angebot verteilt, Mindestbedarf überall gedeckt.\n")

# (c) Vergleich mit Faustregel: 1. Mindestbedarf decken, 2. Rest an nächsten Abnehmer verteilen
vorrat = dict(angebot)
total_greedy = 0
greedy = {(i, j): 0 for (i, j) in verbindungen}

# Schritt 1: Mindestbedarf der Abgabestellen decken
for j in D:
    noch = bedarf[j]
    for i in sorted([s for s in S if (s, j) in kosten], key=lambda s: kosten[(s, j)]):
        n = min(noch, vorrat[i])
        if n > 0:
            greedy[(i, j)] += n
            vorrat[i] -= n
            noch -= n
            total_greedy += n * kosten[(i, j)]
        if noch == 0:
            break
    assert noch == 0, "Faustregel findet keine Lösung für Mindestbedarf"

# Schritt 2: Verbleibende Paletten der Standorte an den jeweils nächsten Abnehmer schicken
for i in S:
    if vorrat[i] > 0:
        ziel = min([d for d in D if (i, d) in kosten], key=lambda d: kosten[(i, d)])
        n = vorrat[i]
        greedy[(i, ziel)] += n
        vorrat[i] = 0
        total_greedy += n * kosten[(i, ziel)]

print(f"Vergleich Faustregel «nächster Standort»: {total_greedy / 100:.1f} Palettenkilometer")
ersparnis = (total_greedy - total_modell) / total_greedy * 100
print(f"Ersparnis durch die Optimierung: {(total_greedy - total_modell) / 100:.1f} "
      f"Palettenkilometer ({ersparnis:.1f} %)")
print(f"Durchschnitt: {total_modell / 100 / sum(angebot.values()):.2f} km pro Palette\n")

genutzt = {i: sum(n for (a, _), n in x.items() if a == i) for i in S}
erhalten = {j: sum(n for (_, b), n in x.items() if b == j) for j in D}

print("Auslastung der Standorte:")
for i in S:
    print(f"  {i:<11} {genutzt[i]:>2} von {angebot[i]} Paletten abgegeben")
print("\nAufnahme der Abgabestellen:")
for j in D:
    print(f"  {j:<15} {erhalten[j]:>2} Paletten erhalten (Mindestbedarf: {bedarf[j]})")

# ------------------------------------------------------------
# 4. Grafik: Karte der Lieferungen
# ------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
C_SUPPLY, C_DEMAND = "#2a78d6", "#eb6834"

lat0, lon0 = 47.38, 8.52
pos = {n: ((lo - lon0) * 111.32 * np.cos(np.radians(la)), (la - lat0) * 110.57)
       for n, (la, lo) in koordinaten.items()}

plt.rcParams.update({"font.size": 11, "text.color": INK})
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

for (i, j) in verbindungen:
    if x[(i, j)] == 0:
        (x0, y0), (x1, y1) = pos[i], pos[j]
        ax.plot([x0, x1], [y0, y1], color="#c3c2b7", lw=1, zorder=1)

for (i, j), n in x.items():
    if n > 0:
        (x0, y0), (x1, y1) = pos[i], pos[j]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.3", color=INK2,
                                    lw=0.8 + 0.3 * n, shrinkA=13, shrinkB=13, mutation_scale=14),
                    zorder=2)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        laenge = np.hypot(x1 - x0, y1 - y0)
        nx_, ny_ = -(y1 - y0) / laenge, (x1 - x0) / laenge
        ax.text(mx + 0.28 * nx_, my + 0.28 * ny_, f"{n}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK, zorder=4,
                path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])

versatz = {
    "Oerlikon": (22, 16, "left"), "Altstetten": (-20, 0, "right"), "Enge": (0, -28, "center"),
    "Aussersihl": (-20, 0, "right"), "Schwamendingen": (20, 0, "left"),
    "Albisrieden": (-20, -4, "right"), "Seebach": (0, 28, "center"),
}
for n in S:
    ax.scatter(*pos[n], s=330, marker="s", color=C_SUPPLY, edgecolor=SURFACE,
               linewidth=2, zorder=5)
    dx, dy, ha = versatz[n]
    ax.annotate(f"{n}\n{genutzt[n]} von {angebot[n]} Pal.", pos[n], xytext=(dx, dy),
                textcoords="offset points", ha=ha, va="center", fontsize=10, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
for n in D:
    ax.scatter(*pos[n], s=330, marker="o", color=C_DEMAND, edgecolor=SURFACE,
               linewidth=2, zorder=5)
    dx, dy, ha = versatz[n]
    ax.annotate(f"{n}\n{erhalten[n]} Pal. (Min. {bedarf[n]})", pos[n], xytext=(dx, dy),
                textcoords="offset points", ha=ha, va="center", fontsize=10, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))

fig.legend(handles=[
    Line2D([0], [0], marker="s", color="none", markerfacecolor=C_SUPPLY, markersize=11,
           label="Detailhandel-Standort (Überschuss)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_DEMAND, markersize=11,
           label="Abgabestelle (Aufnahme)"),
    Line2D([0], [0], color=INK2, lw=3, label="Lieferung, Zahl = Paletten"),
    Line2D([0], [0], color="#c3c2b7", lw=1.2, label="erlaubt, aber nicht genutzt"),
], loc="lower center", ncol=2, frameon=False, fontsize=9.5)

fig.suptitle(
    f"Optimale Verteilung (52 Paletten): {total_modell / 100:.0f} Palettenkilometer "
    f"({ersparnis:.0f} % weniger als «nächster Standort»)",
    fontsize=12, color=INK2, y=0.96
)
ax.set_aspect("equal")
ax.margins(0.16)
ax.axis("off")
plt.tight_layout(rect=(0, 0.10, 1, 0.93))
plt.savefig("m05_02_lebensmittelrettung.png", dpi=200, facecolor=SURFACE)