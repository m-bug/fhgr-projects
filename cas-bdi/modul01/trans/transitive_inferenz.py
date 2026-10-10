"""
Thema: Transitive Inferenz
Was ich untersuche: Wie viele Vergleiche muss man wirklich stellen?
Formel: Bei n Alternativen gibt es n*(n-1)/2 Paare.
Anwendung: Wenn A > B und B > C gilt, folgt A > C. Dieses Paar muss man nicht mehr fragen.

Hint: Die  Simulation wurde mit Gemini erarbeitet.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
# Alternativen, sortiert nach der (fiktiven) Präferenz des Teams (entspricht meiner Meinung)
# Die ersten drei und GitHub Projects entsprechen der Reihenfolge der Score Card Folie.
TOOLS = [
    "OpenProject", "Redmine", "Jira Cloud", "Taiga", "GitLab Issues",
    "YouTrack", "Plane", "Trello", "GitHub Projects",
]
N_BEISPIEL = len(TOOLS)   # 9 Alternativen -> 36 Paare
N_MAX = 20                # Kurve bis 20 Alternativen
LAEUFE = 800              # Simulationen pro Punkt der Kurve
SEED = 42

# Farben (kopiet von stackoverflow)
FLAECHE = "#ffffff"
TINTE = "#0b0b0b"         # text
TINTE_2 = "#52514e"       # 2. text
GITTER = "#e1e0d9"
ACHSE = "#c3c2b7"
BLAU = "#2a78d6"          # direkt gefragt
ORANGE = "#eb6834"        # gefolgert (gespart)
NEUTRAL = "#898781"       # Referenzlinie: alle Paare

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 2. Simulation: Paare in zufälliger Reihenfolge, gefolgerte Paare überspringen
# ---------------------------------------------------------------------------
def durchlauf(n, rng):
    """Gibt die direkt gefragten Paare zurück und den Rang jeder Alternative.

    R[x, y] = True bedeutet: "x ist besser als y" ist bekannt (gefragt oder gefolgert).
    """
    rang = rng.permutation(n)                 # rang[i] = Platz von Alternative i (0 = beste)
    paare = [(i, j) for i in range(n) for j in range(i + 1, n)]
    reihenfolge = rng.permutation(len(paare))
    R = np.zeros((n, n), dtype=bool)
    gefragt = []
    for k in reihenfolge:
        i, j = paare[k]
        if R[i, j] or R[j, i]:                # schon bekannt -> wird gefolgert
            continue
        besser, schlechter = (i, j) if rang[i] < rang[j] else (j, i)   # das Team urteilt
        gefragt.append((i, j))
        oben = R[:, besser].copy()
        oben[besser] = True                   # alle, die mindestens so gut wie "besser" sind
        unten = R[schlechter, :].copy()
        unten[schlechter] = True              # alle, die höchstens so gut wie "schlechter" sind
        R |= np.outer(oben, unten)            # transitive Folgerung
    return gefragt, rang


# ---------------------------------------------------------------------------
# 3. Kurve: Mittelwert der gestellten Fragen für n = 2 ... N_MAX
# ---------------------------------------------------------------------------
ns = np.arange(2, N_MAX + 1)
alle_paare = ns * (ns - 1) // 2
mittel = np.array([
    np.mean([len(durchlauf(n, rng)[0]) for _ in range(LAEUFE)]) for n in ns
])

print("n | alle Paare | Ø gefragt | gespart")
for n, a, m in zip(ns, alle_paare, mittel):
    print(f"{n:2d} | {a:10d} | {m:9.1f} | {100 * (1 - m / a):5.0f} %")

# ---------------------------------------------------------------------------
# 4. Beispiel-Durchlauf für 9 Alternativen (typisch: nahe am Mittelwert)
# ---------------------------------------------------------------------------
ziel = int(round(mittel[N_BEISPIEL - 2]))
while True:
    gefragt, rang = durchlauf(N_BEISPIEL, rng)
    if len(gefragt) == ziel:
        break

# Anzeige sortiert nach Rang (bester oben/links)
gefragt_felder = {tuple(sorted((rang[i], rang[j]))) for i, j in gefragt}
n_paare = N_BEISPIEL * (N_BEISPIEL - 1) // 2
n_gefragt = len(gefragt_felder)
n_gefolgert = n_paare - n_gefragt
print(f"\nBeispiel: {n_gefragt} von {n_paare} Paaren direkt gefragt, {n_gefolgert} gefolgert")

# ---------------------------------------------------------------------------
# 5. Grafik
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": ACHSE,
    "axes.labelcolor": TINTE_2,
    "xtick.color": TINTE_2,
    "ytick.color": TINTE_2,
    "figure.facecolor": FLAECHE,
    "axes.facecolor": FLAECHE,
})

fig = plt.figure(figsize=(5.4, 6.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 0.88], hspace=0.38,
                      left=0.27, right=0.90, top=0.93, bottom=0.08)

# --- Teil 1: Matrix -------------------------------------------------------
ax1 = fig.add_subplot(gs[0])
LUECKE = 0.09
for i in range(N_BEISPIEL):
    for j in range(i + 1, N_BEISPIEL):
        farbe = BLAU if (i, j) in gefragt_felder else ORANGE
        ax1.add_patch(Rectangle((j - 0.5 + LUECKE / 2, i - 0.5 + LUECKE / 2),
                                1 - LUECKE, 1 - LUECKE, facecolor=farbe, edgecolor="none"))

ax1.set_xlim(-0.5, N_BEISPIEL - 0.5)
ax1.set_ylim(N_BEISPIEL - 0.5, -0.5)
ax1.set_aspect("equal")
for s in ax1.spines.values():
    s.set_visible(False)
ax1.set_xticks(range(N_BEISPIEL))
ax1.set_xticklabels([str(k + 1) for k in range(N_BEISPIEL)], fontsize=8)
ax1.xaxis.tick_top()
ax1.set_yticks(range(N_BEISPIEL))
ax1.set_yticklabels([f"{k + 1}  {name}" for k, name in enumerate(TOOLS)], fontsize=8)
ax1.tick_params(length=0, pad=3)
ax1.set_title(f"So sieht es aus: {N_BEISPIEL} Alternativen, {n_paare} Paare",
              loc="left", fontsize=9, fontweight="bold", color=TINTE, pad=16, x=-0.62)

# Legende im leeren Dreieck unten links
for zeile, (farbe, text) in enumerate([
        (BLAU, f"{n_gefragt} direkt gefragt"),
        (ORANGE, f"{n_gefolgert} gefolgert")]):
    y = N_BEISPIEL - 2 + zeile
    ax1.add_patch(Rectangle((0.0, y - 0.25), 0.5, 0.5, facecolor=farbe, edgecolor="none"))
    ax1.text(0.85, y, text, va="center", ha="left", fontsize=8.5, color=TINTE)

# --- Teil 2: Kurve --------------------------------------------------------
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(ns, mittel, alle_paare, color=ORANGE, alpha=0.22, linewidth=0,
                 label="gefolgert (gespart)")
ax2.plot(ns, alle_paare, color=NEUTRAL, linewidth=2, label="alle Paare")
ax2.plot(ns, mittel, color=BLAU, linewidth=2, label="direkt gefragt (Ø)")

# Markierung für das Beispiel mit 9 Alternativen
m9 = mittel[N_BEISPIEL - 2]
ax2.plot([N_BEISPIEL], [n_paare], "o", color=NEUTRAL, markersize=7,
         markeredgecolor=FLAECHE, markeredgewidth=2)
ax2.plot([N_BEISPIEL], [m9], "o", color=BLAU, markersize=7,
         markeredgecolor=FLAECHE, markeredgewidth=2)

# Direkte Beschriftung am Linienende
ax2.annotate(f"{alle_paare[-1]}", (ns[-1], alle_paare[-1]), xytext=(5, 0),
             textcoords="offset points", va="center", fontsize=8.5, color=TINTE)
ax2.annotate(f"{mittel[-1]:.0f}", (ns[-1], mittel[-1]), xytext=(5, 0),
             textcoords="offset points", va="center", fontsize=8.5, color=TINTE)

ax2.set_xlim(2, N_MAX)
ax2.set_ylim(0, alle_paare[-1] * 1.05)
ax2.set_xticks([2, 5, 9, 15, 20])
ax2.set_xlabel("Anzahl Alternativen", fontsize=8.5)
ax2.set_ylabel("Anzahl Vergleiche", fontsize=8.5)
ax2.grid(axis="y", color=GITTER, linewidth=0.8)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.tick_params(length=0)
ax2.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=TINTE)
ax2.set_title(f"So viel spart es: Mittelwert aus {LAEUFE} Simulationen",
              loc="left", fontsize=9, fontweight="bold", color=TINTE, pad=8, x=-0.2)

fig.savefig("m01_03_transitive_inferenz.png", dpi=300, facecolor=FLAECHE)
print("gespeichert: transitive_inferenz.png")
