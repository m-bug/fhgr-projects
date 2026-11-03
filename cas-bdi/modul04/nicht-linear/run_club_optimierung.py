import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

## Getting Started:
# pip install numpy scipy matplot

# ==============================================================================
# 1. DATEN: WOHNORTE DER LÄUFERINNEN UND LÄUFER (ZÜRICH QUARTIERE)
# Coordinates im Swiss LV95 System (von Hand kopiert von https://data.stadt-zuerich.ch)
# ==============================================================================
# Format: "Quartier": (X-Koordinate [km], Y-Koordinate [km], Anzahl Läufer/Gewicht w_i)
quartiere_data = {
    "Oerlikon": (683.5, 251.5, 8),
    "Altstetten": (679.0, 248.0, 6),
    "Aussersihl": (682.0, 247.5, 7),
    "Seefeld": (684.5, 246.0, 4),
    "Enge": (682.5, 245.5, 3),
    "Höngg": (679.5, 250.5, 4),
    "Wipkingen": (681.5, 249.5, 3)
}

# etrahieren in numpy arrays
names = list(quartiere_data.keys())
coords = np.array([[v[0], v[1]] for v in quartiere_data.values()])
weights = np.array([v[2] for v in quartiere_data.values()])

total_members = np.sum(weights)

# ==============================================================================
# 2. ZIELFUNKTION (NICHT-LINEAR)
# Minimalisierung der summierten Distanzen (Wurzel aus Quadratsummen)
# f(X, Y) = sum( w_i * sqrt((X - x_i)^2 + (Y - y_i)^2) )
# source: https://de.wikipedia.org/wiki/Webersches_Standortmodell
# ==============================================================================
def objective_function(point):
    """
    Berechnet die gewichtete Gesamtdistanz aller Läufer zum Kandidaten-Treffpunkt (point).
    point: [X, Y] Koordinaten des Treffpunkts
    """
    X, Y = point
    # Euklidische Distanz zu jedem Quartier (nicht-linear durch np.sqrt) --> eins-zu-eins übersetzt von https://de.wikipedia.org/wiki/Webersches_Standortmodell
    distances = np.sqrt((coords[:, 0] - X)**2 + (coords[:, 1] - Y)**2)
    # Gewichtete Summe aller Wege
    total_distance = np.sum(weights * distances)
    return total_distance

# ==============================================================================
# 3. OPIMIERUNG UND LÖSUNG
# ==============================================================================
# Startwert für den Solver (z. B. einfacher Schwerpunkts-Mittelwert)
initial_guess = np.mean(coords, axis=0)

# Geografische Nebenbedingungen / Schranken (Bounds) innerhalb Zürichs (in km) --> kann unscharf sein, da von Hand von der Website kopiert
bounds = [(675.0, 690.0), (240.0, 255.0)]

# Nicht-lineare Optimierung durchführen mit SLSQP
# Ansatz habe ich von hier: https://arxiv.org/html/2408.13420v1
result = minimize(
    objective_function, 
    initial_guess, 
    method='SLSQP', 
    bounds=bounds
)

# Optimaler Treffpunkt
opt_X, opt_Y = result.x
min_total_dist = result.fun

# Print der Ergebnisse in der Konsole
print("=" * 60)
print("  ERGEBNIS DER NICHT-LINEAREN STANDORTOPTIMIERUNG (RUN-CLUB)")
print("=" * 60)
print(f"Anzahl registrierte Mitglieder: {total_members} Personen")
print(f"Optimaler Treffpunkt (LV95): X = {opt_X:.3f} km, Y = {opt_Y:.3f} km")
print(f"Minimierte Gesamtdistanz (Summe aller Wege): {min_total_dist:.2f} km")
print(f"Durchschnittliche Distanz pro Mitglied: {min_total_dist / total_members:.2f} km")
print("-" * 60)

# Einzelabschnitte berechnen
distances_to_opt = np.sqrt((coords[:, 0] - opt_X)**2 + (coords[:, 1] - opt_Y)**2)
for name, dist, w in zip(names, distances_to_opt, weights):
    print(f"  • Quartier {name:<12}: {w} Läufer, Distanz: {dist:.2f} km (Einzelweg)")
print("=" * 60)

# ==============================================================================
# 4. MATPLOTLIB
# ==============================================================================
plt.figure(figsize=(10, 8))

# 1. Quartiere
plt.scatter(
    coords[:, 0], coords[:, 1], 
    s=weights * 40, 
    color='royalblue', 
    alpha=0.7, 
    edgecolors='black', 
    label='Quartiere'
)

# 2. Treffpunkt --> Quartiere
for i, name in enumerate(names):
    plt.plot([opt_X, coords[i, 0]], [opt_Y, coords[i, 1]], 'k--', alpha=0.3, linewidth=1)
    plt.annotate(
        f"{name} ({weights[i]})", 
        (coords[i, 0], coords[i, 1]), 
        textcoords="offset points", 
        xytext=(8, 5), 
        ha='left', 
        fontsize=9,
        weight='bold'
    )

# 3. Optimaler Treffpunkt
plt.scatter(
    opt_X, opt_Y, 
    s=250, 
    color='crimson', 
    marker='*', 
    edgecolors='black', 
    zorder=5, 
    label=f'Optimaler Treffpunkt\n(X: {opt_X:.2f}, Y: {opt_Y:.2f})'
)

# Diagramm
plt.title('Run-Club Zürich: Standortoptimierung (Weber-Problem)', fontsize=14, pad=15)
plt.xlabel('Ostwert / X-Koordinate [km]', fontsize=11)
plt.ylabel('Nordwert / Y-Koordinate [km]', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Grafik
plt.savefig('m04_03_runclub_nichtlinear.png', dpi=300)
plt.show()