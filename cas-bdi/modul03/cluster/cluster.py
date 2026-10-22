import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse

# Das Thema habe ich schonmal in einem anderen Modul bearbeitet. Ich konnte gewisse Codestellen davon übernehmen.

## GETTING STARTED
# pip install numpy matplotlib scikit-learn

# 1. Daten erzeugen
np.random.seed(42)
n_samples = 400

# Synthetische Daten für 4 Kundengruppen
cluster_0 = np.random.multivariate_normal(
    [180, 240], [[1000, 200], [200, 800]], 100
)  # Stammkunden
cluster_1 = np.random.multivariate_normal(
    [355, 185], [[1200, -100], [-100, 600]], 120
)  # Herbst-Kandidaten (Zielgruppe)
cluster_2 = np.random.multivariate_normal(
    [580, 110], [[2000, 300], [300, 500]], 90
)  # Abwanderer
cluster_3 = np.random.multivariate_normal(
    [310, 140], [[1500, 100], [100, 700]], 90
)  # Gelegenheitskunden

X = np.vstack([cluster_0, cluster_1, cluster_2, cluster_3])

# 2. Scaling & Vorbereitung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- GRAFIK 1: Ellbogen-Methode (Elbow Method) - Kompakt ---
inertias = []
K_range = range(1, 8)
for k in K_range:
    kmodel = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmodel.fit(X_scaled)
    inertias.append(kmodel.inertia_)

# figsize in der Höhe auf 2.2 reduziert für ein schmaleres Format
fig_elbow, ax_elbow = plt.subplots(figsize=(4, 2.2), dpi=300)
ax_elbow.plot(K_range, inertias, marker='o', color='#2b5c8f', lw=1.5, ms=4)
ax_elbow.axvline(x=4, color='#d9534f', linestyle='--', lw=1.2, label='Optimales K=4')
ax_elbow.set_title('Ellbogen-Methode', fontsize=8, fontweight='bold', pad=4)
ax_elbow.set_xlabel('Anzahl Cluster (K)', fontsize=7)
ax_elbow.set_ylabel('Varianz', fontsize=7)
ax_elbow.tick_params(labelsize=6)
ax_elbow.legend(fontsize=6, loc='upper right')

# Zusätzliches Padding minimieren, damit es kompakt bleibt
plt.tight_layout(pad=0.5)
plt.savefig('m03_01_elbow_plot.png', dpi=300)
plt.close(fig_elbow)
print("Kompaktes Ellbogen-Diagramm erfolgreich als 'elbow_plot.png' gespeichert!")


# --- GRAFIK 2: Kundensegmentierung (Dein angepasster Original-Code) ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

colors = ["#2b5c8f", "#d9534f", "#8e44ad", "#27ae60"] # Target cluster is Red (#d9534f)
cluster_names = [
    "Cluster 0: Stammkunden",
    "Cluster 1: Herbst-Kandidaten (ZIEL)",
    "Cluster 2: Inaktive",
    "Cluster 3: Gelegenheitskunden",
]

for i in range(4):
    ax.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        c=colors[i],
        label=cluster_names[i],
        alpha=0.65,
        s=35,
        edgecolors="none",
    )

# Zentroiden berechnen (Original-Skala)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=120,
    c="black",
    edgecolors="white",
    linewidth=1.5,
    label="Zentroiden",
    zorder=10,
)

# Highlight Ellipse um Cluster 1 (Herbst-Kandidaten)
target_centroid = centroids[1]
ellipse = Ellipse(
    xy=target_centroid,
    width=140,
    height=110,
    angle=-15,
    edgecolor="#d9534f",
    fc="none",
    lw=2,
    ls="--",
)
ax.add_patch(ellipse)

# Target annotation arrow
ax.annotate(
    "Mailing Target:\nReifenwechsel Reminder",
    xy=(target_centroid[0] + 15, target_centroid[1] + 10),
    xytext=(target_centroid[0] + 60, target_centroid[1] + 50),
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="#d9534f", lw=1.5
    ),
    fontsize=8,
    fontweight="bold",
    color="#d9534f",
    bbox=dict(
        boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#d9534f", lw=1
    ),
)

ax.set_title(
    "K-Means Kundensegmentierung (Reifenservice)", fontsize=10, fontweight="bold", pad=10
)
ax.set_xlabel("Tage seit letztem Service", fontsize=8.5)
ax.set_ylabel("Ø Umsatz pro Besuch (Fr.)", fontsize=8.5)
ax.tick_params(labelsize=8)
ax.legend(fontsize=6.5, loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig("m03_01_cluster_plot.png", dpi=300)
plt.close(fig)
print("Cluster-Diagramm erfolgreich als 'm03_01_cluster_plot.png' gespeichert!")