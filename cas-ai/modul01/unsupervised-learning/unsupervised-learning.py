import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

###
# Getting started:
# 
# pip install matplotlib pandas numpy scikit-learn
# 
##

# ------------------------
# 1. Simulierte Bankdaten
# ------------------------
np.random.seed(42)

# Cluster 1: Junge, digital affine Kunden
cluster_1 = np.random.normal(
    loc=[28, 45, 120, 14],   # Alter, Anzahl Zahlungen, Betrag, Uhrzeit
    scale=[4, 10, 40, 2],
    size=(40, 4)
)

# Cluster 2: Berufstätige, klassische Nutzung
cluster_2 = np.random.normal(
    loc=[45, 20, 450, 18],
    scale=[5, 6, 120, 2],
    size=(40, 4)
)

# Cluster 3: Ältere, wenige aber hohe Zahlungen
cluster_3 = np.random.normal(
    loc=[62, 8, 1500, 11],
    scale=[6, 3, 300, 1.5],
    size=(40, 4)
)

data = np.vstack([cluster_1, cluster_2, cluster_3])

df = pd.DataFrame(
    data,
    columns=[
        "Alter",
        "Anzahl_Zahlungen",
        "Durchschnittsbetrag",
        "Durchschnittliche_Uhrzeit"
    ]
)

# ------------------------
# 2. Clustering
# ------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df)

# ------------------------
# 3. Visualisierung
# ------------------------
plt.figure(figsize=(6,5))

plt.scatter(
    df["Anzahl_Zahlungen"],
    df["Durchschnittsbetrag"],
    c=df["Cluster"],
    s=80,
    alpha=0.7
)

plt.xlabel("Anzahl ausgehender Zahlungen / Monat")
plt.ylabel("Durchschnittlicher Betrag [CHF]")
plt.title("Unsupervised Learning: Zahlungsbasierte Kundensegmente")
plt.grid(True)
plt.tight_layout()

# ------------------------
# 4. Export für Beamer
# ------------------------
plt.savefig("bank_unsupervised_clusters.png", dpi=300)
plt.show()

