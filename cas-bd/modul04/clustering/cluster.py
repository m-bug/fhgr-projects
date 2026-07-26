"""
Verhaltensbasiertes Clustering – Netflix Users Dataset (Kaggle, synthetisch)
=============================================================================

Idee: RFM-analoges Clustering auf Streaming-Verhalten statt E-Commerce.

    Recency-Analog    -> Tage seit Last_Login
    Frequency-Analog  -> Watch_Time_Hours (Nutzungsintensität im letzten Monat)
    Monetary-Analog   -> Subscription_Type (Basic < Standard < Premium, ordinal codiert)

Erwartete Spalten in der CSV:
    User_ID, Name, Age, Country, Subscription_Type,
    Watch_Time_Hours, Favorite_Genre, Last_Login

Installation:
    pip install pandas numpy scikit-learn matplotlib seaborn

Ablauf:
1. Daten laden & prüfen
2. Recency/Frequency/Monetary-Analoge Kennzahlen ableiten
3. Log-Transformation + Skalierung
4. Optimale Clusteranzahl bestimmen (Elbow + Silhouette)
5. K-Means Clustering
6. Visualisierung (2D-Scatter, 3D-Scatter, PCA, Boxplots)
7. Cluster-Profile inkl. Zusatzinfos (Age, Country, Favorite_Genre)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (für 3D-Plot benötigt)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# 0. PATH
# source: https://www.kaggle.com/datasets/smayanj/netflix-users-database
# ---------------------------------------------------------------------------
DATA_PATH = "netflix_users.csv"

# ---------------------------------------------------------------------------
# 1. Daten laden
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["Last_Login"] = pd.to_datetime(df["Last_Login"])

print(f"Anzahl Nutzer: {len(df):,}")
print(df.head())
print("\nFehlende Werte je Spalte:")
print(df.isna().sum())

# Fehlende Werte in relevanten Spalten vorsichtshalber entfernen
df = df.dropna(subset=["Last_Login", "Watch_Time_Hours", "Subscription_Type"]).copy()

# ---------------------------------------------------------------------------
# 2. Verhaltenskennzahlen ableiten (RFM-Analog)
# ---------------------------------------------------------------------------
# Recency: Tage seit letztem Login (bezogen auf den jüngsten Login im Datensatz + 1 Tag)
snapshot_date = df["Last_Login"].max() + pd.Timedelta(days=1)
df["Recency"] = (snapshot_date - df["Last_Login"]).dt.days

# Frequency-Analog: Watch_Time_Hours direkt übernehmen
df["Frequency"] = df["Watch_Time_Hours"]

# Monetary-Analog: Subscription_Type ordinal codieren (Basic < Standard < Premium)
plan_order = {"Basic": 1, "Standard": 2, "Premium": 3}
df["Monetary"] = df["Subscription_Type"].map(plan_order)

if df["Monetary"].isna().any():
    unbekannt = df.loc[df["Monetary"].isna(), "Subscription_Type"].unique()
    raise ValueError(f"Unbekannte Subscription_Type-Werte gefunden: {unbekannt}. "
                      f"Bitte plan_order-Mapping anpassen.")

rfm = df[["User_ID", "Recency", "Frequency", "Monetary",
          "Age", "Country", "Favorite_Genre", "Subscription_Type"]].copy()

print("\nVerhaltenskennzahlen (erste Zeilen):")
print(rfm.head())
print("\nStatistische Kennzahlen:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

# ---------------------------------------------------------------------------
# 3. Log-Transformation & Skalierung
# ---------------------------------------------------------------------------
rfm["Recency_log"] = np.log1p(rfm["Recency"])
rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
# Monetary ist bereits ordinal (1-3), Log-Transformation hier nicht nötig
rfm["Monetary_scaled_input"] = rfm["Monetary"]

features = ["Recency_log", "Frequency_log", "Monetary_scaled_input"]
X = rfm[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 4. Optimale Clusteranzahl bestimmen (Elbow + Silhouette)
# ---------------------------------------------------------------------------
k_range = range(2, 9)
inertias = []
silhouettes = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow-Methode")
axes[0].set_xlabel("Anzahl Cluster (k)")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(list(k_range), silhouettes, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Anzahl Cluster (k)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig("01_k_auswahl.png", bbox_inches="tight")
plt.show()

best_k = list(k_range)[int(np.argmax(silhouettes))]
print(f"\nEmpfohlene Clusteranzahl (bester Silhouette Score): k = {best_k}")

# --> Hier ggf. manuell überschreiben, z.B.: best_k = 4
K = best_k

# ---------------------------------------------------------------------------
# 5. K-Means Clustering
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"\nClustergrößen (k={K}):")
print(rfm["Cluster"].value_counts().sort_index())

# ---------------------------------------------------------------------------
# 6. Visualisierung der Cluster
# ---------------------------------------------------------------------------
palette = sns.color_palette("Set2", K)

# --- 6a. 2D-Scatterplots ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.scatterplot(data=rfm, x="Recency", y="Frequency", hue="Cluster",
                 palette=palette, ax=axes[0], alpha=0.5, s=20)
axes[0].set_title("Recency vs. Watch_Time_Hours")
axes[0].set_ylabel("Watch_Time_Hours")

sns.scatterplot(data=rfm, x="Monetary", y="Frequency", hue="Cluster",
                 palette=palette, ax=axes[1], alpha=0.5, s=20)
axes[1].set_title("Subscription_Type vs. Watch_Time_Hours")
axes[1].set_xlabel("Subscription_Type (1=Basic, 2=Standard, 3=Premium)")
axes[1].set_ylabel("Watch_Time_Hours")

sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster",
                 palette=palette, ax=axes[2], alpha=0.5, s=20)
axes[2].set_title("Recency vs. Subscription_Type")
axes[2].set_ylabel("Subscription_Type (1=Basic, 2=Standard, 3=Premium)")

plt.tight_layout()
plt.savefig("02_cluster_scatter_2d.png", bbox_inches="tight")
plt.show()

# --- 6b. 3D-Scatterplot ---
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for c in sorted(rfm["Cluster"].unique()):
    subset = rfm[rfm["Cluster"] == c]
    ax.scatter(subset["Recency"], subset["Frequency"], subset["Monetary"],
               label=f"Cluster {c}", alpha=0.5, s=15, color=palette[c])

ax.set_xlabel("Recency (Tage seit Login)")
ax.set_ylabel("Watch_Time_Hours")
ax.set_zlabel("Subscription_Type (1-3)")
ax.set_title("Netflix-Nutzer-Cluster (3D)")
ax.legend()
plt.tight_layout()
plt.savefig("03_cluster_scatter_3d.png", bbox_inches="tight")
plt.show()

# --- 6c. PCA-Projektion ---
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
rfm["PCA1"] = pca_coords[:, 0]
rfm["PCA2"] = pca_coords[:, 1]

plt.figure(figsize=(7, 6))
sns.scatterplot(data=rfm, x="PCA1", y="PCA2", hue="Cluster",
                 palette=palette, alpha=0.5, s=20)
plt.title(f"Cluster in PCA-Projektion (erklärte Varianz: "
          f"{pca.explained_variance_ratio_.sum():.1%})")
plt.tight_layout()
plt.savefig("04_cluster_pca.png", bbox_inches="tight")
plt.show()

# --- 6d. Boxplots je Kennzahl und Cluster ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, label in zip(
    axes,
    ["Recency", "Frequency", "Monetary"],
    ["Recency (Tage)", "Watch_Time_Hours", "Subscription_Type (1-3)"]
):
    sns.boxplot(data=rfm, x="Cluster", y=col, palette=palette, ax=ax, showfliers=False)
    ax.set_title(f"{label} je Cluster")
    ax.set_ylabel(label)
plt.tight_layout()
plt.savefig("05_cluster_boxplots.png", bbox_inches="tight")
plt.show()

# --- 6e. Zusatzauswertung: Favorite_Genre-Verteilung je Cluster ---
genre_dist = pd.crosstab(rfm["Cluster"], rfm["Favorite_Genre"], normalize="index")
genre_dist.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20")
plt.title("Verteilung Favorite_Genre je Cluster (Anteile)")
plt.ylabel("Anteil")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("06_genre_verteilung.png", bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 7. Cluster-Profile
# ---------------------------------------------------------------------------
profile = rfm.groupby("Cluster").agg(
    Nutzeranzahl=("User_ID", "count"),
    Recency_mean=("Recency", "mean"),
    WatchTime_mean=("Frequency", "mean"),
    SubType_mean=("Monetary", "mean"),
    Age_mean=("Age", "mean")
).round(1).sort_values("WatchTime_mean", ascending=False)

# Häufigster Subscription_Type und häufigstes Genre je Cluster
profile["Häufigster_Plan"] = rfm.groupby("Cluster")["Subscription_Type"] \
    .agg(lambda x: x.mode().iloc[0])
profile["Häufigstes_Genre"] = rfm.groupby("Cluster")["Favorite_Genre"] \
    .agg(lambda x: x.mode().iloc[0])

print("\n=== Cluster-Profile ===")
print(profile)

profile.to_csv("netflix_cluster_profile.csv")
rfm.to_csv("netflix_users_mit_clustern.csv", index=False)

print("\nFertig. Ergebnisse gespeichert als:")
print(" - netflix_users_mit_clustern.csv (Nutzerdaten + Clusterzuordnung)")
print(" - netflix_cluster_profile.csv (aggregiertes Profil je Cluster)")
print(" - 01_k_auswahl.png ... 06_genre_verteilung.png")