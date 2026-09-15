"""
Datensatz: Breast Cancer Wisconsin (Diagnostic), 
source: via sklearn.datasets --> from 


Getting started: pip install lazypredict pandas matplotlib scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

# ---------------------------------------------------------------------------
# 1. Daten laden
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target

print(f"Datensatz: {X.shape[0]} Beobachtungen, {X.shape[1]} Features")
print(f"Zielvariable: {data.target_names[0]} vs. {data.target_names[1]}")
print(f"Klassenverteilung: {pd.Series(y).value_counts().to_dict()}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 2. LazyPredict: automatisches Benchmarking ueber >30 Modelle
# ---------------------------------------------------------------------------
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Ergebnistabelle aufbereiten
results = models.reset_index().rename(columns={"index": "Model"})
results = results.sort_values("Accuracy", ascending=False).reset_index(drop=True)

print(f"Anzahl getesteter Modelle: {len(results)}\n")
print(results.head(10).to_string(index=False))

results.to_csv("m07_02_automl_results.csv", index=False)
print("\n-> Vollstaendige Ergebnistabelle gespeichert: m06_07_automl_results.csv")

# ---------------------------------------------------------------------------
# 3. Visualisierung: Top-10 Modelle nach Accuracy
# ---------------------------------------------------------------------------
top10 = results.head(10).sort_values("Accuracy", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.barh(top10["Model"], top10["Accuracy"], color="#4C72B0", edgecolor="black", linewidth=0.5)

for bar, acc in zip(bars, top10["Accuracy"]):
    ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
            f"{acc:.3f}", va="center", ha="right", color="white", fontsize=9, fontweight="bold")

ax.set_xlim(0.85, 1.0)
ax.set_xlabel("Accuracy (Test-Set)")
ax.set_title("LazyPredict: Top-10 Modelle - Breast Cancer Wisconsin", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("m07_02_automl_top10.png", dpi=300)
print("-> Balkendiagramm gespeichert: m06_07_automl_top10.png")
