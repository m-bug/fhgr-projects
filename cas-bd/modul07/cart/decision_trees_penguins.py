"""
Modul 6 - Entscheidungsbaeume (CART) am Beispiel Palmer Penguins
=================================================================

Theorie-Bezug: Entscheidungsbaeume (Decision Trees) sind eine Alternative
zur Linearen Regression (-> Regressionsbaum / Regression Tree) und zur
Logistischen Regression (-> Klassifikationsbaum / Classification Tree).
Beide Verfahren werden gemeinsam als CART (Classification and Regression
Trees) bezeichnet. In R nutzt man dafuer das Paket rpart mit dem Argument
method = "class" (Klassifikation) bzw. method = "anova" (Regression).
In Python entspricht dies sklearn.tree.DecisionTreeClassifier bzw.
sklearn.tree.DecisionTreeRegressor.

Dieses Skript zeigt beide Varianten anhand des Palmer-Penguins-Datensatzes:
  1. Classification Tree: Zielvariable = species (kategorisch)
  2. Regression Tree:     Zielvariable = body_mass_g (kontinuierlich)

Library credits:
  - palmerpenguins (Horst, Hill, Gorman 2020) fuer den Datensatz
  - scikit-learn (Pedregosa et al. 2011) fuer CART-Implementierung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    plot_tree,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
)

RANDOM_STATE = 42
OUT_DIR = "."  # Anpassen: Zielordner fuer Plots

# -----------------------------------------------------------------------
# 1. Daten laden und aufbereiten
# -----------------------------------------------------------------------
df = load_penguins()
df = df.dropna().reset_index(drop=True)  # CART braucht vollstaendige Faelle
print(f"Datensatz nach dropna(): {df.shape[0]} Beobachtungen")
print(df.head())

# Kategorische Merkmale one-hot-encodieren (island, sex)
df_encoded = pd.get_dummies(df, columns=["island", "sex"], drop_first=True)

# -----------------------------------------------------------------------
# 2. Classification Tree: species ~ .
#    (method = "class" in rpart)
# -----------------------------------------------------------------------
target_clf = "species"
features_clf = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
] + [c for c in df_encoded.columns if c.startswith("island_") or c.startswith("sex_")]

X_clf = df_encoded[features_clf]
y_clf = df_encoded[target_clf]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=RANDOM_STATE, stratify=y_clf
)

clf_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
clf_tree.fit(X_train_c, y_train_c)

y_pred_c = clf_tree.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred_c)
print(f"\n[Classification Tree] Accuracy auf Testdaten: {acc:.3f}")

cm = confusion_matrix(y_test_c, y_pred_c, labels=clf_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_tree.classes_)
fig, ax = plt.subplots(figsize=(4.5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
plt.title("Confusion Matrix – Classification Tree (species)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/m06_04_decision_trees_confusion.png", dpi=200)
plt.close()

# Baum visualisieren
fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(
    clf_tree,
    feature_names=features_clf,
    class_names=clf_tree.classes_,
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
)
plt.title("Classification Tree: species ~ .")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/m06_04_decision_trees_classtree.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------
# 3. Regression Tree: body_mass_g ~ .
#    (method = "anova" in rpart)
# -----------------------------------------------------------------------
target_reg = "body_mass_g"
features_reg = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
] + [c for c in df_encoded.columns if c.startswith("island_") or c.startswith("sex_")] \
  + [c for c in pd.get_dummies(df["species"], prefix="species", drop_first=True).columns]

df_encoded_reg = pd.concat(
    [df_encoded, pd.get_dummies(df["species"], prefix="species", drop_first=True)], axis=1
)

X_reg = df_encoded_reg[features_reg]
y_reg = df_encoded_reg[target_reg]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=RANDOM_STATE
)

reg_tree = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
reg_tree.fit(X_train_r, y_train_r)

y_pred_r = reg_tree.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)
print(f"\n[Regression Tree] RMSE: {rmse:.1f} g, R²: {r2:.3f}")

fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(
    reg_tree,
    feature_names=features_reg,
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
)
plt.title("Regression Tree: body_mass_g ~ .")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/m06_04_decision_trees_regtree.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------
# 4. Feature Importance – Vergleich beider Baeume
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

imp_c = pd.Series(clf_tree.feature_importances_, index=features_clf).sort_values()
imp_c.plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Feature Importance\nClassification Tree (species)")

imp_r = pd.Series(reg_tree.feature_importances_, index=features_reg).sort_values()
imp_r.plot(kind="barh", ax=axes[1], color="#DD8452")
axes[1].set_title("Feature Importance\nRegression Tree (body_mass_g)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/m06_04_decision_trees_importance.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------
# 5. Optional: Cost-Complexity Pruning (Analogon zu printcp()/prune() in R)
# -----------------------------------------------------------------------
# path = clf_tree.cost_complexity_pruning_path(X_train_c, y_train_c)
# ccp_alphas = path.ccp_alphas
# scores = []
# for alpha in ccp_alphas:
#     t = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=alpha)
#     t.fit(X_train_c, y_train_c)
#     scores.append(accuracy_score(y_test_c, t.predict(X_test_c)))
# plt.plot(ccp_alphas, scores, marker="o")
# plt.xlabel("ccp_alpha")
# plt.ylabel("Test Accuracy")
# plt.title("Pruning: Accuracy vs. ccp_alpha")
# plt.savefig(f"{OUT_DIR}/m06_04_decision_trees_pruning.png", dpi=200)
# plt.close()

print("\nFertig. Plots gespeichert:")
print(" - m06_04_decision_trees_classtree.png")
print(" - m06_04_decision_trees_regtree.png")
print(" - m06_04_decision_trees_confusion.png")
print(" - m06_04_decision_trees_importance.png")
