"""
Overfitting Demo
================
Folgendes Script zeigt schrttweise:
1) Underfitting
2) Guten Fit
3) Overfitting
4) Analyse über Train-/Test-Fehler
5) Lösung durch Regularisierung

"""

# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

###
# Getting started:
# pip install numpy matplotlib scikit-learn
###

# -------------------------------------------------------
# 1. Daten generieren (fix, ändern sich nie)
# -------------------------------------------------------
np.random.seed(42)

X = np.linspace(-3, 3, 80).reshape(-1, 1)
y = X[:, 0]**2 + np.random.normal(0, 0.5, size=len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# -------------------------------------------------------
# 2. Hilfsfunktion: Modell fitten & visualisieren
# -------------------------------------------------------
def fit_and_plot(degree, title):
    """
    Trainiert ein Polynommodell mit gegebenem Grad,
    visualisiert die Vorhersage und zeigt Train-/Test-Fehler.
    """
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    # Vorhersagen
    x_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_plot = model.predict(x_plot)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, label="Trainingsdaten")
    plt.scatter(X_test, y_test, label="Testdaten")
    plt.plot(x_plot, y_plot, label=f"Polynom Grad {degree}")

    plt.title(
        f"({title} Grad {degree} | Train MSE: {train_mse:.2f} | Test MSE: {test_mse:.2f}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    # Grafik speichern
    plt.savefig(f"m03_03_overfitting_{degree}.png", dpi=250, bbox_inches="tight")
    plt.show(block=True)


# -------------------------------------------------------
# 3. Demo-Schritte (einzeln ausführen)
# -------------------------------------------------------

# 3.1 Underfitting (Modell zu einfach)
fit_and_plot(degree=1, title="G1")

# 3.2 Guter Fit (passende Modellkomplexität)
fit_and_plot(degree=2, title="G2")

# 3.3 Beginnendes Overfitting
fit_and_plot(degree=6, title="G3")

# 3.4 Starkes Overfitting
fit_and_plot(degree=15, title="G4")


# -------------------------------------------------------
# 4. Analyse: Train- vs. Test-Fehler über Modellkomplexität
# -------------------------------------------------------
train_errors = []
test_errors = []
degrees = range(1, 20)

for d in degrees:
    model = make_pipeline(
        PolynomialFeatures(d),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    train_errors.append(
        mean_squared_error(y_train, model.predict(X_train))
    )
    test_errors.append(
        mean_squared_error(y_test, model.predict(X_test))
    )

plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, label="Train MSE")
plt.plot(degrees, test_errors, label="Test MSE")
plt.xlabel("Modellkomplexität (Polynomgrad)")
plt.ylabel("Fehler")
plt.title("Erkennen von Overfitting")
plt.legend()
plt.tight_layout()
plt.savefig(f"m03_03_overfitting_fehler.png", dpi=250, bbox_inches="tight")
plt.show()


# -------------------------------------------------------
# 5. Lösung: Regularisierung (Ridge Regression)
# -------------------------------------------------------
def fit_and_plot_ridge(degree, alpha):
    """
    Gleiches Modell wie zuvor, aber mit L2-Regularisierung.
    """
    model = make_pipeline(
        PolynomialFeatures(degree),
        Ridge(alpha=alpha)
    )
    model.fit(X_train, y_train)

    x_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_plot = model.predict(x_plot)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, label="Trainingsdaten")
    plt.plot(x_plot, y_plot, label=f"Grad {degree}, alpha={alpha}")

    plt.title("Regularisierung reduziert Overfitting")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"m03_03_overfitting_solution.png", dpi=250, bbox_inches="tight")
    plt.show()


# Beispiel: Overfitting-Modell stabilisieren
fit_and_plot_ridge(degree=15, alpha=10)