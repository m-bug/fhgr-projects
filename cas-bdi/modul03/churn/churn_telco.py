"""
Abwanderungsprognose (Churn Prediction) mit logistischer Regression

Call: python3 churn_telco.py <pfad_zu_csv> <ausgabeordner>
example: python3 churn_telco.py WA_Fn-UseC_-Telco-Customer-Churn.csv .

Die Diagramme wurden mit dem R-Code vom Unterricht + Gemini erstellt

Data source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

#GETTING STARTED
# pip install numpy pandas matplotlib scikit-learn
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RS = 42  # Zufallsstartwert für Reproduzierbarkeit
N_FOLDS = 5
N_BOOTSTRAP = 200 # anzahl zufällige simulationen

# Farben (validierte Standardpalette: Slot 1 Blau, Slot 2 Orange)
BLAU = "#2a78d6"
ORANGE = "#eb6834"
GRAU = "#8a8985"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
OBERFLAECHE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": OBERFLAECHE,
        "axes.facecolor": OBERFLAECHE,
        "savefig.facecolor": OBERFLAECHE,
        "axes.edgecolor": GRAU,
        "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "text.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 11,
    }
)


def speichern(fig, ordner, name):
    pfad = ordner / name
    fig.savefig(pfad, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafik gespeichert: {pfad.name}")


def design_matrix(df):
    """Bereitet die Merkmale für die logistische Regression auf.

    - Ja/Nein-Merkmale werden zu 1/0.
    - "No internet service" und "No phone service" zählen als "Nein" (die Information
      steckt schon in InternetService bzw. PhoneService).
    - Mehrstufige Merkmale werden zu 0/1-Spalten; die erste Stufe ist die Referenz.
    - TotalCharges entfällt: Es ergibt sich fast aus Kundendauer x Monatspreis
      (Korrelation mit der Kundendauer: 0.83) und würde die Koeffizienten verfälschen.
    - MonthlyCharges entfällt: Der Monatspreis lässt sich zu 99.9 % aus den gebuchten
      Diensten berechnen (R² = 0.9988). Sein Einfluss steckt also schon in den
      Dienst-Merkmalen; mit beiden zusammen wären die Koeffizienten instabil.
    - Kundendauer in Jahren, damit die Odds Ratios gut lesbar sind.
    """
    X = pd.DataFrame(index=df.index)
    X["Kundendauer (pro Jahr)"] = df["tenure"] / 12
    X["Senior"] = df["SeniorCitizen"]
    X["Frau"] = (df["gender"] == "Female").astype(int)
    X["Partner"] = (df["Partner"] == "Yes").astype(int)
    X["Kinder/Angehörige im Haushalt"] = (df["Dependents"] == "Yes").astype(int)
    X["Telefondienst"] = (df["PhoneService"] == "Yes").astype(int)
    X["Rechnung papierlos"] = (df["PaperlessBilling"] == "Yes").astype(int)
    for spalte, name in [
        ("MultipleLines", "Mehrere Telefonleitungen"),
        ("OnlineSecurity", "Online-Sicherheit"),
        ("OnlineBackup", "Online-Backup"),
        ("DeviceProtection", "Geräteschutz"),
        ("TechSupport", "Technischer Support"),
        ("StreamingTV", "Streaming TV"),
        ("StreamingMovies", "Streaming Filme"),
    ]:
        X[name] = (df[spalte] == "Yes").astype(int)
    X["Internet: Glasfaser (statt DSL)"] = (df["InternetService"] == "Fiber optic").astype(int)
    X["Internet: keiner (statt DSL)"] = (df["InternetService"] == "No").astype(int)
    X["Vertrag: 1 Jahr (statt monatlich)"] = (df["Contract"] == "One year").astype(int)
    X["Vertrag: 2 Jahre (statt monatlich)"] = (df["Contract"] == "Two year").astype(int)
    X["Zahlung: Electronic Check (statt Banküberweisung)"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    X["Zahlung: Post-Check (statt Banküberweisung)"] = (df["PaymentMethod"] == "Mailed check").astype(int)
    X["Zahlung: Kreditkarte (statt Banküberweisung)"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)
    return X


def modell():
    # C sehr gross = praktisch keine Regularisierung (klassische logistische Regression)
    return LogisticRegression(C=1e6, max_iter=5000)


def gain_kurve(y, p):
    """Gain: Anteil der erfassten Kündiger, wenn man die Kunden nach Risiko sortiert
    von oben her anspricht."""
    ordnung = np.argsort(-p, kind="mergesort")
    y_sortiert = np.asarray(y)[ordnung]
    n = len(y_sortiert)
    x = np.arange(0, n + 1) / n
    g = np.concatenate([[0], np.cumsum(y_sortiert)]) / y_sortiert.sum()
    return x, g, y_sortiert


def main(csv_pfad, ausgabe):
    ausgabe = Path(ausgabe)
    ausgabe.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1) Daten laden und prüfen
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("1) DATEN LADEN UND PRÜFEN")
    print("=" * 70)
    df = pd.read_csv(csv_pfad)
    print(f"Zeilen: {len(df)}, Spalten: {df.shape[1]}, doppelte customerID: {df['customerID'].duplicated().sum()}")
    # TotalCharges ist als Text gespeichert; 11 Einträge sind leer (alle mit tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    leer = df["TotalCharges"].isna()
    print(f"Leere TotalCharges: {leer.sum()} (davon tenure == 0: {(df.loc[leer, 'tenure'] == 0).sum()})")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"Korrelation tenure und TotalCharges: {df['tenure'].corr(df['TotalCharges']):.2f}")

    y = (df["Churn"] == "Yes").astype(int).values
    n, quote = len(y), y.mean()
    print(f"Kündigungsquote gesamt: {quote:.1%} ({y.sum()} von {n} Kunden)")

    # -----------------------------------------------------------------------
    # 2) Kündigungsquote nach Merkmalen
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2) KÜNDIGUNGSQUOTE NACH MERKMALEN")
    print("=" * 70)

    def quote_nach(spalte):
        return df.groupby(spalte, observed=True)["Churn"].apply(lambda s: (s == "Yes").mean())

    df["Kundendauer"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0–12", "13–24", "25–48", "49–72"])
    q_vertrag = quote_nach("Contract")
    q_dauer = quote_nach("Kundendauer")
    q_internet = quote_nach("InternetService")
    q_zahlung = quote_nach("PaymentMethod")
    for name, q in [("Vertragsart", q_vertrag), ("Kundendauer (Monate)", q_dauer),
                    ("Internetdienst", q_internet), ("Zahlungsart", q_zahlung),
                    ("Technischer Support", quote_nach("TechSupport"))]:
        print(f"\n{name}:")
        print(q.map("{:.1%}".format).to_string())

    fig, achsen = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Kündigungsquote nach Merkmalen (gesamt: {quote:.1%})", fontsize=14, y=0.98)

    def balken(ax, serie, titel, sortieren=True, drehen=0):
        if sortieren:
            serie = serie.sort_values(ascending=False)
        werte = serie.values * 100
        pos = np.arange(len(serie))
        ax.bar(pos, werte, color=BLAU, width=0.55)
        ax.axhline(quote * 100, color=ORANGE, linewidth=1.6, linestyle="--")
        ax.set_xticks(pos)
        ax.set_xticklabels(serie.index, rotation=drehen, ha="right" if drehen else "center")
        ax.set_ylabel("Kündigungsquote (%)")
        ax.set_title(titel, loc="left", fontsize=11.5)
        ax.grid(axis="x", visible=False)
        for p, w in zip(pos, werte):
            ax.text(p, w + 0.8, f"{w:.0f}%", ha="center", va="bottom", fontsize=9.5, color=TEXT_2)
        ax.set_ylim(0, max(werte) * 1.18)

    balken(achsen[0, 0], q_vertrag, "Vertragsart")
    balken(achsen[0, 1], q_dauer, "Kundendauer (Monate)", sortieren=False)
    balken(achsen[1, 0], q_internet, "Internetdienst")
    balken(achsen[1, 1], q_zahlung, "Zahlungsart", drehen=20)
    achsen[1, 1].text(0.99, 0.97, "gestrichelt: Durchschnitt aller Kunden", transform=achsen[1, 1].transAxes,
                      ha="right", va="top", fontsize=9, color=ORANGE)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    speichern(fig, ausgabe, "m03_02_churn_eda.png")

    # -----------------------------------------------------------------------
    # 3) Logistische Regression: Odds Ratios
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3) LOGISTISCHE REGRESSION (alle Kunden): ODDS RATIOS")
    print("=" * 70)
    X = design_matrix(df)
    print(f"{X.shape[1]} Merkmale nach der Aufbereitung")
    lr = modell().fit(X, y)
    koeff = lr.coef_[0]

    rng = np.random.default_rng(RS)
    boot = np.empty((N_BOOTSTRAP, X.shape[1]))
    Xv = X.values
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        boot[b] = modell().fit(Xv[idx], y[idx]).coef_[0]
    unten, oben = np.percentile(np.exp(boot), [2.5, 97.5], axis=0)

    odds = pd.DataFrame(
        {"Merkmal": X.columns, "Odds Ratio": np.exp(koeff), "95%-Intervall von": unten, "bis": oben}
    )
    odds["Intervall enthält 1"] = (odds["95%-Intervall von"] <= 1) & (odds["bis"] >= 1)
    odds = odds.assign(s=np.abs(koeff)).sort_values("s", ascending=False).drop(columns="s").reset_index(drop=True)
    odds.round(3).to_csv(ausgabe / "m03_03_churn_odds_ratios.csv", index=False)
    print("Odds Ratio > 1: höheres Kündigungsrisiko, < 1: tieferes (bei sonst gleichen Merkmalen).")
    print("Intervalle: Bootstrap mit", N_BOOTSTRAP, "Wiederholungen. Zusammenhänge, keine Ursachen.\n")
    print(odds.round(2).to_string(index=False))

    # -----------------------------------------------------------------------
    # 4) Überprüfung mit Cross-Validation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4) ÜBERPRÜFUNG MIT 5-FACHER CROSS-VALIDATION")
    print("=" * 70)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)
    p = np.zeros(n)
    for i, (train, test) in enumerate(skf.split(Xv, y), start=1):
        m = modell().fit(Xv[train], y[train])
        p[test] = m.predict_proba(Xv[test])[:, 1]
        x_f, g_f, _ = gain_kurve(y[test], p[test])
        g20 = np.interp(0.20, x_f, g_f)
        print(f"  Fold {i}: AUC {roc_auc_score(y[test], p[test]):.3f} | Gain bei 20 %: {g20:.1%}")
    auc = roc_auc_score(y, p)
    print(f"AUC über alle Kunden (Vorhersagen aus der Cross-Validation): {auc:.3f}")

    # -----------------------------------------------------------------------
    # 5) Gain- und Lift-Chart
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5) GAIN- UND LIFT-CHART")
    print("=" * 70)
    x, g, y_sort = gain_kurve(y, p)
    ideal = np.minimum(x / quote, 1.0)
    lift = np.divide(g, x, out=np.full_like(g, np.nan), where=x > 0)

    zeilen = []
    vorher = 0
    for d in range(1, 11):
        bis = int(round(d * n / 10))
        kuend = int(y_sort[vorher:bis].sum())
        zeilen.append(
            {
                "Dezil": d,
                "Kunden bis hier (%)": 10 * d,
                "Kündiger im Dezil": kuend,
                "Quote im Dezil (%)": 100 * kuend / (bis - vorher),
                "Erfasste Kündiger kumuliert (%)": 100 * g[bis],
                "Lift kumuliert": lift[bis],
            }
        )
        vorher = bis
    dezile = pd.DataFrame(zeilen)
    dezile.round(2).to_csv(ausgabe / "m03_02_churn_dezile.csv", index=False)
    print(dezile.round(1).to_string(index=False))
    for ziel in (0.5, 0.8):
        anteil = x[np.searchsorted(g, ziel)]
        print(f"Um {ziel:.0%} der Kündiger zu erreichen, muss man {anteil:.0%} der Kunden ansprechen "
              f"(Zufall: {ziel:.0%}).")
    print(f"Höchster möglicher Lift (perfektes Modell): {1 / quote:.2f}")

    # Gain-Chart und Lift-Chart (einzeln und zusammen)
    def zeichne_gain(ax):
        ax.plot(x * 100, g * 100, color=BLAU, linewidth=2.4, label="Modell (logistische Regression)")
        ax.plot(x * 100, ideal * 100, color=ORANGE, linestyle=":", linewidth=2, label="perfektes Modell")
        ax.plot([0, 100], [0, 100], color=GRAU, linestyle="--", linewidth=1.6, label="Zufall")
        i20 = int(round(0.2 * n))
        ax.scatter([20], [g[i20] * 100], color=BLAU, s=55, zorder=5, edgecolor=OBERFLAECHE, linewidth=2)
        ax.annotate(f"20 % der Kunden\n→ {g[i20]:.0%} der Kündiger", xy=(20, g[i20] * 100),
                    xytext=(47, 27), fontsize=10, color=TEXT_2,
                    arrowprops={"arrowstyle": "-", "color": GRAU})
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Angesprochene Kunden, höchstes Risiko zuerst (%)")
        ax.set_ylabel("Erfasste Kündiger (%)")
        ax.set_title("Gain-Chart", loc="left")
        ax.legend(loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 0.0))

    def zeichne_lift(ax):
        start = int(round(0.05 * n))  # ganz links ist die Kurve wegen weniger Kunden unruhig
        ax.plot(x[start:] * 100, lift[start:], color=BLAU, linewidth=2.4, label="Modell (logistische Regression)")
        ax.axhline(1, color=GRAU, linestyle="--", linewidth=1.6, label="Zufall (Lift = 1)")
        i20 = int(round(0.2 * n))
        ax.scatter([20], [lift[i20]], color=BLAU, s=55, zorder=5, edgecolor=OBERFLAECHE, linewidth=2)
        ax.annotate(f"20 % der Kunden\n→ Lift {lift[i20]:.1f}", xy=(20, lift[i20]),
                    xytext=(38, lift[i20] + 0.55), fontsize=10, color=TEXT_2,
                    arrowprops={"arrowstyle": "-", "color": GRAU})
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1 / quote + 0.25)
        ax.set_xlabel("Angesprochene Kunden, höchstes Risiko zuerst (%)")
        ax.set_ylabel("Lift (Faktor gegenüber Zufall)")
        ax.set_title("Lift-Chart (kumuliert)", loc="left")
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    zeichne_gain(ax)
    speichern(fig, ausgabe, "m03_03_churn_gain.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    zeichne_lift(ax)
    speichern(fig, ausgabe, "m03_03_churn_lift.png")

    fig, achsen = plt.subplots(1, 2, figsize=(12.4, 5.2))
    zeichne_gain(achsen[0])
    zeichne_lift(achsen[1])
    fig.tight_layout(w_pad=2.5)
    speichern(fig, ausgabe, "m03_03_churn_gain_lift.png")

    print("\nFertig.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Aufruf: python3 churn_telco.py <pfad_zur_csv> <ausgabeordner>")
    main(sys.argv[1], sys.argv[2])
