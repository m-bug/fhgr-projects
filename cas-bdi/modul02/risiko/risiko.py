import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

## GETTING STARTED
# pip install numpy matplotlib scikit-learn

#### Konfig
GW = 0.5            # Grenzwert
P_SCREEN = 0.01     # Szenario: 1% der Gruppe ist betroffen (wie im Beispiel in 2.2)
N_GRUPPE = 100_000  # Gruppengrösse für das Rechenbeispiel

# ---------------------------------------------------------------- Daten
# data source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
data = load_breast_cancer()
X = data.data
y = (data.target == 0).astype(int)   # 1 = bösartig (Positive), 0 = gutartig (Negative)
n = len(y)
prev_data = y.mean()

# Modell (Cross-Validation) --> mit Hilfe von Google.. https://scikit-learn.org/stable/modules/cross_validation.html
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kws = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]  # Wahrscheinlichkeit "bösartig"
pred = (kws >= GW).astype(int)

# Vierfeldertafel und Kennzahlen (Skript 2.3)
tp = int(((pred == 1) & (y == 1)).sum())
fp = int(((pred == 1) & (y == 0)).sum())
tn = int(((pred == 0) & (y == 0)).sum())
fn = int(((pred == 0) & (y == 1)).sum())

sens = tp / (tp + fn)            # Sensitivität = Recall = 1 - beta
spez = tn / (tn + fp)
alpha = fp / (tn + fp)           # Fehler 1. Art
beta = fn / (tp + fn)            # Fehler 2. Art
prec = tp / (tp + fp)            # Precision = TP-Rate
f1 = 2 * prec * sens / (prec + sens)
auc = roc_auc_score(y, kws)

# Szenario 1% betroffen (Formel aus dem Script 2.2)
# TP = Sensitivität x Anteil Kranke,  FP = alpha x Anteil Gesunde
tp_s = sens * P_SCREEN
fp_s = alpha * (1 - P_SCREEN)
prec_s = tp_s / (tp_s + fp_s)

# Rechenbeispiel mit Personen (gerundet)
krank = N_GRUPPE * P_SCREEN
gesund = N_GRUPPE - krank
bsp_tp = round(sens * krank)
bsp_fn = round(krank - bsp_tp)
bsp_fp = round(alpha * gesund)
bsp_tn = round(gesund - bsp_fp)

def precision_bei(p, a, s=sens):
    return s * p / (s * p + a * (1 - p))

# Wie robust ist das Resultat? Die Fehlalarm-Rate beruht auf nur 'fp' beobachteten Fehlalarmen.
robust = {k: precision_bei(P_SCREEN, k / (tn + fp)) for k in (fp - 1, fp, fp + 1)}

# Ausgabe
print("=== Datensatz ===")
print(f"Proben: {n}, bösartig: {y.sum()} ({prev_data:.1%}), gutartig: {(y == 0).sum()}")
print("\n=== Vierfeldertafel (Grenzwert 0.5) ===")
print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
print(f"Sensitivität = {tp}/{tp + fn} = {sens:.1%}")
print(f"Spezifität   = {tn}/{tn + fp} = {spez:.1%}")
print(f"alpha = {fp}/{tn + fp} = {alpha:.2%}   beta = {fn}/{tp + fn} = {beta:.1%}")
print(f"Precision    = {tp}/{tp + fp} = {prec:.1%}")
print(f"F1 = {f1:.3f}   AUC = {auc:.3f}")
print(f"\n=== Szenario: {P_SCREEN:.0%} betroffen, Gruppe von {N_GRUPPE:,} Personen ===")
print(f"TP = {sens:.3f} x {P_SCREEN:.0%} = {tp_s:.3%}   FP = {alpha:.3%} x {1 - P_SCREEN:.0%} = {fp_s:.3%}")
print(f"Precision = TP/(TP+FP) = {prec_s:.1%}")
print(f"Personen: krank {krank:.0f} -> TP {bsp_tp}, FN {bsp_fn}; gesund {gesund:.0f} -> FP {bsp_fp}, TN {bsp_tn}")
print(f"Kontrolle: {bsp_tp}/({bsp_tp}+{bsp_fp}) = {bsp_tp / (bsp_tp + bsp_fp):.1%}")
print("\n=== Robustheit: Precision bei 1% je nach Anzahl beobachteter Fehlalarme ===")
for k, v in robust.items():
    print(f"FP = {k}: alpha = {k / (tn + fp):.2%} -> Precision {v:.1%}")

# Grafiken
BLAU, ROT, GRAU = "#1f5fa8", "#c0392b", "#7f8c8d"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

# 1) ROC-Kurve
fig, ax = plt.subplots(figsize=(6.4, 3.5))
fpr, tpr, _ = roc_curve(y, kws)
ax.plot(fpr, tpr, color=BLAU, lw=2.2, label=f"Modell (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=GRAU, ls="--", lw=1.2, label="Zufall (AUC = 0.5)")
ax.scatter([alpha], [sens], color=ROT, zorder=5, s=60)
ax.annotate(f"Grenzwert 0.5\nSensitivität {sens:.1%}\nSpezifität {spez:.1%}",
            (alpha, sens), xytext=(0.12, 0.50), color=ROT,
            arrowprops=dict(arrowstyle="-", color=ROT))
ax.set_xlabel("1 - Spezifität (Anteil Fehlalarme)")
ax.set_ylabel("Sensitivität (Trefferquote)")
ax.set_title("ROC-Kurve des Modells")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("bdi_01_risiko_roc.png", dpi=200)
plt.close(fig)

# 2) Precision in Abhängigkeit vom Anteil Betroffener
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ps = np.logspace(np.log10(0.002), np.log10(0.6), 200)
ax.plot(ps * 100, precision_bei(ps, alpha) * 100, color=BLAU, lw=2.2)
ax.set_xscale("log")
ax.set_xticks([0.2, 1, 5, 10, 37.3])
ax.set_xticklabels(["0.2%", "1%", "5%", "10%", "37%"])
for p, c, txt, dx, dy in [(prev_data, BLAU, "Datensatz", -10, -16),
                          (P_SCREEN, ROT, "Szenario", 12, -6)]:
    v = precision_bei(p, alpha) * 100
    ax.scatter([p * 100], [v], color=c, zorder=5, s=60)
    ax.annotate(f"{txt}: {v:.1f}%", (p * 100, v), xytext=(dx, dy),
                textcoords="offset points", color=c, fontweight="bold",
                ha="right" if dx < 0 else "left")
ax.set_ylim(0, 105)
ax.set_xlabel("Anteil Betroffener (logarithmisch)")
ax.set_ylabel("Precision: Anteil richtiger\npositiver Befunde (%)")
ax.set_title("Precision je nach Anteil Betroffener")
fig.tight_layout()
fig.savefig("bdi_01_risiko_precision.png", dpi=200)
plt.close(fig)
print("\nGrafiken gespeichert: bdi_01_risiko_roc.png, bdi_01_risiko_precision.png")
