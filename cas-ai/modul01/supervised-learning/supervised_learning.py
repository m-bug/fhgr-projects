import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

###
# Getting started:
# 
# pip install matplotlib pandas numpy
# 
##

# ------------------------
# 1. Randomisierte Daten
# ------------------------
np.random.seed(42)

# Features: Einkommen (in 1000) und Schulden (in 1000)
income = np.random.randint(2000, 10000, 50)
debt = np.random.randint(0, 7000, 50)

# Label: Kredit zurückgezahlt? (Ja=1, Nein=0)
# Wir definieren eine einfache Regel für Simulation
label = (income - debt > 3000).astype(int)  # einfache Trennung

# DataFrame für Übersicht
df = pd.DataFrame({'Einkommen': income, 'Schulden': debt, 'Kredit_ok': label})
print(df.head())

# ------------------------
# 2. Scatterplot für Beamer
# ------------------------
plt.figure(figsize=(6,5))
colors = ['red' if l==0 else 'green' for l in label]
plt.scatter(income, debt, c=colors, s=80, alpha=0.6)
plt.xlabel('Einkommen [CHF]')
plt.ylabel('Schulden [CHF]')
plt.title('Supervised Learning Beispiel: Kreditvorhersage')
plt.grid(True)
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Nein', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Ja', markerfacecolor='green', markersize=10)
])
plt.tight_layout()

# ------------------------
# 3. Grafik speichern
# ------------------------
plt.savefig('supervised_example.png', dpi=300)
plt.show()
