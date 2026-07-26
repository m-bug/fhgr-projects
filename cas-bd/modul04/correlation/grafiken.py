import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Kopie aus der Konsole von Script "corelation.py"
data = {
    'Wahlkampf: Interessantheit': [1.000, 0.191, -0.065, 0.060, -0.141, -0.081],
    'Wahlergebnis: Zufriedenheit': [0.191, 1.000, 0.032, 0.008, -0.504, -0.221],
    'Wahl: Schwierigkeit': [-0.065, 0.032, 1.000, -0.465, -0.108, -0.169],
    'Wahl: Zufriedenheit': [0.060, 0.008, -0.465, 1.000, 0.008, 0.033],
    'Koalition: GroKo (CDU/SPD)': [-0.141, -0.504, -0.108, 0.008, 1.000, 0.423],
    'Koalition: Schwarz-Grün': [-0.081, -0.221, -0.169, 0.033, 0.423, 1.000]
}

df_corr = pd.DataFrame(data, index=data.keys())

# 2. Maske erstellen, um die doppelte obere Hälfte auszublenden (aufgeräumtere Optik)
mask = np.triu(np.ones_like(df_corr, dtype=bool))

# 3. Heatmap plotten
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_corr, 
    mask=mask, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    vmin=-1, vmax=1, 
    cbar_kws={'label': 'Spearman-Korrelation (r)'},
    linewidths=1,
    square=True
)

plt.title('GLES BTW 2025: Zusammenhänge zwischen Wahlentscheidung & Koalitionen', fontsize=13, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('m04_02_gles_korrelationen_btw2025.png', dpi=300)
plt.show()