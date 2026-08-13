import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# 1. DATEN EINLESEN & AUFBEREITEN
# source: https://www.kaggle.com/datasets/stefancomanita/hourly-electricity-consumption-and-production
# ==========================================
file_path = "electricity_consumption.csv" 

print("Lade Datensatz...")
df = pd.read_csv(file_path)

df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Duplikate behandeln
if df.index.duplicated().sum() > 0:
    df = df.groupby(df.index).mean()

df = df.sort_index().asfreq('h')

if df['Consumption'].isnull().sum() > 0:
    df['Consumption'] = df['Consumption'].interpolate(method='linear')

# --- JETZT: Direkt auf Tagesbasis aggregieren ---
df_daily = df['Consumption'].resample('D').mean()

# ==========================================
# 2. TRAIN / TEST SPLIT (Tagesbasis)
# ==========================================
max_date = df_daily.index.max()
split_date = max_date.to_period('M').start_time  # Anfang März 2026

train_daily = df_daily.loc[df_daily.index < split_date]
test_daily = df_daily.loc[df_daily.index >= split_date]

print(f"\n--- TRAIN/TEST SPLIT (Holt-Winters auf Tagesebene) ---")
print(f"Trainings-Daten: {train_daily.index.min().strftime('%Y-%m-%d')} bis {train_daily.index.max().strftime('%Y-%m-%d')} ({len(train_daily)} Tage)")
print(f"Test-Daten:     {test_daily.index.min().strftime('%Y-%m-%d')} bis {test_daily.index.max().strftime('%Y-%m-%d')} ({len(test_daily)} Tage)")

# ==========================================
# 3. HOLT-WINTERS MODELL-TRAINING
# ==========================================
# Da wir Tagesdaten nutzen, ist seasonal_periods = 7 (exakt 1 Woche!)
print("\nTrainiere Holt-Winters Modell (Tagesbasis, 7-Tage Saisonalität)...")

hw_model = ExponentialSmoothing(
    train_daily,
    trend='add',
    seasonal='add',
    seasonal_periods=7,
    initialization_method='estimated'
)

hw_fit = hw_model.fit()

# Vorhersage für die Anzahl der Tage im Test-Set
forecast_steps = len(test_daily)
predictions = hw_fit.forecast(steps=forecast_steps)

daily_results = pd.DataFrame({
    'Consumption': test_daily,
    'Prediction': predictions
}, index=test_daily.index)

# ==========================================
# 4. EVALUATION & METRIKEN
# ==========================================
rmse_daily = np.sqrt(mean_squared_error(daily_results['Consumption'], daily_results['Prediction']))
mae_daily = mean_absolute_error(daily_results['Consumption'], daily_results['Prediction'])
mape_daily = np.mean(np.abs((daily_results['Consumption'] - daily_results['Prediction']) / daily_results['Consumption'])) * 100

print("\n--- MODELL-EVALUATION HOLT-WINTERS (TAGESEBENE / TESTMONAT) ---")
print(f"RMSE (Root Mean Squared Error): {rmse_daily:.2f} MW")
print(f"MAE  (Mean Absolute Error):     {mae_daily:.2f} MW")
print(f"MAPE (Mean Absolute % Error):   {mape_daily:.2f} %")

# ==========================================
# 5. GRAFISCHER VERGLEICH (TAGESBASIS)
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(daily_results.index, daily_results['Consumption'], marker='o', label='Echter Tagesverbrauch (Actual)', color='black', linewidth=2)
plt.plot(daily_results.index, daily_results['Prediction'], marker='^', linestyle='--', label='Prognostizierter Tagesverbrauch (Holt-Winters)', color='tab:blue', linewidth=2)

plt.title(f'Stromverbrauch Rumänien: Echte Werte vs. Holt-Winters Vorhersage ({split_date.strftime("%B %Y")})')
plt.xlabel('Datum')
plt.ylabel('Durchschnittlicher Tagesverbrauch (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(daily_results.index[::2], daily_results.index.strftime('%d.%m.').values[::2], rotation=45)
plt.tight_layout()
plt.savefig('m05_01_tagesbasis_holt_winters.png')
plt.show()