import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. DATEN EINLESEN & AUFBEREITEN
# source: https://www.kaggle.com/datasets/stefancomanita/hourly-electricity-consumption-and-production
# ==========================================
# Name der CSV-Datei (ggf. Pfad anpassen)
file_path = "electricity_consumption.csv" 

print("Lade Datensatz...")
df = pd.read_csv(file_path)

df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Duplikate behandeln & Frequenz erzwingen
if df.index.duplicated().sum() > 0:
    df = df.groupby(df.index).mean()

df = df.sort_index()
df = df.asfreq('h')

if df['Consumption'].isnull().sum() > 0:
    df['Consumption'] = df['Consumption'].interpolate(method='linear')

# ==========================================
# 2. FEATURE ENGINEERING (Nur Kalender-Features für Option B)
# ==========================================
def create_calendar_features(data):
    data = data.copy()
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
    data['month'] = data.index.month
    data['dayofyear'] = data.index.dayofyear
    data['quarter'] = data.index.quarter
    return data

df_featured = create_calendar_features(df)

# ==========================================
# 3. TRAIN / TEST SPLIT (Letzter kompletter Monat)
# ==========================================
# Bestimmung des letzten Monats im Datensatz
max_date = df_featured.index.max()
split_date = max_date.to_period('M').start_time  # Anfang des letzten Monats

train = df_featured.loc[df_featured.index < split_date]
test = df_featured.loc[df_featured.index >= split_date]

print(f"\n--- TRAIN/TEST SPLIT (Echte Langzeit-Prognose) ---")
print(f"Trainings-Daten: {train.index.min()} bis {train.index.max()} ({len(train)} Stunden)")
print(f"Test-Daten:     {test.index.min()} bis {test.index.max()} ({len(test)} Stunden)")

feature_cols = ['hour', 'dayofweek', 'is_weekend', 'month', 'dayofyear', 'quarter']

X_train, y_train = train[feature_cols], train['Consumption']
X_test, y_test = test[feature_cols], test['Consumption']

# ==========================================
# 4. MODELL-TRAINING & STÜNDLICHE VORHERSAGE
# ==========================================
print("\nTrainiere Random Forest Regressor (reines Kalendermodell)...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

test_results = test.copy()
test_results['Prediction'] = model.predict(X_test)

# ==========================================
# 5. AGGREGATION AUF TAGESEBENE
# ==========================================
# Mittlerer Tagesverbrauch (MW) oder Tagessumme (MWh)
daily_results = test_results[['Consumption', 'Prediction']].resample('D').mean()

# Evaluation auf Tagesbasis
rmse_daily = np.sqrt(mean_squared_error(daily_results['Consumption'], daily_results['Prediction']))
mae_daily = mean_absolute_error(daily_results['Consumption'], daily_results['Prediction'])
mape_daily = np.mean(np.abs((daily_results['Consumption'] - daily_results['Prediction']) / daily_results['Consumption'])) * 100

print("\n--- MODELL-EVALUATION (TAGESEBENE / TESTMONAT) ---")
print(f"RMSE (Root Mean Squared Error): {rmse_daily:.2f} MW")
print(f"MAE  (Mean Absolute Error):     {mae_daily:.2f} MW")
print(f"MAPE (Mean Absolute % Error):   {mape_daily:.2f} %")

# ==========================================
# 6. GRAFISCHER VERGLEICH (TAGESBASIS)
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(daily_results.index, daily_results['Consumption'], marker='o', label='Echter Tagesverbrauch (Actual)', color='black', linewidth=2)
plt.plot(daily_results.index, daily_results['Prediction'], marker='s', linestyle='--', label='Prognostizierter Tagesverbrauch (Predicted)', color='tab:red', linewidth=2)

plt.title(f'Stromverbrauch Rumänien: Echte Werte vs. Vorhersage auf Tagesbasis ({split_date.strftime("%B %Y")})')
plt.xlabel('Datum')
plt.ylabel('Durchschnittlicher Tagesverbrauch (MW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(daily_results.index[::2], daily_results.index.strftime('%d.%m.').values[::2], rotation=45)
plt.tight_layout()
plt.savefig('m05_01_tagesbasis_testmonat.png')
plt.show()