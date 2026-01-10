# ==========================================
# Neural Network – Mietpreisvorhersage (prediction of renting prices of properties in india in currency INR)
# Dataset: House_Rent_Dataset.csv
# Dataset Source: https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset
# ==========================================

###
# Getting started:
# pip install pandas numpy scikit-learn tensorflow
###

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------------------------
# 1. Daten laden
# ------------------------------------------

df = pd.read_csv("House_Rent_Dataset.csv")

# ------------------------------------------
# 2. Feature-Auswahl (hier werden die Features bewusst reduziert)
# ------------------------------------------

features = [
    "BHK",
    "Size",
    "Bathroom",
    "City",
    "Furnishing Status",
    "Area Type"
]

target = "Rent"

df = df[features + [target]]
df.dropna(inplace=True)

# ------------------------------------------
# 3. Feature-Typen definieren
# ------------------------------------------

categorical_features = ["City", "Furnishing Status", "Area Type"]
numerical_features = ["BHK", "Size", "Bathroom"]

# ------------------------------------------
# 4. Preprocessing (One-Hot-Encoding)
# ------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

X = df[features]
y = df[target]

# ------------------------------------------
# 5. Train / Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# 6. Transformation & Skalierung
# ------------------------------------------

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------
# 7. Neural Network definieren
# ------------------------------------------

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1)  # Regression Output
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# ------------------------------------------
# 8. Training
# ------------------------------------------

model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ------------------------------------------
# 9. Evaluation
# ------------------------------------------


loss, mae = model.evaluate(X_test, y_test, verbose=0)

# Wechselkurs (INR → CHF)
inr_to_chf = 1 / 110  # z.B. 1 CHF = 110 INR
prediction_chf = mae * inr_to_chf

print(f"\nDurchschnittlicher Fehler (MAE): {mae:.0f} INR (CHF: {prediction_chf:.0f})") 

# ------------------------------------------
# 10. Beispiel-Vorhersage
# ------------------------------------------

example = pd.DataFrame([{
    "BHK": 4,
    "Size": 900,
    "Bathroom": 2,
    "City": "Kolkata",
    "Furnishing Status": "Semi-Furnished",
    "Area Type": "Super Area"
}])

example_transformed = preprocessor.transform(example)
example_scaled = scaler.transform(example_transformed)

prediction_inr = model.predict(example_scaled)[0][0]

prediction_chf = prediction_inr * inr_to_chf

print(
    f"Geschätzte Miete für Beispielwohnung: "
    f"{prediction_inr:.0f} INR "
    f"({prediction_chf:.0f} CHF)"
)
