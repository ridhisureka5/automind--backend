import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor


BASE_DIR = os.getcwd()

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "vehicles.csv"))

# Features
X = df[[
    "mileage",
    "age",
    "service_gap",
    "battery",
    "errors"
]]

# Target
y = df["health"]

# Train
model = RandomForestRegressor()
model.fit(X, y)

# Save
joblib.dump(model, "vehicle_model.pkl")

print("✅ Vehicle ML trained")
