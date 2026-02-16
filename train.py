import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor


# Get current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build csv path
csv_path = os.path.join(BASE_DIR, "venv", "data.csv")

print("CSV PATH:", csv_path)

# Load data
df = pd.read_csv(csv_path)


# Features + Target
X = df[["appointments", "capacity"]]
y = df["utilization"]


# Train model
model = RandomForestRegressor()
model.fit(X, y)


# Save model
model_path = os.path.join(BASE_DIR, "load_model.pkl")

joblib.dump(model, model_path)

print("✅ Model trained & saved")
