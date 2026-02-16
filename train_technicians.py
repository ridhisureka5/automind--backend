import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample training data
data = {
    "experience": [2,5,8,12,15,3,6,10,14,7],
    "jobs_completed": [120,340,520,840,950,180,420,600,900,480],
    "current_load": [30,60,55,70,90,40,65,75,85,58],
    "rating": [4.1,4.4,4.7,4.9,4.8,4.2,4.5,4.7,4.9,4.6]
}

df = pd.DataFrame(data)

X = df[["experience","jobs_completed","current_load"]]
y = df["rating"]

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "technician_model.pkl")

print("✅ Technician Model Saved")
