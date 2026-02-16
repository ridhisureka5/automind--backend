import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ===========================
# SAMPLE INVENTORY DATA
# ===========================

data = [
    # stock, max_stock, price, monthly_sales, weekly_demand

    [24, 50, 89.99, 48, 12],
    [45, 100, 12.99, 72, 18],
    [8, 30, 149.99, 40, 10],
    [12, 40, 34.99, 24, 6],
    [5, 60, 18.99, 60, 15],
    [32, 75, 24.99, 32, 8],
    [18, 50, 29.99, 28, 7],
    [25, 40, 19.99, 20, 5],

    # Extra synthetic samples
    [10, 50, 55.00, 35, 9],
    [5, 40, 70.00, 45, 11],
    [40, 80, 15.00, 90, 22],
    [60, 100, 9.99, 120, 30],
    [15, 60, 25.00, 42, 10],
    [20, 70, 35.00, 30, 7],
    [8, 25, 120.00, 18, 4],
    [3, 20, 150.00, 12, 3],
]


columns = [
    "stock",
    "max_stock",
    "price",
    "monthly_sales",
    "weekly_demand"
]


df = pd.DataFrame(data, columns=columns)


# ===========================
# FEATURES / TARGET
# ===========================

X = df.drop("weekly_demand", axis=1)
y = df["weekly_demand"]


# ===========================
# TRAIN TEST SPLIT
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ===========================
# MODEL
# ===========================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)


# ===========================
# TRAIN
# ===========================

model.fit(X_train, y_train)


# ===========================
# EVALUATE
# ===========================

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("Inventory Model MAE:", round(mae,2))


# ===========================
# SAVE MODEL
# ===========================

joblib.dump(model, "inventory_model.pkl")

print("inventory_model.pkl saved successfully!")
