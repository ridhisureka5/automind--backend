from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import joblib
import pandas as pd
import os
import time

# ===========================
# APP SETUP
# ===========================

app = FastAPI(title="AutoMind Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================
# LOAD ML MODEL (SAFE LOAD)
# ===========================

MODEL_PATH = os.path.join(BASE_DIR, "vehicle_model.pkl")

if os.path.exists(MODEL_PATH):
    vehicle_model = joblib.load(MODEL_PATH)
    print("✅ ML model loaded")
else:
    vehicle_model = None
    print("⚠️ vehicle_model.pkl not found — running in demo mode")

API_KEY = "AUTOMIND_DEMO_KEY"

# ===========================
# LIVE TELEMETRY STORAGE
# ===========================

live_vehicle_data: Dict[str, dict] = {}

# ===========================
# DATA MODEL (MATCHES ESP32)
# ===========================

class OBD(BaseModel):
    rpm: int
    speed_kmph: int
    coolant_c: int
    oil_temp_c: int
    oil_pressure_kpa: int
    throttle_pct: int
    battery_v: float
    fuel_level_pct: int
    dtc_count: int

class Vibration(BaseModel):
    engine_rms: float
    wheel_rms: float

class Flags(BaseModel):
    misfire_risk: int
    overheat_risk: int
    low_oil_risk: int

class Telemetry(BaseModel):
    vin: str
    request_id: str
    ts_ms: int
    mode: int
    obd: OBD
    vibration: Vibration
    health_flags: Flags


# ===========================
# ROOT HEALTH CHECK (RENDER NEEDS THIS)
# ===========================

@app.get("/")
def root():
    return {"status": "AutoMind backend running"}


# ===========================
# RECEIVE ESP32 DATA
# ===========================

@app.post("/api/telemetry")
def receive_telemetry(data: Telemetry, x_api_key: Optional[str] = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    live_vehicle_data[data.vin] = {
        "time": time.time(),
        "data": data
    }

    print(f"🚗 Received data from {data.vin} | RPM {data.obd.rpm}")

    return {"status": "received"}


# ===========================
# LIVE VEHICLE HEALTH AI
# ===========================

@app.get("/vehicle-health")
def vehicle_health():

    results = []

    for vin, pack in live_vehicle_data.items():
        v = pack["data"]

        # Convert IoT → ML features
        X = pd.DataFrame([{
            "mileage": v.obd.rpm * 2,
            "age": 5,
            "service_gap": v.obd.coolant_c / 2,
            "battery": v.obd.fuel_level_pct,
            "errors": v.health_flags.misfire_risk + v.obd.dtc_count
        }])

        # Predict health score
        if vehicle_model:
            score = float(vehicle_model.predict(X)[0])
        else:
            score = 75.0   # fallback demo score if model missing

        # Status classification
        if score > 85:
            status = "Healthy"
        elif score > 65:
            status = "Warning"
        else:
            status = "Critical"

        results.append({
            "vin": vin,
            "health": int(score),
            "status": status,
            "rpm": v.obd.rpm,
            "speed": v.obd.speed_kmph,
            "coolant": v.obd.coolant_c,
            "oil_temp": v.obd.oil_temp_c,
            "oil_pressure": v.obd.oil_pressure_kpa,
            "fuel": v.obd.fuel_level_pct,
            "battery": v.obd.battery_v,
            "vibration": round(v.vibration.engine_rms, 2),
            "mode": v.mode
        })

    return {
        "vehicles_online": len(results),
        "vehicles": results
    }
