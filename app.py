from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import joblib
import pandas as pd
import os
import time
import random

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
# MODEL LOADER
# ===========================

def safe_load(name):
    path = os.path.join(BASE_DIR, name)
    if os.path.exists(path):
        print(f"✅ {name} loaded")
        return joblib.load(path)
    else:
        print(f"⚠️ {name} missing — demo mode")
        return None

# ===========================
# VEHICLE MODEL
# ===========================

vehicle_model = safe_load("vehicle_model.pkl")

# ===========================
# FEEDBACK MODELS
# ===========================

feedback_bundle = safe_load("website_feedback_update.pkl")
rating_bundle = safe_load("service_center_rating_update.pkl")
experience_bundle = safe_load("customer_experience_log.pkl")

def unpack(bundle):
    if isinstance(bundle, dict):
        return bundle.get("model"), bundle.get("vectorizer")
    return bundle, None

feedback_model, feedback_vectorizer = unpack(feedback_bundle)
rating_model, rating_vectorizer = unpack(rating_bundle)
experience_model, experience_vectorizer = unpack(experience_bundle)

# ===========================
# MANUFACTURING MODELS
# ===========================

service_history_model = safe_load("service_history_update.pkl")
learning_signal_model = safe_load("learning_signal.pkl")
quality_alerts_model = safe_load("quality_alerts_dashboard.pkl")

# ===========================
# UEBA MODELS
# ===========================

ueba_trust_model = safe_load("ueba_website_output.pkl")
ueba_rules_model = safe_load("ueba_rules.pkl")
ueba_summary_model = safe_load("ueba_backend_summary.pkl")

# ===========================
# MASTER AGENT MODELS
# ===========================

master_web_model = safe_load("master_website_output.pkl")
master_app_model = safe_load("master_app_output.pkl")

API_KEY = "AUTOMIND_DEMO_KEY"

# ===========================
# LIVE TELEMETRY STORAGE
# ===========================

live_vehicle_data: Dict[str, dict] = {}

# ===========================
# DATA MODELS
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
# ROOT
# ===========================

@app.get("/")
def root():
    return {"status": "AutoMind backend running"}

# ===========================
# MASTER ORCHESTRATOR
# ===========================

@app.get("/master-agent")
def master_agent():

    vehicles=len(live_vehicle_data)
    anomalies=random.randint(5,40)
    trust=random.randint(70,98)
    failures=random.randint(10,120)

    X=pd.DataFrame([{
        "vehicles":vehicles,
        "anomalies":anomalies,
        "trust":trust,
        "failures":failures
    }])

    intelligence=float(master_web_model.predict(X)[0]) if master_web_model else random.randint(80,97)
    decision=float(master_app_model.predict(X)[0]) if master_app_model else random.randint(0,3)

    decision_map={
        0:"Normal Monitoring",
        1:"Schedule Service Campaign",
        2:"Issue OTA Fix",
        3:"Trigger Recall Investigation"
    }

    return {
        "fleet":{
            "vehicles_online":vehicles,
            "active_alerts":anomalies,
            "avg_trust":trust,
            "open_failures":failures
        },
        "ai":{
            "intelligence_score":round(intelligence,1),
            "decision":decision_map.get(int(decision),"Monitoring")
        }
    }

# ===========================
# TELEMETRY INGEST API
# ===========================

@app.post("/api/telemetry")
def receive_telemetry(data: Telemetry, x_api_key: str = Header(None)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    vin = data.vin

    live_vehicle_data[vin] = {
        "ts": int(time.time()),
        "mode": data.mode,
        "rpm": data.obd.rpm,
        "speed": data.obd.speed_kmph,
        "coolant": data.obd.coolant_c,
        "oil_temp": data.obd.oil_temp_c,
        "oil_pressure": data.obd.oil_pressure_kpa,
        "battery": data.obd.battery_v,
        "fuel": data.obd.fuel_level_pct,
        "dtc": data.obd.dtc_count,
        "misfire": data.health_flags.misfire_risk,
        "overheat": data.health_flags.overheat_risk,
        "low_oil": data.health_flags.low_oil_risk,
        "vibration": data.vibration.engine_rms
    }

    return {"status": "received", "vin": vin}

# ===========================
# VEHICLE HEALTH API (FIXED)
# ===========================

@app.get("/vehicle-health")
def vehicle_health(vin: Optional[str] = None):

    if len(live_vehicle_data) == 0:
        raise HTTPException(status_code=404, detail="No telemetry received yet")

    # Single vehicle lookup
    if vin:
        if vin not in live_vehicle_data:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        return {
            "vin": vin,
            "data": live_vehicle_data[vin]
        }

    # Return full fleet
    return {
        "fleet_size": len(live_vehicle_data),
        "vehicles": [
            {"vin": v, "data": d}
            for v, d in live_vehicle_data.items()
        ]
    }
# ===========================
# AGENT GOVERNANCE (UEBA)
# ===========================

@app.get("/agent-governance")
def agent_governance():

    agent_ids = [
        "data-agent",
        "diagnostics-agent",
        "customer-agent",
        "scheduling-agent",
        "manufacturing-agent",
        "security-agent"
    ]

    agents_output = []

    trust_values = []
    anomalies_total = 0
    actions_total = 0

    for aid in agent_ids:

        # --------------------------
        # Simulated behavioral logs
        # --------------------------
        actions = random.randint(20, 120)
        anomalies = random.randint(0, 12)
        failed_logins = random.randint(0, 5)
        unusual_hours = random.randint(0, 10)
        data_access = random.randint(10, 200)

        X = pd.DataFrame([{
            "actions": actions,
            "anomalies": anomalies,
            "failed_logins": failed_logins,
            "unusual_hours": unusual_hours,
            "data_access": data_access
        }])

        # --------------------------
        # TRUST SCORE (ML)
        # --------------------------
        if ueba_trust_model:
            trust = float(ueba_trust_model.predict(X)[0])
        else:
            trust = 100 - (anomalies * 4 + failed_logins * 5 + unusual_hours * 2)

        trust = max(50, min(100, round(trust,1)))

        # --------------------------
        # RISK LEVEL
        # --------------------------
        if trust > 90:
            risk = "LOW"
            status = "Healthy"
        elif trust > 75:
            risk = "MEDIUM"
            status = "Monitor"
        else:
            risk = "HIGH"
            status = "Restricted"

        agents_output.append({
            "id": aid,
            "name": aid.replace("-", " ").title(),
            "trust": trust,
            "actions": actions,
            "anomalies": anomalies,
            "risk": risk,
            "status": status
        })

        trust_values.append(trust)
        anomalies_total += anomalies
        actions_total += actions

    # --------------------------
    # SUMMARY (ML)
    # --------------------------
    summary_X = pd.DataFrame([{
        "agents": len(agent_ids),
        "avg_trust": sum(trust_values)/len(trust_values),
        "total_anomalies": anomalies_total,
        "total_actions": actions_total
    }])

    if ueba_summary_model:
        summary_pred = ueba_summary_model.predict(summary_X)[0]
    else:
        summary_pred = sum(trust_values)/len(trust_values)

    summary = {
        "active_agents": len(agent_ids),
        "avg_trust": round(sum(trust_values)/len(trust_values),1),
        "anomalies": anomalies_total,
        "actions": actions_total
    }

    return {
        "agents": agents_output,
        "summary": summary
    }
