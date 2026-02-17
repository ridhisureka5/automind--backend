from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import joblib
import pandas as pd
import os
import time
import random
from fastapi.responses import JSONResponse

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
# LOAD VEHICLE ML MODEL
# ===========================

MODEL_PATH = os.path.join(BASE_DIR, "vehicle_model.pkl")

if os.path.exists(MODEL_PATH):
    vehicle_model = joblib.load(MODEL_PATH)
    print("✅ vehicle_model.pkl loaded")
else:
    vehicle_model = None
    print("⚠️ vehicle_model.pkl not found — running in demo mode")

# ===========================
# LOAD FEEDBACK MODELS
# ===========================

def safe_load(name):
    path = os.path.join(BASE_DIR, name)
    if os.path.exists(path):
        print(f"✅ {name} loaded")
        return joblib.load(path)
    else:
        print(f"⚠️ {name} missing — demo mode")
        return None

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

API_KEY = "AUTOMIND_DEMO_KEY"

# ===========================
# LIVE TELEMETRY STORAGE
# ===========================

live_vehicle_data: Dict[str, dict] = {}

# ===========================
# DATA MODEL
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


@app.get("/")
def root():
    return {"status": "AutoMind backend running"}


@app.post("/api/telemetry")
def receive_telemetry(data: Telemetry, x_api_key: Optional[str] = Header(None, alias="X-API-Key")):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    live_vehicle_data[data.vin] = {
        "time": time.time(),
        "data": data
    }

    print(f"🚗 Received data from {data.vin} | RPM {data.obd.rpm}")
    return {"status": "received"}


# ===========================
# VEHICLE HEALTH AI
# ===========================

@app.get("/vehicle-health")
def vehicle_health():

    results = []
    TIMEOUT = 20
    now = time.time()

    for vin, pack in live_vehicle_data.items():

        if now - pack["time"] > TIMEOUT:
            continue

        v = pack["data"]

        X = pd.DataFrame([{
            "mileage": v.obd.rpm * 2,
            "age": 5,
            "service_gap": v.obd.coolant_c / 2,
            "battery": v.obd.fuel_level_pct,
            "errors": v.health_flags.misfire_risk + v.obd.dtc_count
        }])

        score = float(vehicle_model.predict(X)[0]) if vehicle_model else random.randint(60,95)

        status = "Healthy" if score>85 else "Warning" if score>65 else "Critical"

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

    return {"vehicles_online": len(results), "vehicles": results}


# ==========================================================
# FEEDBACK ANALYTICS AI
# ==========================================================

def predict_sentiment(text):
    try:
        if feedback_model:
            if feedback_vectorizer:
                X = feedback_vectorizer.transform([text])
                return int(feedback_model.predict(X)[0])
            return int(feedback_model.predict([text])[0])
    except Exception as e:
        print("Sentiment fallback:", e)

    t=text.lower()
    if any(w in t for w in ["excellent","great","perfect","good","love"]): return 2
    if any(w in t for w in ["slow","bad","dirty","not","overcharged"]): return 0
    return 1

def predict_category(text):
    try:
        if experience_model:
            if experience_vectorizer:
                X = experience_vectorizer.transform([text])
                return str(experience_model.predict(X)[0])
            return str(experience_model.predict([text])[0])
    except Exception as e:
        print("Category fallback:", e)

    t=text.lower()
    if "engine" in t: return "Engine"
    if "ac" in t: return "Electrical"
    if "oil" in t: return "Maintenance"
    if "staff" in t: return "Service"
    return "General"


@app.get("/feedback-analytics")
def feedback_analytics():

    comments=[
        "Service was excellent and fast",
        "Very slow repair time",
        "Engine problem not solved",
        "Staff was polite and helpful",
        "Overcharged for oil change",
        "Car returned dirty",
        "Perfect experience loved it",
        "AC still not working",
        "Great service center",
        "Bad experience never coming again"
    ]

    complaints=[]
    pos=neu=neg=0
    ratings=[]

    for i,text in enumerate(comments):

        s=predict_sentiment(text)

        if s==2: sentiment="positive";pos+=1;rating=random.randint(4,5)
        elif s==1: sentiment="neutral";neu+=1;rating=random.randint(3,4)
        else: sentiment="negative";neg+=1;rating=random.randint(1,2)

        ratings.append(rating)
        cat=predict_category(text)

        complaints.append({
            "name":f"Customer {i+1}",
            "rating":rating,
            "category":cat,
            "text":text,
            "date":"2026-02-17",
            "sentiment":sentiment
        })

    total=len(complaints)
    avg=round(sum(ratings)/total,2)

    cats={}
    for c in complaints:
        cats[c["category"]]=cats.get(c["category"],0)+1

    return {
        "total_feedback":total,
        "avg_rating":avg,
        "sentiment":{
            "positive_percent":round((pos/total)*100,1),
            "positive":pos,"neutral":neu,"negative":neg
        },
        "categories":cats,
        "complaints":complaints
    }
