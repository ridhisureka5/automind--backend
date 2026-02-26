from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import numpy as np
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
model = joblib.load("predictive_model.pkl")




   


df = pd.read_csv("vehicle_dataset.csv")
def severity_level(prob):
    if prob > 0.8:
        return "critical"
    elif prob > 0.6:
        return "high"
    elif prob > 0.4:
        return "medium"
    else:
        return "low"

# ===========================
# MODEL LOADER (SAFE CLOUD LOADER)
# ===========================
def predict_sentiment(text):

    if feedback_sentiment_model:
        try:
            pred = feedback_sentiment_model.predict([text])[0]
            mapping = {0:"negative",1:"neutral",2:"positive"}
            return mapping.get(pred,"neutral")
        except:
            pass

    # fallback demo
    if any(w in text.lower() for w in ["bad","worst","delay","issue"]):
        return "negative"
    if any(w in text.lower() for w in ["ok","fine"]):
        return "neutral"
    return "positive"


def predict_category(text):

    if category_model and category_vectorizer:
        try:
            X = category_vectorizer.transform([text])
            return category_model.predict(X)[0]
        except:
            pass

    keywords={
        "service":"Service",
        "engine":"Engine",
        "delay":"Delivery",
        "price":"Pricing",
        "app":"App Experience"
    }

    for k,v in keywords.items():
        if k in text.lower():
            return v
    return "General"


def predict_rating(text):

    if experience_model and experience_vectorizer:
        try:
            X=experience_vectorizer.transform([text])
            return int(experience_model.predict(X)[0])
        except:
            pass

    if "excellent" in text.lower(): return 5
    if "good" in text.lower(): return 4
    if "ok" in text.lower(): return 3
    if "bad" in text.lower(): return 2
    return 1


def safe_load(name):
    try:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            print(f"✅ {name} loaded")
            return joblib.load(path)
        else:
            print(f"⚠️ {name} missing — demo mode")
            return None
    except Exception as e:
        print(f"❌ Failed loading {name}: {e}")
        return None
# ===========================
# TECHNICIAN ANALYTICS MODELS
# ===========================

technician_skill_model = safe_load("technician_model.pkl")
technician_perf_model = safe_load("technician_performance_model.pkl")
service_history_model = safe_load("service_history_update.pkl")

skills = ["Engine","Electrical","Brakes","Transmission","Diagnostics","HVAC"]

def predict_skill(exp, jobs):
    try:
        X=pd.DataFrame([[exp,jobs]],columns=["experience","jobs"])
        idx=int(technician_skill_model.predict(X)[0]) if technician_skill_model else random.randint(0,5)
        return skills[idx%len(skills)]
    except:
        return random.choice(skills)

def predict_rating(exp, jobs):
    try:
        X=pd.DataFrame([[exp,jobs]],columns=["experience","jobs"])
        r=float(technician_perf_model.predict(X)[0]) if technician_perf_model else random.uniform(3.5,5)
        return round(max(3,min(5,r)),1)
    except:
        return round(random.uniform(3.5,5),1)

def predict_load(exp, jobs):
    try:
        X=pd.DataFrame([[exp,jobs]],columns=["experience","jobs"])
        load=float(service_history_model.predict(X)[0]) if service_history_model else random.random()
        return int(min(100,max(5,load*100)))
    except:
        return random.randint(10,95)

# ===========================
# VEHICLE MODEL
# ===========================

vehicle_model = safe_load("vehicle_model.pkl")
# ===========================
# INVENTORY AI MODEL
# ===========================

inventory_model = safe_load("inventory_model.pkl")

# ===========================
# SERVICE CENTER AI DASHBOARD
# ===========================

fleet_health_model = safe_load("fleet_health_model.pkl")
predictive_model = safe_load("predictive_maintenance_model.pkl")
alert_priority_model = safe_load("alert_priority_model.pkl")
rul_model = safe_load("rul_model.pkl")
technician_perf_model = safe_load("technician_performance_model.pkl")


# ===========================
# FEEDBACK MODELS
# ===========================

feedback_bundle = safe_load("website_feedback_update.pkl")
rating_bundle = safe_load("service_center_rating_update.pkl")
experience_bundle = safe_load("customer_experience_log.pkl")



# ===========================
# ADVANCED MANUFACTURING AI MODELS (NEW)
# ===========================

defect_forecast_model = safe_load("defect_forecast_model.pkl")
rca_pattern_model = safe_load("rca_pattern_model.pkl")
supplier_risk_model = safe_load("supplier_risk_model.pkl")
wokwi_anomaly_model = safe_load("wokwi_sensor_anomaly_model.pkl")


# ===========================
# MANUFACTURING MODELS
# ===========================

service_history_model = safe_load("service_history_update.pkl")
learning_signal_model = safe_load("learning_signal.pkl")
quality_alerts_model = safe_load("quality_alerts_dashboard.pkl")
# ===========================
# FEEDBACK ANALYTICS MODELS
# ===========================

feedback_sentiment_model = safe_load("feedback_model.pkl")
feedback_category_bundle = safe_load("website_feedback_update.pkl")
experience_bundle = safe_load("customer_experience_log.pkl")

def unpack(bundle):
    if isinstance(bundle, dict):
        return bundle.get("model"), bundle.get("vectorizer")
    return bundle, None

category_model, category_vectorizer = unpack(feedback_category_bundle)
experience_model, experience_vectorizer = unpack(experience_bundle)

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
# SERVICE CENTER LOAD MODELS
# ===========================

load_model = safe_load("load_model.pkl")
history_model = safe_load("service_history_update.pkl")
rating_bundle = safe_load("service_center_rating_update.pkl")
def unpack(bundle):
    if isinstance(bundle, dict):
        return bundle.get("model"), bundle.get("vectorizer")
    return bundle, None

rating_model, rating_vectorizer = unpack(rating_bundle)
def load_label(value):
    if value >= 0.75: return "HIGH"
    if value >= 0.45: return "MEDIUM"
    return "LOW"

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

    try:
        intelligence=float(master_web_model.predict(X)[0]) if master_web_model else random.randint(80,97)
        decision=float(master_app_model.predict(X)[0]) if master_app_model else random.randint(0,3)
    except:
        intelligence=random.randint(80,97)
        decision=random.randint(0,3)

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
# VEHICLE HEALTH API
# ===========================

@app.get("/vehicle-health")
def vehicle_health(vin: Optional[str] = None):

    if len(live_vehicle_data) == 0:
        raise HTTPException(status_code=404, detail="No telemetry received yet")

    if vin:
        if vin not in live_vehicle_data:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        return {"vin": vin, "data": live_vehicle_data[vin]}

    return {
        "fleet_size": len(live_vehicle_data),
        "vehicles": [{"vin": v, "data": d} for v, d in live_vehicle_data.items()]
    }

# ===========================
# AGENT GOVERNANCE (UEBA) — FIXED
# ===========================

@app.get("/agent-governance")
def agent_governance():

    try:
        agent_ids = [
            "data-agent",
            "diagnostics-agent",
            "customer-agent",
            "scheduling-agent",
            "manufacturing-agent",
            "security-agent"
        ]

        agents_output=[]
        trust_values=[]
        anomalies_total=0
        actions_total=0

        for aid in agent_ids:

            actions=random.randint(20,120)
            anomalies=random.randint(0,12)
            failed_logins=random.randint(0,5)
            unusual_hours=random.randint(0,10)
            data_access=random.randint(10,200)

            X=pd.DataFrame([[actions,anomalies,failed_logins,unusual_hours,data_access]],
                           columns=["actions","anomalies","failed_logins","unusual_hours","data_access"])

            trust=None
            if ueba_trust_model is not None:
                try:
                    trust=float(ueba_trust_model.predict(X)[0])
                except Exception as e:
                    print("UEBA model error:",e)

            if trust is None:
                trust=100-(anomalies*4+failed_logins*5+unusual_hours*2)

            trust=max(50,min(100,round(trust,1)))

            if trust>90:
                risk="LOW"; status="Healthy"
            elif trust>75:
                risk="MEDIUM"; status="Monitor"
            else:
                risk="HIGH"; status="Restricted"

            agents_output.append({
                "id":aid,
                "name":aid.replace("-"," ").title(),
                "trust":trust,
                "actions":actions,
                "anomalies":anomalies,
                "risk":risk,
                "status":status
            })

            trust_values.append(trust)
            anomalies_total+=anomalies
            actions_total+=actions

        summary={
            "active_agents":len(agent_ids),
            "avg_trust":round(sum(trust_values)/len(trust_values),1),
            "anomalies":anomalies_total,
            "actions":actions_total
        }

        return {"agents":agents_output,"summary":summary}

    except Exception as e:
        print("AGENT GOVERNANCE ERROR:",e)
        return {"agents":[],"summary":{}}
@app.get("/manufacturing/defect-forecast")
def defect_forecast():

    components = ["Engine","Transmission","Brakes","Suspension","Electrical","HVAC"]
    results=[]

    for comp in components:
        X=pd.DataFrame([{
            "component":len(comp),
            "vehicles":random.randint(100,500),
            "usage":random.randint(20,90),
            "temperature":random.randint(40,120)
        }])

        try:
            risk=float(defect_forecast_model.predict(X)[0]) if defect_forecast_model else random.random()
        except:
            risk=random.random()

        results.append({
            "component":comp,
            "failure_probability":round(min(max(risk,0),1),2)
        })

    return {"forecast":results}

@app.get("/manufacturing/rca-analysis")
def rca_analysis():

    issues=["Brake wear","Oil leak","Sensor fault","Overheating","Loose wiring"]
    output=[]

    for issue in issues:

        X=pd.DataFrame([{
            "freq":random.randint(1,20),
            "temp":random.randint(30,120),
            "vibration":random.random()*10
        }])

        try:
            cause=int(rca_pattern_model.predict(X)[0]) if rca_pattern_model else random.randint(0,2)
        except:
            cause=random.randint(0,2)

        mapping={
            0:"Manufacturing Defect",
            1:"Supplier Quality Issue",
            2:"Design Limitation"
        }

        output.append({
            "issue":issue,
            "root_cause":mapping[cause]
        })

    return {"rca":output}


@app.get("/manufacturing/supplier-risk")
def supplier_risk():

    suppliers=["AutoParts Global","SupplierTech Inc","Elite Manufacturing","Precision Components"]

    results=[]

    for s in suppliers:
        X=pd.DataFrame([{
            "defects":random.randint(1,40),
            "delay_days":random.randint(0,15),
            "returns":random.randint(0,20)
        }])

        try:
            score=float(supplier_risk_model.predict(X)[0]) if supplier_risk_model else random.random()
        except:
            score=random.random()

        level="LOW"
        if score>0.7: level="HIGH"
        elif score>0.4: level="MEDIUM"

        results.append({
            "supplier":s,
            "risk_score":round(score,2),
            "risk_level":level
        })

    return {"suppliers":results}

@app.get("/manufacturing/live-anomaly")
def live_anomaly():

    X=pd.DataFrame([{
        "temperature":random.randint(20,120),
        "vibration":random.random()*15,
        "pressure":random.randint(20,200)
    }])

    try:
        anomaly=int(wokwi_anomaly_model.predict(X)[0]) if wokwi_anomaly_model else random.randint(0,1)
    except:
        anomaly=random.randint(0,1)

    return {
        "factory_alert":"ANOMALY DETECTED" if anomaly==1 else "NORMAL",
        "severity":"HIGH" if anomaly==1 else "OK"
    }

@app.get("/service-dashboard")
def service_dashboard():

    vehicles = len(live_vehicle_data) if live_vehicle_data else random.randint(5,12)

    X = pd.DataFrame([{
        "vehicles":vehicles,
        "avg_temp":random.randint(60,110),
        "dtc":random.randint(0,5),
        "vibration":random.random()*10
    }])

    try:
        health = float(fleet_health_model.predict(X)[0]) if fleet_health_model else random.randint(70,95)
    except:
        health = random.randint(70,95)

    try:
        uptime = float(technician_perf_model.predict(X)[0]) if technician_perf_model else random.uniform(97,99.9)
    except:
        uptime = random.uniform(97,99.9)

    return {
        "fleet_health": round(health,1),
        "system_uptime": round(uptime,2),
        "total_vehicles": vehicles,
        "active_alerts": random.randint(2,6),
        "upcoming_services": random.randint(2,6)
    }


@app.get("/service-alerts")
def service_alerts():

    alerts=[]

    names=["Transmission Fluid","Brake Pads","Engine Cooling","Air Filter"]

    for n in names:

        X=pd.DataFrame([[random.randint(50,120),random.random()*10]],
                       columns=["temp","vibration"])

        try:
            level=int(alert_priority_model.predict(X)[0]) if alert_priority_model else random.randint(0,3)
        except:
            level=random.randint(0,3)

        mapping=["low","medium","high","critical"]
        alerts.append({"title":n,"level":mapping[level]})

    return {"alerts":alerts}


@app.get("/service-predictions")
def service_predictions():

    services=[]

    for i in range(4):

        X=pd.DataFrame([[random.randint(1000,20000),random.randint(60,120)]],
                       columns=["km","temp"])

        try:
            urgency=int(predictive_model.predict(X)[0]) if predictive_model else random.randint(0,2)
        except:
            urgency=random.randint(0,2)

        tag=["routine","repair","recall"][urgency]

        services.append({
            "vin":f"VIN{i+1}AUTO",
            "city":random.choice(["Delhi","Noida","Gurgaon","Rohini"]),
            "tag":tag
        })

    return {"services":services}


@app.get("/inventory-analytics")
def inventory_analytics():

    parts = [
        ("Brake Pads","BP-221","Brakes",120),
        ("Oil Filter","OF-332","Engine",40),
        ("Air Filter","AF-876","HVAC",35),
        ("Spark Plug","SP-111","Engine",25),
        ("Clutch Plate","CP-909","Transmission",220),
        ("Coolant Pump","CP-450","Cooling",180),
        ("Battery","BT-222","Electrical",150),
        ("ABS Sensor","ABS-01","Electrical",95)
    ]

    inventory=[]

    for name,sku,category,price in parts:

        usage=random.randint(10,80)
        vehicles=random.randint(50,300)
        season=random.randint(0,1)  # seasonal demand

        X=pd.DataFrame([[usage,vehicles,season]],
                       columns=["usage","vehicles","season"])

        try:
            demand=float(inventory_model.predict(X)[0]) if inventory_model else random.randint(2,12)
        except:
            demand=random.randint(2,12)

        demand=max(1,round(demand))

        max_stock=random.randint(20,80)
        current_stock=max(1,max_stock - random.randint(0,max_stock))

        inventory.append({
            "name":name,
            "sku":sku,
            "category":category,
            "stock":f"{current_stock}/{max_stock}",
            "ai_demand":f"{demand}/week",
            "price":price
        })

    return {"inventory":inventory}

@app.get("/feedback-analytics")
def feedback_analytics():

    base_feedback = [
        "service was quick",
        "engine problem fixed",
        "late delivery",
        "app confusing",
        "price high",
        "very helpful staff",
        "bad response time",
        "excellent support",
        "average experience",
        "booking failed"
    ]

    users=["Rahul","Amit","Sneha","Karan","Neha","Riya","Vikram","Ankit","Meera","Arjun"]

    modifiers=[
        "very","extremely","slightly","unexpectedly","surprisingly",
        "really","quite","somewhat","barely","properly"
    ]

    complaints=[]
    categories={}
    positive=neutral=negative=0
    rating_sum=0

    for i in range(len(users)):

        # 🔹 Create dynamic sentence each time
        text=f"{random.choice(modifiers)} {random.choice(base_feedback)}"

        sentiment=predict_sentiment(text)
        category=predict_category(text)
        rating=predict_rating(text)

        if sentiment=="positive": positive+=1
        elif sentiment=="neutral": neutral+=1
        else: negative+=1

        rating_sum+=rating
        categories[category]=categories.get(category,0)+1

        complaints.append({
            "name":users[i],
            "rating":rating,
            "category":category,
            "text":text,
            "date":time.strftime("%d %b %Y"),
            "sentiment":sentiment
        })

    total=len(complaints)
    avg_rating=round(rating_sum/total,2)

    return {
        "total_feedback":total,
        "avg_rating":avg_rating,
        "sentiment":{
            "positive":positive,
            "neutral":neutral,
            "negative":negative,
            "positive_percent":round((positive/total)*100)
        },
        "categories":categories,
        "complaints":complaints
    }

@app.get("/load-prediction")
def load_prediction():

    centers_data = [
        ("AutoMind Prime","Delhi"),
        ("RapidFix Motors","Gurgaon"),
        ("CityCare Service","Noida"),
        ("NorthZone Auto","Rohini"),
        ("Metro Garage","Janakpuri"),
        ("Elite AutoWorks","Dwarka")
    ]

    centers=[]
    loads=[]

    for i,(name,city) in enumerate(centers_data):

        bookings=random.randint(10,80)
        technicians=random.randint(3,12)
        repeat=random.randint(0,30)

        X=pd.DataFrame([[bookings,technicians,repeat]],
                       columns=["bookings","technicians","repeat"])

        try:
            load=float(load_model.predict(X)[0]) if load_model else random.random()
        except:
            load=random.random()

        label=load_label(load)

        utilization=min(100,round(load*100))

        advice={
            "HIGH":"Add technicians or redirect bookings",
            "MEDIUM":"Monitor and prepare backup staff",
            "LOW":"Promote offers to increase bookings"
        }[label]

        centers.append({
            "id":i+1,
            "name":name,
            "city":city,
            "utilization":utilization,
            "predicted_load":label,
            "recommendation":advice,
            "slots_filling_fast":utilization>85
        })

        loads.append(load)

    avg_load=sum(loads)/len(loads)
    global_label=load_label(avg_load)

    global_advice={
        "HIGH":"Shift bookings to nearby centers",
        "MEDIUM":"Balanced load across centers",
        "LOW":"Increase marketing campaigns"
    }[global_label]

    return {
        "global":{
            "predicted_load":global_label,
            "recommendation":global_advice
        },
        "centers":centers
    }
@app.get("/technician-analytics")
def technician_analytics():

    names=[
        "Rahul Sharma","Amit Verma","Sneha Kapoor",
        "Karan Mehta","Neha Singh","Vikram Patel"
    ]

    technicians=[]
    working=0
    available=0
    rating_sum=0

    for i,name in enumerate(names):

        exp=random.randint(1,15)
        jobs=random.randint(5,120)

        skill=predict_skill(exp,jobs)
        rating=predict_rating(exp,jobs)
        load=predict_load(exp,jobs)

        status="Working" if load>55 else "Available"

        if status=="Working": working+=1
        else: available+=1

        rating_sum+=rating

        technicians.append({
            "id":i+1,
            "name":name,
            "experience":exp,
            "skill":skill,
            "rating":rating,
            "jobs":jobs,
            "status":status,
            "load":load
        })

    total=len(technicians)

    return {
        "summary":{
            "total":total,
            "available":available,
            "working":working,
            "avg_rating":round(rating_sum/total,2)
        },
        "technicians":technicians
    }
@app.get("/predict-alerts")
def predict_alerts():

    results = []

    for _, row in df.iterrows():

        features = np.array([[ 
            row.mileage,
            row.engine_temp,
            row.brake_wear,
            row.battery_voltage,
            row.error_codes
        ]])

        prob = model.predict_proba(features)[0][1]
        level = severity_level(prob)

        if level in ["medium","high","critical"]:

            results.append({
                "car_name": row.car_name,
                "vin": row.vin,
                "failure_probability": round(float(prob),2),
                "level": level
            })

    return {"alerts": results}


@app.get("/predict-services")
def predict_services():

    services = []

    for _, row in df.iterrows():

        days_left = max(0, 100000 - row.mileage) // 1000
        tag = "AI Predicted" if row.failure == 1 else "Scheduled"

        services.append({
            "car_name": row.car_name,
            "vin": row.vin,
            "city": "Auto Assigned",
            "tag": tag,
            "service_due_in_km": days_left
        })

    return {"services": services}



