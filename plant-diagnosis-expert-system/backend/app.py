from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json, os, joblib
from typing import Optional, Dict, Any

app = FastAPI(title="Plant Diagnosis Expert System")

BASE_DIR = os.path.dirname(__file__)
RULES_PATH = os.path.join(os.path.dirname(BASE_DIR), "knowledge", "rules.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(RULES_PATH) as f:
    RULES = json.load(f)

# Pydantic model for input
class Symptoms(BaseModel):
    leaf_color: str
    spots: str
    wilt: str

def apply_rules(symptoms: Dict[str, str]):
    matches = []
    for r in RULES:
        cond = r.get("conditions", {})
        ok = True
        for k, v in cond.items():
            if symptoms.get(k) != v:
                ok = False
                break
        if ok:
            matches.append({"disease": r["disease"], "confidence": r.get("confidence", 0.5), "advice": r.get("advice","")})
    # sort by confidence
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    return matches

# try load ML model if present
MODEL = None
if os.path.exists(MODEL_PATH):
    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        MODEL = None

@app.post("/api/predict")
def predict(symptoms: Symptoms):
    s = symptoms.dict()
    rule_results = apply_rules(s)
    ml_result = None
    if MODEL is not None:
        # map inputs to features in same order as training
        X = [[s["leaf_color"], s["spots"], s["wilt"]]]
        # our training encodes categorical features as strings; model pipeline should handle this
        try:
            pred = MODEL.predict(X)
            ml_result = {"disease": str(pred[0])}
        except Exception as e:
            ml_result = {"error": "Model failed to predict: " + str(e)}
    return {"rules": rule_results, "ml": ml_result}

@app.get("/")
def root():
    return {"message":"Plant Diagnosis Expert System. POST /api/predict with symptoms."}
