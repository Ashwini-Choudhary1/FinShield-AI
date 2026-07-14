import os
import sys
import json
import pickle
import duckdb
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure root workspace directory is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from backend.api_models import TransactionSimulationRequest, SimulationResponse, SHAPContribution
except ImportError:
    from api_models import TransactionSimulationRequest, SimulationResponse, SHAPContribution

app = FastAPI(
    title="FinShield-AI Command Center API",
    description="Real-Time Financial Crime, Graph Analytics & SHAP Explainable AI Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
model_artifact = None
shap_explainer = None
model_metrics = {}
graph_metadata = {}

@app.on_event("startup")
def load_artifacts():
    global model_artifact, shap_explainer, model_metrics, graph_metadata
    try:
        with open("models/xgboost_fraud_model.pkl", "rb") as f:
            model_artifact = pickle.load(f)
        with open("models/shap_explainer.pkl", "rb") as f:
            shap_explainer = pickle.load(f)
        if os.path.exists("models/model_metrics.json"):
            with open("models/model_metrics.json", "r") as f:
                model_metrics = json.load(f)
        if os.path.exists("models/graph_metadata.json"):
            with open("models/graph_metadata.json", "r") as f:
                graph_metadata = json.load(f)
        print("[+] Successfully loaded ML and Graph artifacts!")
    except Exception as e:
        print(f"[!] Warning during startup artifact loading: {e}")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_loaded": model_artifact is not None}

@app.get("/api/analytics/kpis")
def get_kpis():
    try:
        conn = duckdb.connect("finshield.duckdb")
        kpis = conn.execute("""
            SELECT 
                SUM(total_tx) as total_transactions,
                SUM(fraud_tx) as total_fraud,
                ROUND(SUM(fraud_tx) * 100.0 / SUM(total_tx), 3) as overall_fraud_rate,
                ROUND(SUM(total_volume_usd), 2) as total_volume_usd,
                ROUND(SUM(fraud_volume_usd), 2) as fraud_volume_usd,
                SUM(count_drained_accounts) as count_drained_accounts,
                SUM(count_dest_unchanged_anomalies) as count_unchanged_anomalies,
                SUM(count_ring_victim_tx) + SUM(count_ring_mule_tx) as count_mule_ring_tx
            FROM fct_fraud_kpis
        """).df().to_dict(orient="records")[0]
        conn.close()
        return kpis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/hourly-trend")
def get_hourly_trend():
    try:
        conn = duckdb.connect("finshield.duckdb")
        trend = conn.execute("""
            SELECT 
                step,
                SUM(total_tx) as total_tx,
                SUM(fraud_tx) as fraud_tx,
                ROUND(SUM(fraud_tx) * 100.0 / SUM(total_tx), 3) as fraud_rate_pct
            FROM fct_fraud_kpis
            GROUP BY step
            ORDER BY step
        """).df().to_dict(orient="records")
        conn.close()
        return trend
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/type-breakdown")
def get_type_breakdown():
    try:
        conn = duckdb.connect("finshield.duckdb")
        types = conn.execute("""
            SELECT 
                transaction_type,
                SUM(total_tx) as volume,
                SUM(fraud_tx) as fraud_count,
                ROUND(SUM(fraud_tx) * 100.0 / SUM(total_tx), 3) as fraud_rate
            FROM fct_fraud_kpis
            GROUP BY transaction_type
        """).df().to_dict(orient="records")
        conn.close()
        return types
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/metadata")
def get_graph_data():
    if os.path.exists("models/graph_metadata.json"):
        with open("models/graph_metadata.json", "r") as f:
            return json.load(f)
    return graph_metadata

@app.get("/api/ml/metrics")
def get_ml_metrics():
    if os.path.exists("models/model_metrics.json"):
        with open("models/model_metrics.json", "r") as f:
            return json.load(f)
    return model_metrics

@app.post("/api/simulate/transaction", response_model=SimulationResponse)
def simulate_transaction(req: TransactionSimulationRequest):
    if not model_artifact or not shap_explainer:
        load_artifacts()
        if not model_artifact:
            raise HTTPException(status_code=500, detail="ML model not loaded.")
            
    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    threshold = model_artifact.get("best_threshold", 0.5)
    
    # Calculate derived features
    error_balance_orig = round(req.old_balance_orig - req.amount - req.new_balance_orig, 2)
    error_balance_dest = round(req.old_balance_dest + req.amount - req.new_balance_dest, 2)
    ratio_old = round(req.amount / req.old_balance_orig, 4) if req.old_balance_orig > 0 else 0.0
    flag_drained = 1 if (req.old_balance_orig == req.amount and req.new_balance_orig == 0) else 0
    flag_dest_unchanged = 1 if (req.transaction_type in ['TRANSFER', 'CASH_IN'] and req.old_balance_dest == req.new_balance_dest and req.amount > 0) else 0
    
    # Build input feature dict
    feat_dict = {
        'amount': req.amount,
        'old_balance_orig': req.old_balance_orig,
        'new_balance_orig': req.new_balance_orig,
        'old_balance_dest': req.old_balance_dest,
        'new_balance_dest': req.new_balance_dest,
        'error_balance_orig': error_balance_orig,
        'error_balance_dest': error_balance_dest,
        'amount_to_oldbalance_ratio': ratio_old,
        'flag_orig_drained': flag_drained,
        'flag_dest_unchanged': flag_dest_unchanged,
        'tx_velocity_1h': req.tx_velocity_1h,
        'is_mule_victim': req.is_mule_victim,
        'is_mule_hop_receiver': req.is_mule_hop_receiver
    }
    
    for t in ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'DEBIT']:
        feat_dict[f'type_{t}'] = 1 if req.transaction_type == t else 0
        
    df_input = pd.DataFrame([feat_dict])[feature_cols]
    
    # Predict
    prob = float(model.predict_proba(df_input)[0, 1])
    
    if prob >= threshold:
        risk_level = "HIGH RISK (FLAGGED)"
    elif prob >= threshold * 0.5:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LEGITIMATE"
        
    # SHAP Explainability
    shap_vals = shap_explainer(df_input)
    shap_data = shap_vals.values[0]
    
    contributions = []
    for col, impact, val in zip(feature_cols, shap_data, df_input.iloc[0]):
        if abs(impact) > 0.001 or col in ['error_balance_orig', 'amount', 'flag_orig_drained', 'tx_velocity_1h']:
            contributions.append(SHAPContribution(
                feature=col,
                impact=round(float(impact), 4),
                value=val
            ))
            
    contributions.sort(key=lambda x: abs(x.impact), reverse=True)
    
    return SimulationResponse(
        risk_score=round(prob, 4),
        risk_level=risk_level,
        decision_threshold=round(threshold, 4),
        shap_contributions=contributions[:8],
        computed_features=feat_dict
    )

# Mount static frontend
os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return "<h1>FinShield-AI Command Center API Running. Please create frontend/index.html.</h1>"

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
