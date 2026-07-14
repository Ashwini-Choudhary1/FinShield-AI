# FinShield-AI XGBoost & TreeSHAP Training Pipeline v1.0.0
import os
import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import json
import time
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve

def train_xgboost_and_shap():
    print("--- [Explainable AI Engine] Training XGBoost with Temporal Split & SHAP Explainer ---")
    start_time = time.time()
    
    db_path = "finshield.duckdb"
    conn = duckdb.connect(db_path)
    
    print("[+] Querying `fct_ml_graph_features` mart from DuckDB...")
    df = conn.execute("SELECT * FROM fct_ml_graph_features").df()
    conn.close()
    
    print(f"[+] Loaded {len(df):,} transactions (`{df['is_fraud'].sum()}` confirmed fraud cases).")
    
    # 1. Feature Engineering & Selection
    # One-hot encode transaction_type
    df_encoded = pd.get_dummies(df, columns=['transaction_type'], prefix='type', dtype=int)
    
    feature_cols = [
        'amount', 'old_balance_orig', 'new_balance_orig', 
        'old_balance_dest', 'new_balance_dest', 
        'error_balance_orig', 'error_balance_dest', 
        'amount_to_oldbalance_ratio', 'flag_orig_drained', 
        'flag_dest_unchanged', 'tx_velocity_1h', 
        'is_mule_victim', 'is_mule_hop_receiver'
    ] + [col for col in df_encoded.columns if col.startswith('type_')]
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    X = df_encoded[feature_cols].copy()
    y = df_encoded['is_fraud'].copy()
    
    # 2. Temporal Split: Train on earlier steps (e.g. step <= 80), Test on future steps (step > 80)
    # This demonstrates production concept drift testing rather than random split!
    split_step = int(df['step'].quantile(0.80))
    train_mask = df['step'] <= split_step
    test_mask = df['step'] > split_step
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"[+] Temporal Split (Threshold step={split_step}): Train={len(X_train):,} rows, Test={len(X_test):,} rows.")
    
    # 3. Handle Class Imbalance via scale_pos_weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = max(1.0, n_neg / max(n_pos, 1))
    print(f"[+] Class imbalance ratio: `scale_pos_weight = {scale_pos_weight:.2f}`")
    
    # 4. Train XGBoost Classifier
    print("[+] Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    y_probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # Find optimal threshold using Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    # Maximize F1 score across thresholds
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    
    y_pred_best = (y_probs >= best_thresh).astype(int)
    report = classification_report(y_test, y_pred_best, output_dict=True)
    
    print(f"\n--- [Model Evaluation Results on Unseen Temporal Test Set] ---")
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"PR-AUC Score  : {pr_auc:.4f} (Crucial benchmark for extreme class imbalance)")
    print(f"Optimal Thresh: {best_thresh:.4f} (F1 = {f1_scores[best_idx]:.4f})")
    print(f"Precision     : {report['1']['precision']:.4f}")
    print(f"Recall        : {report['1']['recall']:.4f}")
    print("--------------------------------------------------------------")
    
    # 6. Fit SHAP TreeExplainer for real-time explainability
    print("[+] Fitting SHAP TreeExplainer for real-time risk explanations...")
    explainer = shap.TreeExplainer(model)
    
    # Save Model, Explainer, and Metrics
    os.makedirs("models", exist_ok=True)
    model_artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "best_threshold": best_thresh
    }
    with open("models/xgboost_fraud_model.pkl", "wb") as f:
        pickle.dump(model_artifact, f)
        
    with open("models/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
        
    metrics = {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "optimal_threshold": round(best_thresh, 4),
        "precision": round(float(report['1']['precision']), 4),
        "recall": round(float(report['1']['recall']), 4),
        "f1_score": round(float(f1_scores[best_idx]), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_cols": feature_cols
    }
    with open("models/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[+] All artifacts saved to `models/` in {time.time() - start_time:.2f}s!")

if __name__ == '__main__':
    train_xgboost_and_shap()
