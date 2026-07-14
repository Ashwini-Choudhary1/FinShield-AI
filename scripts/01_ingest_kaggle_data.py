# FinShield-AI Data Ingestion & Synthetic Generation Pipeline v1.0.0
import os
import duckdb
import pandas as pd
import numpy as np
import time

def generate_synthetic_paysim(n_rows=100000, seed=42):
    """
    Generates a high-fidelity synthetic PaySim / Online Payment Fraud dataset
    matching the exact schema and statistical patterns of the Kaggle dataset.
    This includes realistic legitimate transactions, financial anomalies, and
    multi-hop money mule transfer rings for graph analysis.
    """
    print(f"Generating {n_rows:,} high-fidelity synthetic transactions (PaySim schema)...")
    np.random.seed(seed)
    
    # Steps: 1 to 100 hours
    steps = np.random.randint(1, 101, size=n_rows)
    
    # Types: TRANSFER (10%), CASH_OUT (35%), CASH_IN (25%), PAYMENT (25%), DEBIT (5%)
    types = np.random.choice(
        ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'DEBIT'],
        size=n_rows,
        p=[0.10, 0.35, 0.25, 0.25, 0.05]
    )
    
    # Amounts: Log-normal distribution roughly matching real mobile transactions
    amounts = np.round(np.random.lognormal(mean=5.5, sigma=1.8, size=n_rows), 2)
    amounts = np.clip(amounts, 1.0, 10_000_000.0)
    
    # Accounts
    name_orig = [f"C{np.random.randint(1000000000, 9999999999)}" for _ in range(n_rows)]
    name_dest = []
    for t in types:
        if t in ['PAYMENT', 'DEBIT']:
            name_dest.append(f"M{np.random.randint(1000000000, 9999999999)}")
        else:
            name_dest.append(f"C{np.random.randint(1000000000, 9999999999)}")
            
    # Balances for legitimate transactions
    oldbalance_org = np.round(amounts + np.random.exponential(scale=50000, size=n_rows), 2)
    newbalance_orig = np.round(oldbalance_org - amounts, 2)
    newbalance_orig = np.where(newbalance_orig < 0, 0.0, newbalance_orig)
    
    oldbalance_dest = np.round(np.random.exponential(scale=100000, size=n_rows), 2)
    newbalance_dest = np.round(oldbalance_dest + amounts, 2)
    
    is_fraud = np.zeros(n_rows, dtype=int)
    
    # --- Inject Fraud Patterns (approx 0.5% to 1.0% of dataset) ---
    n_fraud = int(n_rows * 0.008)
    fraud_indices = np.random.choice(np.where(np.isin(types, ['TRANSFER', 'CASH_OUT']))[0], size=n_fraud, replace=False)
    
    for idx in fraud_indices:
        is_fraud[idx] = 1
        # Anomaly 1: Account completely drained (amount == oldbalance_org) and newbalance_orig == 0
        oldbalance_org[idx] = amounts[idx]
        newbalance_orig[idx] = 0.0
        # Anomaly 2: Recipient balance doesn't increase (hidden account clearing or anomaly)
        if np.random.rand() > 0.3:
            newbalance_dest[idx] = oldbalance_dest[idx]
            
    # --- Inject Money Mule Rings (Multi-Hop Graph Patterns) ---
    # E.g., Account A -> TRANSFER -> Account B at step T, then Account B -> CASH_OUT -> Account C at step T or T+1
    n_rings = 50
    print(f"Injecting {n_rings} multi-hop Money Mule transfer rings for Graph ML analysis...")
    for i in range(n_rings):
        idx_t = np.random.randint(0, n_rows - 2)
        idx_c = idx_t + 1
        
        step_t = steps[idx_t]
        mule_account = f"C_MULE_{np.random.randint(100000, 999999)}"
        sender_account = f"C_VICTIM_{np.random.randint(100000, 999999)}"
        cashout_dest = f"C_SHADOW_{np.random.randint(100000, 999999)}"
        
        ring_amount = np.round(np.random.uniform(50000, 500000), 2)
        
        # Leg 1: Victim transfers to Mule
        types[idx_t] = 'TRANSFER'
        amounts[idx_t] = ring_amount
        name_orig[idx_t] = sender_account
        name_dest[idx_t] = mule_account
        oldbalance_org[idx_t] = ring_amount
        newbalance_orig[idx_t] = 0.0
        steps[idx_t] = step_t
        is_fraud[idx_t] = 1
        
        # Leg 2: Mule cashes out within 1-2 hours
        types[idx_c] = 'CASH_OUT'
        amounts[idx_c] = ring_amount
        name_orig[idx_c] = mule_account
        name_dest[idx_c] = cashout_dest
        oldbalance_org[idx_c] = ring_amount
        newbalance_orig[idx_c] = 0.0
        steps[idx_c] = step_t + np.random.choice([0, 1])
        is_fraud[idx_c] = 1

    df = pd.DataFrame({
        'step': steps,
        'type': types,
        'amount': amounts,
        'nameOrig': name_orig,
        'oldbalanceOrg': oldbalance_org,
        'newbalanceOrig': newbalance_orig,
        'nameDest': name_dest,
        'oldbalanceDest': oldbalance_dest,
        'newbalanceDest': newbalance_dest,
        'isFraud': is_fraud
    })
    
    df = df.sort_values('step').reset_index(drop=True)
    return df

def main():
    db_path = "finshield.duckdb"
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Check for raw Kaggle CSV
    csv_candidates = [
        os.path.join(raw_dir, "onlinefraud.csv"),
        os.path.join("data", "onlinefraud.csv"),
        os.path.join(raw_dir, "PS_20174392719_1491204439457_log.csv")
    ]
    
    target_csv = None
    for c in csv_candidates:
        if os.path.exists(c):
            target_csv = c
            break
            
    conn = duckdb.connect(db_path)
    
    if target_csv:
        print(f"[+] Found raw Kaggle CSV at {target_csv}. Loading into DuckDB...")
        start_time = time.time()
        conn.execute(f"""
            CREATE OR REPLACE TABLE raw_transactions AS 
            SELECT * FROM read_csv_auto('{target_csv}')
        """)
        elapsed = time.time() - start_time
        count = conn.execute("SELECT COUNT(*) FROM raw_transactions").fetchone()[0]
        print(f"[+] Successfully ingested {count:,} rows from CSV in {elapsed:.2f}s!")
    else:
        print("[!] No raw Kaggle CSV found in data/raw/. Auto-generating synthetic PaySim dataset...")
        df = generate_synthetic_paysim(n_rows=100000)
        sample_path = os.path.join(raw_dir, "sample_onlinefraud.csv")
        df.to_csv(sample_path, index=False)
        print(f"[+] Saved sample dataset to {sample_path} ({len(df):,} rows)")
        
        start_time = time.time()
        conn.execute("CREATE OR REPLACE TABLE raw_transactions AS SELECT * FROM df")
        elapsed = time.time() - start_time
        print(f"[+] Successfully loaded {len(df):,} rows into DuckDB (`{db_path}`) in {elapsed:.2f}s!")
        
    # Verify table schema & summary
    summary = conn.execute("""
        SELECT 
            COUNT(*) as total_rows,
            SUM(isFraud) as total_fraud,
            ROUND(SUM(isFraud) * 100.0 / COUNT(*), 3) as fraud_percentage,
            MIN(step) as min_step,
            MAX(step) as max_step
        FROM raw_transactions
    """).df()
    
    print("\n--- DuckDB `raw_transactions` Summary ---")
    print(summary.to_string(index=False))
    print("------------------------------------------")
    conn.close()

if __name__ == '__main__':
    main()
