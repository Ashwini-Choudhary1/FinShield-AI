import os
import duckdb
import networkx as nx
import json
import time

def build_transaction_graph():
    print("--- [Graph ML Engine] Building Transaction Network Graph ---")
    start_time = time.time()
    
    db_path = "finshield.duckdb"
    conn = duckdb.connect(db_path)
    
    # 1. Load edges from staging / intermediate
    edges_df = conn.execute("""
        SELECT 
            name_orig, 
            name_dest, 
            amount, 
            transaction_type, 
            step, 
            is_fraud
        FROM stg_transactions
        WHERE transaction_type IN ('TRANSFER', 'CASH_OUT', 'PAYMENT')
        LIMIT 50000 -- Sample edges for rapid graph centrality computation
    """).df()
    
    print(f"[+] Ingested {len(edges_df):,} transaction edges for graph construction.")
    
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['name_orig'], 
            row['name_dest'], 
            weight=row['amount'], 
            tx_type=row['transaction_type'], 
            step=row['step'], 
            is_fraud=row['is_fraud']
        )
        
    print(f"[+] Graph constructed with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")
    
    # 2. Query exact money mule rings from our dbt intermediate table
    mule_rings_df = conn.execute("""
        SELECT * FROM int_money_mule_graph
    """).df()
    
    print(f"[+] Identified {len(mule_rings_df)} verified 2-Hop Money Mule Rings (`Account A -> Mule -> Cashout`).")
    
    # 3. Compute node degree centralities for top suspicious hubs
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # Export structured graph metadata & sample rings for our live Web Dashboard
    sample_rings = []
    for _, ring in mule_rings_df.head(20).iterrows():
        sample_rings.append({
            "transfer_step": int(ring['transfer_step']),
            "cashout_step": int(ring['cashout_step']),
            "victim": str(ring['victim_account_id']),
            "mule": str(ring['mule_account_id']),
            "cashout_dest": str(ring['cashout_destination_id']),
            "transfer_amount": float(ring['transfer_amount']),
            "cashout_amount": float(ring['cashout_amount'])
        })
        
    graph_metadata = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "money_mule_rings_detected": len(mule_rings_df),
        "top_out_degree_hubs": sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5],
        "top_in_degree_hubs": sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5],
        "sample_mule_rings": sample_rings
    }
    
    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "graph_metadata.json")
    with open(out_path, "w") as f:
        json.dump(graph_metadata, f, indent=2)
        
    print(f"[+] Saved graph metadata and mule rings to `{out_path}` in {time.time() - start_time:.2f}s!")
    conn.close()

if __name__ == '__main__':
    build_transaction_graph()
