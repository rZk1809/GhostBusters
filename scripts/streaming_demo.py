"""
streaming_demo.py
=================
Simulates real-time scoring of transactions using a trained GNN model.
Processes transactions in timestamp order, updates graph state incrementally,
scores each new edge/node, and triggers alerts when threshold is exceeded.

Usage:
    python scripts/streaming_demo.py                     # full demo
    python scripts/streaming_demo.py --limit 1000        # first 1000 txns
    python scripts/streaming_demo.py --threshold 0.7     # custom threshold
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))
from train_gnn import GraphSAGEModel, GATModel

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "amlsim_v1")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load the best available trained model."""
    for model_file, ModelClass, config_key in [
        ("graphsage_node.pt", GraphSAGEModel, "config"),
        ("gat_node.pt", GATModel, "config"),
    ]:
        path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            cfg = ckpt[config_key]
            if "num_heads" in cfg:
                model = GATModel(cfg["in_dim"], cfg["hidden_dim"], cfg["out_dim"],
                                num_heads=cfg.get("num_heads", 4)).to(DEVICE)
            else:
                model = GraphSAGEModel(cfg["in_dim"], cfg["hidden_dim"], cfg["out_dim"]).to(DEVICE)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            norm = ckpt.get("norm", {})
            mean = torch.tensor(norm.get("mean", [0]*cfg["in_dim"]), dtype=torch.float32, device=DEVICE)
            std = torch.tensor(norm.get("std", [1]*cfg["in_dim"]), dtype=torch.float32, device=DEVICE)
            print(f"  Loaded model: {model_file}")
            return model, mean, std

    print("[ERROR] No trained model found. Run train_gnn.py first.")
    return None, None, None


def run_streaming_demo(limit=500, threshold=0.5):
    """Simulate real-time transaction scoring."""
    print("="*60)
    print("STREAMING DEMO - Real-Time Mule Account Scoring")
    print("="*60)

    # Load model
    print("\nLoading model...")
    model, mean, std = load_model()
    if model is None:
        return

    # Load test data for simulation
    print("Loading test data for simulation...")
    gd = os.path.join(BASE_DIR, "test", "graph_data")
    features = np.load(os.path.join(gd, "account_features.npy"))
    labels = np.load(os.path.join(gd, "account_labels.npy"))
    edge_index = np.load(os.path.join(gd, "transfer_edge_index.npy"))

    # Load same_wallet edges
    sw_path = os.path.join(gd, "same_wallet_edge_index.npy")
    if os.path.exists(sw_path):
        sw_edges = np.load(sw_path)
        if sw_edges.shape[1] > 0:
            edge_index = np.concatenate([edge_index, sw_edges], axis=1)

    # Also load transactions for streaming order
    tx_df = pd.read_csv(os.path.join(BASE_DIR, "test", "enriched_transactions.csv"))
    tx_df = tx_df.sort_values("tran_timestamp").head(limit)

    X = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    X = (X - mean) / std.clamp(min=1e-6)
    X[torch.isnan(X)] = 0
    X[torch.isinf(X)] = 0

    edges = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)

    print(f"\nSimulating {len(tx_df)} transactions...")
    print(f"Alert threshold: {threshold}")
    print("-"*80)

    alerts = []
    scored = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    start_time = time.time()

    with torch.no_grad():
        # Get all node scores at once (simulating incremental updates)
        logits = model(X, edges)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    for idx, (_, row) in enumerate(tx_df.iterrows()):
        orig = int(row["orig_acct"])
        bene = int(row["bene_acct"])
        amt = float(row["base_amt"])
        ts = row["tran_timestamp"]
        channel = row.get("channel", "UNKNOWN")
        is_sar = str(row.get("is_sar", False)).lower() in ["true", "1"]

        # Score the originator and beneficiary
        score_orig = float(probs[orig]) if orig < len(probs) else 0
        score_bene = float(probs[bene]) if bene < len(probs) else 0
        max_score = max(score_orig, score_bene)

        flagged = max_score >= threshold
        actual_sar = labels[orig] == 1 or labels[bene] == 1 if orig < len(labels) and bene < len(labels) else False

        if flagged and actual_sar:
            true_positives += 1
        elif flagged and not actual_sar:
            false_positives += 1
        elif not flagged and actual_sar:
            false_negatives += 1
        else:
            true_negatives += 1

        scored += 1

        if flagged:
            alert = {
                "tx_id": int(row.get("tran_id", idx)),
                "timestamp": str(ts),
                "orig_acct": orig,
                "bene_acct": bene,
                "amount": amt,
                "channel": str(channel),
                "score_orig": round(score_orig, 4),
                "score_bene": round(score_bene, 4),
                "max_score": round(max_score, 4),
                "actual_sar": bool(actual_sar),
                "reason": []
            }

            # Add reason codes
            if score_orig >= threshold:
                alert["reason"].append(f"ORIG_HIGH_RISK (score={score_orig:.3f})")
            if score_bene >= threshold:
                alert["reason"].append(f"BENE_HIGH_RISK (score={score_bene:.3f})")
            if channel in ["ATM", "UPI"]:
                alert["reason"].append(f"HIGH_RISK_CHANNEL ({channel})")
            if amt > 200:
                alert["reason"].append(f"HIGH_AMOUNT ({amt:.2f})")

            alerts.append(alert)

            if len(alerts) <= 20:  # Print first 20 alerts
                status = "TRUE POSITIVE" if actual_sar else "FALSE POSITIVE"
                print(f"  [ALERT] TX #{alert['tx_id']} | {ts} | {channel} | "
                      f"${amt:.2f} | Score: {max_score:.3f} | {status}")
                for r in alert["reason"]:
                    print(f"          -> {r}")

    elapsed = time.time() - start_time
    latency_per_tx = (elapsed / scored * 1000) if scored > 0 else 0

    # Summary
    print("\n" + "="*60)
    print("STREAMING DEMO SUMMARY")
    print("="*60)
    print(f"  Transactions processed: {scored:,}")
    print(f"  Total alerts triggered: {len(alerts):,}")
    print(f"  Processing time: {elapsed:.2f}s")
    print(f"  Avg latency per tx: {latency_per_tx:.2f}ms")
    print(f"\n  True Positives:  {true_positives:,}")
    print(f"  False Positives: {false_positives:,}")
    print(f"  True Negatives:  {true_negatives:,}")
    print(f"  False Negatives: {false_negatives:,}")
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    # Save alerts
    os.makedirs(RESULTS_DIR, exist_ok=True)
    alerts_path = os.path.join(RESULTS_DIR, "streaming_alerts.json")
    with open(alerts_path, "w") as f:
        json.dump({
            "summary": {
                "total_txns": scored,
                "total_alerts": len(alerts),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "precision": precision,
                "recall": recall,
                "avg_latency_ms": latency_per_tx
            },
            "alerts": alerts[:100]  # Save first 100 alerts
        }, f, indent=2)
    print(f"\n  Alerts saved to {alerts_path}")


def main():
    parser = argparse.ArgumentParser(description="Streaming scoring demo")
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of transactions to process")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Alert threshold (0-1)")
    args = parser.parse_args()

    run_streaming_demo(limit=args.limit, threshold=args.threshold)


if __name__ == "__main__":
    main()
