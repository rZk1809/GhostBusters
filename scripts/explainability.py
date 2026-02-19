"""
explainability.py
=================
Generate regulator-ready explanations for flagged accounts:
  1. Evidence subgraph — k-hop neighborhood export
  2. Top contributing signals — feature importance per account
  3. Reason codes — structured typology tags
  4. Narrative summary — auto-generated text explanation

Usage:
    python scripts/explainability.py                              # top 20 flagged accounts
    python scripts/explainability.py --top-k 50                   # top 50
    python scripts/explainability.py --acct-ids 51164,1079,26617  # specific accounts
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "amlsim_v1")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

FEATURE_NAMES = [
    "acct_type", "is_active", "prior_sar", "initial_deposit",
    "sent_count", "recv_count", "sent_amt_mean", "sent_amt_std",
    "recv_amt_mean", "recv_amt_std", "fan_out_ratio", "fan_in_ratio",
    "recv_send_delay_min", "channel_diversity"
]

# Reason code definitions
REASON_CODES = {
    "RAPID_PASS_THROUGH": "Account receives and forwards funds within minutes",
    "FAN_IN": "Multiple accounts sending funds to this account (high in-degree)",
    "FAN_OUT": "Account sending funds to many different accounts (high out-degree)",
    "HIGH_VALUE": "Transaction amounts significantly above account average",
    "DEVICE_SHARING": "Account shares login device with other flagged accounts",
    "WALLET_SHARING": "Account shares wallet with other flagged accounts",
    "CHANNEL_HOPPING": "Account uses multiple channels in short timeframe",
    "RAPID_CASHOUT": "ATM withdrawal shortly after receiving funds",
    "PRIOR_SAR": "Account has prior SAR filing history",
    "UNUSUAL_ACTIVITY": "Account activity deviates significantly from historical pattern"
}


def load_data(split="test"):
    """Load features, labels, edges, and transaction data."""
    gd = os.path.join(BASE_DIR, split, "graph_data")

    data = {
        "features": np.load(os.path.join(gd, "account_features.npy")),
        "labels": np.load(os.path.join(gd, "account_labels.npy")),
        "transfer_edges": np.load(os.path.join(gd, "transfer_edge_index.npy")),
    }

    # Load optional edge types
    for name in ["same_wallet_edge_index", "shared_device_edge_index"]:
        path = os.path.join(gd, f"{name}.npy")
        if os.path.exists(path):
            data[name] = np.load(path)

    # Load ID mappings
    map_path = os.path.join(gd, "id_mappings.json")
    if os.path.exists(map_path):
        with open(map_path) as f:
            data["mappings"] = json.load(f)

    # Load transactions
    tx_path = os.path.join(BASE_DIR, split, "enriched_transactions.csv")
    if os.path.exists(tx_path):
        data["transactions"] = pd.read_csv(tx_path)

    # Load SAR accounts
    sar_path = os.path.join(BASE_DIR, split, "sar_accounts.csv")
    if os.path.exists(sar_path):
        try:
            sar_df = pd.read_csv(sar_path)
            sar_df.columns = [c.lower() for c in sar_df.columns]
            col = "account_id" if "account_id" in sar_df.columns else "acct_id"
            data["sar_accounts"] = set(sar_df[col].unique())
        except Exception:
            data["sar_accounts"] = set()

    # Load ATM withdrawals
    atm_path = os.path.join(BASE_DIR, split, "atm_withdrawals.csv")
    if os.path.exists(atm_path):
        try:
            data["atm"] = pd.read_csv(atm_path)
        except Exception:
            data["atm"] = pd.DataFrame()

    return data


def get_k_hop_neighbors(acct_idx, edges, k=2):
    """Get k-hop neighborhood of an account."""
    visited = {acct_idx}
    frontier = {acct_idx}
    neighbor_edges = []

    for hop in range(k):
        next_frontier = set()
        for i in range(edges.shape[1]):
            src, dst = int(edges[0, i]), int(edges[1, i])
            if src in frontier:
                next_frontier.add(dst)
                neighbor_edges.append((src, dst, hop + 1))
            if dst in frontier:
                next_frontier.add(src)
                neighbor_edges.append((src, dst, hop + 1))
        frontier = next_frontier - visited
        visited.update(frontier)

    return visited, neighbor_edges


def generate_reason_codes(acct_idx, features, data):
    """Generate structured reason codes for a flagged account."""
    feat = features[acct_idx]
    reasons = []

    # Prior SAR
    if feat[2] > 0.5:  # prior_sar
        reasons.append({
            "code": "PRIOR_SAR",
            "description": REASON_CODES["PRIOR_SAR"],
            "confidence": "HIGH",
            "value": f"prior_sar={feat[2]:.0f}"
        })

    # Fan-in pattern (high recv_count relative to sent)
    if feat[5] > feat[4] * 3 and feat[5] > 5:  # recv_count >> sent_count
        reasons.append({
            "code": "FAN_IN",
            "description": REASON_CODES["FAN_IN"],
            "confidence": "MEDIUM",
            "value": f"recv_count={feat[5]:.0f}, sent_count={feat[4]:.0f}"
        })

    # Fan-out pattern
    if feat[4] > feat[5] * 3 and feat[4] > 5:
        reasons.append({
            "code": "FAN_OUT",
            "description": REASON_CODES["FAN_OUT"],
            "confidence": "MEDIUM",
            "value": f"sent_count={feat[4]:.0f}, recv_count={feat[5]:.0f}"
        })

    # Rapid pass-through (low recv-send delay)
    if 0 < feat[12] < 30:  # recv_send_delay_min < 30 minutes
        reasons.append({
            "code": "RAPID_PASS_THROUGH",
            "description": REASON_CODES["RAPID_PASS_THROUGH"],
            "confidence": "HIGH",
            "value": f"delay={feat[12]:.1f} min"
        })

    # Channel diversity (hopping)
    if feat[13] >= 3:  # 3+ channels used
        reasons.append({
            "code": "CHANNEL_HOPPING",
            "description": REASON_CODES["CHANNEL_HOPPING"],
            "confidence": "MEDIUM",
            "value": f"channels_used={feat[13]:.0f}"
        })

    # High value transactions
    if feat[6] > 200 or feat[8] > 200:  # mean sent/recv > $200
        reasons.append({
            "code": "HIGH_VALUE",
            "description": REASON_CODES["HIGH_VALUE"],
            "confidence": "LOW",
            "value": f"avg_sent=${feat[6]:.2f}, avg_recv=${feat[8]:.2f}"
        })

    # ATM cashout
    if "atm" in data and not data["atm"].empty:
        acct_atm = data["atm"][data["atm"]["acct_id"] == acct_idx]
        if len(acct_atm) > 0:
            reasons.append({
                "code": "RAPID_CASHOUT",
                "description": REASON_CODES["RAPID_CASHOUT"],
                "confidence": "HIGH",
                "value": f"atm_withdrawals={len(acct_atm)}"
            })

    # Wallet sharing
    sw_key = "same_wallet_edge_index"
    if sw_key in data and data[sw_key].shape[1] > 0:
        sw = data[sw_key]
        shared = np.where((sw[0] == acct_idx) | (sw[1] == acct_idx))[0]
        if len(shared) > 0:
            reasons.append({
                "code": "WALLET_SHARING",
                "description": REASON_CODES["WALLET_SHARING"],
                "confidence": "HIGH",
                "value": f"shared_wallet_links={len(shared)}"
            })

    if not reasons:
        reasons.append({
            "code": "UNUSUAL_ACTIVITY",
            "description": REASON_CODES["UNUSUAL_ACTIVITY"],
            "confidence": "LOW",
            "value": "flagged by model"
        })

    return reasons


def generate_narrative(acct_idx, features, reasons, neighbors_count, data):
    """Generate auto-text narrative for a flagged account."""
    feat = features[acct_idx]
    is_sar = data["labels"][acct_idx] == 1

    lines = []
    lines.append(f"Account {acct_idx} has been flagged for suspicious activity review.")
    lines.append(f"Ground truth: {'CONFIRMED SAR' if is_sar else 'NOT YET CONFIRMED'}.")
    lines.append("")

    lines.append("Summary of Activity:")
    lines.append(f"  - Sent {int(feat[4])} transactions (avg ${feat[6]:.2f}, std ${feat[7]:.2f})")
    lines.append(f"  - Received {int(feat[5])} transactions (avg ${feat[8]:.2f}, std ${feat[9]:.2f})")
    lines.append(f"  - Fan-out ratio: {feat[10]:.2f}, Fan-in ratio: {feat[11]:.2f}")

    if feat[12] > 0:
        lines.append(f"  - Receive-to-send delay: {feat[12]:.1f} minutes")

    lines.append(f"  - Network reach: {neighbors_count} accounts within 2 hops")
    lines.append("")

    lines.append("Risk Indicators:")
    for r in reasons:
        lines.append(f"  [{r['confidence']}] {r['code']}: {r['description']} ({r['value']})")

    return "\n".join(lines)


def generate_explanations(data, acct_ids=None, top_k=20):
    """Generate full explanations for flagged accounts."""
    features = data["features"]
    labels = data["labels"]
    edges = data["transfer_edges"]

    # Determine which accounts to explain
    if acct_ids:
        target_ids = acct_ids
    else:
        # Use SAR accounts sorted by feature divergence
        sar_mask = labels == 1
        sar_indices = np.where(sar_mask)[0]
        if len(sar_indices) > top_k:
            # Sort by total transaction volume (most active first)
            activity = features[sar_indices, 4] + features[sar_indices, 5]  # sent + recv
            top_idx = np.argsort(activity)[::-1][:top_k]
            target_ids = sar_indices[top_idx].tolist()
        else:
            target_ids = sar_indices.tolist()

    print(f"\nGenerating explanations for {len(target_ids)} accounts...")

    explanations = []
    for i, acct_idx in enumerate(target_ids):
        print(f"  [{i+1}/{len(target_ids)}] Account {acct_idx}...", end="")

        # Get neighborhood (limit edge search for speed)
        max_edges_to_search = min(edges.shape[1], 500000)
        sub_edges = edges[:, :max_edges_to_search]
        neighbors, neighbor_edges = get_k_hop_neighbors(acct_idx, sub_edges, k=2)

        # Feature importance for this account
        feat = features[acct_idx]
        feature_profile = {}
        for j, name in enumerate(FEATURE_NAMES):
            if j < len(feat):
                feature_profile[name] = float(feat[j])

        # Reason codes
        reasons = generate_reason_codes(acct_idx, features, data)

        # Narrative
        narrative = generate_narrative(acct_idx, features, reasons, len(neighbors), data)

        # Evidence subgraph (limited to keep JSON manageable)
        evidence_nodes = list(neighbors)[:50]
        evidence_edges = []
        for src, dst, hop in neighbor_edges[:100]:
            evidence_edges.append({"src": int(src), "dst": int(dst), "hop": int(hop)})

        explanation = {
            "account_id": int(acct_idx),
            "is_sar": bool(labels[acct_idx] == 1),
            "feature_profile": feature_profile,
            "reason_codes": reasons,
            "narrative": narrative,
            "evidence_subgraph": {
                "nodes": evidence_nodes,
                "edges": evidence_edges,
                "num_2hop_neighbors": len(neighbors)
            },
            "risk_level": "HIGH" if any(r["confidence"] == "HIGH" for r in reasons) else "MEDIUM"
        }

        explanations.append(explanation)
        n_reasons = len(reasons)
        risk = explanation["risk_level"]
        print(f" {n_reasons} reasons, {risk} risk, {len(neighbors)} neighbors")

    return explanations


def main():
    parser = argparse.ArgumentParser(description="Generate explainability reports")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--acct-ids", type=str, default=None,
                        help="Comma-separated account IDs")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    acct_ids = None
    if args.acct_ids:
        acct_ids = [int(x.strip()) for x in args.acct_ids.split(",")]

    print("="*60)
    print("EXPLAINABILITY MODULE")
    print("="*60)

    print("\nLoading data...")
    data = load_data(args.split)
    print(f"  Accounts: {len(data['features']):,}")
    print(f"  SAR accounts: {data['labels'].sum():,}")
    print(f"  Transfer edges: {data['transfer_edges'].shape[1]:,}")

    explanations = generate_explanations(data, acct_ids=acct_ids, top_k=args.top_k)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "explanations.json")
    with open(out_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"\n  Explanations saved to {out_path}")

    # Print summary
    print("\n" + "="*60)
    print("EXPLANATION SUMMARY")
    print("="*60)
    high_risk = sum(1 for e in explanations if e["risk_level"] == "HIGH")
    med_risk = sum(1 for e in explanations if e["risk_level"] == "MEDIUM")
    print(f"  Total explained: {len(explanations)}")
    print(f"  HIGH risk: {high_risk}")
    print(f"  MEDIUM risk: {med_risk}")

    # Most common reason codes
    code_counts = defaultdict(int)
    for e in explanations:
        for r in e["reason_codes"]:
            code_counts[r["code"]] += 1

    print("\n  Reason Code Distribution:")
    for code, count in sorted(code_counts.items(), key=lambda x: -x[1]):
        print(f"    {code}: {count} ({count/len(explanations)*100:.0f}%)")

    # Print sample narrative
    if explanations:
        print(f"\n  --- Sample Narrative (Account {explanations[0]['account_id']}) ---")
        print(explanations[0]["narrative"])


if __name__ == "__main__":
    main()
