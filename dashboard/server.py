"""
GhostBusters AML Dashboard — Flask Backend
============================================
Serves pre-computed model results, flagged account explanations,
evidence subgraphs, and simulated streaming transaction scoring.
"""

import os
import sys
import json
import time
import random
import csv
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, Response

# ── Paths ──────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "amlsim_v1"
RESULTS_DIR = BASE / "results"
MODELS_DIR = BASE / "models"

app = Flask(__name__, static_folder="static", static_url_path="/static")


# ── Data Loaders (cached) ─────────────────────────────────────

_cache = {}

def _load_json(path, key):
    if key not in _cache:
        with open(path, encoding="utf-8") as f:
            _cache[key] = json.load(f)
    return _cache[key]

def get_all_results():
    return _load_json(RESULTS_DIR / "all_results.json", "all_results")

def get_explanations():
    return _load_json(RESULTS_DIR / "explanations.json", "explanations")

def get_training_histories():
    return _load_json(RESULTS_DIR / "gnn_training_histories.json", "histories")

def get_graph_stats():
    stats = {}
    for split in ["train", "val", "test"]:
        p = DATA_DIR / split / "graph_data" / "graph_stats.json"
        if p.exists():
            with open(p) as f:
                stats[split] = json.load(f)
    return stats

def get_transactions_sample():
    """Load a sample of test transactions for streaming simulation."""
    if "txn_sample" in _cache:
        return _cache["txn_sample"]
    txn_path = DATA_DIR / "test" / "transactions.csv"
    rows = []
    if txn_path.exists():
        with open(txn_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 2000:
                    break
                rows.append(row)
    _cache["txn_sample"] = rows
    return rows


# ── Static / SPA ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── API: Overview / KPIs ──────────────────────────────────────

@app.route("/api/overview")
def api_overview():
    results = get_all_results()
    explanations = get_explanations()
    stats = get_graph_stats()

    test_stats = stats.get("test", {})
    total_accounts = test_stats.get("num_accounts", 0)
    sar_accounts = test_stats.get("num_sar_accounts", 0)
    total_edges = test_stats.get("total_edges", 0)

    # Best model by PR-AUC
    best_model = {"name": "N/A", "pr_auc": 0}
    for category in ["baseline", "gnn"]:
        for name, metrics in results.get(category, {}).items():
            pr = metrics.get("pr_auc", 0)
            if pr > best_model["pr_auc"]:
                best_model = {"name": name, "pr_auc": pr, "type": category}

    # Risk distribution from explanations
    risk_dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for ex in explanations:
        risk_dist[ex.get("risk_level", "MEDIUM")] += 1

    # Channel stats from graph edges
    edge_stats = test_stats.get("edges", {})

    return jsonify({
        "total_accounts": total_accounts,
        "sar_accounts": sar_accounts,
        "sar_rate": round(sar_accounts / max(total_accounts, 1) * 100, 2),
        "total_edges": total_edges,
        "flagged_accounts": len(explanations),
        "best_model": best_model,
        "risk_distribution": risk_dist,
        "edge_stats": edge_stats,
        "graph_stats": stats,
    })


# ── API: Models ───────────────────────────────────────────────

@app.route("/api/models")
def api_models():
    results = get_all_results()
    histories = get_training_histories()
    return jsonify({
        "baseline": results.get("baseline", {}),
        "gnn": results.get("gnn", {}),
        "training_histories": histories,
        "gpu_config": results.get("gpu_config", {}),
        "training_config": results.get("training_config", {}),
    })


# ── API: Alerts ───────────────────────────────────────────────

@app.route("/api/alerts")
def api_alerts():
    explanations = get_explanations()
    # Return summary list (without evidence_subgraph to keep response small)
    summaries = []
    for ex in explanations:
        summaries.append({
            "account_id": ex["account_id"],
            "is_sar": ex.get("is_sar", False),
            "risk_level": ex.get("risk_level", "MEDIUM"),
            "reason_codes": ex.get("reason_codes", []),
            "feature_profile": ex.get("feature_profile", {}),
        })
    return jsonify(summaries)


@app.route("/api/alerts/<int:account_id>")
def api_alert_detail(account_id):
    explanations = get_explanations()
    for ex in explanations:
        if ex["account_id"] == account_id:
            return jsonify(ex)
    return jsonify({"error": "Account not found"}), 404


# ── API: Evidence Subgraph ────────────────────────────────────

@app.route("/api/graph/<int:account_id>")
def api_graph(account_id):
    explanations = get_explanations()
    for ex in explanations:
        if ex["account_id"] == account_id:
            subgraph = ex.get("evidence_subgraph", {})
            nodes = subgraph.get("nodes", [])
            edges = subgraph.get("edges", [])

            # Enrich nodes with SAR labels
            node_data = []
            for n in nodes[:50]:  # Limit for visualization
                node_data.append({
                    "id": n,
                    "is_target": n == account_id,
                    "is_sar": n == account_id,  # simplified — target is SAR
                })

            edge_data = []
            node_ids = set(n["id"] for n in node_data)
            for e in edges[:80]:  # Limit edges for perf
                if e["src"] in node_ids or e["dst"] in node_ids:
                    edge_data.append(e)

            return jsonify({
                "center_node": account_id,
                "nodes": node_data,
                "edges": edge_data,
                "total_2hop": subgraph.get("num_2hop_neighbors", 0),
            })
    return jsonify({"error": "Account not found"}), 404


# ── API: Streaming SSE ────────────────────────────────────────

@app.route("/api/stream")
def api_stream():
    """Server-Sent Events: simulated real-time transaction scoring."""
    def generate():
        txns = get_transactions_sample()
        channels = ["APP", "UPI", "ATM", "WALLET", "WEB"]
        risk_labels = ["NORMAL", "NORMAL", "NORMAL", "NORMAL",
                       "NORMAL", "NORMAL", "NORMAL", "SUSPICIOUS",
                       "SUSPICIOUS", "HIGH_RISK"]

        i = 0
        while True:
            if txns:
                txn = txns[i % len(txns)]
                amount = float(txn.get("amount", random.uniform(50, 5000)))
                src = txn.get("src", f"ACCT-{random.randint(1000,9999)}")
                dst = txn.get("dst", f"ACCT-{random.randint(1000,9999)}")
            else:
                amount = round(random.uniform(50, 5000), 2)
                src = f"ACCT-{random.randint(1000, 9999)}"
                dst = f"ACCT-{random.randint(1000, 9999)}"

            score = round(random.random(), 4)
            risk = random.choice(risk_labels) if score < 0.7 else "HIGH_RISK"
            channel = random.choice(channels)

            event = {
                "id": i,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "src": str(src),
                "dst": str(dst),
                "amount": round(amount, 2),
                "channel": channel,
                "score": score,
                "risk": risk,
            }

            yield f"data: {json.dumps(event)}\n\n"
            i += 1
            time.sleep(random.uniform(0.3, 1.2))

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  GhostBusters AML Dashboard")
    print(f"{'='*60}")
    print(f"  Data dir:    {DATA_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Models dir:  {MODELS_DIR}")
    print(f"  Open:        http://localhost:5000")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
