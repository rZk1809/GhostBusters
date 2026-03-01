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


# ── Graph Forensics ──────────────────────────────────────────

def get_transfer_graph():
    """Load transfer edges for money tracing."""
    if "transfer_adj" in _cache:
        return _cache["transfer_adj"]
    
    import numpy as np
    edge_path = DATA_DIR / "test" / "graph_data" / "transfer_edge_index.npy"
    
    adj_out = {}
    adj_in = {}
    
    if edge_path.exists():
        # Load [2, E] edge index
        edge_index = np.load(edge_path)
        src, dst = edge_index[0], edge_index[1]
        
        # Build adjacency lists (using simple dicts for speed on 320k edges)
        for s, d in zip(src, dst):
            s, d = int(s), int(d)
            if s not in adj_out: adj_out[s] = []
            if d not in adj_in: adj_in[d] = []
            adj_out[s].append(d)
            adj_in[d].append(s)
            
    _cache["transfer_adj"] = (adj_out, adj_in)
    return adj_out, adj_in

@app.route("/api/trace/<int:account_id>")
def api_trace_funds(account_id):
    """Trace funds upstream (source) and downstream (destination) 3 hops."""
    adj_out, adj_in = get_transfer_graph()
    
    def bfs(start_node, adj_map, max_depth=3):
        paths = []
        queue = [(start_node, [start_node])]
        visited = {start_node}
        
        while queue:
            node, path = queue.pop(0)
            if len(path) > max_depth + 1:
                continue
            
            # Save path if length > 1
            if len(path) > 1:
                paths.append(path)
                
            if len(path) <= max_depth:
                current_depth = len(path)
                neighbors = adj_map.get(node, [])
                # Limit branching factor to avoid explosion
                for neighbor in neighbors[:5]: 
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        return paths

    upstream = bfs(account_id, adj_in, max_depth=3)   # Who sent money TO me?
    downstream = bfs(account_id, adj_out, max_depth=3) # Who did I send money TO?
    
    return jsonify({
        "account_id": account_id,
        "upstream": upstream,
        "downstream": downstream
    })

@app.route("/api/community/<int:account_id>")
def api_community(account_id):
    """Detect mule ring (connected component in transfer graph)."""
    adj_out, adj_in = get_transfer_graph()
    
    # Simple BFS to find connected component (ignoring direction for community)
    # limit to 50 nodes max for visualization
    visited = {account_id}
    queue = [account_id]
    nodes = []
    edges = []
    
    while queue and len(visited) < 50:
        u = queue.pop(0)
        nodes.append(u)
        
        # Outgoing neighbors
        for v in adj_out.get(u, []):
            if len(visited) >= 50: break
            if v not in visited:
                visited.add(v)
                queue.append(v)
            if v in visited:
                edges.append({"src": u, "dst": v})
                
        # Incoming neighbors
        for v in adj_in.get(u, []):
            if len(visited) >= 50: break
            if v not in visited:
                visited.add(v)
                queue.append(v)
            if v in visited:
                edges.append({"src": v, "dst": u})
                
    return jsonify({
        "community_id": account_id, # Simplified
        "nodes": list(visited),
        "edges": edges
    })


# ── Ollama Integration ────────────────────────────────────────

import urllib.request

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

def ollama_available():
    """Check if Ollama is running."""
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False

def ollama_generate(prompt, stream=False):
    """Call Ollama generate API."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {"temperature": 0.7, "num_predict": 1024}
    }).encode("utf-8")
    
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    if stream:
        return urllib.request.urlopen(req, timeout=120)
    else:
        resp = urllib.request.urlopen(req, timeout=120)
        return json.loads(resp.read().decode("utf-8"))


# ── Agentic AI Investigator (Ollama-Powered) ─────────────────

def _gather_intelligence(account_id):
    """Gather all evidence for an account (profile, graph, community, trace)."""
    explanations = get_explanations()
    profile = next((ex for ex in explanations if ex["account_id"] == account_id), None)
    
    adj_out, adj_in = get_transfer_graph()
    
    # Community size
    visited = {account_id}
    queue = [account_id]
    while queue and len(visited) < 100:
        u = queue.pop(0)
        for v in adj_out.get(u, []) + adj_in.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    community_size = len(visited)
    
    # Trace counts
    def count_hops(adj, depth=2):
        q = [(account_id, 0)]
        seen = {account_id}
        count = 0
        while q:
            u, d = q.pop(0)
            if d >= depth: continue
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    count += 1
                    q.append((v, d+1))
        return count
    
    upstream = count_hops(adj_in)
    downstream = count_hops(adj_out)
    
    # Risk scoring
    score = 0.1
    risk_level = profile.get("risk_level", "LOW") if profile else "UNKNOWN"
    if profile:
        score += 0.4
        if risk_level == "HIGH": score += 0.3
    if community_size > 10: score += 0.3
    if upstream > 0 and downstream > 0: score += 0.2
    
    if score > 0.8: verdict = "HIGH_RISK"
    elif score > 0.5: verdict = "SUSPICIOUS"
    else: verdict = "LOW_RISK"
    
    return {
        "profile": profile,
        "risk_level": risk_level,
        "community_size": community_size,
        "upstream": upstream,
        "downstream": downstream,
        "score": round(score, 2),
        "verdict": verdict,
        "is_flagged": profile is not None,
        "reason_codes": [r.get("code", "") for r in (profile or {}).get("reason_codes", [])],
    }


@app.route("/api/agent/investigate/<int:account_id>")
def api_agent_investigate(account_id):
    """
    AI Agent investigation — uses Ollama qwen2.5 if available,
    falls back to rule-based reasoning.
    Returns SSE stream if Ollama is live, JSON otherwise.
    """
    intel = _gather_intelligence(account_id)
    
    use_llm = ollama_available()
    
    if use_llm:
        # ── SSE Stream with Ollama ──
        def generate_sse():
            from datetime import datetime
            
            # Step 1: Rule-based reasoning steps (fast)
            reasoning_steps = [
                f"Initializing GhostBusters AI Agent v3.0 (qwen2.5)...",
                f"Loading profile for Account #{account_id}...",
            ]
            
            if intel["is_flagged"]:
                reasoning_steps.append(f"⚠ COMPLIANCE ALERT: Risk Level = {intel['risk_level']}")
                reasoning_steps.extend([f"  Reason: {rc}" for rc in intel["reason_codes"][:3]])
            else:
                reasoning_steps.append("No prior compliance alerts. Zero-trust scan initiated.")
            
            reasoning_steps.extend([
                f"Scanning transaction graph...",
                f"Network: {intel['community_size']} linked accounts in cluster.",
            ])
            
            if intel["community_size"] > 10:
                reasoning_steps.append(f"❗ ANOMALY: Large cluster ({intel['community_size']} nodes) — possible mule ring.")
            
            reasoning_steps.extend([
                f"Tracing funds: {intel['upstream']} sources → Account → {intel['downstream']} destinations.",
            ])
            
            if intel["upstream"] > 0 and intel["downstream"] > 0:
                reasoning_steps.append("🚩 Pass-through pattern detected (Layering Phase).")
            
            reasoning_steps.extend([
                f"Risk Score: {intel['score']:.2f} / 1.00",
                f"VERDICT: {intel['verdict']}",
            ])
            
            # Emit reasoning steps
            for step in reasoning_steps:
                yield f"data: {json.dumps({'type': 'step', 'text': step})}\n\n"
            
            yield f"data: {json.dumps({'type': 'step', 'text': '🤖 Generating AI analysis with qwen2.5...'})}\n\n"
            
            # Step 2: LLM analysis
            prompt = f"""You are a financial crime investigator AI for an Anti-Money Laundering (AML) system called GhostBusters.

EVIDENCE FOR ACCOUNT #{account_id}:
- Risk Level: {intel['risk_level']}
- SAR Status: {'Confirmed SAR' if intel['is_flagged'] else 'Not flagged'}
- Connected cluster size: {intel['community_size']} accounts
- Upstream fund sources: {intel['upstream']}
- Downstream fund destinations: {intel['downstream']}
- Risk Score: {intel['score']:.2f}/1.00
- Reason Codes: {', '.join(intel['reason_codes'][:5]) or 'None'}

Based on this evidence, write a concise Suspicious Activity Report (SAR) narrative. Include:
1. Subject description
2. Suspicious activity summary
3. Financial flow analysis
4. Network analysis findings
5. Recommended actions

Keep it professional, factual, and under 300 words. Format as a proper SAR document."""

            try:
                resp = ollama_generate(prompt, stream=True)
                full_response = []
                
                for line in resp:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        full_response.append(token)
                        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                    if chunk.get("done", False):
                        break
                
                sar_text = "".join(full_response)
                
            except Exception as e:
                sar_text = f"[LLM Error: {str(e)}. Falling back to template.]"
                # Template fallback
                sar_text = _generate_template_sar(account_id, intel)
                yield f"data: {json.dumps({'type': 'token', 'text': sar_text})}\n\n"
            
            # Final result
            yield f"data: {json.dumps({'type': 'result', 'sar_text': sar_text, 'verdict': intel['verdict'], 'score': intel['score']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return Response(
            generate_sse(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
    
    else:
        # ── Fallback: Rule-based JSON ──
        steps = [
            f"Initializing GhostBusters AI Agent v2.0 (offline mode)...",
            f"Loading profile for Account #{account_id}...",
        ]
        
        if intel["is_flagged"]:
            steps.append(f"⚠ COMPLIANCE ALERT FOUND: Risk Level = {intel['risk_level']}")
        else:
            steps.append("No prior compliance alerts found. Zero-trust scan initiated.")
        
        steps.extend([
            f"Scanning transaction graph...",
            f"Network Analysis: {intel['community_size']} linked accounts.",
        ])
        
        if intel["community_size"] > 10:
            steps.append(f"❗ ANOMALY: Large cluster ({intel['community_size']} nodes).")
        
        steps.extend([
            f"Tracing funds: {intel['upstream']} sources, {intel['downstream']} destinations.",
        ])
        
        if intel["upstream"] > 0 and intel["downstream"] > 0:
            steps.append("🚩 Pass-through behavior detected (Layering Phase).")
        
        action = "FILE_SAR" if intel["verdict"] == "HIGH_RISK" else "ENHANCED_DUE_DILIGENCE" if intel["verdict"] == "SUSPICIOUS" else "CLOSE_CASE"
        
        steps.extend([
            f"Risk Score: {intel['score']:.2f} / 1.00",
            f"VERDICT: {intel['verdict']}",
            f"RECOMMENDED ACTION: {action}",
        ])
        
        sar_text = _generate_template_sar(account_id, intel)
        
        return jsonify({
            "account_id": account_id,
            "steps": steps,
            "result": {
                "verdict": intel["verdict"],
                "score": intel["score"],
                "action": action,
                "sar_text": sar_text
            }
        })


def _generate_template_sar(account_id, intel):
    """Generate a template-based SAR when Ollama is unavailable."""
    from datetime import datetime
    action = "FILE SAR" if intel["verdict"] == "HIGH_RISK" else "ENHANCED DUE DILIGENCE" if intel["verdict"] == "SUSPICIOUS" else "CLOSE CASE"
    return f"""SUSPICIOUS ACTIVITY REPORT (SAR)
--------------------------------
DATE: {datetime.now().strftime('%Y-%m-%d')}
SUBJECT: Account #{account_id}
RISK RATING: {intel['verdict']} ({intel['score']:.2f})

NARRATIVE:
GhostBusters AI has identified suspicious activity involving Account #{account_id}.
The subject is part of a connected cluster of {intel['community_size']} accounts,
exhibiting characteristics consistent with a mule network.

Financial Flow Analysis:
- Inbound Transfers: {intel['upstream']} distinct sources identified.
- Outbound Transfers: {intel['downstream']} distinct destinations identified.
- Flow Pattern: High-velocity pass-through activity detected.

The account appears to be acting as a node in a larger money laundering scheme.
The network topology suggests organized layering.

RECOMMENDATION:
{action}. Immediate freezing of funds recommended pending LEA review."""


# ── Agent Chat (Ollama Q&A) ──────────────────────────────────

from flask import request

@app.route("/api/agent/chat", methods=["POST"])
def api_agent_chat():
    """Free-form chat with the AI agent about an account."""
    body = request.get_json(force=True)
    message = body.get("message", "")
    account_id = body.get("account_id")
    
    if not message:
        return jsonify({"response": "Please provide a message."}), 400
    
    # Gather context
    context = ""
    if account_id:
        intel = _gather_intelligence(account_id)
        context = f"""
Context for Account #{account_id}:
- Risk Level: {intel['risk_level']}
- SAR Flagged: {intel['is_flagged']}
- Connected Cluster Size: {intel['community_size']} accounts
- Upstream Sources: {intel['upstream']}
- Downstream Destinations: {intel['downstream']}
- Risk Score: {intel['score']:.2f}/1.00
- Verdict: {intel['verdict']}
- Reason Codes: {', '.join(intel['reason_codes'][:5]) or 'None'}
"""
    
    prompt = f"""You are GhostBusters AI, an expert financial crime investigator for an Anti-Money Laundering system.
{context}

User Question: {message}

Answer concisely and professionally. Focus on AML investigation insights. Keep your response under 200 words."""

    if ollama_available():
        try:
            result = ollama_generate(prompt, stream=False)
            response_text = result.get("response", "No response generated.")
        except Exception as e:
            response_text = f"LLM Error: {str(e)}. Please try again."
    else:
        response_text = f"Ollama is currently offline. Based on rule-based analysis: Account #{account_id} has a risk score of {intel['score']:.2f} ({intel['verdict']}). The account is connected to {intel['community_size']} accounts with {intel['upstream']} upstream sources and {intel['downstream']} downstream destinations."
    
    return jsonify({"response": response_text})


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  GhostBusters AML Platform")
    print(f"{'='*60}")
    print(f"  Data dir:    {DATA_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Models dir:  {MODELS_DIR}")
    print(f"  Ollama:      {OLLAMA_URL} ({OLLAMA_MODEL})")
    print(f"  Open:        http://localhost:5000")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
