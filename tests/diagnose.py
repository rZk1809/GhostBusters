"""Quick diagnostic: test every API and check data shapes."""
import urllib.request, json

BASE = "http://localhost:5000"

def get(path):
    r = urllib.request.urlopen(f"{BASE}{path}", timeout=15)
    return json.loads(r.read())

print("=== API Diagnostics ===\n")

# 1. Overview
ov = get("/api/overview")
print(f"[overview] total_accounts={ov.get('total_accounts')}, flagged={ov.get('flagged_accounts')}, best_model={ov.get('best_model',{}).get('name')}")
print(f"           risk_dist={ov.get('risk_distribution')}, edge_stats keys={list(ov.get('edge_stats',{}).keys())}")
print(f"           total_edges={ov.get('total_edges')}, sar_rate={ov.get('sar_rate')}")

# 2. Alerts
alerts = get("/api/alerts")
print(f"\n[alerts] count={len(alerts)}")
if alerts:
    a = alerts[0]
    print(f"  sample: id={a['account_id']}, risk={a['risk_level']}, is_sar={a['is_sar']}")
    rc = a.get("reason_codes", [])
    print(f"  reason_codes[0] type={type(rc[0]) if rc else 'N/A'}")
    if rc:
        print(f"  reason_codes[0] = {rc[0]}")

# 3. Alert Detail
    aid = a["account_id"]
    det = get(f"/api/alerts/{aid}")
    print(f"\n[detail #{aid}] risk={det.get('risk_level')}, narrative_len={len(det.get('narrative',''))}")
    print(f"  feature_profile keys: {list(det.get('feature_profile',{}).keys())[:6]}")
    sg = det.get("evidence_subgraph", {})
    print(f"  evidence_subgraph: {len(sg.get('nodes',[]))} nodes, {len(sg.get('edges',[]))} edges")

# 4. Graph
    g = get(f"/api/graph/{aid}")
    print(f"\n[graph #{aid}] {len(g.get('nodes',[]))} nodes, {len(g.get('edges',[]))} edges")
    if g.get("nodes"):
        print(f"  node sample: {g['nodes'][0]}")
    if g.get("edges"):
        print(f"  edge sample: {g['edges'][0]}")

# 5. Trace
    t = get(f"/api/trace/{aid}")
    print(f"\n[trace #{aid}] upstream={len(t.get('upstream',[]))}, downstream={len(t.get('downstream',[]))}")

# 6. Community
    c = get(f"/api/community/{aid}")
    print(f"\n[community #{aid}] nodes={len(c.get('nodes',[]))}, edges={len(c.get('edges',[]))}")

# 7. Models
    m = get("/api/models")
    print(f"\n[models] baseline={list(m.get('baseline',{}).keys())}")
    print(f"         gnn={list(m.get('gnn',{}).keys())}")
    print(f"         training_histories={list(m.get('training_histories',{}).keys())}")
    bl_sample = list(m.get("baseline",{}).values())
    if bl_sample:
        print(f"  baseline sample keys: {list(bl_sample[0].keys())[:8]}")
    gnn_sample = list(m.get("gnn",{}).values())
    if gnn_sample:
        print(f"  gnn sample keys: {list(gnn_sample[0].keys())[:8]}")

# 8. Stream test
    req = urllib.request.urlopen(f"{BASE}/api/stream", timeout=5)
    chunk = req.read(500).decode()
    req.close()
    print(f"\n[stream] first 200 chars: {chunk[:200]}")

print("\n=== Done ===")
