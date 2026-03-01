"""Comprehensive dashboard verification."""
import urllib.request
import json
import time

def check(name, ok):
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {name}")
    return ok

def main():
    print("=== Comprehensive Dashboard Verification ===\n")
    passed = 0
    total = 0

    # 1. HTML Structure
    r = urllib.request.urlopen("http://localhost:5000")
    html = r.read().decode()
    checks = [
        ("Landing page", "page-landing" in html),
        ("Dashboard page", "page-dashboard" in html),
        ("Alerts page", "page-alerts" in html),
        ("Investigation page", "page-investigate" in html),
        ("Models page", "page-models" in html),
        ("Streaming page", "page-streaming" in html),
        ("Chart.js CDN", "chart.js" in html.lower()),
        ("D3.js CDN", "d3" in html),
        ("app.js loaded", "app.js" in html),
        ("style.css loaded", "style.css" in html),
    ]
    print("HTML Structure:")
    for name, ok in checks:
        total += 1
        if check(name, ok):
            passed += 1

    # 2. CSS Theme
    r2 = urllib.request.urlopen("http://localhost:5000/static/css/style.css")
    css = r2.read().decode()
    css_checks = [
        ("Light bg-void (#f0f2f5)", "#f0f2f5" in css),
        ("White card bg", "rgba(255, 255, 255" in css),
        ("Dark text (#1e293b)", "#1e293b" in css),
        ("Accent violet (#7c3aed)", "#7c3aed" in css),
        ("Agent console dark bg", "#1e293b" in css and ".agent-console" in css),
        ("Chat panel styles", ".chat-panel" in css),
        ("Chat messages styles", ".chat-messages" in css),
        ("Chat input styles", ".chat-input" in css),
        ("Landing page styles", ".landing-page" in css),
        ("Responsive styles", "@media" in css),
        ("Print styles", "@media print" in css),
        ("No dark #030614", "#030614" not in css),
    ]
    print("\nCSS Theme:")
    for name, ok in css_checks:
        total += 1
        if check(name, ok):
            passed += 1

    # 3. JS Functions
    r3 = urllib.request.urlopen("http://localhost:5000/static/js/app.js")
    js = r3.read().decode()
    js_checks = [
        ("openChatPanel function", "function openChatPanel" in js),
        ("sendChat function", "function sendChat" in js),
        ("closeAgentConsole", "function closeAgentConsole" in js),
        ("cleanupCurrentPage", "function cleanupCurrentPage" in js),
        ("initDashboard", "function initDashboard" in js),
        ("initAlerts", "function initAlerts" in js),
        ("initModels", "function initModels" in js),
        ("initStreaming", "function initStreaming" in js),
        ("toggleStream", "function toggleStream" in js),
        ("renderEvidenceGraph", "function renderEvidenceGraph" in js),
        ("navigateToInvestigate", "function navigateToInvestigate" in js),
        ("Investigate nav case", "case 'investigate'" in js),
        ("openChatPanel onclick", "openChatPanel()" in js),
    ]
    print("\nJS Functions:")
    for name, ok in js_checks:
        total += 1
        if check(name, ok):
            passed += 1

    # 4. API Endpoints
    apis = ["/api/overview", "/api/alerts", "/api/models"]
    print("\nAPI Endpoints:")
    for api in apis:
        total += 1
        try:
            r = urllib.request.urlopen(f"http://localhost:5000{api}")
            data = json.loads(r.read().decode())
            if check(f"{api} -> {r.status}", True):
                passed += 1
        except Exception as e:
            check(f"{api} -> {e}", False)

    # 5. Chat API
    print("\nChat API:")
    total += 1
    try:
        req = urllib.request.Request(
            "http://localhost:5000/api/agent/chat",
            data=json.dumps({"message": "hello", "account_id": 22}).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        r = urllib.request.urlopen(req, timeout=60)
        d = json.loads(r.read().decode())
        elapsed = time.time() - t0
        ok = r.status == 200 and len(d.get("response", "")) > 10
        if check(f"POST /api/agent/chat -> {r.status} ({elapsed:.1f}s)", ok):
            passed += 1
        print(f"    Response preview: {d['response'][:100]}...")
    except Exception as e:
        check(f"Chat API: {e}", False)

    print(f"\n=== Results: {passed}/{total} checks passed ===")
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
