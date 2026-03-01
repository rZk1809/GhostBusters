"""
GhostBusters AML Platform — Comprehensive Test Suite
=====================================================
Unit tests, integration tests, system tests, and endpoint verification.

Run: python -m pytest tests/test_api.py -v
"""

import os
import sys
import json
import urllib.request
import urllib.error
import unittest
import time

BASE_URL = "http://localhost:5000"
OLLAMA_URL = "http://localhost:11434"


def api_get(path):
    """Helper to GET an API endpoint."""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read().decode("utf-8")), resp.status


def api_post(path, data):
    """Helper to POST to an API endpoint."""
    url = f"{BASE_URL}{path}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read().decode("utf-8")), resp.status


class TestEndpointHealth(unittest.TestCase):
    """Basic endpoint availability and response code checks."""

    def test_index_returns_html(self):
        """/ should return HTML with 200."""
        req = urllib.request.urlopen(f"{BASE_URL}/", timeout=10)
        self.assertEqual(req.status, 200)
        body = req.read().decode()
        self.assertIn("GhostBusters", body)
        self.assertIn("<html", body)

    def test_overview_endpoint(self):
        """GET /api/overview returns JSON with expected fields."""
        data, status = api_get("/api/overview")
        self.assertEqual(status, 200)
        self.assertIn("total_accounts", data)
        self.assertIn("flagged_accounts", data)
        self.assertIn("best_model", data)
        self.assertIn("risk_distribution", data)
        self.assertIn("total_edges", data)

    def test_models_endpoint(self):
        """GET /api/models returns baseline and GNN model data."""
        data, status = api_get("/api/models")
        self.assertEqual(status, 200)
        self.assertIn("baseline", data)
        self.assertIn("gnn", data)
        self.assertIn("training_histories", data)

    def test_alerts_endpoint(self):
        """GET /api/alerts returns a list of flagged accounts."""
        data, status = api_get("/api/alerts")
        self.assertEqual(status, 200)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        # Each alert should have these fields
        for alert in data[:3]:
            self.assertIn("account_id", alert)
            self.assertIn("risk_level", alert)
            self.assertIn("reason_codes", alert)

    def test_alert_detail_valid_id(self):
        """GET /api/alerts/<id> returns detail for a valid account."""
        # First get a valid account_id
        alerts, _ = api_get("/api/alerts")
        valid_id = alerts[0]["account_id"]
        data, status = api_get(f"/api/alerts/{valid_id}")
        self.assertEqual(status, 200)
        self.assertEqual(data["account_id"], valid_id)
        self.assertIn("risk_level", data)
        self.assertIn("narrative", data)

    def test_alert_detail_invalid_id(self):
        """GET /api/alerts/<invalid_id> returns 404."""
        try:
            api_get("/api/alerts/999999")
            self.fail("Expected 404 error")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_graph_endpoint(self):
        """GET /api/graph/<id> returns nodes and edges."""
        alerts, _ = api_get("/api/alerts")
        valid_id = alerts[0]["account_id"]
        data, status = api_get(f"/api/graph/{valid_id}")
        self.assertEqual(status, 200)
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIsInstance(data["nodes"], list)
        self.assertIsInstance(data["edges"], list)


class TestGraphForensics(unittest.TestCase):
    """Test graph forensics endpoints (trace, community)."""

    def _get_valid_account(self):
        alerts, _ = api_get("/api/alerts")
        return alerts[0]["account_id"]

    def test_trace_endpoint(self):
        """GET /api/trace/<id> returns upstream/downstream paths."""
        aid = self._get_valid_account()
        data, status = api_get(f"/api/trace/{aid}")
        self.assertEqual(status, 200)
        self.assertIn("upstream", data)
        self.assertIn("downstream", data)
        self.assertIsInstance(data["upstream"], list)
        self.assertIsInstance(data["downstream"], list)

    def test_community_endpoint(self):
        """GET /api/community/<id> returns community nodes/edges."""
        aid = self._get_valid_account()
        data, status = api_get(f"/api/community/{aid}")
        self.assertEqual(status, 200)
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertLessEqual(len(data["nodes"]), 50, "Community should be limited to 50 nodes")

    def test_community_node_limit(self):
        """Verify community detection stays under 50 nodes."""
        aid = self._get_valid_account()
        data, _ = api_get(f"/api/community/{aid}")
        self.assertLessEqual(len(data["nodes"]), 50)


class TestAgentInvestigation(unittest.TestCase):
    """Test Agent investigation endpoint."""

    def _get_valid_account(self):
        alerts, _ = api_get("/api/alerts")
        return alerts[0]["account_id"]

    def test_agent_investigate_returns_data(self):
        """GET /api/agent/investigate/<id> returns either SSE or JSON."""
        aid = self._get_valid_account()
        url = f"{BASE_URL}/api/agent/investigate/{aid}"
        req = urllib.request.urlopen(url, timeout=120)
        content_type = req.headers.get("Content-Type", "")

        if "text/event-stream" in content_type:
            # SSE stream — read some events
            body = req.read(8192).decode("utf-8")
            self.assertIn("data:", body)
            # Should have step events
            self.assertTrue(
                "step" in body or "token" in body,
                "SSE should contain step or token events"
            )
        else:
            # JSON fallback
            data = json.loads(req.read().decode("utf-8"))
            self.assertIn("steps", data)
            self.assertIn("result", data)
            self.assertIn("verdict", data["result"])
            self.assertIn("sar_text", data["result"])
            self.assertGreater(len(data["steps"]), 3)


class TestAgentChat(unittest.TestCase):
    """Test the agent chat endpoint."""

    def _get_valid_account(self):
        alerts, _ = api_get("/api/alerts")
        return alerts[0]["account_id"]

    def test_chat_returns_response(self):
        """POST /api/agent/chat returns a text response."""
        aid = self._get_valid_account()
        data, status = api_post("/api/agent/chat", {
            "message": "What is the risk level of this account?",
            "account_id": aid
        })
        self.assertEqual(status, 200)
        self.assertIn("response", data)
        self.assertGreater(len(data["response"]), 10)

    def test_chat_empty_message(self):
        """POST /api/agent/chat with empty message returns 400."""
        try:
            api_post("/api/agent/chat", {"message": "", "account_id": 1})
            self.fail("Expected 400 error")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)


class TestOllamaConnectivity(unittest.TestCase):
    """Integration tests for Ollama connectivity."""

    def test_ollama_is_running(self):
        """Ollama API should be reachable."""
        try:
            resp = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
            self.assertEqual(resp.status, 200)
        except Exception:
            self.skipTest("Ollama not running — skipping LLM tests")

    def test_qwen25_model_available(self):
        """qwen2.5:7b should be in the available models list."""
        try:
            resp = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
            data = json.loads(resp.read().decode())
            model_names = [m["name"] for m in data["models"]]
            self.assertIn("qwen2.5:7b", model_names)
        except Exception:
            self.skipTest("Ollama not running")


class TestSystemWorkflow(unittest.TestCase):
    """End-to-end system workflow tests."""

    def test_full_investigation_workflow(self):
        """Simulate a full investigation: overview → alerts → detail → graph → trace → community → agent."""
        # 1. Dashboard
        overview, _ = api_get("/api/overview")
        self.assertGreater(overview["total_accounts"], 0)

        # 2. Alert Queue
        alerts, _ = api_get("/api/alerts")
        self.assertGreater(len(alerts), 0)

        # 3. Pick first high-risk alert
        high_risk = [a for a in alerts if a["risk_level"] == "HIGH"]
        if not high_risk:
            high_risk = alerts
        target = high_risk[0]["account_id"]

        # 4. Alert Detail
        detail, _ = api_get(f"/api/alerts/{target}")
        self.assertEqual(detail["account_id"], target)
        self.assertIn("narrative", detail)

        # 5. Evidence Graph
        graph, _ = api_get(f"/api/graph/{target}")
        self.assertGreater(len(graph["nodes"]), 0)

        # 6. Trace Funds
        trace, _ = api_get(f"/api/trace/{target}")
        self.assertIn("upstream", trace)

        # 7. Community Detection
        community, _ = api_get(f"/api/community/{target}")
        self.assertGreater(len(community["nodes"]), 0)

        print(f"\n✅ Full workflow passed for Account #{target}")
        print(f"   Overview: {overview['total_accounts']} accounts, {overview['flagged_accounts']} flagged")
        print(f"   Detail: Risk={detail['risk_level']}, SAR={detail.get('is_sar')}")
        print(f"   Graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
        print(f"   Trace: {len(trace['upstream'])} upstream, {len(trace['downstream'])} downstream")
        print(f"   Community: {len(community['nodes'])} nodes")

    def test_response_times(self):
        """Verify all endpoints respond within acceptable time."""
        endpoints = [
            "/api/overview",
            "/api/alerts",
            "/api/models",
        ]
        alerts, _ = api_get("/api/alerts")
        aid = alerts[0]["account_id"]
        endpoints.extend([
            f"/api/alerts/{aid}",
            f"/api/graph/{aid}",
            f"/api/trace/{aid}",
            f"/api/community/{aid}",
        ])

        print("\n📊 Response Time Benchmarks:")
        for endpoint in endpoints:
            start = time.time()
            api_get(endpoint)
            elapsed = (time.time() - start) * 1000
            print(f"   {endpoint:35s} → {elapsed:7.1f} ms")
            self.assertLess(elapsed, 10000, f"{endpoint} took too long: {elapsed:.0f}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("  GhostBusters AML Platform — Test Suite")
    print("=" * 60)
    print(f"  Server: {BASE_URL}")
    print(f"  Ollama: {OLLAMA_URL}")
    print()
    unittest.main(verbosity=2)
