/* ============================================================
   GhostBusters AML Platform — Main Application Controller
   Handles: SPA routing, data fetching, D3 graphs, AI agent,
            Ollama chat, particle canvas, streaming
   ============================================================ */

// ── Global State ──────────────────────────────────────────────
let chartInstances = {};
let alertData = [];
let currentAccountId = null;
let originalGraphData = null;
let agentData = null;

// EventSource refs — track & cleanup
let liveFeedSource = null;
let streamSource = null;
let streamActive = false;
let streamStats = { total: 0, flagged: 0, totalScore: 0, startTime: null };
let histBuckets = new Array(10).fill(0);

// Guard flags for idempotent init
let alertListenersAttached = false;

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initParticles();           // Landing page particle background
    animateLandingCounters();  // KPI count-up animation

    // Attach sidebar nav clicks once
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            navigateTo(page);
        });
    });
});

// ── Landing Page ──────────────────────────────────────────────
function enterDashboard() {
    const landing = document.getElementById('page-landing');
    const shell = document.getElementById('app-shell');
    if (landing) landing.style.display = 'none';
    if (shell) shell.style.display = 'flex';
    navigateTo('dashboard');
}

// ── Particle Canvas ───────────────────────────────────────────
function initParticles() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let particles = [];
    const count = 80;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            r: Math.random() * 2 + 0.5,
            alpha: Math.random() * 0.5 + 0.1,
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(139, 92, 246, ${0.08 * (1 - dist / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        // Draw particles
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(139, 92, 246, ${p.alpha})`;
            ctx.fill();
        });

        requestAnimationFrame(draw);
    }
    draw();
}

function animateLandingCounters() {
    document.querySelectorAll('.landing-stat .stat-number').forEach(el => {
        const target = parseInt(el.dataset.target);
        if (isNaN(target)) return;
        const suffix = el.dataset.suffix || '';
        const duration = 2000;
        const start = performance.now();

        function update(now) {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            el.textContent = Math.floor(target * eased).toLocaleString() + suffix;
            if (progress < 1) requestAnimationFrame(update);
        }
        // Delay until visible
        setTimeout(() => requestAnimationFrame(update), 800);
    });
}

// ── SPA Navigation ────────────────────────────────────────────
function navigateTo(page) {
    // Cleanup current page resources
    cleanupCurrentPage();

    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.style.display = 'none');

    // Deactivate all nav items
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    // Show the selected page
    const pageEl = document.getElementById(`page-${page}`);
    if (pageEl) {
        pageEl.style.display = 'block';
        // Re-trigger page animation
        pageEl.style.animation = 'none';
        pageEl.offsetHeight; // force reflow
        pageEl.style.animation = '';
    }

    // Activate nav item
    const navItem = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (navItem) navItem.classList.add('active');

    // Load data for the page
    switch (page) {
        case 'dashboard': initDashboard(); break;
        case 'alerts': initAlerts(); break;
        case 'investigate':
            if (currentAccountId) {
                loadInvestigation(currentAccountId);
            } else {
                const inv = document.getElementById('page-investigate');
                if (inv && !inv.querySelector('.profile-header')) {
                    inv.innerHTML = '<div class="page-header"><h2>Investigation</h2><p>Select an account from the Alert Queue to begin investigation</p></div><div style="text-align:center; padding:80px 20px; color:var(--text-muted);"><span style="font-size:4rem; display:block; margin-bottom:20px;">🔍</span>No account selected. Go to <a href="#" onclick="navigateTo(\'alerts\'); return false;" style="color:var(--accent-violet);">Alert Queue</a> to pick an account.</div>';
                }
            }
            break;
        case 'models': initModels(); break;
        case 'streaming': initStreaming(); break;
    }
}

function cleanupCurrentPage() {
    // Close live feed EventSource when leaving Command Center
    if (liveFeedSource) {
        liveFeedSource.close();
        liveFeedSource = null;
    }

    // Close streaming EventSource when leaving Streaming page
    if (streamSource) {
        streamSource.close();
        streamSource = null;
        streamActive = false;
    }
}

// ── Page 1: Command Center ────────────────────────────────────
async function initDashboard() {
    try {
        const res = await fetch('/api/overview');
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        const data = await res.json();

        const kpiGrid = document.getElementById('kpi-grid');
        if (!kpiGrid) return;

        kpiGrid.innerHTML = `
            <div class="kpi-card violet">
                <div class="kpi-icon">📡</div>
                <div class="kpi-value">${(data.total_accounts || 0).toLocaleString()}</div>
                <div class="kpi-label">Monitored Accounts</div>
            </div>
            <div class="kpi-card rose">
                <div class="kpi-icon">🚨</div>
                <div class="kpi-value">${data.flagged_accounts || 0}</div>
                <div class="kpi-label">Flagged Accounts</div>
            </div>
            <div class="kpi-card cyan">
                <div class="kpi-icon">🏆</div>
                <div class="kpi-value">${data.best_model?.pr_auc?.toFixed(4) || '—'}</div>
                <div class="kpi-label">Best PR-AUC (${data.best_model?.name || '—'})</div>
            </div>
            <div class="kpi-card emerald">
                <div class="kpi-icon">🔗</div>
                <div class="kpi-value">${(data.total_edges || 0).toLocaleString()}</div>
                <div class="kpi-label">Transaction Edges</div>
            </div>
            <div class="kpi-card amber">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-value">${data.sar_rate || 0}%</div>
                <div class="kpi-label">SAR Rate</div>
            </div>
        `;

        // Animate KPI values
        kpiGrid.querySelectorAll('.kpi-value').forEach((el, idx) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(10px)';
            setTimeout(() => {
                el.style.transition = 'all 0.5s cubic-bezier(0.16, 1, 0.3, 1)';
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, 100 + idx * 80);
        });

        // Risk distribution donut
        const rd = data.risk_distribution;
        if (rd) {
            renderChart('chart-risk', 'doughnut', {
                labels: ['HIGH', 'MEDIUM', 'LOW'],
                datasets: [{
                    data: [rd.HIGH || 0, rd.MEDIUM || 0, rd.LOW || 0],
                    backgroundColor: ['#f43f5e', '#f59e0b', '#10b981'],
                    borderColor: 'transparent',
                    borderWidth: 0,
                }]
            }, {
                cutout: '70%',
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 16, font: { size: 11 } } }
                }
            });
        }

        // Channel chart
        const edgeStats = data.edge_stats || {};
        const edgeKeys = Object.keys(edgeStats);
        if (edgeKeys.length > 0) {
            const barColors = ['rgba(139,92,246,0.6)', 'rgba(6,182,212,0.6)', 'rgba(16,185,129,0.6)', 'rgba(245,158,11,0.6)', 'rgba(244,63,94,0.6)'];
            const borderColors = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e'];
            renderChart('chart-channels', 'bar', {
                labels: edgeKeys,
                datasets: [{
                    label: 'Transaction Count',
                    data: Object.values(edgeStats),
                    backgroundColor: edgeKeys.map((_, i) => barColors[i % barColors.length]),
                    borderColor: edgeKeys.map((_, i) => borderColors[i % borderColors.length]),
                    borderWidth: 1,
                    borderRadius: 6,
                }]
            }, {
                scales: {
                    x: { ticks: { color: '#64748b' }, grid: { display: false } },
                    y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.03)' } }
                },
                plugins: { legend: { display: false } }
            });
        }

        // Start SSE live feed (close any existing first)
        if (liveFeedSource) liveFeedSource.close();
        startLiveFeed();

        // Update alert badge
        try {
            const alertRes = await fetch('/api/alerts');
            if (alertRes.ok) {
                const alerts = await alertRes.json();
                const badge = document.getElementById('alert-badge');
                if (badge) badge.textContent = alerts.length;
            }
        } catch (_) { /* non-critical */ }

    } catch (e) {
        console.error('Dashboard error:', e);
        const kpiGrid = document.getElementById('kpi-grid');
        if (kpiGrid) kpiGrid.innerHTML = `<div style="padding:20px; color:var(--accent-rose);">Error loading dashboard: ${e.message}</div>`;
    }
}

function startLiveFeed() {
    const feed = document.getElementById('live-feed');
    const statusEl = document.getElementById('feed-status');
    if (!feed) return;

    liveFeedSource = new EventSource('/api/stream');

    liveFeedSource.onopen = () => {
        if (statusEl) {
            statusEl.textContent = 'Live';
            statusEl.className = 'badge badge-low';
        }
    };

    liveFeedSource.onmessage = (e) => {
        try {
            const txn = JSON.parse(e.data);
            const rClass = txn.risk === 'HIGH_RISK' ? 'risk-high' : txn.risk === 'SUSPICIOUS' ? 'risk-suspicious' : 'risk-normal';
            const scoreColor = txn.risk === 'HIGH_RISK' ? 'var(--accent-rose)' : txn.risk === 'SUSPICIOUS' ? 'var(--accent-amber)' : 'var(--accent-emerald)';

            const item = document.createElement('div');
            item.className = `feed-item ${rClass}`;
            item.innerHTML = `
                <span class="feed-time">${(txn.timestamp || '').split(' ')[1] || '—'}</span>
                <span class="feed-detail"><strong>${txn.src || '?'}</strong> → <strong>${txn.dst || '?'}</strong> · $${(txn.amount || 0).toLocaleString()}</span>
                <span class="feed-score" style="color:${scoreColor}">${(txn.score || 0).toFixed(4)}</span>
            `;

            feed.insertBefore(item, feed.firstChild);
            while (feed.children.length > 50) feed.removeChild(feed.lastChild);
        } catch (parseErr) {
            console.warn('Feed parse error:', parseErr);
        }
    };

    liveFeedSource.onerror = () => {
        if (statusEl) {
            statusEl.textContent = 'Reconnecting...';
            statusEl.className = 'badge badge-medium';
        }
    };
}


// ── Page 2: Alert Queue ───────────────────────────────────────
async function initAlerts() {
    try {
        const res = await fetch('/api/alerts');
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        alertData = await res.json();
        renderAlertTable(alertData);

        // Attach listeners only once
        if (!alertListenersAttached) {
            const searchEl = document.getElementById('alert-search');
            const filterEl = document.getElementById('alert-filter');
            if (searchEl) searchEl.addEventListener('input', filterAlerts);
            if (filterEl) filterEl.addEventListener('change', filterAlerts);
            alertListenersAttached = true;
        }
    } catch (e) {
        console.error('Alerts error:', e);
        const tbody = document.getElementById('alert-tbody');
        if (tbody) tbody.innerHTML = `<tr><td colspan="6" style="color:var(--accent-rose); padding:20px;">Error loading alerts: ${e.message}</td></tr>`;
    }
}

function filterAlerts() {
    const searchEl = document.getElementById('alert-search');
    const filterEl = document.getElementById('alert-filter');
    const q = (searchEl ? searchEl.value : '').toLowerCase();
    const risk = filterEl ? filterEl.value : 'all';

    const filtered = alertData.filter(a => {
        const matchSearch = String(a.account_id).includes(q) || (a.risk_level || '').toLowerCase().includes(q);
        const matchRisk = risk === 'all' || a.risk_level === risk;
        return matchSearch && matchRisk;
    });
    renderAlertTable(filtered);
}

function renderAlertTable(data) {
    const tbody = document.getElementById('alert-tbody');
    if (!tbody) return;

    tbody.innerHTML = data.map(a => {
        const rc0 = (a.reason_codes && a.reason_codes[0]) || {};
        const topCode = rc0.code || rc0.reason || '—';
        const confidence = typeof rc0.confidence === 'number' ? rc0.confidence.toFixed(2) : '—';
        const riskLevel = a.risk_level || 'MEDIUM';

        return `
            <tr onclick="navigateToInvestigate(${a.account_id})" style="cursor:pointer;">
                <td style="font-family:'JetBrains Mono',monospace; font-weight:600;">#${a.account_id}</td>
                <td><span class="badge badge-${riskLevel.toLowerCase()}">${riskLevel}</span></td>
                <td>${a.is_sar ? '<span style="color:var(--accent-rose);">⚠️ SAR</span>' : '<span style="color:var(--accent-emerald);">✅ Normal</span>'}</td>
                <td style="font-size:0.78rem; color:var(--text-secondary);">${topCode}</td>
                <td style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:var(--accent-cyan);">${confidence}</td>
                <td><button class="btn btn-outline" style="padding:4px 14px; font-size:0.72rem;">Investigate →</button></td>
            </tr>
        `;
    }).join('');
}


// ── Page 3: Investigation ─────────────────────────────────────
function navigateToInvestigate(accountId) {
    currentAccountId = accountId;
    navigateTo('investigate');
    loadInvestigation(accountId);
}

async function loadInvestigation(accountId) {
    const container = document.getElementById('page-investigate');
    if (!container) return;
    container.innerHTML = '<div style="text-align:center; padding:60px; color:var(--text-muted);">Loading investigation...</div>';

    try {
        const [detailRes, graphRes] = await Promise.all([
            fetch(`/api/alerts/${accountId}`),
            fetch(`/api/graph/${accountId}`),
        ]);

        if (!detailRes.ok) throw new Error(`Alert detail API returned ${detailRes.status}`);
        if (!graphRes.ok) throw new Error(`Graph API returned ${graphRes.status}`);

        const detail = await detailRes.json();
        const graph = await graphRes.json();
        originalGraphData = graph;

        if (detail.error) {
            container.innerHTML = `<div style="padding:40px; color:var(--accent-rose);">Account #${accountId} not found.</div>`;
            return;
        }

        // Build reason cards
        const reasons = detail.reason_codes || [];
        const reasonsHtml = reasons.map(r => {
            const conf = typeof r.confidence === 'number' ? r.confidence : 0;
            const severity = conf > 0.7 ? 'high' : conf > 0.4 ? 'medium' : 'low';
            const icon = severity === 'high' ? '🔴' : severity === 'medium' ? '🟡' : '🟢';
            return `
                <div class="reason-card ${severity}">
                    <div class="reason-icon">${icon}</div>
                    <div class="reason-body">
                        <h4>${r.code || r.reason || 'Unknown'}</h4>
                        <p>${r.description || ''}</p>
                        <div class="reason-value">Confidence: ${(conf * 100).toFixed(0)}%</div>
                    </div>
                </div>
            `;
        }).join('');

        // Build feature profile
        const features = detail.feature_profile || {};
        const featureHtml = Object.entries(features).map(([k, v]) => `
            <div class="feature-item">
                <div class="feature-label">${k.replace(/_/g, ' ')}</div>
                <div class="feature-value">${typeof v === 'number' ? v.toFixed(4) : String(v)}</div>
            </div>
        `).join('');

        const riskLevel = detail.risk_level || 'MEDIUM';
        container.innerHTML = `
            <div class="page-header">
                <h2>Investigation</h2>
                <p>Detailed forensic analysis for Account #${accountId}</p>
            </div>

            <!-- Profile Header -->
            <div class="profile-header">
                <div class="profile-avatar sar">👻</div>
                <div class="profile-info">
                    <h3>Account #${detail.account_id}</h3>
                    <div class="profile-meta">
                        <span class="badge badge-${riskLevel.toLowerCase()}">${riskLevel} Risk</span>
                        <span>${detail.is_sar ? '⚠️ Confirmed SAR' : '✅ Normal'}</span>
                    </div>
                </div>
                <div style="margin-left: auto; display:flex; gap:8px;">
                    <button class="btn btn-primary" onclick="runAgentInvestigation()">🤖 AI Investigate</button>
                </div>
            </div>

            <div class="grid-1-2">
                <!-- Reason Codes -->
                <div class="card">
                    <div class="card-header">
                        <h3>🚩 Risk Indicators</h3>
                    </div>
                    <div class="reason-list">${reasonsHtml || '<div style="color:var(--text-muted); padding:12px;">No risk indicators found.</div>'}</div>
                </div>

                <!-- Evidence Graph -->
                <div class="card">
                    <div class="card-header">
                        <h3>🔗 Evidence Subgraph</h3>
                        <div style="display:flex; gap:8px;">
                            <button class="btn btn-outline" style="padding:4px 10px; font-size:0.7rem;" onclick="traceFunds('upstream')">⬅ Source</button>
                            <button class="btn btn-outline" style="padding:4px 10px; font-size:0.7rem;" onclick="traceFunds('downstream')">Dest ➡</button>
                            <button class="btn btn-outline" style="padding:4px 10px; font-size:0.7rem;" onclick="detectCommunity()">🕸 Ring</button>
                            <button class="btn btn-outline" style="padding:4px 10px; font-size:0.7rem;" onclick="resetGraph()">↺</button>
                        </div>
                    </div>
                    <div class="graph-container" id="evidence-graph-container"></div>
                </div>
            </div>

            <div class="grid-2">
                <!-- Features -->
                <div class="card">
                    <div class="card-header">
                        <h3>📊 Feature Profile</h3>
                    </div>
                    <div class="feature-grid">${featureHtml || '<div style="color:var(--text-muted);">No features available.</div>'}</div>
                </div>

                <!-- Narrative -->
                <div class="card">
                    <div class="card-header">
                        <h3>📝 Regulator-Ready Narrative</h3>
                        <button class="btn btn-outline" style="padding: 6px 14px; font-size: 0.72rem;" onclick="printNarrative()">🖨 Print Report</button>
                    </div>
                    <div class="narrative-box">${detail.narrative || 'No narrative available. Click "AI Investigate" to generate one.'}</div>
                </div>
            </div>

            <!-- Agent Console Overlay -->
            <div id="agent-console-overlay" class="agent-overlay" style="display:none;">
                <div class="agent-console">
                    <div class="console-header">
                        <span>🤖 GhostBusters AI Agent</span>
                        <button class="btn-close" onclick="closeAgentConsole()">×</button>
                    </div>
                    <div class="console-body" id="agent-terminal"></div>
                    <div class="chat-panel" id="agent-chat-panel" style="display:none;">
                        <div class="chat-messages" id="chat-messages"></div>
                        <div class="chat-input-row">
                            <input class="chat-input" id="chat-input" type="text" placeholder="Ask a follow-up question..." onkeydown="if(event.key==='Enter')sendChat()">
                            <button class="btn btn-primary" style="padding:8px 16px;" onclick="sendChat()">Send</button>
                        </div>
                    </div>
                    <div class="console-actions" id="agent-actions" style="display:none;">
                        <button class="btn btn-outline" onclick="openChatPanel()">💬 Chat with Agent</button>
                        <button class="btn btn-primary" onclick="printNarrative()">📋 Export SAR</button>
                        <button class="btn btn-outline" onclick="closeAgentConsole()">Dismiss</button>
                    </div>
                </div>
            </div>
        `;

        // Render D3 graph after DOM is in place
        setTimeout(() => {
            if (graph.nodes && graph.nodes.length > 0) {
                renderEvidenceGraph(graph);
            } else {
                const gc = document.getElementById('evidence-graph-container');
                if (gc) gc.innerHTML = '<div style="display:flex; align-items:center; justify-content:center; height:100%; color:var(--text-muted);">No graph data available</div>';
            }
        }, 100);

    } catch (e) {
        console.error('Investigation error:', e);
        container.innerHTML = `<div style="padding:40px; color:var(--accent-rose);">Error loading investigation: ${e.message}</div>`;
    }
}


// ── D3 Evidence Graph ─────────────────────────────────────────
function renderEvidenceGraph(graphData, highlightNodes = null, traceLinks = null) {
    const container = document.getElementById('evidence-graph-container');
    if (!container) return;
    container.innerHTML = '';

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 450;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Glow filter
    const defs = svg.append('defs');
    const filter = defs.append('filter').attr('id', 'glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    const nodes = graphData.nodes.map(n => ({ ...n }));
    const edges = graphData.edges.map(e => ({
        source: e.src,
        target: e.dst,
        is_trace: traceLinks ? traceLinks.some(t => t.src === e.src && t.dst === e.dst) : false,
    }));

    // Filter edges to only include existing node IDs
    const nodeIds = new Set(nodes.map(n => n.id));
    const validEdges = edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));

    const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(validEdges).id(d => d.id).distance(50))
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide(15));

    const link = svg.append('g')
        .selectAll('line')
        .data(validEdges)
        .join('line')
        .attr('stroke', d => d.is_trace ? '#f59e0b' : 'rgba(139, 92, 246, 0.15)')
        .attr('stroke-width', d => d.is_trace ? 2.5 : 1)
        .attr('stroke-dasharray', d => d.is_trace ? '6,3' : 'none');

    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', d => d.is_target ? 14 : (highlightNodes && highlightNodes.has(d.id) ? 9 : 6))
        .attr('fill', d => {
            if (d.is_target) return '#f43f5e';
            if (highlightNodes && highlightNodes.has(d.id)) return '#8b5cf6';
            return '#06b6d4';
        })
        .attr('stroke', d => d.is_target ? '#f43f5e' : 'transparent')
        .attr('stroke-width', d => d.is_target ? 3 : 0)
        .attr('opacity', d => {
            if (!highlightNodes) return 0.8;
            return highlightNodes.has(d.id) || d.is_target ? 1 : 0.15;
        })
        .style('filter', d => d.is_target ? 'url(#glow)' : 'none')
        .style('cursor', 'pointer')
        .call(d3.drag()
            .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
        );

    // Labels for important nodes
    const label = svg.append('g')
        .selectAll('text')
        .data(nodes.filter(n => n.is_target || (highlightNodes && highlightNodes.has(n.id))))
        .join('text')
        .text(d => `#${d.id}`)
        .attr('font-size', '10px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .attr('fill', '#94a3b8')
        .attr('dx', 14)
        .attr('dy', 4);

    sim.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('cx', d => d.x).attr('cy', d => d.y);
        label.attr('x', d => d.x).attr('y', d => d.y);
    });

    // Tooltip
    node.append('title').text(d => `Account #${d.id}${d.is_target ? ' (Target)' : ''}`);
}


// ── Graph Forensics ───────────────────────────────────────────
async function traceFunds(direction) {
    if (!currentAccountId) return;

    try {
        const res = await fetch(`/api/trace/${currentAccountId}`);
        if (!res.ok) throw new Error(`Trace API error: ${res.status}`);
        const data = await res.json();
        const paths = direction === 'upstream' ? data.upstream : data.downstream;

        if (!paths || paths.length === 0) {
            alert('No paths found in this direction.');
            return;
        }

        const highlightNodes = new Set();
        const traceLinks = [];

        paths.forEach(path => {
            path.forEach(n => highlightNodes.add(n));
            for (let i = 0; i < path.length - 1; i++) {
                traceLinks.push({ src: path[i], dst: path[i + 1] });
            }
        });

        if (!originalGraphData) return;
        const existingIds = new Set(originalGraphData.nodes.map(n => n.id));
        const mergedNodes = [...originalGraphData.nodes];
        highlightNodes.forEach(nid => {
            if (!existingIds.has(nid)) {
                mergedNodes.push({ id: nid, is_target: nid === currentAccountId, is_sar: false });
            }
        });

        const mergedEdges = [...originalGraphData.edges];
        traceLinks.forEach(t => {
            if (!mergedEdges.some(e => e.src === t.src && e.dst === t.dst)) {
                mergedEdges.push(t);
            }
        });

        renderEvidenceGraph({ nodes: mergedNodes, edges: mergedEdges }, highlightNodes, traceLinks);
    } catch (e) {
        console.error('Trace error:', e);
    }
}

async function detectCommunity() {
    if (!currentAccountId) return;

    try {
        const res = await fetch(`/api/community/${currentAccountId}`);
        if (!res.ok) throw new Error(`Community API error: ${res.status}`);
        const data = await res.json();

        const highlightNodes = new Set(data.nodes);
        const communityNodes = data.nodes.map(n => ({
            id: n,
            is_target: n === currentAccountId,
            is_sar: n === currentAccountId,
        }));

        renderEvidenceGraph({ nodes: communityNodes, edges: data.edges }, highlightNodes);
    } catch (e) {
        console.error('Community error:', e);
    }
}

function resetGraph() {
    if (originalGraphData) {
        renderEvidenceGraph(originalGraphData);
    }
}


// ── Agentic AI Logic (Ollama-Powered SSE) ─────────────────────
async function runAgentInvestigation() {
    if (!currentAccountId) return;

    const overlay = document.getElementById('agent-console-overlay');
    const terminal = document.getElementById('agent-terminal');
    const actions = document.getElementById('agent-actions');
    const chatPanel = document.getElementById('agent-chat-panel');

    if (!overlay || !terminal || !actions) return;

    overlay.style.display = 'flex';
    terminal.innerHTML = '<div class="console-line" style="color:var(--accent-violet);">🤖 Initializing GhostBusters AI Agent...</div>';
    actions.style.display = 'none';
    if (chatPanel) chatPanel.style.display = 'none';

    try {
        const res = await fetch(`/api/agent/investigate/${currentAccountId}`);

        // Check if SSE stream or JSON
        const contentType = res.headers.get('content-type') || '';

        if (contentType.includes('text/event-stream')) {
            // SSE streaming from Ollama
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let sarText = '';

            terminal.innerHTML = '';
            let currentTokenLine = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Process complete SSE lines
                while (buffer.includes('\n')) {
                    const newlineIdx = buffer.indexOf('\n');
                    const line = buffer.substring(0, newlineIdx).trim();
                    buffer = buffer.substring(newlineIdx + 1);

                    if (line.startsWith('data: ')) {
                        const payload = line.slice(6);
                        try {
                            const evt = JSON.parse(payload);

                            if (evt.type === 'step') {
                                // New reasoning step
                                currentTokenLine = null; // reset token line
                                const stepLine = document.createElement('div');
                                stepLine.className = 'console-line';
                                stepLine.style.color = 'var(--accent-emerald)';
                                stepLine.textContent = '> ' + evt.text;
                                terminal.appendChild(stepLine);
                                terminal.scrollTop = terminal.scrollHeight;
                            } else if (evt.type === 'token') {
                                // LLM token streaming
                                if (!currentTokenLine) {
                                    currentTokenLine = document.createElement('div');
                                    currentTokenLine.className = 'console-line';
                                    currentTokenLine.style.color = 'var(--text-secondary)';
                                    terminal.appendChild(currentTokenLine);
                                }
                                currentTokenLine.textContent += evt.text;
                                sarText += evt.text;
                                terminal.scrollTop = terminal.scrollHeight;

                                // Handle newlines in token text
                                if (evt.text.includes('\n')) {
                                    currentTokenLine = null;
                                }
                            } else if (evt.type === 'result') {
                                agentData = evt;
                                const narrativeBox = document.querySelector('.narrative-box');
                                if (narrativeBox) narrativeBox.textContent = evt.sar_text || sarText || '';
                            } else if (evt.type === 'done') {
                                actions.style.display = 'flex';

                                // Add completion message
                                const doneMsg = document.createElement('div');
                                doneMsg.className = 'console-system';
                                doneMsg.textContent = '✅ ANALYSIS COMPLETE';
                                terminal.appendChild(doneMsg);
                                terminal.scrollTop = terminal.scrollHeight;
                            }
                        } catch (parseErr) {
                            // Ignore JSON parse errors on incomplete chunks
                        }
                    }
                }
            }
            // Ensure actions are shown even if no explicit 'done' event
            actions.style.display = 'flex';
        } else {
            // Fallback: Regular JSON response (rule-based)
            agentData = await res.json();
            terminal.innerHTML = '';

            if (agentData.steps && agentData.steps.length > 0) {
                for (const step of agentData.steps) {
                    await typeLine(terminal, '> ' + step);
                    await sleep(300);
                }
            }

            await typeLine(terminal, '', 'console-spacer');
            await typeLine(terminal, '✅ ANALYSIS COMPLETE', 'console-system');

            const narrativeBox = document.querySelector('.narrative-box');
            if (narrativeBox && agentData.result) {
                narrativeBox.textContent = agentData.result.sar_text || '';
            }
            actions.style.display = 'flex';
        }

    } catch (e) {
        console.error('Agent error:', e);
        terminal.innerHTML += `<div class="console-error">Error: ${e.message}</div>`;
        actions.style.display = 'flex';
    }
}

function typeLine(container, text, className = 'console-line') {
    return new Promise(resolve => {
        const line = document.createElement('div');
        line.className = className;
        container.appendChild(line);
        container.scrollTop = container.scrollHeight;

        let i = 0;
        const speed = 18;

        function type() {
            if (i < text.length) {
                line.textContent += text.charAt(i);
                i++;
                container.scrollTop = container.scrollHeight;
                setTimeout(type, speed);
            } else {
                resolve();
            }
        }
        type();
    });
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function closeAgentConsole() {
    const overlay = document.getElementById('agent-console-overlay');
    if (overlay) overlay.style.display = 'none';
}

function openChatPanel() {
    const chatPanel = document.getElementById('agent-chat-panel');
    const chatInput = document.getElementById('chat-input');
    const agentConsole = document.querySelector('.agent-console');

    if (chatPanel) {
        chatPanel.style.display = 'block';
        // Scroll the agent console to make sure chat is visible
        if (agentConsole) agentConsole.scrollTop = agentConsole.scrollHeight;
        // Focus the input
        if (chatInput) setTimeout(() => chatInput.focus(), 100);
    }
}


// ── Chat with Agent (Ollama) ──────────────────────────────────
async function sendChat() {
    const input = document.getElementById('chat-input');
    const messages = document.getElementById('chat-messages');
    if (!input || !messages) return;

    const text = input.value.trim();
    if (!text) return;

    // Show user message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-msg user';
    userMsg.textContent = text;
    messages.appendChild(userMsg);
    input.value = '';
    messages.scrollTop = messages.scrollHeight;

    // Show typing indicator
    const aiMsg = document.createElement('div');
    aiMsg.className = 'chat-msg ai';
    aiMsg.textContent = '⏳ Thinking...';
    messages.appendChild(aiMsg);
    messages.scrollTop = messages.scrollHeight;

    try {
        const res = await fetch('/api/agent/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                account_id: currentAccountId,
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.response || `API error: ${res.status}`);
        }

        const data = await res.json();
        aiMsg.textContent = data.response || 'No response from agent.';
    } catch (e) {
        aiMsg.textContent = `Error: ${e.message}`;
        aiMsg.style.color = 'var(--accent-rose)';
    }
    messages.scrollTop = messages.scrollHeight;
}


// ── Page 4: Model Analytics ───────────────────────────────────
async function initModels() {
    try {
        const res = await fetch('/api/models');
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        const data = await res.json();

        // PR-AUC bars
        const praucBars = document.getElementById('prauc-bars');
        if (!praucBars) return;

        const allModels = [];

        for (const [name, metrics] of Object.entries(data.gnn || {})) {
            allModels.push({ name, pr_auc: metrics.pr_auc || 0, type: 'gnn' });
        }
        for (const [name, metrics] of Object.entries(data.baseline || {})) {
            allModels.push({ name, pr_auc: metrics.pr_auc || 0, type: 'baseline' });
        }

        allModels.sort((a, b) => b.pr_auc - a.pr_auc);

        praucBars.innerHTML = allModels.map(m => `
            <div class="prauc-bar-row ${m.type}">
                <div class="model-name">${m.name}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width: 0%;" data-target="${m.pr_auc * 100}">
                        ${m.pr_auc.toFixed(4)}
                    </div>
                </div>
            </div>
        `).join('');

        // Animate bars
        setTimeout(() => {
            praucBars.querySelectorAll('.bar-fill').forEach(bar => {
                const target = parseFloat(bar.dataset.target) || 0;
                bar.style.width = target + '%';
            });
        }, 100);

        // Training histories
        const histories = data.training_histories || {};
        const historyKeys = Object.keys(histories);
        if (historyKeys.length > 0) {
            const colors = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e'];

            // PR-AUC history
            const praucDatasets = historyKeys.map((name, i) => ({
                label: name,
                data: histories[name].val_pr_auc || [],
                borderColor: colors[i % colors.length],
                backgroundColor: 'transparent',
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2,
            }));

            const sampleData = praucDatasets[0]?.data || [];
            const epochs = sampleData.map((_, i) => i + 1);

            if (epochs.length > 0) {
                renderChart('chart-prauc-history', 'line', {
                    labels: epochs,
                    datasets: praucDatasets,
                }, {
                    scales: {
                        x: { title: { display: true, text: 'Epoch', color: '#64748b' }, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.03)' } },
                        y: { title: { display: true, text: 'Val PR-AUC', color: '#64748b' }, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.03)' } }
                    },
                    plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 12, font: { size: 10 } } } }
                });

                // Loss history
                const lossDatasets = historyKeys.map((name, i) => ({
                    label: name,
                    data: histories[name].train_loss || [],
                    borderColor: colors[i % colors.length],
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                }));

                renderChart('chart-loss-history', 'line', {
                    labels: epochs,
                    datasets: lossDatasets,
                }, {
                    scales: {
                        x: { title: { display: true, text: 'Epoch', color: '#64748b' }, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.03)' } },
                        y: { title: { display: true, text: 'Train Loss', color: '#64748b' }, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.03)' } }
                    },
                    plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 12, font: { size: 10 } } } }
                });
            }
        }

        // Metrics table
        const metricsTbody = document.getElementById('metrics-tbody');
        if (metricsTbody) {
            metricsTbody.innerHTML = allModels.map(m => {
                const src = m.type === 'gnn' ? (data.gnn[m.name] || {}) : (data.baseline[m.name] || {});
                const prAuc = typeof src.pr_auc === 'number' ? src.pr_auc.toFixed(4) : '—';
                const rocAuc = typeof src.roc_auc === 'number' ? src.roc_auc.toFixed(4) : '—';

                // Safely extract f1 and recall
                let f1 = '—', recall = '—';
                const cr = src.classification_report;
                if (cr && cr['1']) {
                    f1 = typeof cr['1']['f1-score'] === 'number' ? cr['1']['f1-score'].toFixed(4) : '—';
                    recall = typeof cr['1']['recall'] === 'number' ? cr['1']['recall'].toFixed(4) : '—';
                } else {
                    if (typeof src.f1_sar === 'number') f1 = src.f1_sar.toFixed(4);
                    if (typeof src.recall_sar === 'number') recall = src.recall_sar.toFixed(4);
                }

                return `
                    <tr>
                        <td style="font-family:'JetBrains Mono',monospace; font-weight:600; color:var(--text-primary);">${m.name}</td>
                        <td><span class="badge ${m.type === 'gnn' ? 'badge-purple' : 'badge-info'}">${m.type.toUpperCase()}</span></td>
                        <td style="font-family:'JetBrains Mono',monospace; color:var(--accent-violet);">${prAuc}</td>
                        <td style="font-family:'JetBrains Mono',monospace;">${rocAuc}</td>
                        <td style="font-family:'JetBrains Mono',monospace;">${f1}</td>
                        <td style="font-family:'JetBrains Mono',monospace;">${recall}</td>
                    </tr>
                `;
            }).join('');
        }

        // Config display
        const configDisplay = document.getElementById('config-display');
        if (configDisplay) {
            const gpuConfig = data.gpu_config || {};
            const trainConfig = data.training_config || {};
            const allConfig = { ...gpuConfig, ...trainConfig };
            const configEntries = Object.entries(allConfig);

            if (configEntries.length > 0) {
                configDisplay.innerHTML = `
                    <div class="feature-grid">
                        ${configEntries.map(([k, v]) => `
                            <div class="feature-item">
                                <div class="feature-label">${k}</div>
                                <div class="feature-value">${v}</div>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else {
                configDisplay.innerHTML = '<div style="color:var(--text-muted); padding:12px;">No config data available.</div>';
            }
        }

    } catch (e) {
        console.error('Models error:', e);
    }
}


// ── Page 5: Streaming Demo ────────────────────────────────────
function initStreaming() {
    // Reset UI state
    const btn = document.getElementById('stream-toggle');
    const status = document.getElementById('stream-status');
    if (btn && !streamActive) btn.innerHTML = '▶ Start Stream';
    if (status && !streamActive) {
        status.textContent = 'Idle';
        status.className = 'badge badge-info';
    }
}

function toggleStream() {
    const btn = document.getElementById('stream-toggle');
    const status = document.getElementById('stream-status');

    if (streamActive) {
        // Stop
        if (streamSource) {
            streamSource.close();
            streamSource = null;
        }
        streamActive = false;
        if (btn) btn.innerHTML = '▶ Start Stream';
        if (status) {
            status.textContent = 'Paused';
            status.className = 'badge badge-medium';
        }
    } else {
        // Start
        streamActive = true;
        streamStats = { total: 0, flagged: 0, totalScore: 0, startTime: Date.now() };
        histBuckets = new Array(10).fill(0);

        if (btn) btn.innerHTML = '⏹ Stop Stream';
        if (status) {
            status.textContent = 'Streaming';
            status.className = 'badge badge-low';
        }

        streamSource = new EventSource('/api/stream');
        streamSource.onmessage = (e) => {
            try {
                const txn = JSON.parse(e.data);
                streamStats.total++;
                streamStats.totalScore += txn.score || 0;
                if (txn.risk === 'HIGH_RISK' || txn.risk === 'SUSPICIOUS') streamStats.flagged++;

                // Update stats
                const stTotal = document.getElementById('st-total');
                const stFlagged = document.getElementById('st-flagged');
                const stAvg = document.getElementById('st-avg');
                const stSpeed = document.getElementById('st-speed');

                if (stTotal) stTotal.textContent = streamStats.total;
                if (stFlagged) stFlagged.textContent = streamStats.flagged;
                if (stAvg) stAvg.textContent = (streamStats.totalScore / streamStats.total).toFixed(4);
                const elapsed = (Date.now() - streamStats.startTime) / 1000;
                if (stSpeed) stSpeed.textContent = (streamStats.total / Math.max(elapsed, 1)).toFixed(1);

                // Update gauge
                const gaugeEl = document.getElementById('stream-gauge');
                const gaugeVal = document.getElementById('gauge-value');
                const gaugeColor = txn.risk === 'HIGH_RISK' ? 'var(--accent-rose)' : txn.risk === 'SUSPICIOUS' ? 'var(--accent-amber)' : 'var(--accent-emerald)';
                if (gaugeVal) {
                    gaugeVal.textContent = (txn.score || 0).toFixed(4);
                    gaugeVal.style.color = gaugeColor;
                }
                if (gaugeEl) {
                    gaugeEl.style.background = `conic-gradient(${gaugeColor} ${(txn.score || 0) * 360}deg, rgba(255,255,255,0.03) 0deg)`;
                }

                // Feed item
                const feed = document.getElementById('stream-feed');
                if (feed) {
                    const rClass = txn.risk === 'HIGH_RISK' ? 'risk-high' : txn.risk === 'SUSPICIOUS' ? 'risk-suspicious' : 'risk-normal';
                    const scoreColor = txn.risk === 'HIGH_RISK' ? 'var(--accent-rose)' : txn.risk === 'SUSPICIOUS' ? 'var(--accent-amber)' : 'var(--accent-emerald)';
                    const item = document.createElement('div');
                    item.className = `feed-item ${rClass}`;
                    item.innerHTML = `
                        <span class="feed-time">${(txn.timestamp || '').split(' ')[1] || '—'}</span>
                        <span class="feed-detail"><strong>${txn.src || '?'}</strong> → <strong>${txn.dst || '?'}</strong> · $${txn.amount || 0}</span>
                        <span class="feed-score" style="color:${scoreColor}">${(txn.score || 0).toFixed(4)}</span>
                    `;
                    feed.insertBefore(item, feed.firstChild);
                    while (feed.children.length > 30) feed.removeChild(feed.lastChild);
                }

                // Update histogram
                updateStreamHistogram(txn.score || 0);
            } catch (parseErr) {
                console.warn('Stream parse error:', parseErr);
            }
        };

        streamSource.onerror = () => {
            if (status) {
                status.textContent = 'Reconnecting...';
                status.className = 'badge badge-medium';
            }
        };
    }
}

function updateStreamHistogram(score) {
    const bucket = Math.min(Math.floor(score * 10), 9);
    histBuckets[bucket]++;

    renderChart('chart-stream-hist', 'bar', {
        labels: ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'],
        datasets: [{
            label: 'Count',
            data: [...histBuckets],
            backgroundColor: histBuckets.map((_, i) => {
                if (i >= 7) return 'rgba(244, 63, 94, 0.6)';
                if (i >= 5) return 'rgba(245, 158, 11, 0.6)';
                return 'rgba(16, 185, 129, 0.6)';
            }),
            borderRadius: 4,
        }]
    }, {
        animation: false,
        scales: {
            x: { ticks: { color: '#64748b', font: { size: 9 } }, grid: { display: false } },
            y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.03)' } }
        },
        plugins: { legend: { display: false } }
    });
}


// ── Chart Helper ──────────────────────────────────────────────
function renderChart(canvasId, type, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing chart
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
        delete chartInstances[canvasId];
    }

    try {
        chartInstances[canvasId] = new Chart(ctx, {
            type,
            data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                ...options,
            }
        });
    } catch (e) {
        console.error(`Chart error (${canvasId}):`, e);
    }
}


// ── Print Narrative ───────────────────────────────────────────
function printNarrative() {
    const narr = document.querySelector('.narrative-box');
    if (!narr) return;
    const content = narr.textContent || 'No narrative available.';
    const win = window.open('', '_blank');
    if (!win) {
        alert('Please allow popups to print the report.');
        return;
    }

    win.document.write(`
        <html><head><title>SAR Report — Account #${currentAccountId || 'Unknown'}</title>
        <style>body { font-family: 'Courier New', monospace; padding: 40px; max-width: 800px; margin: 0 auto; }
        h1 { font-size: 18px; border-bottom: 2px solid #333; padding-bottom: 10px; }
        pre { white-space: pre-wrap; line-height: 1.8; }</style></head>
        <body><h1>GhostBusters AML — SAR Report</h1><pre>${content}</pre></body></html>
    `);
    win.document.close();
    win.print();
}
