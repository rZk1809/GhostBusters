/* ============================================================
   GhostBusters AML Dashboard — Main SPA Controller
   ============================================================ */

// ── State ─────────────────────────────────────────────────────
let currentPage = 'dashboard';
let feedEventSource = null;
let streamEventSource = null;
let alertsData = [];
let streamStats = { total: 0, flagged: 0, scoreSum: 0, startTime: null };
let chartsReady = {};

// ── Navigation ────────────────────────────────────────────────
function navigateTo(page, params) {
    currentPage = page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const pageEl = document.getElementById('page-' + page);
    const navEl = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (pageEl) pageEl.classList.add('active');
    if (navEl) navEl.classList.add('active');

    // Close mobile sidebar
    document.querySelector('.sidebar').classList.remove('open');

    // Page-specific init
    if (page === 'dashboard') initDashboard();
    else if (page === 'alerts') initAlerts();
    else if (page === 'investigate') initInvestigate(params);
    else if (page === 'models') initModels();
    else if (page === 'streaming') initStreaming();
}

// ── Utility ───────────────────────────────────────────────────
function fmt(n) {
    if (n === undefined || n === null) return '—';
    if (typeof n === 'number') {
        if (Number.isInteger(n)) return n.toLocaleString();
        return n.toFixed(4);
    }
    return String(n);
}

function fmtPct(n) {
    return (n * 100).toFixed(2) + '%';
}

function fmtShort(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toLocaleString();
}

const REASON_ICONS = {
    'PRIOR_SAR': '📋', 'FAN_IN': '📥', 'FAN_OUT': '📤',
    'CHANNEL_HOPPING': '🔀', 'HIGH_VALUE': '💰', 'RAPID_CASHOUT': '🏧',
    'WALLET_SHARING': '👛', 'RAPID_PASS_THROUGH': '⚡',
    'UNUSUAL_ACTIVITY': '❓', 'STRUCTURING': '🧩',
    'DEVICE_SHARING': '📱', 'DEFAULT': '🔶'
};

function getReasonIcon(code) {
    return REASON_ICONS[code] || REASON_ICONS.DEFAULT;
}

// Chart color palette
const CHART_COLORS = {
    cyan: 'rgba(0, 212, 255, 0.8)',
    cyanDim: 'rgba(0, 212, 255, 0.15)',
    red: 'rgba(255, 71, 87, 0.8)',
    redDim: 'rgba(255, 71, 87, 0.15)',
    green: 'rgba(46, 213, 115, 0.8)',
    greenDim: 'rgba(46, 213, 115, 0.15)',
    amber: 'rgba(255, 165, 2, 0.8)',
    amberDim: 'rgba(255, 165, 2, 0.15)',
    purple: 'rgba(168, 85, 247, 0.8)',
    purpleDim: 'rgba(168, 85, 247, 0.15)',
};

const GNN_COLORS = ['#00d4ff', '#2ed573', '#ffa502', '#ff4757', '#a855f7'];

// ══════════════════════════════════════════════════════════════
//  PAGE 1: COMMAND CENTER
// ══════════════════════════════════════════════════════════════
async function initDashboard() {
    try {
        const res = await fetch('/api/overview');
        const data = await res.json();

        // KPIs with count-up animation
        animateValue('kpi-accounts', data.total_accounts);
        animateValue('kpi-sar', data.sar_accounts);
        animateValue('kpi-flagged', data.flagged_accounts);
        document.getElementById('kpi-best-prauc').textContent = data.best_model.pr_auc.toFixed(4);
        document.getElementById('kpi-best-model').textContent = data.best_model.name;
        animateValue('kpi-edges', data.total_edges);
        document.getElementById('alert-count').textContent = data.flagged_accounts;

        // Channel chart
        renderChannelChart(data.edge_stats);
        renderRiskChart(data.risk_distribution);
    } catch (e) {
        console.error('Dashboard load error:', e);
    }
}

function animateValue(id, end) {
    const el = document.getElementById(id);
    if (!el) return;
    const duration = 1200;
    const start = 0;
    const startTime = performance.now();

    function tick(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const val = Math.round(start + (end - start) * eased);
        el.textContent = fmtShort(val);
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

function renderChannelChart(edges) {
    if (chartsReady['channels']) chartsReady['channels'].destroy();

    const labels = {
        'transfer': 'Transfers', 'logged_in': 'Logins', 'linked_wallet': 'Wallets',
        'has_vpa': 'VPA', 'withdrew_atm': 'ATM', 'belongs_to_bank': 'Bank',
        'shared_device': 'Shared Device', 'same_wallet': 'Same Wallet'
    };

    const keys = Object.keys(edges);
    const colors = ['#00d4ff', '#2ed573', '#ffa502', '#ff4757', '#a855f7', '#e056fd', '#1abc9c', '#f39c12'];

    chartsReady['channels'] = new Chart(document.getElementById('chart-channels'), {
        type: 'bar',
        data: {
            labels: keys.map(k => labels[k] || k),
            datasets: [{
                data: keys.map(k => edges[k]),
                backgroundColor: colors.map(c => c + '33'),
                borderColor: colors,
                borderWidth: 1.5,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#64748b', font: { family: 'JetBrains Mono', size: 10 },
                        callback: v => fmtShort(v)
                    },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                x: {
                    ticks: { color: '#94a3b8', font: { size: 10 }, maxRotation: 45 },
                    grid: { display: false }
                }
            }
        }
    });
}

function renderRiskChart(dist) {
    if (chartsReady['risk']) chartsReady['risk'].destroy();

    chartsReady['risk'] = new Chart(document.getElementById('chart-risk'), {
        type: 'doughnut',
        data: {
            labels: ['High', 'Medium', 'Low'],
            datasets: [{
                data: [dist.HIGH, dist.MEDIUM, dist.LOW],
                backgroundColor: [
                    'rgba(255, 71, 87, 0.7)',
                    'rgba(255, 165, 2, 0.7)',
                    'rgba(46, 213, 115, 0.7)'
                ],
                borderColor: ['#ff4757', '#ffa502', '#2ed573'],
                borderWidth: 2,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', padding: 16, font: { size: 12 } }
                }
            }
        }
    });
}

// Live Feed
let feedRunning = false;
function toggleFeed() {
    if (feedRunning) {
        if (feedEventSource) feedEventSource.close();
        feedEventSource = null;
        feedRunning = false;
        document.getElementById('btn-toggle-feed').textContent = '▶ Start Feed';
        return;
    }

    feedRunning = true;
    document.getElementById('btn-toggle-feed').textContent = '⏸ Stop Feed';
    document.getElementById('live-feed').innerHTML = '';

    feedEventSource = new EventSource('/api/stream');
    feedEventSource.onmessage = (e) => {
        const txn = JSON.parse(e.data);
        addFeedItem('live-feed', txn, 20);
    };
}

function addFeedItem(containerId, txn, maxItems) {
    const feed = document.getElementById(containerId);
    const riskClass = txn.risk === 'HIGH_RISK' ? 'risk-high' :
        txn.risk === 'SUSPICIOUS' ? 'risk-suspicious' : 'risk-normal';
    const scoreColor = txn.score > 0.7 ? 'var(--accent-red)' :
        txn.score > 0.4 ? 'var(--accent-amber)' : 'var(--accent-green)';

    const item = document.createElement('div');
    item.className = `feed-item ${riskClass}`;
    item.innerHTML = `
        <span class="feed-time">${txn.timestamp.split(' ')[1]}</span>
        <span class="feed-detail">
            <strong>${txn.src}</strong> → <strong>${txn.dst}</strong>
            &nbsp;$${txn.amount.toLocaleString()}
            &nbsp;<span class="badge badge-info">${txn.channel}</span>
        </span>
        <span class="feed-score" style="color:${scoreColor}">${txn.score.toFixed(3)}</span>
    `;

    feed.insertBefore(item, feed.firstChild);
    while (feed.children.length > maxItems) {
        feed.removeChild(feed.lastChild);
    }
}


// ══════════════════════════════════════════════════════════════
//  PAGE 2: ALERT QUEUE
// ══════════════════════════════════════════════════════════════
async function initAlerts() {
    if (alertsData.length > 0) { renderAlerts(alertsData); return; }

    try {
        const res = await fetch('/api/alerts');
        alertsData = await res.json();
        document.getElementById('alert-count').textContent = alertsData.length;
        renderAlerts(alertsData);
    } catch (e) {
        console.error('Alerts load error:', e);
    }
}

function renderAlerts(data) {
    const tbody = document.getElementById('alerts-tbody');
    tbody.innerHTML = '';

    data.forEach(acct => {
        const tr = document.createElement('tr');
        const reasons = acct.reason_codes.slice(0, 3).map(r =>
            `<span class="badge badge-${r.confidence.toLowerCase()}">${getReasonIcon(r.code)} ${r.code}</span>`
        ).join(' ');

        const prof = acct.feature_profile;
        tr.innerHTML = `
            <td><strong style="color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace;">#${acct.account_id}</strong></td>
            <td>${acct.is_sar ? '<span class="badge badge-high">⚠ SAR</span>' : '<span class="badge badge-low">Normal</span>'}</td>
            <td><span class="badge badge-${acct.risk_level.toLowerCase()}">${acct.risk_level}</span></td>
            <td>${reasons}</td>
            <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">${fmt(prof.sent_count)} / ${fmt(prof.recv_count)}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${prof.fan_in_ratio ? prof.fan_in_ratio.toFixed(2) : '—'}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${fmt(prof.channel_diversity)}</td>
            <td><button class="btn btn-outline" style="padding: 6px 12px; font-size: 0.72rem;" onclick="investigateAccount(${acct.account_id})">Investigate →</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function filterAlerts() {
    const search = document.getElementById('alert-search').value.toLowerCase();
    const level = document.getElementById('alert-filter').value;

    const filtered = alertsData.filter(a => {
        const matchSearch = search === '' ||
            String(a.account_id).includes(search) ||
            a.reason_codes.some(r => r.code.toLowerCase().includes(search));
        const matchLevel = level === 'all' || a.risk_level === level;
        return matchSearch && matchLevel;
    });
    renderAlerts(filtered);
}

function investigateAccount(id) {
    navigateTo('investigate', { account_id: id });
}


// ══════════════════════════════════════════════════════════════
//  PAGE 3: INVESTIGATION
// ══════════════════════════════════════════════════════════════
async function initInvestigate(params) {
    if (!params || !params.account_id) return;

    const container = document.getElementById('investigate-content');
    container.innerHTML = '<div class="loading"><div class="spinner"></div> Loading investigation data...</div>';

    try {
        const [detailRes, graphRes] = await Promise.all([
            fetch(`/api/alerts/${params.account_id}`),
            fetch(`/api/graph/${params.account_id}`)
        ]);
        const detail = await detailRes.json();
        const graph = await graphRes.json();

        renderInvestigation(detail, graph);
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><div class="empty-icon">❌</div><p>Error loading account data</p></div>`;
        console.error(e);
    }
}

function renderInvestigation(detail, graph) {
    const container = document.getElementById('investigate-content');
    const prof = detail.feature_profile;

    // Build reason cards HTML
    const reasonsHtml = detail.reason_codes.map(r => `
        <div class="reason-card ${r.confidence.toLowerCase()}">
            <div class="reason-icon">${getReasonIcon(r.code)}</div>
            <div class="reason-body">
                <h4>${r.code.replace(/_/g, ' ')}</h4>
                <p>${r.description}</p>
                <div class="reason-value">${r.value}</div>
            </div>
            <span class="badge badge-${r.confidence.toLowerCase()}" style="margin-left: auto; align-self: flex-start;">${r.confidence}</span>
        </div>
    `).join('');

    // Feature items
    const featureHtml = Object.entries(prof).map(([k, v]) => `
        <div class="feature-item">
            <div class="feature-label">${k.replace(/_/g, ' ')}</div>
            <div class="feature-value">${typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4)) : v}</div>
        </div>
    `).join('');

    container.innerHTML = `
        <!-- Profile Header -->
        <div class="profile-header">
            <div class="profile-avatar sar">👻</div>
            <div class="profile-info">
                <h3>Account #${detail.account_id}</h3>
                <div class="profile-meta">
                    <span class="badge badge-${detail.risk_level.toLowerCase()}">${detail.risk_level} Risk</span>
                    <span>${detail.is_sar ? '⚠️ Confirmed SAR' : '✅ Normal'}</span>
                    <span>🔗 ${graph.total_2hop ? graph.total_2hop.toLocaleString() : '—'} accounts within 2 hops</span>
                </div>
            </div>
        </div>

        <div class="grid-1-2">
            <!-- Reason Codes -->
            <div class="card">
                <div class="card-header">
                    <h3>🚩 Risk Indicators</h3>
                </div>
                <div class="reason-list">${reasonsHtml}</div>
            </div>

            <!-- Evidence Graph -->
            <div class="card">
                <div class="card-header">
                    <h3>🔗 Evidence Subgraph</h3>
                    <span class="badge badge-info">${graph.nodes ? graph.nodes.length : 0} nodes · ${graph.edges ? graph.edges.length : 0} edges</span>
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
                <div class="feature-grid">${featureHtml}</div>
            </div>

            <!-- Narrative -->
            <div class="card">
                <div class="card-header">
                    <h3>📝 Regulator-Ready Narrative</h3>
                    <button class="btn btn-outline" style="padding: 6px 14px; font-size: 0.72rem;" onclick="printNarrative()">🖨 Print Report</button>
                </div>
                <div class="narrative-box">${detail.narrative || 'No narrative available.'}</div>
            </div>
        </div>
    `;

    // Render D3 graph
    if (graph.nodes && graph.nodes.length > 0) {
        renderEvidenceGraph(graph);
    }
}

function printNarrative() {
    const narr = document.querySelector('.narrative-box');
    if (!narr) return;
    const win = window.open('', '_blank');
    win.document.write(`
        <html><head><title>GhostBusters — Investigation Report</title>
        <style>body{font-family:monospace;padding:40px;line-height:1.8;white-space:pre-wrap;}
        h1{font-family:sans-serif;}</style></head>
        <body><h1>🔍 GhostBusters Investigation Report</h1><hr>
        ${narr.textContent}</body></html>
    `);
    win.document.close();
    win.print();
}


// ── D3 Evidence Graph ─────────────────────────────────────────
function renderEvidenceGraph(graphData) {
    const container = document.getElementById('evidence-graph-container');
    if (!container) return;
    container.innerHTML = '';

    const width = container.clientWidth;
    const height = container.clientHeight || 440;

    const svg = d3.select(container).append('svg')
        .attr('width', width)
        .attr('height', height);

    // Defs for glow effect
    const defs = svg.append('defs');
    const filter = defs.append('filter').attr('id', 'glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Build node/link data
    const nodeMap = new Map();
    graphData.nodes.forEach(n => { nodeMap.set(n.id, { ...n }); });

    const links = [];
    graphData.edges.forEach(e => {
        if (nodeMap.has(e.src) && nodeMap.has(e.dst)) {
            links.push({ source: e.src, target: e.dst, hop: e.hop });
        } else if (nodeMap.has(e.dst)) {
            // Src not in our limited node set, add it
            nodeMap.set(e.src, { id: e.src, is_target: false, is_sar: false });
            links.push({ source: e.src, target: e.dst, hop: e.hop });
        }
    });

    const nodes = Array.from(nodeMap.values());

    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(60))
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(18));

    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', 'rgba(0, 212, 255, 0.2)')
        .attr('stroke-width', 1);

    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', d => d.is_target ? 16 : 7)
        .attr('fill', d => d.is_target ? '#ff4757' : '#00d4ff')
        .attr('stroke', d => d.is_target ? '#ff6b81' : 'rgba(0, 212, 255, 0.4)')
        .attr('stroke-width', d => d.is_target ? 3 : 1)
        .attr('filter', d => d.is_target ? 'url(#glow)' : null)
        .call(d3.drag()
            .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
            .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
        );

    // Labels for target node
    const labels = svg.append('g')
        .selectAll('text')
        .data(nodes.filter(n => n.is_target))
        .join('text')
        .attr('text-anchor', 'middle')
        .attr('dy', -22)
        .attr('fill', '#ff4757')
        .attr('font-size', '11px')
        .attr('font-weight', '700')
        .attr('font-family', 'JetBrains Mono')
        .text(d => `#${d.id}`);

    // Tooltip
    node.append('title').text(d => `Account ${d.id}${d.is_target ? ' (TARGET)' : ''}`);

    simulation.on('tick', () => {
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('cx', d => d.x).attr('cy', d => d.y);
        labels.attr('x', d => d.x).attr('y', d => d.y);
    });
}


// ══════════════════════════════════════════════════════════════
//  PAGE 4: MODEL ANALYTICS
// ══════════════════════════════════════════════════════════════
async function initModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();

        renderPRAUCBars(data);
        renderTrainingCurves(data.training_histories);
        renderMetricsTable(data);
        renderTrainingConfig(data);
    } catch (e) {
        console.error('Models load error:', e);
    }
}

function renderPRAUCBars(data) {
    const container = document.getElementById('prauc-bars');
    container.innerHTML = '';

    const models = [];
    for (const [name, m] of Object.entries(data.baseline || {})) {
        models.push({ name, pr_auc: m.pr_auc, type: 'baseline' });
    }
    for (const [name, m] of Object.entries(data.gnn || {})) {
        models.push({ name, pr_auc: m.pr_auc, type: 'gnn' });
    }
    models.sort((a, b) => b.pr_auc - a.pr_auc);

    models.forEach((m, i) => {
        const pct = (m.pr_auc * 100).toFixed(2);
        const row = document.createElement('div');
        row.className = 'model-bar-row';
        row.innerHTML = `
            <span class="bar-label">${m.name}</span>
            <div class="bar-track">
                <div class="bar-fill ${m.type}" style="width: 0%"></div>
            </div>
            <span class="bar-value">${pct}%</span>
        `;
        container.appendChild(row);

        // Animate bar
        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = `${m.pr_auc * 100}%`;
        }, i * 80 + 100);
    });
}

function renderTrainingCurves(histories) {
    if (!histories || Object.keys(histories).length === 0) return;

    // PR-AUC curves
    if (chartsReady['train-prauc']) chartsReady['train-prauc'].destroy();
    const prDatasets = [];
    let i = 0;
    for (const [name, h] of Object.entries(histories)) {
        if (h.val_pr_auc) {
            prDatasets.push({
                label: name,
                data: h.val_pr_auc,
                borderColor: GNN_COLORS[i % GNN_COLORS.length],
                backgroundColor: GNN_COLORS[i % GNN_COLORS.length] + '20',
                borderWidth: 2, tension: 0.3, pointRadius: 0, fill: false,
            });
        }
        i++;
    }

    chartsReady['train-prauc'] = new Chart(document.getElementById('chart-train-prauc'), {
        type: 'line',
        data: { labels: prDatasets[0] ? prDatasets[0].data.map((_, idx) => idx + 1) : [], datasets: prDatasets },
        options: chartOpts('PR-AUC')
    });

    // Loss curves
    if (chartsReady['train-loss']) chartsReady['train-loss'].destroy();
    const lossDatasets = [];
    i = 0;
    for (const [name, h] of Object.entries(histories)) {
        if (h.loss) {
            lossDatasets.push({
                label: name,
                data: h.loss,
                borderColor: GNN_COLORS[i % GNN_COLORS.length],
                borderWidth: 2, tension: 0.3, pointRadius: 0, fill: false,
            });
        }
        i++;
    }

    chartsReady['train-loss'] = new Chart(document.getElementById('chart-train-loss'), {
        type: 'line',
        data: { labels: lossDatasets[0] ? lossDatasets[0].data.map((_, idx) => idx + 1) : [], datasets: lossDatasets },
        options: chartOpts('Loss')
    });
}

function chartOpts(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: '#94a3b8', font: { size: 11 } } }
        },
        scales: {
            y: {
                ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } },
                grid: { color: 'rgba(255,255,255,0.04)' },
                title: { display: true, text: yLabel, color: '#64748b' }
            },
            x: {
                ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 15 },
                grid: { display: false },
                title: { display: true, text: 'Epoch', color: '#64748b' }
            }
        }
    };
}

function renderMetricsTable(data) {
    const tbody = document.getElementById('model-metrics-tbody');
    tbody.innerHTML = '';

    const addRow = (name, type, m) => {
        const isBest = m.pr_auc >= 0.999;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong style="color: ${type === 'gnn' ? 'var(--accent-cyan)' : 'var(--accent-purple)'};">${name}</strong></td>
            <td><span class="badge ${type === 'gnn' ? 'badge-info' : 'badge-purple'}">${type.toUpperCase()}</span></td>
            <td style="font-family: 'JetBrains Mono', monospace; ${isBest ? 'color: var(--accent-green); font-weight:700;' : ''}">${m.pr_auc.toFixed(4)}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${m.roc_auc.toFixed(4)}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${m.f1_sar.toFixed(4)}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${m.precision_sar.toFixed(4)}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${m.recall_sar.toFixed(4)}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${(m['recall@1x'] || 0).toFixed(4)}</td>
            <td style="font-family: 'JetBrains Mono', monospace;">${m.accuracy.toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
    };

    // Sort all models by PR-AUC
    const all = [];
    for (const [name, m] of Object.entries(data.gnn || {})) all.push({ name, type: 'gnn', ...m });
    for (const [name, m] of Object.entries(data.baseline || {})) all.push({ name, type: 'baseline', ...m });
    all.sort((a, b) => b.pr_auc - a.pr_auc);
    all.forEach(m => addRow(m.name, m.type, m));
}

function renderTrainingConfig(data) {
    const grid = document.getElementById('training-config-grid');
    grid.innerHTML = '';

    const config = { ...data.training_config, ...data.gpu_config };
    const display = {
        'gpu_name': '🖥 GPU', 'gpu_memory_gb': '💾 GPU Memory',
        'epochs': '🔄 Epochs', 'lr': '📈 Learning Rate',
        'hidden_dim': '🧠 Hidden Dim', 'patience': '⏱ Patience',
        'use_amp': '⚡ AMP', 'use_bf16': '🔢 BF16',
        'use_tf32': '🔢 TF32', 'batch_size': '📦 Batch Size'
    };

    for (const [key, label] of Object.entries(display)) {
        if (config[key] !== undefined) {
            const item = document.createElement('div');
            item.className = 'feature-item';
            const val = typeof config[key] === 'boolean' ? (config[key] ? '✅ Yes' : '❌ No') : config[key];
            item.innerHTML = `<div class="feature-label">${label}</div><div class="feature-value">${val}</div>`;
            grid.appendChild(item);
        }
    }
}


// ══════════════════════════════════════════════════════════════
//  PAGE 5: STREAMING DEMO
// ══════════════════════════════════════════════════════════════
let streamRunning = false;
let streamHistData = new Array(10).fill(0); // histogram buckets [0-0.1, 0.1-0.2, ...]
let streamChart = null;

function initStreaming() {
    if (!streamChart) {
        streamChart = new Chart(document.getElementById('chart-stream-hist'), {
            type: 'bar',
            data: {
                labels: ['0-.1', '.1-.2', '.2-.3', '.3-.4', '.4-.5', '.5-.6', '.6-.7', '.7-.8', '.8-.9', '.9-1'],
                datasets: [{
                    label: 'Score Distribution',
                    data: [...streamHistData],
                    backgroundColor: streamHistData.map((_, i) => {
                        if (i < 4) return 'rgba(46, 213, 115, 0.6)';
                        if (i < 7) return 'rgba(255, 165, 2, 0.6)';
                        return 'rgba(255, 71, 87, 0.6)';
                    }),
                    borderColor: streamHistData.map((_, i) => {
                        if (i < 4) return '#2ed573';
                        if (i < 7) return '#ffa502';
                        return '#ff4757';
                    }),
                    borderWidth: 1.5,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        title: { display: true, text: 'Count', color: '#64748b' }
                    },
                    x: {
                        ticks: { color: '#94a3b8', font: { size: 10 } },
                        grid: { display: false },
                        title: { display: true, text: 'Risk Score', color: '#64748b' }
                    }
                }
            }
        });
    }
}

function toggleStreaming() {
    if (streamRunning) {
        if (streamEventSource) streamEventSource.close();
        streamEventSource = null;
        streamRunning = false;
        document.getElementById('btn-stream').textContent = '▶ Start Streaming';
        return;
    }

    streamRunning = true;
    streamStats = { total: 0, flagged: 0, scoreSum: 0, startTime: Date.now() };
    streamHistData = new Array(10).fill(0);
    document.getElementById('btn-stream').textContent = '⏸ Stop Streaming';
    document.getElementById('stream-feed').innerHTML = '';

    streamEventSource = new EventSource('/api/stream');
    streamEventSource.onmessage = (e) => {
        const txn = JSON.parse(e.data);
        processStreamTxn(txn);
    };
}

function processStreamTxn(txn) {
    streamStats.total++;
    streamStats.scoreSum += txn.score;
    if (txn.risk === 'HIGH_RISK' || txn.risk === 'SUSPICIOUS') streamStats.flagged++;

    // Update stats
    document.getElementById('stream-total').textContent = streamStats.total;
    document.getElementById('stream-flagged').textContent = streamStats.flagged;
    document.getElementById('stream-avg-score').textContent = (streamStats.scoreSum / streamStats.total).toFixed(3);

    const elapsed = (Date.now() - streamStats.startTime) / 1000;
    document.getElementById('stream-rate').textContent = (streamStats.total / Math.max(elapsed, 1)).toFixed(1);

    // Update histogram
    const bucket = Math.min(Math.floor(txn.score * 10), 9);
    streamHistData[bucket]++;
    if (streamChart) {
        streamChart.data.datasets[0].data = [...streamHistData];
        streamChart.update('none');
    }

    // Update gauge
    const gaugeEl = document.getElementById('score-gauge');
    const scoreColor = txn.score > 0.7 ? '#ff4757' : txn.score > 0.4 ? '#ffa502' : '#2ed573';
    gaugeEl.style.background = `conic-gradient(${scoreColor} ${txn.score * 360}deg, rgba(255,255,255,0.05) ${txn.score * 360}deg)`;
    gaugeEl.querySelector('.gauge-value').textContent = txn.score.toFixed(2);
    gaugeEl.querySelector('.gauge-value').style.color = scoreColor;

    document.getElementById('latest-txn-details').innerHTML = `
        <strong>${txn.src}</strong> → <strong>${txn.dst}</strong><br>
        $${txn.amount.toLocaleString()} via <span class="badge badge-info">${txn.channel}</span><br>
        <span class="badge badge-${txn.risk === 'HIGH_RISK' ? 'high' : txn.risk === 'SUSPICIOUS' ? 'medium' : 'low'}">${txn.risk}</span>
    `;

    // Add to feed
    addFeedItem('stream-feed', txn, 15);
}


// ── Init on Load ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
});
