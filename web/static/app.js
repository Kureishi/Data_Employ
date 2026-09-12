// ML Agent Web UI JavaScript

// ============ State ============
let currentTab = 'data';
let preprocessQueue = [];
let connectedTables = [];
let currentDataSource = null;
let lastAnalyzeResult = null;
let lastPredictions = null;
let lastPredictColumns = null;
let lastTrainResult = null;

// ============ Tier D (item 15): persistent dashboard state ============
const UI_STATE_KEY = 'ml_agent_ui_state';
function saveUIState() {
    const els = {
        analyzeTarget: 'analyze-target', analyzeType: 'analyze-type',
        dbConnection: 'db-connection', trainTarget: 'train-target',
    };
    const s = { tab: currentTab, dataSource: currentDataSource };
    for (const [k, id] of Object.entries(els)) {
        const el = document.getElementById(id);
        if (el) s[k] = el.value;
    }
    try { localStorage.setItem(UI_STATE_KEY, JSON.stringify(s)); } catch (e) { /* ignore */ }
}
function loadUIState() {
    let s = null;
    try { s = JSON.parse(localStorage.getItem(UI_STATE_KEY) || 'null'); } catch (e) { s = null; }
    if (!s) return s;
    if (s.tab && typeof switchTab === 'function') switchTab(s.tab);
    else currentTab = s.tab || 'data';
    currentDataSource = s.dataSource || null;
    const els = {
        analyzeTarget: 'analyze-target', analyzeType: 'analyze-type',
        dbConnection: 'db-connection', trainTarget: 'train-target',
    };
    for (const [k, id] of Object.entries(els)) {
        if (!(k in s)) continue;
        const el = document.getElementById(id);
        if (el) el.value = s[k];
    }
    return s;
}

// ============ Helpers ============

async function api(url, method = 'GET', body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) {
        options.body = JSON.stringify(body);
    }
    const res = await fetch(url, options);
    const data = await res.json();
    if (!res.ok && data.error) {
        throw new Error(data.error);
    }
    return data;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    while (container.children.length >= 5) container.firstChild.remove();
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function setStatus(id, text, state) {
    const el = document.getElementById(id);
    if (!el) return;
    const label = el.querySelector('.label') || el;
    label.textContent = text;
    label.title = text;
    el.className = 'status-badge';
    if (state) el.classList.add(state);
    updatePageTitle();
}

// Item 12: keep the browser-tab title reflective of connection state.
function updatePageTitle() {
    const conn = document.getElementById('db-status');
    const dbText = conn && conn.querySelector ? (conn.querySelector('.label').textContent || '') : '';
    const connected = /connected|ok/i.test(dbText) && !/not/i.test(dbText);
    const tb = document.getElementById('data-table-select');
    const src = (tb && tb.value) ? tb.value : '';
    const suffix = connected ? (src ? ` · ${src}` : ' · Connected') : ' · Not connected';
    try { document.title = 'ML Agent' + suffix; } catch (e) { /* ignore */ }
}

// ============ Loading state helpers ============

function setLoading(btn, loading) {
    if (!btn) return;
    const sp = btn.querySelector('.spinner');
    if (loading) {
        if (sp) return; // already loading
        btn.disabled = true;
        btn.setAttribute('aria-busy', 'true');
        const spinner = document.createElement('span');
        spinner.className = 'spinner';
        spinner.setAttribute('aria-hidden', 'true');
        btn.prepend(spinner);
    } else {
        btn.disabled = false;
        btn.removeAttribute('aria-busy');
        if (sp) sp.remove();
    }
}

function fmtNum(val) {
    if (val === null || val === undefined) return 'N/A';
    if (typeof val === 'number') {
        if (Number.isInteger(val)) return val.toLocaleString('en-US');
        if (Math.abs(val) >= 1000) return val.toLocaleString('en-US', { maximumFractionDigits: 2 });
        return val.toFixed(4);
    }
    return String(val);
}

function formatJson(val) {
    if (val === null || val === undefined) return 'N/A';
    if (typeof val === 'number') return fmtNum(val);
    if (typeof val === 'object') return JSON.stringify(val);
    return String(val);
}

function escapeHtml(str) {
    const amp = String.fromCharCode(38);
    const lt = String.fromCharCode(60);
    const gt = String.fromCharCode(62);
    const quot = String.fromCharCode(34);
    return String(str)
        .replace(/&/g, amp + 'amp;')
        .replace(/</g, lt + 'lt;')
        .replace(/>/g, gt + 'gt;')
        .replace(/"/g, quot + 'quot;')
        .replace(/'/g, '&#039;');
}

function renderTable(data, columns) {
    if (!data || data.length === 0) return '<p class="hint">No data to display.</p>';
    const cols = columns || Object.keys(data[0]);
    // Detect numeric columns from the first row so values right-align (Item 8).
    const first = data[0] || {};
    const numeric = new Set(cols.filter(c => typeof first[c] === 'number'));
    let html = '<table class="dataframe"><thead><tr>';
    html += cols.map(c => `<th class="${numeric.has(c) ? 'num' : ''}">${escapeHtml(c)}</th>`).join('');
    html += '</tr></thead><tbody>';
    for (const row of data) {
        html += '<tr>';
        html += cols.map(c => `<td class="${numeric.has(c) ? 'num' : ''}">${escapeHtml(formatJson(row[c]))}</td>`).join('');
        html += '</tr>';
    }
    html += '</tbody></table>';
    return html;
}

function renderJson(obj) {
    return JSON.stringify(obj, null, 2);
}

function renderMetrics(metrics) {
    if (!metrics || Object.keys(metrics).length === 0) return '';
    let html = '<div class="metric-grid">';
    for (const [k, v] of Object.entries(metrics)) {
        html += `<div class="metric-card"><div class="metric-label">${escapeHtml(k)}</div><div class="metric-value">${escapeHtml(formatJson(v))}</div></div>`;
    }
    html += '</div>';
    return html;
}

// ============ Tab switching ============

function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-content').forEach(c => {
        c.classList.toggle('active', c.id === `tab-${tab}`);
    });
    saveUIState(); // Tier D: persist dashboard state
}

// ============ State refresh ============

async function refreshState() {
    try {
        const state = await api('/api/state');
        if (state.connected) {
            setStatus('db-status', 'Connected', 'connected');
            if (state.tables) {
                connectedTables = state.tables;
                renderTablesList(state.tables);
                populateTableSelect(state.tables);
            }
            if (state.loaded_data) {
                const info = document.getElementById('data-info');
                info.innerHTML = `<strong>Loaded:</strong> ${state.loaded_data.shape[0]} rows × ${state.loaded_data.shape[1]} cols<br><strong>Columns:</strong> ${state.loaded_data.columns.join(', ')}`;
            }
            if (state.model) {
                const trainResult = document.getElementById('train-result');
                trainResult.innerHTML = `<strong>Model:</strong> ${escapeHtml(state.model.best_model)}<br><strong>Task:</strong> ${escapeHtml(state.model.task_type)}<br><strong>CV Score:</strong> ${formatJson(state.model.best_cv_score)}<br>${renderMetrics(state.model.test_metrics)}`;
            }
        } else {
            setStatus('db-status', 'Not Connected', '');
            document.getElementById('tables-list').innerHTML = '<p class="hint">Connect to a database to see tables.</p>';
        }

        if (state.llm) {
            if (state.llm.available) {
                setStatus('llm-status', `LLM: ${state.llm.detail}`, 'connected');
            } else {
                setStatus('llm-status', 'LLM: Unavailable', 'error');
            }
        } else {
            setStatus('llm-status', 'LLM: Not enabled', '');
        }
    } catch (e) {
        setStatus('db-status', 'Not Connected', 'error');
        setStatus('llm-status', 'LLM: Not enabled', '');
    }
}

function skeletonBlock(text) {
    return '<div class="empty-state"><span class="skeleton-line"></span><p class="hint">' + escapeHtml(text) + '</p></div>';
}

function renderTablesList(tables) {
    const list = document.getElementById('tables-list');
    if (!tables || tables.length === 0) {
        list.innerHTML = '<p class="hint">No tables found. Connect to a database to list tables.</p>';
        return;
    }
    list.innerHTML = tables.map(t => `
        <div class="table-item" data-table="${escapeHtml(t)}" title="${escapeHtml(t)} — click to load" tabindex="0" role="button">
            <span class="table-name">${escapeHtml(t)}</span>
            <span class="table-spark"></span>
            <span class="table-rows">load</span>
        </div>
    `).join('');

    const activate = (el) => {
        document.querySelectorAll('.table-item').forEach(x => x.classList.remove('active'));
        el.classList.add('active');
    };

    const loadTable = async (el) => {
        const table = el.dataset.table;
        currentDataSource = table;
        activate(el);
        const preview = document.getElementById('data-preview');
        if (preview) preview.innerHTML = skeletonBlock(`Loading ${table}…`);
        try {
            const res = await api('/api/load', 'POST', { table });
            const info = document.getElementById('data-info');
            info.className = 'info-box success';
            info.innerHTML = `<strong>Loaded:</strong> ${escapeHtml(table)}<br><strong>Rows:</strong> ${fmtNum(res.rows)}<br><strong>Columns:</strong> ${escapeHtml(res.columns.join(', '))}`;
            if (preview) preview.innerHTML = renderTable(res.data, res.columns);
            showToast(`Loaded ${fmtNum(res.rows)} rows from ${table}`, 'success');
            await refreshState();
            await autoSuggestTarget();
            saveUIState();
            try { await runQuickView(); } catch (e) { /* best-effort */ }
        } catch (e) {
            if (preview) preview.innerHTML = '<div class="empty-state error"><p class="hint">Failed to load ' + escapeHtml(table) + '.</p></div>';
            showToast(e.message, 'error');
        }
    };

    document.querySelectorAll('.table-item').forEach(el => {
        el.addEventListener('click', () => loadTable(el));
        el.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); loadTable(el); }
        });
    });
    renderSidebarPreviews(tables);
}

// ============ Tier D (item 16): sidebar sparkline previews ============
let __sidebarPreviewsLoaded = false;
function renderSidebarPreviews(tables) {
    if (__sidebarPreviewsLoaded || !tables || !tables.length) return;
    __sidebarPreviewsLoaded = true;
    // Lazy-load a small numeric preview per table (first ~6) to draw a sparkline.
    tables.slice(0, 6).forEach((t, i) => {
        setTimeout(async () => {
            try {
                const res = await api(`/api/tables/${encodeURIComponent(t)}/preview?limit=40`, 'GET');
                const rows = res.data || [];
                const numeric = (res.columns || []).map(c => c.name)
                    .find(name => rows.length && typeof rows[0][name] === 'number');
                if (!numeric || !rows.length) return;
                const el = document.querySelector(`.table-item[data-table="${escapeHtml(t)}"] .table-spark`);
                if (!el) return;
                const vmin = Math.min(...rows.map(r => r[numeric])), vmax = Math.max(...rows.map(r => r[numeric]));
                const span = (vmax > vmin) ? (vmax - vmin) : 1;
                const pts = rows.slice(0, 40).map((r, idx) => {
                    const x = idx * 1.4, y = 18 - ((Number(r[numeric]) - vmin) / span) * 14;
                    return `${x.toFixed(1)},${y.toFixed(1)}`;
                }).join(' ');
                el.innerHTML = `<svg width="${(rows.length - 1) * 1.4}" height="20" xmlns="http://www.w3.org/2000/svg" class="sparkline" role="img" aria-label="Preview of ${escapeHtml(numeric)}"><title>${escapeHtml(t)} · ${escapeHtml(numeric)} ${fmtNum(vmin)}–${fmtNum(vmax)}</title><polyline points="${pts}" fill="none" stroke="#4f46e5" stroke-width="1.5"/></svg>`;
            } catch (e) { /* non-fatal */ }
        }, 250 + i * 150);
    });
}

// ============ Tier D (item 13): Quick View ============
async function runQuickView() {
    const box = document.getElementById('quick-view-result');
    if (!box) return;
    box.innerHTML = '<p class="hint">Running quick view…</p>';
    try {
        const target = ((document.getElementById('analyze-target') || {}).value || '').trim() || null;
        const res = await api('/api/analyze', 'POST', { type: 'summary', target_column: target, use_cache: true });
        const rep = res.analysis;
        box.innerHTML = renderSummary(rep);
        if (res.analysis && res.analysis.basic_stats) {
            box.innerHTML += `<p class="hint">Source: ${escapeHtml(res.analysis.basic_stats.rows || '')} rows · ${escapeHtml(res.analysis.basic_stats.columns || '')} cols</p>`;
        }
    } catch (err) {
        box.innerHTML = '';
        showToast(err.message, 'error');
    }
}
document.getElementById('btn-quick-view').addEventListener('click', () => runQuickView());

function populateTableSelect(tables) {
    const select = document.getElementById('data-table-select');
    select.innerHTML = '<option value="">Select a table...</option>' +
        tables.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
    const predictSelect = document.getElementById('predict-table-select');
    if (predictSelect) {
        const cur = predictSelect.value;
        predictSelect.innerHTML = '<option value="">Select a table...</option>' +
            tables.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
        if (cur) predictSelect.value = cur;
    }
    // Schema tab table selects
    for (const id of ['schema-table-select', 'sample-table-select']) {
        const s = document.getElementById(id);
        if (s) {
            s.innerHTML = '<option value="">Select a table...</option>' +
                tables.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
        }
    }
    // Auto-join multiselect
    const jt = document.getElementById('join-tables');
    if (jt) {
        jt.innerHTML = tables.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
    }
    // Deep-features base table
    const dft = document.getElementById('deepfeat-table');
    if (dft) {
        dft.innerHTML = '<option value="">Select a base table...</option>' +
            tables.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
    }
    // Anomaly detection + model monitoring table selects
    for (const id of ['anomaly-table', 'monitor-table']) {
        const s = document.getElementById(id);
        if (s) {
            const cur = s.value;
            s.innerHTML = '<option value="">Loaded data / select table...</option>' +
                tables.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
            if (cur) s.value = cur;
        }
    }
    // Re-predict (production table) select
    const rp = document.getElementById('repredict-table-select');
    if (rp) {
        const cur = rp.value;
        rp.innerHTML = '<option value="">Select a table...</option>' +
            tables.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
        if (cur) rp.value = cur;
    }
}

// ============ Tier 1 efficiency: target suggestion / 1-click auto-train ============

async function autoSuggestTarget() {
    // Heuristic target-column suggestions on the current data. Populates the
    // datalist and pre-fills the target input with the top suggestion.
    try {
        const res = await api('/api/suggest-targets', 'GET');
        const suggestions = res.suggestions || [];
        const dl = document.getElementById('target-options');
        if (dl) {
            dl.innerHTML = suggestions.map(s => `<option value="${escapeHtml(s.column)}"></option>`).join('');
        }
        if (suggestions.length > 0) {
            const top = suggestions[0];
            const targetInput = document.getElementById('train-target');
            if (targetInput && !targetInput.value.trim()) {
                targetInput.value = top.column;
            }
            const hint = document.getElementById('auto-train-hint');
            if (hint) {
                hint.textContent = `Suggested target: ${top.column} (${top.task_type}, CV-score-able)`;
            }
        }
    } catch (e) {
        // silent — suggestion is best-effort
    }
}

// 1-click pipeline: suggest target → auto-prepare → reuse cached experiment if
// possible (champion-proposal) → otherwise kick off asynchronised training.
async function runAutoTrain() {
    const btn = document.getElementById('btn-auto-train');
    setLoading(btn, true);
    const hint = document.getElementById('auto-train-hint');
    try {
        let target = document.getElementById('train-target').value.trim() || null;
        const tuning = document.getElementById('train-tuning').value || 'off';
        const nJobs = getTrainNJobs();

        if (!target) {
            const res = await api('/api/suggest-targets', 'GET');
            target = (res.suggestions || [])[0]?.column || null;
            if (!target) {
                showToast('Could not infer a target column — enter one manually.', 'error');
                return;
            }
        }

        // 1) Auto-prepare (drop constant / high-cardinality ID columns)
        if (hint) hint.textContent = `Auto-preparing with target ${target}…`;
        const prep = await api('/api/auto-prepare', 'POST', { target_column: target });
        document.getElementById('preprocess-result').innerHTML =
            renderJson({ auto_prepared: prep.prepared, kept: prep.kept_columns.length, dropped: prep.dropped_columns });

        // 2) Champion-proposal: offer a cached matching experiment instead of retraining.
        const prop = await api('/api/experiments/propose', 'POST', {
            data_source: currentDataSource || undefined,
            target_column: target,
        });
        if (prop.found && prop.experiment) {
            const ex = prop.experiment;
            const path = ex.model_path;
            if (path && window.confirm(
                `A matching trained model already exists for "${target}" (${ex.best_model || 'model'}` +
                `${ex.best_cv_score != null ? ', CV ' + Number(ex.best_cv_score).toFixed(4) : ''}).\n\nLoad it instead of retraining?`
            )) {
                await api('/api/load-model', 'POST', { path });
                document.getElementById('train-result').innerHTML = renderTrainResults({
                    best_model: ex.best_model,
                    task_type: ex.task_type,
                    target_column: ex.target_column,
                });
                showToast('Loaded existing model (champion-proposal).', 'success');
                if (hint) hint.textContent = 'Reused cached experiment — no retraining needed.';
                await refreshState();
                return;
            }
        }

        // 3) Otherwise train (auto-prepare already cleaned the data).
        if (hint) hint.textContent = `Training on auto-prepared data (target: ${target})…`;
        await startTraining(target, document.getElementById('train-task-type').value || null, tuning, nJobs);
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(btn, false);
    }
}

function getTrainNJobs() {
    const v = document.getElementById('train-njobs').value;
    const n = parseInt(v, 10);
    return !v || isNaN(n) || n < 1 ? null : n;
}

function getTrainEarlyStop() {
    const v = document.getElementById('train-early-stop').value;
    const n = parseInt(v, 10);
    return !v || isNaN(n) || n < 1 ? null : n;
}

// Start a backend async operation job and poll it to completion.
async function runOp(operation, params, { progress } = {}) {
    const res = await api('/api/op/start', 'POST', { operation, params });
    const jobId = res.job_id;
    for (;;) {
        await new Promise(r => setTimeout(r, 800));
        const st = await api(`/api/op/status/${jobId}`, 'GET');
        if (progress) progress(st);
        if (st.status === 'done') return st.result;
        if (st.status === 'error') throw new Error(st.error || 'Operation failed.');
    }
}

// Poll the shared notification drain and surface completion signals as toasts.
async function pollNotifications() {
    try {
        const res = await fetch('/api/notifications');
        const data = await res.json();
        for (const n of (data.notifications || [])) {
            showToast(n.message || `${n.operation} finished`, n.level || 'info');
        }
    } catch (e) {
        // silent — next poll will retry
    }
}

// ============ CSV / chart helpers ============

function downloadCsv(filename, rows) {
    if (!rows || rows.length === 0) {
        showToast('Nothing to export.', 'warning');
        return;
    }
    const cols = Object.keys(rows[0]);
    const esc = v => {
        if (v === null || v === undefined) return '';
        const s = String(v);
        return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const lines = [cols.join(',')];
    for (const r of rows) {
        lines.push(cols.map(c => esc(r[c])).join(','));
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Generic text/blob download.
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadText(filename, text, mime) {
    downloadBlob(new Blob([text], { type: mime || 'text/plain;charset=utf-8;' }), filename);
}

// Serialize an inline <svg> to a downloadable PNG (Tier B item 9).
async function downloadSvgAsPng(svg, filename) {
    const clone = svg.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    const width = parseFloat(svg.getAttribute('width')) || svg.getBoundingClientRect().width || 600;
    const height = parseFloat(svg.getAttribute('height')) || svg.getBoundingClientRect().height || 400;
    clone.setAttribute('width', width);
    clone.setAttribute('height', height);
    // Copy any stylesheet rules so colours/fonts resolve in the isolated SVG.
    const styles = Array.from(document.styleSheets)
        .map(s => { try { return Array.from(s.cssRules).map(r => r.cssText).join('\n'); } catch (e) { return ''; } })
        .join('\n');
    const styleEl = document.createElement('style');
    styleEl.textContent = styles;
    clone.insertBefore(styleEl, clone.firstChild);

    const xml = new XMLSerializer().serializeToString(clone);
    const svgDataUrl = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml);

    try {
        const img = new Image();
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = svgDataUrl;
        });
        const scale = Math.min(2, Math.max(1, 3000 / Math.max(width, height))); // up to 2x for crispness
        const canvas = document.createElement('canvas');
        canvas.width = Math.round(width * scale);
        canvas.height = Math.round(height * scale);
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(b => { if (b) downloadBlob(b, filename); }, 'image/png');
    } catch (e) {
        // Fallback: download the raw SVG if PNG rasterisation fails.
        downloadBlob(new Blob([xml], { type: 'image/svg+xml;charset=utf-8;' }), (filename || 'chart.png').replace(/\.png$/i, '.svg'));
    }
}

// ============ Export helpers (Tier B: reports & markdown) ============

function renderSummaryToReport(report, target) {
    // Build a compact, printable body from an analysis summary.
    const b = report.basic_stats || {};
    const meta = `<div class="health-summary">` +
        `<div class="health-card"><div class="cell-label">Rows</div><div class="cell-value">${b.rows}</div></div>` +
        `<div class="health-card"><div class="cell-label">Columns</div><div class="cell-value">${b.columns}</div></div>` +
        `<div class="health-card"><div class="cell-label">Missing</div><div class="cell-value">${b.missing_values}</div></div>` +
        `<div class="health-card"><div class="cell-label">Duplicates</div><div class="cell-value">${b.duplicate_rows}</div></div>` +
        (target ? `<div class="health-card"><div class="cell-label">Target</div><div class="cell-value">${escapeHtml(target)}</div></div>` : '') +
        `</div>`;
    return meta + renderCorrelations(report.correlations) +
        (report.column_insights && report.column_insights.length ? '<h4>Column Insights</h4>' + renderInsights(report.column_insights) : '') +
        (report.target_analysis && !report.target_analysis.error ? '<h4>Target Analysis</h4>' + renderTargetAnalysis(report.target_analysis) : '');
}

// Data-rich HTML tables useful in a shareable report / Markdown export.
function metricsToHtmlTable(metrics) {
    if (!metrics || !Object.keys(metrics).length) return '';
    const rows = Object.entries(metrics).map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${formatJson(v)}</td></tr>`).join('');
    return '<div class="table-container"><table class="dataframe"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
}

function markdownTable(headers, rows) {
    const h = '| ' + headers.join(' | ') + ' |';
    const sep = '| ' + headers.map(() => '---').join(' | ') + ' |';
    const body = rows.map(r => '| ' + headers.map((_, i) => r[i] !== undefined && r[i] !== null ? String(r[i]) : '').join(' | ') + ' |').join('\n');
    return h + '\n' + sep + '\n' + body;
}

// Render the current training result as a shareable Markdown document (Tier B item 8).
function trainResultsToMarkdown(t) {
    if (!t) return '';
    const md = [];
    md.push('# Training Report');
    md.push('');
    md.push(`- **Task type**: ${t.task_type || 'N/A'}`);
    md.push(`- **Best model**: ${t.best_model || 'N/A'}`);
    md.push(`- **CV score**: ${formatJson(t.best_cv_score)}`);
    if (t.target_column) md.push(`- **Target column**: ${t.target_column}`);
    if (t.tuning && t.tuning !== 'off') md.push(`- **Tuning**: ${t.tuning}`);

    if (t.test_metrics && Object.keys(t.test_metrics).length) {
        md.push('');
        md.push('## Test Metrics');
        md.push('');
        md.push(markdownTable(['Metric', 'Value'], Object.entries(t.test_metrics).map(([k, v]) => [k, formatJson(v)])));
    }

    if (t.model_scores && Object.keys(t.model_scores).length) {
        md.push('');
        md.push('## Candidate Model Scores');
        md.push('');
        const entries = Object.entries(t.model_scores).sort((a, b) => b[1] - a[1]);
        md.push(markdownTable(['Model', 'CV Score'], entries.map(([k, v]) => [k, formatJson(v)])));
    }

    if (t.feature_importance && t.feature_importance.length) {
        md.push('');
        md.push('## Top Feature Importances');
        md.push('');
        md.push(markdownTable(['Feature', 'Importance'], t.feature_importance.slice(0, 10).map(fi => [fi.feature, formatJson(fi.importance)])));
    }

    if (t.tuning && t.tuning !== 'off' && t.best_params && Object.keys(t.best_params).length) {
        md.push('');
        md.push('## Best Hyperparameters');
        md.push('');
        md.push('```json');
        md.push(JSON.stringify(t.best_params, null, 2));
        md.push('```');
    }
    return md.join('\n') + '\n';
}

// Assemble a self-contained, shareable HTML analysis report (Tier B item 7).
async function exportHtmlReport() {
    const targetEl = document.getElementById('analyze-target');
    const target = targetEl ? (targetEl.value.trim() || null) : null;
    let report;
    try {
        const res = await api('/api/analyze', 'POST', { type: 'summary', target_column: target, use_cache: true });
        report = res.analysis;
        // Also surface the same view in the Analyze tab so the user sees what's exported.
        const out = document.getElementById('analyze-result');
        if (out) out.innerHTML = renderSummary(report);
        if (lastAnalyzeResult === null) lastAnalyzeResult = report;
    } catch (err) {
        showToast('Could not generate report: ' + err.message, 'error');
        return;
    }

    const body = renderSummaryToReport(report, target);
    let css = '';
    try { css = await (await fetch('/static/style.css')).text(); } catch (e) { css = ''; }
    const stamp = new Date().toLocaleString();
    const source = (report.basic_stats && report.basic_stats.source) ? report.basic_stats.source : '';
    const filename = 'ml-agent-analysis-report.html';
    const doc = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">' +
        '<meta name="viewport" content="width=device-width, initial-scale=1">' +
        '<title>ML Agent — Analysis Report</title>' +
        '<style>html{max-width:980px;margin:0 auto;padding:24px;}@media print{body{padding:0}}' + css + '</style>' +
        '</head><body class="report-body">' +
        `<h1>Data Analysis Report</h1>` +
        `<p class="hint">Generated ${escapeHtml(stamp)} ${source ? '· Source: ' + escapeHtml(source) : ''}</p>` +
        body + '</body></html>';
    downloadBlob(new Blob([doc], { type: 'text/html;charset=utf-8;' }), filename);
}

// Delegated handler: any .chart-png-btn downloads the sibling SVG as PNG (Item 9).
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.chart-png-btn');
    if (!btn) return;
    const chart = btn.closest('.chart');
    const svg = chart && chart.querySelector('svg');
    if (!svg) return;
    const fname = svg.getAttribute('data-filename') || 'chart.png';
    downloadSvgAsPng(svg, fname).catch(() => showToast('Chart download failed.', 'error'));
});

document.getElementById('btn-export-train-md').addEventListener('click', () => {
    if (!lastTrainResult) {
        showToast('No training results to export yet. Train a model first.', 'warning');
        return;
    }
    downloadText('training-report.md', trainResultsToMarkdown(lastTrainResult), 'text/markdown;charset=utf-8;');
    showToast('Training report exported as Markdown.', 'success');
});

document.getElementById('btn-export-report').addEventListener('click', async (e) => {
    const _b = e.currentTarget; setLoading(_b, true);
    try {
        await exportHtmlReport();
        showToast('Analysis report exported as HTML.', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_b, false);
    }
});

function svgBarChart(items, opts) {
    opts = opts || {};
    if (!items || items.length === 0) return '';
    const width = opts.width || 520;
    const barH = opts.barH || 18;
    const gap = 4;
    const maxVal = Math.max(...items.map(i => Math.abs(i.value) || 0)) || 1;
    const padding = { top: 6, bottom: 6 };
    const height = items.length * (barH + gap) + padding.top + padding.bottom;
    const labelW = 130;
    const valueW = 46;
    const plotW = width - labelW - valueW;
    const color = opts.color || '#4f46e5';

    let bars = '';
    // Consistent chart language (Item 13): muted top reference gridline + baseline.
    bars += `<line x1="${labelW}" y1="${padding.top}" x2="${width - valueW}" y2="${padding.top}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="3 3"></line>`;
    bars += `<line x1="${labelW}" y1="${padding.top}" x2="${labelW}" y2="${height - padding.bottom}" stroke="#d1d5db" stroke-width="1"></line>`;
    items.forEach((it, idx) => {
        const w = (Math.abs(it.value) / maxVal) * plotW;
        const y = padding.top + idx * (barH + gap);
        const isNeg = it.value < 0;
        const x = labelW + (isNeg ? plotW - w : 0);
        const disp = (it.label || '').length > 18 ? it.label.slice(0, 16) + '…' : it.label;
        bars += `<text x="${labelW - 6}" y="${y + barH - 4}" text-anchor="end" font-size="11" fill="#6b7280">${escapeHtml(disp)}</text>`;
        bars += `<rect x="${x}" y="${y}" width="${Math.max(w, it.value === 0 ? 0 : 2)}" height="${barH}" fill="${color}" rx="2"></rect>`;
        bars += `<text x="${x + Math.max(w, 2) + 4}" y="${y + barH - 4}" font-size="11" fill="#1f2937">${formatJson(it.value)}</text>`;
    });

    const fname = opts.filename || 'chart.png';
    const aria = escapeHtml(opts.title || 'Bar chart');
    return `<div class="chart"><div class="chart-head"><div class="chart-title">${escapeHtml(opts.title || '')}</div>` +
        `<button type="button" class="btn btn-outline btn-sm chart-png-btn" title="Download this chart as PNG" aria-label="Download ${aria} as PNG">⤓ PNG</button></div>` +
        `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${aria}" data-filename="${escapeHtml(fname)}">${bars}</svg></div>`;
}

function renderConfusionMatrix(cm, classLabels) {
    if (!cm || !cm.length) return '';
    const n = cm.length;
    const labels = classLabels || Array.from({ length: n }, (_, i) => i);
    let mx = 0;
    for (const row of cm) for (const v of row) mx = Math.max(mx, v || 0);
    let html = '<h4>Confusion Matrix</h4><div class="table-container"><table class="cm-table wide">';
    html += '<tr><th></th>' + labels.map(l => `<th>Pred ${escapeHtml(String(l))}</th>`).join('') + '</tr>';
    for (let i = 0; i < n; i++) {
        html += `<tr><th>Act ${escapeHtml(String(labels[i]))}</th>`;
        for (let j = 0; j < n; j++) {
            const v = cm[i][j] || 0;
            const a = mx ? v / mx : 0;
            const bg = 'rgba(16,185,129,' + (a * 0.88 + 0.07).toFixed(3) + ')';
            html += `<td style="background:${bg}">${v}</td>`;
        }
        html += '</tr>';
    }
    html += '</table></div>';
    return html;
}

function renderClassificationReport(report, classMapping) {
    if (!report) return '';
    const classes = Object.keys(report).filter(k => !['accuracy', 'macro avg', 'weighted avg'].includes(k));
    if (classes.length === 0) return '';
    const map = v => (classMapping && classMapping[v] !== undefined) ? classMapping[v] : v;
    let html = '<h4>Classification Report</h4><table class="cm-table"><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>';
    for (const c of classes) {
        const row = report[c];
        html += `<tr><td>${escapeHtml(String(map(c)))}</td><td>${formatJson(row.precision)}</td><td>${formatJson(row.recall)}</td><td>${formatJson(row['f1-score'])}</td><td>${row.support}</td></tr>`;
    }
    html += '</table>';
    return html;
}

function renderColumnHealthTable(cols) {
    if (!cols || !cols.length) return '';
    let html = '<div class="table-container"><table class="dataframe"><thead><tr>' +
        '<th>Column</th><th>Type</th><th>Null</th><th>Null %</th><th>Unique</th><th>Cardinality</th></tr></thead><tbody>';
    for (const c of cols) {
        html += `<tr><td>${escapeHtml(c.column)}</td><td>${escapeHtml(c.dtype)}</td><td>${c.null_count}</td><td>${c.null_pct}%</td><td>${c.unique_values}</td><td>${c.cardinality}</td></tr>`;
    }
    html += '</tbody></table></div>';
    return html;
}

// ============ Render helpers for training / health ============

function classLabelsFromMapping(m) {
    if (!m) return null;
    return Object.keys(m).sort((a, b) => Number(a) - Number(b)).map(k => m[k]);
}

function renderTrainResults(t) {
    if (!t) return '';
    let html = `<strong>Task:</strong> ${escapeHtml(t.task_type)}<br>`;
    html += `<strong>Best Model:</strong> ${escapeHtml(t.best_model)}<br>`;
    html += `<strong>CV Score:</strong> ${formatJson(t.best_cv_score)}<br>`;
    html += renderMetrics(t.test_metrics);

    if (t.tuning && t.tuning !== 'off' && t.best_params && Object.keys(t.best_params).length) {
        const clean = {};
        for (const [k, v] of Object.entries(t.best_params)) {
            const key = k.startsWith('model__') ? k.slice('model__'.length) : k;
            clean[key] = (v === null) ? 'None' : v;
        }
        html += '<br><strong>Tuned Parameters (best):</strong><br>';
        html += '<code>' + escapeHtml(JSON.stringify(clean)) + '</code><br>';
    }

    if (t.model_scores && Object.keys(t.model_scores).length) {
        const items = Object.entries(t.model_scores)
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => ({ label: k, value: v }));
        html += svgBarChart(items, { title: 'Candidate Model Scores', color: '#10b981' });
    }

    if (t.feature_importance && t.feature_importance.length) {
        const items = t.feature_importance.slice(0, 10).map(fi => ({ label: fi.feature, value: fi.importance }));
        html += svgBarChart(items, { title: 'Top Feature Importances', color: '#4f46e5' });
    }

    const classLabels = classLabelsFromMapping(t.class_mapping);
    html += renderConfusionMatrix(t.confusion_matrix, classLabels);
    html += renderClassificationReport(t.classification_report, t.class_mapping);
    return html;
}

function renderHealthReport(h) {
    if (!h) return '';
    let html = '<div class="health-summary">';
    html += `<div class="health-card"><div class="cell-label">Rows</div><div class="cell-value">${h.rows}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Columns</div><div class="cell-value">${h.columns}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Missing</div><div class="cell-value">${h.missing_cells_pct}%</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Duplicates</div><div class="cell-value">${h.duplicate_pct}%</div></div>`;
    html += '</div>';

    const nullItems = (h.column_health || [])
        .map(c => ({ label: c.column, value: c.null_count }))
        .filter(i => i.value > 0);
    if (nullItems.length) {
        html += svgBarChart(nullItems, { title: 'Missing Values by Column', color: '#ef4444' });
    }

    if (h.warnings && h.warnings.length) {
        html += '<h4>Warnings</h4><ul class="warning-list">' + h.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('') + '</ul>';
    } else {
        html += '<p class="hint">No data quality warnings detected. 🎉</p>';
    }

    html += '<h4>Column Health</h4>' + renderColumnHealthTable(h.column_health);

    if (h.class_balance && Object.keys(h.class_balance).length) {
        const items = Object.entries(h.class_balance).map(([k, v]) => ({ label: k, value: v.count }));
        html += svgBarChart(items, { title: 'Class Balance (counts)', color: '#f59e0b' });
    }
    return html;
}

// ============ Tier A: SVG-based analysis charts ============

// Diverging color (red = +, blue = -) for a correlation value in [-1, 1].
function corrFill(v) {
    const a = Math.min(1, Math.abs(v || 0));
    if (v >= 0) {
        return 'rgba(220,38,38,' + (a * 0.85 + 0.12).toFixed(3) + ')';
    }
    return 'rgba(37,99,235,' + (a * 0.85 + 0.12).toFixed(3) + ')';
}

// Correlation heatmap from {col: {col: value}}.
function renderCorrelations(corr) {
    if (!corr) return '';
    if (corr.error) return `<p class="hint">${escapeHtml(corr.error)}</p>`;
    const cols = Object.keys(corr);
    if (cols.length < 2) return renderJson(corr);
    let html = '<div class="chart"><div class="chart-title">Correlation Heatmap</div>';
    html += '<div class="heatmap" style="grid-template-columns:auto repeat(' + cols.length + ', minmax(36px,1fr))">';
    html += '<div></div>';
    for (const c of cols) html += `<div class="hm-label">${escapeHtml(c)}</div>`;
    for (const c of cols) {
        html += `<div class="hm-label">${escapeHtml(c)}</div>`;
        for (const c2 of cols) {
            const v = (corr[c] && corr[c][c2] !== undefined) ? corr[c][c2] : 0;
            html += `<div class="hm-cell" style="background:${corrFill(v)}" title="${escapeHtml(c)} vs ${escapeHtml(c2)}: ${v}">${formatJson(v)}</div>`;
        }
    }
    html += '</div></div>';
    return html;
}

// Horizontal mini distribution band for a numeric column's min/mean/±1σ/max.
function distBand(info) {
    if (info == null || info.min == null || info.max == null || info.min === info.max) return '';
    const min = info.min, max = info.max, mean = info.mean, std = info.std || 0;
    const range = max - min;
    const p = (x) => ((x - min) / range * 100).toFixed(1);
    const lo = Math.min(max, Math.max(min, mean - std));
    const hi = Math.max(min, Math.min(max, mean + std));
    const w = Math.max(0, p(hi) - p(lo));
    return `<div class="dist-band"><div class="dist-track"><div class="dist-one" style="left:${p(lo)}%;width:${w}%"></div>` +
        `<div class="dist-mean" style="left:${p(mean)}%"></div></div>` +
        `<div class="dist-ticks"><span>${formatJson(min)}</span><span>${formatJson(mean)}</span><span>${formatJson(max)}</span></div></div>`;
}



// Column insights: numeric -> distribution band + stats; categorical -> top-value bars.
function renderInsights(insights) {
    if (!insights || !insights.length) return '';
    let html = '<div class="chart"><div class="chart-title">Column Insights</div>';
    html += '<div class="table-container"><table class="dataframe"><thead><tr>' +
        '<th>Column</th><th>Type</th><th>Null</th><th>Distribution / Top Values</th><th>Stats</th></tr></thead><tbody>';
    for (const c of insights) {
        let vis = '<span class="hint">—</span>';
        let stats = '';
        if (c.mean !== undefined) {
            vis = distBand(c);
            stats = 'μ ' + formatJson(c.mean) + ' · σ ' + formatJson(c.std) + ' · skew ' + formatJson(c.skew);
        } else if (c.top_values && Object.keys(c.top_values).length) {
            vis = svgBarChart(Object.entries(c.top_values).map(([k, v]) => ({ label: k, value: v })),
                { color: '#6366f1' });
            stats = c.unique_values + ' unique';
        } else {
            stats = (c.unique_values != null ? c.unique_values : '?') + ' unique';
        }
        html += `<tr><td>${escapeHtml(c.column)}</td><td>${escapeHtml(c.dtype)}</td><td>${c.null_count}</td><td>${vis}</td><td class="hint">${stats}</td></tr>`;
    }
    html += '</tbody></table></div></div>';
    return html;
}

function renderDistCard(d) {
    if (!d || d.min == null) return '';
    let html = '<div class="health-summary">';
    for (const k of ['mean', 'median', 'std', 'min', 'max', 'skew']) {
        if (d[k] !== undefined) {
            html += `<div class="health-card"><div class="cell-label">${escapeHtml(k)}</div><div class="cell-value">${formatJson(d[k])}</div></div>`;
        }
    }
    html += '</div>';
    return html + distBand(d);
}

// Target analysis: numeric distribution band OR categorical value bars, plus
// correlation-with-target horizontal bars.
function renderTargetAnalysis(ta) {
    if (!ta) return '';
    if (ta.error) return `<p class="hint">${escapeHtml(ta.error)}</p>`;
    let html = `<strong>Target:</strong> ${escapeHtml(ta.target_column)} <span class="hint">(${escapeHtml(ta.dtype)}; ${ta.null_count} null)</span><br>`;
    const d = ta.distribution || {};
    if (d.min !== undefined && d.max !== undefined) {
        html += renderDistCard(d);
    } else if (typeof d === 'object' && Object.keys(d).length) {
        html += svgBarChart(Object.entries(d).map(([k, v]) => ({ label: k, value: v })),
            { title: 'Target Class Distribution', color: '#8b5cf6' });
        if (ta.unique_classes !== undefined) html += '<p class="hint">' + ta.unique_classes + ' unique classes</p>';
    }
    const corr = ta.correlations_with_numeric_features;
    if (corr && Object.keys(corr).length) {
        const items = Object.entries(corr)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 12)
            .map(([k, v]) => ({ label: k, value: v }));
        html += svgBarChart(items, { title: 'Correlation with Target', color: '#0ea5e9' });
    }
    return html;
}

// Summary: stat cards + correlation heatmap + column insights + target analysis.
function renderSummary(s) {
    if (!s) return '';
    const b = s.basic_stats || {};
    const num = (b.numeric_columns || []).length;
    const cat = (b.categorical_columns || []).length;
    let html = '<div class="health-summary">';
    html += `<div class="health-card"><div class="cell-label">Rows</div><div class="cell-value">${b.rows}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Columns</div><div class="cell-value">${b.columns}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Numeric</div><div class="cell-value">${num}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Categorical</div><div class="cell-value">${cat}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Missing</div><div class="cell-value">${b.missing_values}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Duplicates</div><div class="cell-value">${b.duplicate_rows}</div></div>`;
    html += '</div>';
    if (s.correlations && !s.correlations.error) html += renderCorrelations(s.correlations);
    if (s.column_insights && s.column_insights.length) html += '<h4>Column Insights</h4>' + renderInsights(s.column_insights);
    if (s.target_analysis && !s.target_analysis.error) html += '<h4>Target Analysis</h4>' + renderTargetAnalysis(s.target_analysis);
    return html;
}

function renderValidation(v) {
    if (!v) return '';
    if (v.valid) {
        let h = `<p class="hint" style="color:var(--success)">✔ Valid ${escapeHtml(v.statement_type || '')} statement.</p>`;
        if (v.warnings && v.warnings.length) {
            h += '<h4>Warnings</h4><ul class="warning-list">' + v.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('') + '</ul>';
        }
        if (v.notes && v.notes.length) {
            h += '<h4>Notes</h4><ul class="warning-list">' + v.notes.map(n => `<li>${escapeHtml(n)}</li>`).join('') + '</ul>';
        }
        return h;
    }
    let h = `<p class="hint" style="color:var(--danger)">✖ Invalid statement</p>`;
    if (v.errors && v.errors.length) {
        h += '<ul class="warning-list">' + v.errors.map(e => `<li style="color:var(--danger)">${escapeHtml(e)}</li>`).join('') + '</ul>';
    }
    return h;
}

function renderAnomalyScatter(an, xf, yf) {
    if (!an) return '';
    const feats = an.features || [];
    if (feats.length < 2) return '<p class="hint">Need at least 2 numeric features to plot a scatter view.</p>';
    xf = xf || feats[0]; yf = yf || feats[1];
    const flagCol = (an.summary && an.summary.flag_column) || 'is_anomaly';
    const pts = (an.data || [])
        .filter(r => r[xf] != null && r[yf] != null && !isNaN(Number(r[xf])) && !isNaN(Number(r[yf])))
        .map(r => ({ x: Number(r[xf]), y: Number(r[yf]), flag: !!r[flagCol] }));
    if (!pts.length) return '<p class="hint">No plottable rows for the selected features.</p>';
    const W = 470, H = 280, P = { l: 52, r: 12, t: 12, b: 32 };
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xsp = (xmax > xmin) ? (xmax - xmin) : 1;
    const ysp = (ymax > ymin) ? (ymax - ymin) : 1;
    const sx = v => P.l + ((v - xmin) / xsp) * (W - P.l - P.r);
    const sy = v => P.t + (1 - (v - ymin) / ysp) * (H - P.t - P.b);
    const nAnom = pts.filter(p => p.flag).length;

    let body = `<line x1="${P.l}" y1="${H - P.b}" x2="${W - P.r}" y2="${H - P.b}" stroke="#d1d5db"/>` +
        `<line x1="${P.l}" y1="${P.t}" x2="${P.l}" y2="${H - P.b}" stroke="#d1d5db"/>`;
    for (const p of pts) {
        body += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${p.flag ? 4.5 : 2.5}" ` +
            `fill="${p.flag ? '#ef4444' : '#9ca3af'}" opacity="${p.flag ? 0.95 : 0.55}" ` +
            `title="${p.flag ? 'Anomaly' : 'Normal'} — ${escapeHtml(xf)}=${formatJson(p.x)}, ${escapeHtml(yf)}=${formatJson(p.y)}"/>`;
    }
    // Axis labels + min/max ticks
    body += `<text x="${P.l + (W - P.l - P.r) / 2}" y="${H - 8}" text-anchor="middle" font-size="11" fill="#6b7280">${escapeHtml(xf)}</text>`;
    body += `<text x="14" y="${P.t + (H - P.t - P.b) / 2}" text-anchor="middle" font-size="11" fill="#6b7280" transform="rotate(-90 14 ${P.t + (H - P.t - P.b) / 2})">${escapeHtml(yf)}</text>`;
    const legend = `<div class="scatter-legend"><span class="sw anom"></span>Anomaly (${nAnom}) <span class="sw norm"></span>Normal (${pts.length - nAnom})</div>`;
    return `<div class="chart"><div class="chart-title">Anomaly Scatter — ${escapeHtml(xf)} × ${escapeHtml(yf)}</div>${legend}` +
        `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" class="anom-svg">${body}</svg></div>`;
}

// Re-plot the anomaly scatter into its container using the selected features.
function plotAnomaly() {
    const an = window.__lastAnomaly;
    const box = document.getElementById('anom-scatter');
    if (!an || !box) return;
    const xf = (document.getElementById('anom-x') || {}).value;
    const yf = (document.getElementById('anom-y') || {}).value;
    box.innerHTML = renderAnomalyScatter(an, xf, yf);
}

function renderAnomaly(an) {
    if (!an) return '';
    window.__lastAnomaly = an;
    let html = `<div class="health-summary">`;
    html += `<div class="health-card"><div class="cell-label">Rows</div><div class="cell-value">${an.total}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Anomalies</div><div class="cell-value">${an.n_anomalies}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Rate</div><div class="cell-value">${(an.summary.anomaly_rate * 100).toFixed(1)}%</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Features</div><div class="cell-value">${an.features.length}</div></div>`;
    html += '</div>';

    if (an.importance && an.importance.length) {
        html += svgBarChart(an.importance.slice(0, 10).map(i => ({ label: i.feature, value: i.importance })),
            { title: 'Anomaly Feature Importance', color: '#ef4444' });
    }

    // Scatter/outlier view (Item 11): feature picker + canvas.
    const feats = an.features || [];
    if (feats.length >= 2) {
        const opts = (sel) => feats.map(f => `<option value="${escapeHtml(f)}" ${f === sel ? 'selected' : ''}>${escapeHtml(f)}</option>`).join('');
        html += '<div class="chart"><div class="chart-title">Outlier Scatter View</div><div class="form-row" style="margin-bottom:6px">' +
            `<label>X <select id="anom-x" onchange="plotAnomaly()">${opts(feats[0])}</select></label>` +
            `<label>Y <select id="anom-y" onchange="plotAnomaly()">${opts(feats[1] || feats[0])}</select></label></div>` +
            '<div id="anom-scatter"></div></div>';
        setTimeout(() => plotAnomaly(), 0);
    }

    const data = an.data || [];
    html += `<p class="hint">Flagged via the ${escapeHtml((an.summary.flag_column || 'is_anomaly'))} column (${an.n_anomalies} rows)</p>`;
    const cols = an.features.concat([an.summary.score_column, an.summary.flag_column]);
    html += renderTable(data.slice(0, 50), cols);
    return html;
}

// SHAP-style diverging bars: positive pushes prediction up, negative down. (Tier D item 14)
function renderContributionBars(contributions) {
    if (!contributions || !contributions.length) return '';
    const maxAbs = Math.max(...contributions.map(c => Math.abs(c.contribution || 0))) || 1;
    const halfW = 220;
    let rows = '';
    for (const c of contributions) {
        const v = c.contribution || 0;
        const w = Math.min(halfW, Math.abs(v) / maxAbs * halfW);
        const neg = v < 0;
        const start = neg ? 50 : 50 + (halfW - w);
        const fill = neg ? '#ef4444' : '#10b981';
        rows += `<div class="contrib-row" title="Contribution ${formatJson(v)}">` +
            `<span class="contrib-label">${escapeHtml(c.feature)}</span>` +
            `<span class="contrib-track">` +
            `<span class="contrib-bar ${neg ? 'neg' : 'pos'}" style="left:${start}px;width:${w}px;background:${fill}"></span>` +
            `<span class="contrib-zero" style="left:50px"></span></span>` +
            `<span class="contrib-val">${v > 0 ? '+' : ''}${formatJson(v)}</span></div>`;
    }
    return `<div class="chart"><div class="chart-title">Feature Contributions (diverging)</div>` +
        `<div class="contrib-hint"><span class="sw pos"></span>Pushes up <span class="sw neg"></span>Pushes down</div>${rows}</div>`;
}

function renderExplain(ex) {
    if (!ex) return '';
    let html = `<strong>${escapeHtml(ex.best_model)}</strong> — prediction: <strong>${escapeHtml(formatJson(ex.prediction))}</strong><br>`;
    html += `<p class="hint">Base <code>${escapeHtml(ex.target_column)}</code> = ${escapeHtml(formatJson(ex.base_prediction))}</p>`;
    if (ex.contributions && ex.contributions.length) {
        const sorted = ex.contributions.slice().sort((a, b) => Math.abs(b.contribution || 0) - Math.abs(a.contribution || 0));
        html += renderContributionBars(sorted.slice(0, 12));
        html += '<h4>Feature contributions (higher = pushes prediction up)</h4><div class="table-container"><table class="dataframe"><thead><tr><th>Feature</th><th>Kind</th><th>Value</th><th>Typical</th><th>Contribution</th></tr></thead><tbody>';
        for (const c of ex.contributions) {
            html += `<tr><td>${escapeHtml(c.feature)}</td><td>${escapeHtml(c.kind || '')}</td><td>${escapeHtml(formatJson(c.value))}</td><td>${escapeHtml(formatJson(c.typical))}</td><td>${escapeHtml(formatJson(c.contribution))}</td></tr>`;
        }
        html += '</tbody></table></div>';
    }
    return html;
}

// Explain a single prediction row (from a batch or typed JSON) into the viewer.
async function explainPredictRow(row) {
    if (!row) { showToast('No row to explain.', 'warning'); return; }
    const box = document.getElementById('predict-explain-result');
    if (box) box.innerHTML = '<p class="hint">Explaining…</p>';
    try {
        const res = await api('/api/explain/prediction', 'POST', { data: row, top_n: 8 });
        if (box) box.innerHTML = renderExplain(res.explanation);
        else document.getElementById('explain-result').innerHTML = renderExplain(res.explanation);
    } catch (err) {
        showToast(err.message, 'error');
        if (box) box.innerHTML = '';
    }
}

function renderWhatIf(wi) {
    if (!wi) return '';
    const arrow = wi.changed ? '→' : '=';
    return `<strong>What-if: ${escapeHtml(wi.feature)} = ${escapeHtml(formatJson(wi.value))}</strong><br>` +
        `<p>Base prediction: <code>${escapeHtml(formatJson(wi.base_prediction_label))}</code> ${arrow} New: <code>${escapeHtml(formatJson(wi.new_prediction_label))}</code></p>` +
        `<p class="hint">${wi.changed ? 'The prediction changed.' : 'The prediction did not change.'}</p>`;
}

function renderPsiBars(features, maxPsi) {
    const items = (features || []).filter(f => f.psi != null).sort((a, b) => (b.psi || 0) - (a.psi || 0));
    if (!items.length) return '';
    const max = maxPsi || Math.max(...items.map(f => f.psi)) || 1;
    const cap = max || 1;
    let rows = '';
    for (const f of items) {
        const statusCls = f.status === 'drift' ? 'drift' : (f.status === 'moderate' ? 'moderate' : 'stable');
        const pct = Math.min(100, Math.round((f.psi / (cap || 1)) * 100));
        rows += `<div class="drift-row" title="PSI ${f.psi} — ${escapeHtml(f.status)}${f.message ? ' · ' + escapeHtml(f.message) : ''}">` +
            `<span class="drift-label">${escapeHtml(f.feature)}</span>` +
            `<span class="drift-track"><span class="drift-fill ${statusCls}" style="width:${pct}%"></span></span>` +
            `<span class="drift-val">${f.psi != null ? formatJson(f.psi) : '-'}</span>` +
            `<span class="drift-badge ${statusCls}">${escapeHtml(f.status)}</span></div>`;
    }
    return `<div class="chart"><div class="chart-title">PSI Distribution Shift by Feature</div>` +
        `<div class="drift-legend"><span class="sw stable"></span>Stable &lt; 0.1` +
        `<span class="sw moderate"></span>0.1–0.25` +
        `<span class="sw drift"></span>&gt; 0.25</div>` +
        rows + '</div>';
}

function renderDrift(dr) {
    if (!dr) return '';
    let html = `<div class="health-summary">`;
    html += `<div class="health-card"><div class="cell-label">Overall</div><div class="cell-value">${escapeHtml(dr.overall_status)}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Drifted</div><div class="cell-value">${dr.n_drifted}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Moderate</div><div class="cell-value">${dr.n_moderate}</div></div>`;
    html += `<div class="health-card"><div class="cell-label">Max PSI</div><div class="cell-value">${formatJson(dr.max_psi)}</div></div>`;
    html += '</div>';
    html += `<p class="hint">Reference: ${dr.rows_reference} rows vs live: ${dr.rows_live} rows. PSI &gt; ${dr.drift_threshold} = drift, ${dr.moderate_threshold}–${dr.drift_threshold} = moderate.</p>`;
    html += renderPsiBars(dr.features, dr.max_psi);
    html += '<div class="table-container"><table class="dataframe"><thead><tr><th>Feature</th><th>PSI</th><th>Status</th><th>Message</th></tr></thead><tbody>';
    for (const f of dr.features) {
        const cls = f.status === 'drift' ? 'var(--danger)' : (f.status === 'moderate' ? 'var(--warning)' : '');
        html += `<tr><td>${escapeHtml(f.feature)}</td><td>${f.psi != null ? formatJson(f.psi) : '-'}</td><td style="color:${cls || 'inherit'}">${escapeHtml(f.status)}</td><td>${f.message ? escapeHtml(f.message) : ''}</td></tr>`;
    }
    html += '</tbody></table></div>';
    return html;
}

// ============ Event handlers ============

// Connect
document.getElementById('btn-connect').addEventListener('click', async (e) => {
    const conn = document.getElementById('db-connection').value.trim();
    if (!conn) {
        showToast('Please enter a database connection string.', 'error');
        return;
    }
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/connect', 'POST', { connection: conn });
        showToast(res.message || 'Connected!', 'success');
        await refreshState();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// Upload database file
document.getElementById('btn-upload-db').addEventListener('click', async (e) => {
    const fileInput = document.getElementById('db-file');
    const file = fileInput.files[0];
    if (!file) {
        showToast('Please select a database file to upload.', 'error');
        return;
    }

    // Validate extension client-side
    const allowedExts = ['.db', '.sqlite', '.sqlite3'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!allowedExts.includes(ext)) {
        showToast(`Unsupported file type '${ext}'. Allowed: .db, .sqlite, .sqlite3`, 'error');
        return;
    }

    const _btn = e.currentTarget; setLoading(_btn, true);
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload-db', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (!res.ok && data.error) {
            throw new Error(data.error);
        }
        showToast(data.message || 'Database uploaded and connected!', 'success');
        fileInput.value = '';
        await refreshState();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// Load table
document.getElementById('btn-load-table').addEventListener('click', async (e) => {
    const table = document.getElementById('data-table-select').value;
    const limit = document.getElementById('data-limit').value || null;
    if (!table) {
        showToast('Please select a table.', 'error');
        return;
    }
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/load', 'POST', { table, limit });
        currentDataSource = table;
        const info = document.getElementById('data-info');
        info.className = 'info-box success';
        info.innerHTML = `<strong>Loaded table:</strong> ${escapeHtml(table)}<br><strong>Rows:</strong> ${res.rows}<br><strong>Columns:</strong> ${res.columns.join(', ')}`;
        document.getElementById('data-preview').innerHTML = renderTable(res.data, res.columns);
        showToast(`Loaded ${res.rows} rows from ${table}`, 'success');
        await refreshState();
        await autoSuggestTarget();
        saveUIState();
        try { await runQuickView(); } catch (e) { /* Quiet View Auto is best-effort */ }
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// Run query
document.getElementById('btn-run-query').addEventListener('click', async (e) => {
    const query = document.getElementById('data-query').value.trim();
    if (!query) {
        showToast('Please enter a SQL query.', 'error');
        return;
    }
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/load-query', 'POST', { query });
        currentDataSource = null;
        const info = document.getElementById('data-info');
        info.className = 'info-box success';
        info.innerHTML = `<strong>Query:</strong> <code>${escapeHtml(query)}</code><br><strong>Rows:</strong> ${res.rows}<br><strong>Columns:</strong> ${res.columns.join(', ')}`;
        document.getElementById('data-preview').innerHTML = renderTable(res.data, res.columns);
        showToast(`Loaded ${res.rows} rows from query`, 'success');
        await refreshState();
        await autoSuggestTarget();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// Overview
document.getElementById('btn-overview').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/overview');
        const overview = res.overview;
        let html = `<strong>Connection:</strong> ${escapeHtml(overview.connection_string)}<br><strong>Tables:</strong> ${overview.table_count}<br><br>`;
        for (const t of overview.tables) {
            html += `<strong>${escapeHtml(t.name)}</strong> (${t.row_count} rows)<br>`;
            html += `Columns: ${t.columns.map(c => escapeHtml(c.name)).join(', ')}<br>`;
            if (t.foreign_keys && t.foreign_keys.length > 0) {
                html += `FKs: ${t.foreign_keys.map(fk => `${fk.constrained_columns} → ${fk.referred_table}.${fk.referred_columns}`).join(', ')}<br>`;
            }
            html += '<br>';
        }
        document.getElementById('data-info').className = 'info-box';
        document.getElementById('data-info').innerHTML = html;
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// LLM check
document.getElementById('btn-llm-check').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/llm/check');
        if (res.status.available) {
            setStatus('llm-status', `LLM: ${res.status.detail}`, 'connected');
            showToast(`LLM available: ${res.status.detail}`, 'success');
        } else {
            setStatus('llm-status', 'LLM: Unavailable', 'error');
            showToast(`LLM unavailable: ${res.status.detail}`, 'warning');
        }
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// ============ Preprocessing ============

document.getElementById('btn-add-op').addEventListener('click', () => {
    const op = document.getElementById('preprocess-op').value;
    const params = document.getElementById('preprocess-params').value.trim();
    const opObj = { op };
    if (params) {
        // Parse "key=value, key2=value2"
        params.split(',').forEach(pair => {
            const [k, v] = pair.split('=').map(s => s.trim());
            if (k && v) {
                // Try to parse numbers and booleans
                if (v === 'true') opObj[k] = true;
                else if (v === 'false') opObj[k] = false;
                else if (!isNaN(Number(v))) opObj[k] = Number(v);
                else opObj[k] = v;
            }
        });
    }
    preprocessQueue.push(opObj);
    renderPreprocessQueue();
    document.getElementById('preprocess-params').value = '';
});

function renderPreprocessQueue() {
    const queue = document.getElementById('preprocess-queue');
    if (preprocessQueue.length === 0) {
        queue.innerHTML = '';
        return;
    }
    queue.innerHTML = preprocessQueue.map((op, i) => `
        <div class="op-item">
            <span>${escapeHtml(JSON.stringify(op))}</span>
            <button class="op-remove" data-index="${i}">✕</button>
        </div>
    `).join('');

    // Attach remove handlers
    document.querySelectorAll('.op-remove').forEach(el => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.index);
            preprocessQueue.splice(idx, 1);
            renderPreprocessQueue();
        });
    });
}

document.getElementById('btn-apply-preprocess').addEventListener('click', async (e) => {
    if (preprocessQueue.length === 0) {
        showToast('No preprocessing operations queued.', 'warning');
        return;
    }
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/preprocess', 'POST', { operations: preprocessQueue });
        document.getElementById('preprocess-result').innerHTML = renderJson(res.summary);
        showToast(`Applied ${res.summary.total_operations} operations`, 'success');
        await refreshState();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-clear-preprocess').addEventListener('click', () => {
    preprocessQueue = [];
    renderPreprocessQueue();
    document.getElementById('preprocess-result').innerHTML = '';
});

document.getElementById('btn-llm-suggest-preprocess').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/llm/suggest-preprocessing', 'POST', {});
        const result = res.result;
        if (result.error) {
            showToast(result.error, 'error');
            return;
        }
        const ops = result.operations || [];
        preprocessQueue = ops.map(op => typeof op === 'string' ? { op } : op);
        renderPreprocessQueue();
        document.getElementById('preprocess-result').innerHTML = renderJson(result);
        showToast(`LLM suggested ${ops.length} operations.`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// ============ Analysis ============

document.getElementById('btn-analyze').addEventListener('click', async (e) => {
    const type = document.getElementById('analyze-type').value;
    const target = document.getElementById('analyze-target').value.trim() || null;
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/analyze', 'POST', { type, target_column: target });
        lastAnalyzeResult = res.analysis;
        const out = document.getElementById('analyze-result');
        if (type === 'health') {
            out.innerHTML = renderHealthReport(res.analysis);
        } else if (type === 'correlations') {
            out.innerHTML = renderCorrelations(res.analysis);
        } else if (type === 'insights') {
            out.innerHTML = renderInsights(res.analysis);
        } else if (type === 'target') {
            out.innerHTML = renderTargetAnalysis(res.analysis);
        } else if (type === 'summary') {
            out.innerHTML = renderSummary(res.analysis);
        } else {
            out.innerHTML = renderJson(res.analysis);
        }
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// ============ Training (async job with progress + cancel) ============

let trainPoller = null;

function renderTrainProgress(status, jobId) {
    const box = document.getElementById('train-progress');
    const p = status.progress || {};
    const total = p.total || 1;
    const pct = Math.round(((p.current || 0) / total) * 100);
    const cancelled = status.status === 'cancelled';
    const done = status.status === 'done' || status.status === 'error' || cancelled;
    const tuning = p.phase === 'tuning';

    let html = '<div class="progress-track">';
    if (tuning && p.current === 0) {
        html += '<div class="progress-fill tuning" style="width:100%"></div>';
    } else {
        html += '<div class="progress-fill" style="width:' + pct + '%"></div>';
    }
    html += '</div>';

    let label;
    if (cancelled) label = 'Training cancelled.';
    else if (tuning) label = `Tuning ${escapeHtml(p.model || 'best model')} hyperparameters…`;
    else label = 'Evaluating ' + pct + '% — ' + escapeHtml(p.model || 'starting…');
    html += '<p class="progress-label">' + label + '</p>';

    if (tuning && p.best_params && Object.keys(p.best_params).length) {
        html += '<p class="progress-label">Best params: <code>' + escapeHtml(JSON.stringify(p.best_params)) + '</code></p>';
    }

    if (p.scores && Object.keys(p.scores).length && !tuning) {
        html += '<div class="progress-scores">' + Object.entries(p.scores)
            .map(([k, v]) => `<span class="progress-score">${escapeHtml(k)}: ${formatJson(v)}</span>`)
            .join('') + '</div>';
    }

    if (status.status === 'error') {
        html += `<p class="progress-label" style="color:var(--danger)">Error: ${escapeHtml(status.error || 'unknown')}</p>`;
    }

    if (!done) {
        html += `<button class="btn btn-danger btn-sm" id="btn-cancel-train">Cancel Training</button>`;
    }
    box.innerHTML = html;

    const cancelBtn = box.querySelector('#btn-cancel-train');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', async () => {
            cancelBtn.disabled = true;
            cancelBtn.textContent = 'Cancelling…';
            try {
                await api(`/api/train/cancel/${jobId}`, 'POST');
            } catch (err) {
                showToast(err.message, 'error');
            }
        });
    }
}

async function startTraining(target, taskType, tuning, nJobs) {
    const _btn = document.getElementById('btn-train');
    setLoading(_btn, true);
    document.getElementById('train-result').innerHTML = '';
    const progressBox = document.getElementById('train-progress');
    progressBox.innerHTML = '<div class="progress-track"><div class="progress-fill" style="width:0%"></div></div><p class="progress-label">Starting training…</p>';

    let jobId = null;
    try {
        const payload = { target_column: target, task_type: taskType, tuning };
        if (nJobs) payload.n_jobs = nJobs;
        const earlyStop = getTrainEarlyStop();
        if (earlyStop) payload.early_stop = earlyStop;
        const res = await api('/api/train', 'POST', payload);
        jobId = res.job_id;
    } catch (err) {
        setLoading(_btn, false);
        progressBox.innerHTML = '';
        showToast(err.message, 'error');
        return;
    }

    if (trainPoller) clearInterval(trainPoller);
    trainPoller = setInterval(async () => {
        let status;
        try {
            const r = await fetch(`/api/train/status/${jobId}`);
            status = await r.json();
        } catch (err) {
            clearInterval(trainPoller);
            trainPoller = null;
            setLoading(_btn, false);
            showToast(err.message, 'error');
            return;
        }

        renderTrainProgress(status, jobId);

        if (status.status === 'done') {
            clearInterval(trainPoller);
            trainPoller = null;
            setLoading(_btn, false);
            document.getElementById('train-result').innerHTML = renderTrainResults(status.training);
            lastTrainResult = status.training;
            showToast(`Trained ${status.training.best_model} (${status.training.task_type})`, 'success');
            await refreshState();
        } else if (status.status === 'error') {
            clearInterval(trainPoller);
            trainPoller = null;
            setLoading(_btn, false);
            showToast(status.error || 'Training failed.', 'error');
        } else if (status.status === 'cancelled') {
            clearInterval(trainPoller);
            trainPoller = null;
            setLoading(_btn, false);
            showToast('Training cancelled.', 'warning');
        }
    }, 700);
}

document.getElementById('btn-train').addEventListener('click', () => {
    const target = document.getElementById('train-target').value.trim();
    const taskType = document.getElementById('train-task-type').value || null;
    const tuning = document.getElementById('train-tuning').value || 'off';
    if (!target) {
        showToast('Please enter a target column.', 'error');
        return;
    }
    startTraining(target, taskType, tuning, getTrainNJobs());
});

document.getElementById('btn-auto-train').addEventListener('click', () => runAutoTrain());

// ============ Prediction ============

function renderPredictions(res, containerId, filenameBase) {
    lastPredictions = res.predictions;
    lastPredictColumns = res.columns;
    const box = document.getElementById(containerId);
    window._lastPredictRows = res.predictions;
    let html = renderPredictRowsWithExplain(res.predictions, res.columns);
    html += `<p class="hint">${res.rows} prediction(s). Use "Export Predictions (CSV)" below to download. Click "Explain" on a row to see what drives that prediction.</p>`;
    box.innerHTML = html;
    window._lastPredictFilename = filenameBase;
}

// Render a prediction table with a per-row "Explain" column (Tier D item 14).
function renderPredictRowsWithExplain(data, columns) {
    if (!data || data.length === 0) return '<p class="hint">No data to display.</p>';
    const cols = columns || Object.keys(data[0]);
    let html = '<table class="dataframe"><thead><tr>' +
        cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') +
        '<th></th></tr></thead><tbody>';
    data.forEach((row, i) => {
        html += '<tr>' + cols.map(c => `<td>${escapeHtml(formatJson(row[c]))}</td>`).join('') +
            `<td><button class="btn btn-outline btn-sm js-explain-row" data-idx="${i}" title="Explain this prediction">Explain</button></td></tr>`;
    });
    html += '</tbody></table>';
    return html;
}

// Delegated handler: per-row Explain button in prediction tables.
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.js-explain-row');
    if (!btn) return;
    const rows = window._lastPredictRows || [];
    const row = rows[parseInt(btn.dataset.idx, 10)];
    if (!row) return;
    explainPredictRow(row);
});

document.getElementById('btn-predict').addEventListener('click', async (e) => {
    const dataText = document.getElementById('predict-data').value.trim();
    if (!dataText) {
        showToast('Please enter prediction data as JSON.', 'error');
        return;
    }
    let data;
    try {
        data = JSON.parse(dataText);
    } catch (e) {
        showToast('Invalid JSON. Please check your input.', 'error');
        return;
    }
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/predict', 'POST', { data });
        renderPredictions(res, 'predict-result', 'predictions.csv');
        showToast(`Made ${res.rows} prediction(s)`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// ---------- Batch prediction ----------

document.getElementById('btn-predict-table').addEventListener('click', async (e) => {
    const table = document.getElementById('predict-table-select').value;
    if (!table) {
        showToast('Please select a table.', 'error');
        return;
    }
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/predict/table', 'POST', { table });
        renderPredictions(res, 'batch-predict-result', `${table}_predictions.csv`);
        showToast(`Predicted ${res.rows} rows from ${table}`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-predict-current').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/predict/current', 'POST');
        renderPredictions(res, 'batch-predict-result', 'loaded_data_predictions.csv');
        showToast(`Predicted ${res.rows} rows from loaded data`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-predict-csv').addEventListener('click', async (e) => {
    const fileInput = document.getElementById('predict-csv');
    const file = fileInput.files[0];
    if (!file) {
        showToast('Please select a CSV file.', 'error');
        return;
    }
    const _btn = e.currentTarget; setLoading(_btn, true);
    const formData = new FormData();
    formData.append('file', file);
    try {
        const res = await fetch('/api/predict/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok && data.error) throw new Error(data.error);
        renderPredictions(data, 'batch-predict-result', `${file.name.replace(/\.csv$/i, '')}_predictions.csv`);
        fileInput.value = '';
        showToast(`Predicted ${data.rows} rows from CSV`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-export-predictions').addEventListener('click', () => {
    if (!lastPredictions) {
        showToast('No predictions to export yet.', 'warning');
        return;
    }
    downloadCsv(window._lastPredictFilename || 'predictions.csv', lastPredictions);
});

// ---------- Export loaded / preprocessed data ----------

document.getElementById('btn-export-data').addEventListener('click', () => {
    window.location.href = '/api/export/csv';
});

document.getElementById('btn-export-preprocessed').addEventListener('click', () => {
    window.location.href = '/api/export/csv';
});

// ============ LLM ============

document.getElementById('btn-llm-enable').addEventListener('click', async (e) => {
    const url = document.getElementById('llm-url').value.trim() || 'http://localhost:1234/v1';
    const model = document.getElementById('llm-model').value.trim() || null;
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/llm/enable', 'POST', { base_url: url, model });
        if (res.status.available) {
            setStatus('llm-status', `LLM: ${res.status.detail}`, 'connected');
            showToast(`LLM enabled: ${res.status.detail}`, 'success');
        } else {
            setStatus('llm-status', 'LLM: Unavailable', 'error');
            showToast(`LLM enabled but unavailable: ${res.status.detail}`, 'warning');
        }
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-llm-sql').addEventListener('click', async (e) => {
    const question = document.getElementById('llm-question').value.trim();
    if (!question) {
        showToast('Please enter a question.', 'error');
        return;
    }
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/llm/sql', 'POST', { question });
        const result = res.result;
        if (result.error) {
            showToast(result.error, 'error');
            return;
        }
        let html = `<strong>Generated SQL:</strong><br><code>${escapeHtml(result.sql)}</code>`;
        if (result.explanation) {
            html += `<br><br><strong>Explanation:</strong><br>${escapeHtml(result.explanation)}`;
        }
        if (result.validation) {
            html += `<br><br><strong>Validation:</strong><br>${renderValidation(result.validation)}`;
        }
        document.getElementById('llm-result').innerHTML = html;
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-llm-suggest-target').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/llm/suggest-target', 'POST', {});
        const result = res.result;
        if (result.error) {
            showToast(result.error, 'error');
            return;
        }
        let html = `<strong>Target Column:</strong> ${escapeHtml(result.target_column)}<br>`;
        html += `<strong>Task Type:</strong> ${escapeHtml(result.task_type)}<br>`;
        html += `<strong>Reasoning:</strong> ${escapeHtml(result.reasoning)}<br>`;
        html += `<strong>Source:</strong> ${escapeHtml(result.source || 'llm')}`;
        document.getElementById('llm-result').innerHTML = html;
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-llm-explain-results').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    try {
        const res = await api('/api/llm/explain-results', 'POST', {});
        const result = res.result;
        if (result.error) {
            showToast(result.error, 'error');
            return;
        }
        let html = `<div class="llm-markdown"><h3>LLM Interpretation</h3>`;
        if (result.explanation) {
            html += `<p>${escapeHtml(result.explanation)}</p>`;
        }
        if (result.highlights && result.highlights.length > 0) {
            html += '<h3>Highlights</h3><ul>';
            for (const hl of result.highlights) {
                html += `<li>${escapeHtml(hl)}</li>`;
            }
            html += '</ul>';
        }
        html += '</div>';
        document.getElementById('llm-result').innerHTML = html;
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// ============ Model persistence ============

document.getElementById('btn-save-model').addEventListener('click', async (e) => {
    const _btn = e.currentTarget;
    const path = document.getElementById('model-path').value.trim() || 'model.joblib';
    setLoading(_btn, true);
    try {
        const res = await api('/api/save-model', 'POST', { path });
        const out = document.getElementById('model-persistence-result');
        out.className = 'info-box success';
        out.innerHTML = `<strong>Model saved.</strong><br>File: <code>${escapeHtml(res.saved_path)}</code><br>Use "Download" to get the portable .joblib file.`;
        showToast(`Model saved to ${res.saved_path}`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-download-model').addEventListener('click', () => {
    const path = document.getElementById('model-path').value.trim() || 'model.joblib';
    window.location.href = `/api/model/download?path=${encodeURIComponent(path)}`;
});

document.getElementById('btn-load-model').addEventListener('click', async (e) => {
    const _btn = e.currentTarget;
    const path = document.getElementById('model-load-path').value.trim();
    if (!path) {
        showToast('Please enter a path to a model file.', 'error');
        return;
    }
    setLoading(_btn, true);
    try {
        const res = await api('/api/load-model', 'POST', { path });
        const info = res.model_info;
        let html = `<strong>Model loaded:</strong> ${escapeHtml(info.best_model)}<br>`;
        html += `<strong>Task:</strong> ${escapeHtml(info.task_type)}<br>`;
        html += `<strong>Target:</strong> ${escapeHtml(info.target_column)}<br>`;
        html += `<strong>CV Score:</strong> ${formatJson(info.best_cv_score)}<br>`;
        html += renderMetrics(info.test_metrics);
        document.getElementById('train-result').innerHTML = html;
        showToast(`Model loaded: ${info.best_model}`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

document.getElementById('btn-upload-model').addEventListener('click', async (e) => {
    const _btn = e.currentTarget;
    const fileInput = document.getElementById('model-file');
    const file = fileInput.files[0];
    if (!file) {
        showToast('Please select a model file to upload.', 'error');
        return;
    }

    const allowedExts = ['.joblib', '.pkl', '.pickle'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!allowedExts.includes(ext)) {
        showToast(`Unsupported file type '${ext}'. Allowed: .joblib, .pkl, .pickle`, 'error');
        return;
    }

    setLoading(_btn, true);
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/api/upload-model', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (!res.ok && data.error) {
            throw new Error(data.error);
        }
        const info = data.model_info;
        let html = `<strong>Model loaded (uploaded):</strong> ${escapeHtml(info.best_model)}<br>`;
        html += `<strong>Task:</strong> ${escapeHtml(info.task_type)}<br>`;
        html += `<strong>Target:</strong> ${escapeHtml(info.target_column)}<br>`;
        html += `<strong>CV Score:</strong> ${formatJson(info.best_cv_score)}<br>`;
        html += renderMetrics(info.test_metrics);
        document.getElementById('train-result').innerHTML = html;
        const out = document.getElementById('model-persistence-result');
        out.className = 'info-box success';
        out.innerHTML = `<strong>Model loaded from uploaded file:</strong> ${escapeHtml(file.name)}`;
        fileInput.value = '';
        showToast(`Model loaded: ${info.best_model}`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(_btn, false);
    }
});

// ============ Schema tab (SQL-specific features) ============
function renderErDiagram(o, rels){
  const ts=(o&&o.tables)||[]; if(!ts.length) return '<p class="hint">No tables.</p>';
  const bW=220,bH=70,gap=40,rGap=60,cols=Math.max(1,Math.ceil(Math.sqrt(ts.length)));
  const pos={}; ts.forEach((t,i)=>{const r=Math.floor(i/cols),c=i%cols;pos[t.name]={x:30+c*(bW+gap),y:40+r*(bH+rGap)};});
  const W=Math.max(780,cols*(bW+gap)+30),H=Math.max(220,Math.ceil(ts.length/cols)*(bH+rGap)+60);
  let s=`<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">`;
  (rels||[]).forEach(e=>{const a=pos[e.from_table],b=pos[e.to_table];if(!a||!b)return;
    const x1=a.x+bW,y1=a.y+bH/2,x2=b.x,y2=b.y+bH/2,m=(x1+x2)/2;
    const lbl=((e.from_columns||[])[0]||'')+'→'+((e.to_columns||[])[0]||'');
    s+=`<path d="M ${x1} ${y1} C ${m} ${y1}, ${m} ${y2}, ${x2} ${y2}" fill="none" stroke="#4f46e5" stroke-width="1.5" marker-end="url(#arr)"></path>`;
    s+=`<text x="${m}" y="${(y1+y2)/2-4}" font-size="10" fill="#6b7280" text-anchor="middle">${escapeHtml(lbl)}</text>`;});
  ts.forEach(t=>{const p=pos[t.name];const ct=(t.columns||[]).map(c=>c.name||c).join(', ');
    const lab=t.name.length>22?t.name.slice(0,20)+'…':t.name, disp=ct.length>40?ct.slice(0,38)+'…':ct;
    s+=`<rect x="${p.x}" y="${p.y}" width="${bW}" height="${bH}" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.5"></rect>`;
    s+=`<text x="${p.x+10}" y="${p.y+22}" font-size="13" font-weight="600" fill="#1f2937">${escapeHtml(lab)}</text>`;
    s+=`<text x="${p.x+10}" y="${p.y+40}" font-size="10" fill="#6b7280">${t.row_count??''} rows</text>`;
    s+=`<text x="${p.x+10}" y="${p.y+56}" font-size="10" fill="#4b5563">${escapeHtml(disp)}</text>`;});
  s=s.replace('<svg ','<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#4f46e5"></path></marker></defs><svg ')+'</svg>';
  return `<div class="chart">${s}</div>`;
}
document.getElementById('btn-render-er').addEventListener('click',async e=>{const _b=e.currentTarget;setLoading(_b,true);
  try{const[o,r]=await Promise.all([api('/api/overview'),api('/api/relationships')]);
    document.getElementById('er-diagram').innerHTML=renderErDiagram(o.overview,r.relationships);}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});
document.getElementById('btn-column-types').addEventListener('click',async e=>{const t=document.getElementById('schema-table-select').value;if(!t)return showToast('Select a table.','error');
  const _b=e.currentTarget;setLoading(_b,true);
  try{const res=await api(`/api/columns/${encodeURIComponent(t)}/types`);
    let h='<div class="table-container"><table class="dataframe"><thead><tr><th>Column</th><th>SQL Type</th><th>Semantic</th><th>PK</th><th>FK</th><th>Null</th></tr></thead><tbody>';
    res.columns.forEach(c=>{h+=`<tr><td>${escapeHtml(c.name)}</td><td>${escapeHtml(c.type)}</td><td><span class="sem-tag">${escapeHtml(c.semantic_type)}</span></td><td>${c.primary_key?'✓':''}</td><td>${c.foreign_key?'✓':''}</td><td>${c.nullable}</td></tr>`;});
    document.getElementById('column-types-result').innerHTML=h+'</tbody></table></div>';}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});
function selectedJoinTables(){return Array.from(document.getElementById('join-tables').selectedOptions).map(o=>o.value);}
document.getElementById('btn-build-join').addEventListener('click',async e=>{const ts=selectedJoinTables();if(ts.length<2)return showToast('Select ≥2 tables.','error');
  const _b=e.currentTarget;setLoading(_b,true);
  try{const res=await api('/api/query/build-join','POST',{tables:ts,join_type:document.getElementById('join-type').value});
    document.getElementById('join-sql').innerHTML=`<code>${escapeHtml(res.query)}</code>`;}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});
document.getElementById('btn-load-join').addEventListener('click',async e=>{const ts=selectedJoinTables();if(ts.length<2)return showToast('Select ≥2 tables.','error');
  const _b=e.currentTarget;setLoading(_b,true);
  try{const res=await api('/api/load-join','POST',{tables:ts,join_type:document.getElementById('join-type').value});
    document.getElementById('join-result').innerHTML=`<strong>Loaded ${res.rows} rows × ${res.columns.length}</strong><br>`+renderTable(res.data||[],res.columns);}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});
document.getElementById('btn-load-sample').addEventListener('click',async e=>{const t=document.getElementById('sample-table-select').value;if(!t)return showToast('Select a table.','error');
  const _b=e.currentTarget;setLoading(_b,true);
  try{const res=await api('/api/load-sample','POST',{table:t,limit:document.getElementById('sample-limit').value||null});
    document.getElementById('sample-result').innerHTML=`<strong>Loaded sample: ${res.rows} rows</strong><br>`+renderTable(res.data||[],res.columns);}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});

// ----- Schema tab: saved queries, profiles, settings -----
async function refreshSavedQueries(){
  try{const res=await api('/api/query/list');const qs=res.queries||[];const el=document.getElementById('saved-query-list');
    if(!qs.length){el.innerHTML='<p class="hint">No saved queries yet.</p>';return;}
    let h='<div class="table-container"><table class="dataframe"><thead><tr><th>Name</th><th>Query</th><th></th></tr></thead><tbody>';
    qs.forEach(q=>{h+=`<tr><td>${escapeHtml(q.name)}</td><td><code>${escapeHtml((q.query||'').slice(0,60))}</code></td><td><button class="btn btn-outline btn-sm js-rq" data-n="${escapeHtml(q.name)}">Run</button> <button class="btn btn-danger btn-sm js-dq" data-n="${escapeHtml(q.name)}">✕</button></td></tr>`;});
    el.innerHTML=h+'</tbody></table></div>';
    el.querySelectorAll('.js-rq').forEach(b=>b.addEventListener('click',async()=>{const nm=b.dataset.n;const q=(await api(`/api/query/get/${encodeURIComponent(nm)}`)).query;document.getElementById('query-store-sql').value=q.query;
      try{const r=await api('/api/load-query','POST',{query:q.query});showToast(`Ran '${nm}': ${r.rows} rows`,'success');}catch(err){showToast(err.message,'error');}}));
    el.querySelectorAll('.js-dq').forEach(b=>b.addEventListener('click',async()=>{try{await api(`/api/query/delete/${encodeURIComponent(b.dataset.n)}`,'POST');showToast('Deleted.','success');await refreshSavedQueries();}catch(err){showToast(err.message,'error');}}));
  }catch(e){}}
document.getElementById('btn-save-query').addEventListener('click',async e=>{const nm=document.getElementById('query-store-name').value.trim(),q=document.getElementById('query-store-sql').value.trim();
  if(!nm||!q)return showToast('Enter a name and query.','error');const _b=e.currentTarget;setLoading(_b,true);
  try{await api('/api/query/save','POST',{name:nm,query:q});showToast(`Saved '${nm}'`,'success');await refreshSavedQueries();}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});
document.getElementById('btn-run-querysaved').addEventListener('click',async e=>{const q=document.getElementById('query-store-sql').value.trim();if(!q)return showToast('Enter a query.','error');
  const _b=e.currentTarget;setLoading(_b,true);
  try{const r=await api('/api/load-query','POST',{query:q});showToast(`Ran query: ${r.rows} rows`,'success');document.getElementById('saved-query-list').innerHTML=`<strong>${r.rows}</strong> rows · columns: ${r.columns.join(', ')}`;}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});
async function refreshProfiles(){
  try{const res=await api('/api/profile/list');const names=res.profiles||[];
    ['profile-a','profile-b'].forEach(id=>{const s=document.getElementById(id);s.innerHTML=names.map(n=>`<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('');});
    document.getElementById('profile-status').innerHTML=names.length?`<p class="hint">${names.length} profile(s).</p>`:'<p class="hint">No profiles yet.</p>';}catch(e){}}
document.getElementById('btn-capture-profile').addEventListener('click',async e=>{const nm=document.getElementById('profile-name').value.trim()||('snap_'+Date.now());
  const _b=e.currentTarget;setLoading(_b,true);
  try{await api('/api/profile/capture','POST',{name:nm});showToast(`Captured '${nm}'`,'success');await refreshProfiles();}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});
document.getElementById('btn-compare-profiles').addEventListener('click',async e=>{const a=document.getElementById('profile-a').value,b=document.getElementById('profile-b').value;if(!a||!b)return showToast('Select two profiles.','error');
  const _b=e.currentTarget;setLoading(_b,true);
  try{const res=await api('/api/profile/compare','POST',{a,b});const d=res.diff;
    let h=`<strong>${escapeHtml(d.from)}</strong> vs <strong>${escapeHtml(d.to)}</strong><br>`;
    h+=(d.summary&&d.summary.length)?('<ul class="warning-list">'+d.summary.map(s=>`<li>${escapeHtml(s)}</li>`).join('')+'</ul>'):'<p class="hint">No changes detected.</p>';
    document.getElementById('profile-status').innerHTML=h;}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});
document.getElementById('btn-save-settings').addEventListener('click',async e=>{const _b=e.currentTarget;setLoading(_b,true);
  try{const p={read_only:document.getElementById('read-only-flag').checked};const to=document.getElementById('query-timeout').value;if(to)p.timeout=Number(to);
    const res=await api('/api/settings','POST',p);showToast(`Settings: read-only=${res.read_only}, timeout=${res.timeout??'none'}`,'success');}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});

// ----- Features 3-6: SQL validation / anomaly / explain / drift -----

document.getElementById('btn-validate-sql').addEventListener('click', async (e) => {
    const q = document.getElementById('data-query').value.trim();
    if (!q) { showToast('Enter a SQL query to validate.', 'error'); return; }
    const _b = e.currentTarget; setLoading(_b, true);
    try {
        const res = await api('/api/query/validate', 'POST', { query: q });
        document.getElementById('validate-result').innerHTML = renderValidation(res.validation);
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_b, false); }
});

document.getElementById('btn-anomaly-detect').addEventListener('click', async (e) => {
    const t = document.getElementById('anomaly-table').value;
    const cont = parseFloat(document.getElementById('anomaly-cont').value || '0.1');
    const _b = e.currentTarget; setLoading(_b, true);
    try {
        const body = { method: 'isolation_forest', contamination: cont };
        if (t) body.table = t;
        const res = await api('/api/anomaly/detect', 'POST', body);
        document.getElementById('anomaly-result').innerHTML = renderAnomaly(res.anomaly);
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_b, false); }
});

document.getElementById('btn-explain-pred').addEventListener('click', async (e) => {
    const dataText = document.getElementById('explain-data').value.trim();
    if (!dataText) { showToast('Enter a JSON record to explain.', 'error'); return; }
    let row;
    try { row = JSON.parse(dataText); } catch (err) { showToast('Invalid JSON.', 'error'); return; }
    const _b = e.currentTarget; setLoading(_b, true);
    try {
        const res = await api('/api/explain/prediction', 'POST', { data: row, top_n: 6 });
        document.getElementById('explain-result').innerHTML = renderExplain(res.explanation);
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_b, false); }
});

// Explain the first row of the last prediction batch (Tier D productivity).
document.getElementById('btn-explain-first').addEventListener('click', async () => {
    const rows = window._lastPredictRows || [];
    if (!rows.length) { showToast('No prediction batch to explain yet. Run predictions first.', 'warning'); return; }
    await explainPredictRow(rows[0]);
});

document.getElementById('btn-whatif').addEventListener('click', async (e) => {
    const dataText = document.getElementById('explain-data').value.trim();
    const feature = document.getElementById('whatif-feature').value.trim();
    const valueText = document.getElementById('whatif-value').value.trim();
    if (!dataText || !feature) { showToast('Provide a JSON record and the feature to change.', 'error'); return; }
    let row, value;
    try {
        row = JSON.parse(dataText);
        value = JSON.parse(valueText);
    } catch (err) { showToast('JSON data / value must be valid JSON.', 'error'); return; }
    const _b = e.currentTarget; setLoading(_b, true);
    try {
        const res = await api('/api/explain/whatif', 'POST', { data: row, feature, value });
        document.getElementById('explain-result').innerHTML += '<hr>' + renderWhatIf(res.whatif);
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_b, false); }
});

// ----- Batch what-if (perturb a feature across all loaded rows) -----
document.getElementById('btn-batch-whatif').addEventListener('click', async (e) => {
    const feature = document.getElementById('batchwi-feature').value.trim();
    const valueText = document.getElementById('batchwi-value').value.trim();
    if (!feature) { showToast('Provide the feature to change.', 'error'); return; }
    let value;
    try { value = JSON.parse(valueText); } catch (err) { showToast('Value must be valid JSON.', 'error'); return; }
    const _b = e.currentTarget; setLoading(_b, true);
    const box = document.getElementById('batch-whatif-result');
    box.innerHTML = '<p class="train-progress"><p class="progress-label">Applying what-if across all rows…</p></p>';
    try {
        const res = await api('/api/explain/batch-whatif', 'POST', { feature, value });
        const bw = res.batch_whatif || {};
        const s = bw.summary || {};
        let html = `<strong>Batch what-if:</strong> ${escapeHtml(s.feature)} = ${escapeHtml(formatJson(s.value))} across ${s.rows || 0} row(s)<br>`;
        if (s.task_type === 'classification') {
            html += `<strong>Rows changed:</strong> ${s.changed_rows || 0} (${formatJson(s.pct_changed)}%)<br>`;
        } else {
            html += `<strong>Base mean:</strong> ${formatJson(s.base_mean)} → <strong>New mean:</strong> ${formatJson(s.mean_prediction)}<br>`;
            html += `<strong>Δ mean prediction:</strong> ${formatJson(s.mean_delta)}<br>`;
            if (s.std_delta != null) html += `<strong>Δ std:</strong> ${formatJson(s.std_delta)}<br>`;
        }
        if (bw.sample && bw.sample.length) {
            html += `<p class="hint">Sample rows:</p><div class="table-container"><table class="dataframe"><thead><tr><th>Row</th><th>Base</th><th>New</th></tr></thead><tbody>` +
                bw.sample.map(x => `<tr><td>${x.index}</td><td>${escapeHtml(formatJson(x.base))}</td><td>${escapeHtml(formatJson(x.new))}</td></tr>`).join('') +
                '</tbody></table></div>';
        }
        box.innerHTML = html;
    } catch (err) { box.innerHTML = ''; showToast(err.message, 'error'); }
    finally { setLoading(_b, false); }
});

// ----- Re-predict a (changed) production table with drift + anomaly flags -----
document.getElementById('btn-repredict').addEventListener('click', async (e) => {
    const table = document.getElementById('repredict-table-select').value;
    if (!table) { showToast('Select a table to re-score.', 'error'); return; }
    const limit = document.getElementById('repredict-limit').value || null;
    const _b = e.currentTarget; setLoading(_b, true);
    const box = document.getElementById('repredict-result');
    box.innerHTML = '<p class="train-progress"><p class="progress-label">Re-scoring table…</p></p>';
    try {
        const res = await api('/api/repredict', 'POST', { table, limit });
        lastPredictions = res.predictions;
        lastPredictColumns = res.columns;
        window._lastPredictFilename = `${table}_rescored.csv`;
        let html = renderDrift(res.drift) + renderAnomaly(res.anomaly);
        html += `<p class="hint">${res.rows} rows re-scored from ${escapeHtml(res.source || table)}.</p>`;
        html += renderTable(res.predictions, res.columns);
        box.innerHTML = html;
        showToast(`Re-scored ${res.rows} rows from ${table}`, 'success');
    } catch (err) { box.innerHTML = ''; showToast(err.message, 'error'); }
    finally { setLoading(_b, false); }
});

document.getElementById('btn-monitor-capture').addEventListener('click', async (e) => {
    const _b = e.currentTarget; setLoading(_b, true);
    try {
        const res = await api('/api/monitor/capture', 'POST', {});
        document.getElementById('monitor-status').innerHTML =
            `<p class="hint" style="color:var(--success)">✔ Reference captured from ${res.rows} rows.</p>`;
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_b, false); }
});

document.getElementById('btn-monitor-check').addEventListener('click', async (e) => {
    const t = document.getElementById('monitor-table').value;
    if (!t) { showToast('Select a table to monitor.', 'error'); return; }
    const _b = e.currentTarget; setLoading(_b, true);
    try {
        const res = await api('/api/monitor/check', 'POST', { table: t });
        document.getElementById('monitor-result').innerHTML = renderDrift(res.drift);
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_b, false); }
});

// ----- Schema tab: relational deep features -----
document.getElementById('btn-synthesize').addEventListener('click',async e=>{const t=document.getElementById('deepfeat-table').value;if(!t)return showToast('Select a base table.','error');
  const _b=e.currentTarget;setLoading(_b,true);
  try{const res=await api('/api/synthesize','POST',{table:t,include_counts:document.getElementById('deepfeat-counts').checked,include_aggregates:document.getElementById('deepfeat-aggs').checked});
    const sum=res.summary||{};
    let h=`<strong>Generated ${res.columns.length} columns (${sum.features?sum.features.length:0} deep features)</strong><br>`;
    if(sum.features&&sum.features.length){h+='<h4>Aggregations</h4><div class="table-container"><table class="dataframe"><thead><tr><th>Feature</th><th>Child</th><th>Op</th><th>Column</th></tr></thead><tbody>';
      sum.features.forEach(f=>{h+=`<tr><td><code>${escapeHtml(f.name)}</code></td><td>${escapeHtml(f.child)}</td><td>${escapeHtml(f.op)}</td><td>${f.column?escapeHtml(f.column):'-'}</td></tr>`;});h+='</tbody></table></div>';}
    document.getElementById('deepfeat-result').innerHTML=h+renderTable(res.data||[],res.columns);}catch(err){showToast(err.message,'error');}finally{setLoading(_b,false);}});

// ----- Schema tab: experiments & champion -----
function fmtEpoch(ts) {
    if (!ts) return '';
    try { return new Date(ts * 1000).toLocaleString(); } catch (e) { return ''; }
}

// Leaderboard bars for experiments with a CV score, champion highlighted (Item 12).
function renderExperimentLeaderboard(ex, champId) {
    const scored = (ex || []).filter(x => x.best_cv_score != null);
    if (!scored.length) return '';
    const max = Math.max(...scored.map(x => x.best_cv_score)) || 1;
    const items = scored.slice().sort((a, b) => b.best_cv_score - a.best_cv_score).slice(0, 10);
    let html = '<div class="chart"><div class="chart-title">Model Leaderboard (CV Score)</div>';
    for (const it of items) {
        const isChamp = champId && it.id === champId;
        const pct = Math.max(2, (it.best_cv_score / max) * 100);
        const color = isChamp ? '#f59e0b' : '#4f46e5';
        html += `<div class="lb-row" title="${escapeHtml((it.target_column || '') + ' · ' + (it.best_model || ''))} | ${formatJson(it.best_cv_score)}">` +
            `<span class="lb-label">${isChamp ? '⭐ ' : ''}${escapeHtml(it.best_model || '-')} <span class="hint">(${escapeHtml(it.target_column || '-')})</span></span>` +
            `<span class="lb-track"><span class="lb-fill" style="width:${pct.toFixed(1)}%;background:${color}"></span></span>` +
            `<span class="lb-val">${formatJson(it.best_cv_score)}</span></div>`;
    }
    return html + '</div>';
}

async function refreshExperiments() {
  try {
    const res = await api('/api/experiments');
    const ex = res.experiments || [];
    const el = document.getElementById('experiments-list');
    const champ = (await api('/api/experiments/champion')).champion;
    const champId = champ ? champ.id : null;
    document.getElementById('btn-rollback-champion').style.display = champ ? '' : 'none';
    document.getElementById('champion-label').textContent = champ
      ? `Champion: ${champ.best_model} (${champ.target_column}) — ${champ.best_cv_score != null ? 'CV ' + champ.best_cv_score.toFixed(4) : ''}`
      : 'No champion set — click Promote on a run to auto-select the best.';
    if (!ex.length) { el.innerHTML = '<p class="hint">No experiments yet. Train a model to record one.</p>'; return; }
    el.innerHTML = renderExperimentLeaderboard(ex, champId);
    el.innerHTML += '<div class="table-container"><table class="dataframe"><thead><tr><th>Run</th><th>Target</th><th>Model</th><th>CV</th><th>Status</th><th></th></tr></thead><tbody>' +
      ex.map(x => `<tr class="${champId && x.id === champId ? 'ex-champ-row' : ''}"><td>${fmtEpoch(x.created)}</td>` +
        `<td>${champId && x.id === champId ? '⭐ ' : ''}${escapeHtml(x.target_column || '')}</td><td>${escapeHtml(x.best_model || '-')}</td><td>${x.best_cv_score != null ? x.best_cv_score.toFixed(4) : '-'}</td><td>${escapeHtml(x.status)}</td>` +
        `<td><button class="btn btn-outline btn-sm js-promote" data-id="${escapeHtml(x.id)}">Promote if Better</button></td></tr>`).join('') +
      '</tbody></table></div>';
    el.querySelectorAll('.js-promote').forEach(b => b.addEventListener('click', async (e) => {
      const _b = e.currentTarget; setLoading(_b, true);
      try {
        const r = await api(`/api/experiments/${encodeURIComponent(b.dataset.id)}/promote`, 'POST', {});
        const d = r.promote || r;
        const why = d.decision === 'promoted'
          ? (d.reason === 'first_champion' ? 'set as first champion' : 'beats current champion')
          : (d.reason === 'not_better' ? 'not better than current champion' : (d.reason === 'no_score' ? 'no CV score' : 'kept'));
        showToast(`Promote: ${d.decision} (${why})${d.reloaded ? ' — model reloaded' : ''}`, d.decision === 'promoted' ? 'success' : 'warning');
        refreshState({ silent: true });
        await refreshExperiments();
      } catch (err) { showToast(err.message, 'error'); }
      finally { setLoading(_b, false); }
    }));
  } catch (e) { }
}
document.getElementById('btn-refresh-experiments').addEventListener('click', () => refreshExperiments());
document.getElementById('btn-rollback-champion').addEventListener('click', async (e) => {
  const _b = e.currentTarget; setLoading(_b, true);
  try {
    const r = await api('/api/experiments/rollback', 'POST', {});
    const d = r.rollback || r;
    showToast(d.rolled_back ? `Rolled back to: ${d.champion && d.champion.best_model}${d.reloaded ? ' (model reloaded)' : ''}` : 'No previous champion to roll back to.', d.rolled_back ? 'success' : 'warning');
    if (d.rolled_back) refreshState({ silent: true });
    await refreshExperiments();
  } catch (err) { showToast(err.message, 'error'); }
  finally { setLoading(_b, false); }
});

// ----- Schema tab: reusable pipeline recipes -----
async function refreshRecipes() {
  try {
    const res = await api('/api/recipe/list');
    const recipes = res.recipes || [];
    const el = document.getElementById('recipe-list');
    if (!el) return;
    if (!recipes.length) { el.innerHTML = '<p class="hint">No recipes saved yet. Train a model, then click "Save Recipe" to capture it.</p>'; return; }
    el.innerHTML = '<div class="table-container"><table class="dataframe"><thead><tr><th>Name</th><th>Target</th><th>Task</th><th>Tuning</th><th></th></tr></thead><tbody>' +
      recipes.map(r => `<tr><td><code>${escapeHtml(r.name)}</code></td><td>${escapeHtml(r.target_column || '-')}</td><td>${escapeHtml(r.task_type || r.task || '-')}</td><td>${escapeHtml(r.tuning || '-')}</td>` +
        `<td><button class="btn btn-primary btn-sm js-recipe-apply" data-name="${escapeHtml(r.name)}">Apply</button> ` +
        `<button class="btn btn-outline btn-sm js-recipe-del" data-name="${escapeHtml(r.name)}">Delete</button></td></tr>`).join('') +
      '</tbody></table></div>';
    el.querySelectorAll('.js-recipe-apply').forEach(b => b.addEventListener('click', async (e) => {
      const _b = e.currentTarget; setLoading(_b, true);
      try {
        showToast(`Applying recipe '${b.dataset.name}' — training…`, 'info');
        await runOp('recipe_apply', { name: b.dataset.name });
        showToast(`Recipe '${b.dataset.name}' applied and trained successfully.`, 'success');
        await refreshState();
      } catch (err) { showToast(err.message, 'error'); }
      finally { setLoading(_b, false); }
    }));
    el.querySelectorAll('.js-recipe-del').forEach(b => b.addEventListener('click', async (e) => {
      const _b = e.currentTarget; setLoading(_b, true);
      try {
        await api(`/api/recipe/delete/${encodeURIComponent(b.dataset.name)}`, 'POST', {});
        showToast(`Deleted recipe '${b.dataset.name}'`, 'success');
        await refreshRecipes();
      } catch (err) { showToast(err.message, 'error'); }
      finally { setLoading(_b, false); }
    }));
  } catch (e) { }
}
document.getElementById('btn-recipe-save').addEventListener('click', async (e) => {
  const _b = e.currentTarget; setLoading(_b, true);
  try {
    const name = document.getElementById('recipe-name').value.trim();
    if (!name) { showToast('Enter a recipe name.', 'error'); return; }
    const p = {
      name,
      auto_prepare: document.getElementById('recipe-auto-prepare').checked,
    };
    const target = document.getElementById('train-target').value.trim();
    if (target) p.target_column = target;
    const taskType = document.getElementById('train-task-type').value;
    if (taskType) p.task_type = taskType;
    const tuning = document.getElementById('train-tuning').value;
    if (tuning) p.tuning = tuning;
    const nJobs = getTrainNJobs();
    if (nJobs) p.n_jobs = nJobs;
    await api('/api/recipe/save', 'POST', p);
    showToast(`Recipe '${name}' saved.`, 'success');
    await refreshRecipes();
  } catch (err) { showToast(err.message, 'error'); }
  finally { setLoading(_b, false); }
});
document.getElementById('btn-recipe-refresh').addEventListener('click', () => refreshRecipes());

// ============ Feature health (Tier 3 #11) ============

function renderFeatureHealth(h) {
    if (!h) return '';
    let html = '<p><strong>' + escapeHtml(h.warnings.join(' · ')) + '</strong></p>';

    if (h.near_constant && h.near_constant.length) {
        html += '<h4>Near-constant columns</h4><ul>';
        for (const c of h.near_constant) {
            html += '<li>' + escapeHtml(c.column) + ' (' + escapeHtml(c.reason) +
                (c.dominance != null ? ', ' + (c.dominance * 100).toFixed(1) + '% one value' : '') + ')</li>';
        }
        html += '</ul>';
    }
    if (h.high_cardinality && h.high_cardinality.length) {
        html += '<h4>High-cardinality / ID-like columns</h4><ul>';
        for (const c of h.high_cardinality) {
            html += '<li>' + escapeHtml(c.column) + ' (' + c.unique_values + ' unique)</li>';
        }
        html += '</ul>';
    }
    if (h.multicollinearity && h.multicollinearity.length) {
        html += '<h4>Highly-correlated pairs (multicollinearity)</h4>' +
            '<table class="dataframe"><thead><tr><th>Feature A</th><th>Feature B</th><th>Correlation</th><th>Suggested drop</th></tr></thead><tbody>';
        for (const m of h.multicollinearity) {
            html += '<tr><td>' + escapeHtml(m.feature_a) + '</td><td>' + escapeHtml(m.feature_b) + '</td>' +
                '<td>' + m.correlation.toFixed(3) + '</td><td>' + escapeHtml(m.suggested_drop) + '</td></tr>';
        }
        html += '</tbody></table>';
    }
    if (!h.near_constant.length && !h.high_cardinality.length && !h.multicollinearity.length) {
        html += '<p class="hint">No obvious filtering problems found — features look healthy.</p>';
    }
    return html;
}

document.getElementById('btn-feature-health').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    const target = document.getElementById('train-target').value.trim() || null;
    const thr = parseFloat(document.getElementById('feature-corr-threshold').value);
    try {
        const res = await api('/api/feature-health', 'POST', {
            target_column: target,
            corr_threshold: isNaN(thr) ? 0.95 : thr,
        });
        document.getElementById('feature-health-result').innerHTML = renderFeatureHealth(res.health);
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_btn, false); }
});

// ============ Result snapshots & diff (Tier 3 #10) ============

async function buildSnapshotPayload(kind) {
    if (kind === 'training') {
        try { return (await api('/api/model-info')).model || {}; }
        catch (e) { return {}; }
    }
    if (kind === 'analysis') return lastAnalyzeResult || {};
    if (kind === 'prediction') {
        return lastPredictions && lastPredictions.length
            ? { predictions: lastPredictions.slice(0, 100), count: lastPredictions.length }
            : {};
    }
    // manual: merge whatever is available
    const m = {};
    if (lastAnalyzeResult) m.analysis = lastAnalyzeResult;
    if (lastPredictions) m.prediction_count = lastPredictions.length;
    return m;
}

function renderSnapshotDiff(d) {
    if (!d) return '';
    let html = '<p><strong>Diff:</strong> ' + escapeHtml(d.a) + ' (' + escapeHtml(d.kind_a || '?') + ') vs ' +
        escapeHtml(d.b) + ' (' + escapeHtml(d.kind_b || '?') + ') — ' + d.change_count + ' change(s)</p>';
    if (!d.changes || !d.changes.length) {
        html += '<p class="hint">No differences in the compared fields.</p>';
        return html;
    }
    html += '<table class="dataframe"><thead><tr><th>Field</th><th>("' + escapeHtml(d.a) + '" → "' + escapeHtml(d.b) + '")</th></tr></thead><tbody>';
    for (const c of d.changes) {
        const field = escapeHtml(c.field);
        let val;
        if (c.kind === 'numeric') {
            val = '<span>' + formatJson(c.old) + ' → <strong>' + formatJson(c.new) + '</strong> (Δ ' + formatJson(c.delta) + ')</span>';
        } else {
            val = '<span>' + escapeHtml(String(c.old)) + ' → <strong>' + escapeHtml(String(c.new)) + '</strong></span>';
        }
        html += '<tr><td>' + field + '</td><td>' + val + '</td></tr>';
    }
    html += '</tbody></table>';
    return html;
}

async function refreshSnapshots() {
    let snaps = [];
    try { snaps = (await api('/api/snapshots')).snapshots || []; }
    catch (e) { snaps = []; }
    const listBox = document.getElementById('snapshot-list');
    if (!listBox) return;
    if (!snaps.length) {
        listBox.innerHTML = '<p class="hint">No snapshots yet. Run an analysis / training / prediction, then Save Current Result to compare before vs after (or two models).</p>';
    } else {
        listBox.innerHTML = '<table class="dataframe"><thead><tr><th>Name</th><th>Kind</th><th>Created</th><th></th></tr></thead><tbody>' +
            snaps.map(s => '<tr><td>' + escapeHtml(s.name) + '</td><td>' + escapeHtml(s.kind) + '</td><td>' + escapeHtml(s.created_iso || '') +
                '</td><td><button class="btn btn-outline btn-sm js-snap-del" data-n="' + escapeHtml(s.name) + '">Delete</button></td></tr>').join('') +
            '</tbody></table>';
        listBox.querySelectorAll('.js-snap-del').forEach(b => b.addEventListener('click', async () => {
            try {
                await api('/api/snapshots/' + encodeURIComponent(b.dataset.n), 'DELETE');
                showToast('Snapshot deleted.', 'success');
                await refreshSnapshots();
            } catch (err) { showToast(err.message, 'error'); }
        }));
    }
    const aSel = document.getElementById('snap-diff-a');
    const bSel = document.getElementById('snap-diff-b');
    if (aSel && bSel) {
        const opts = snaps.map(s => '<option value="' + escapeHtml(s.name) + '">' + escapeHtml(s.name) + '</option>').join('');
        aSel.innerHTML = opts;
        bSel.innerHTML = opts;
    }
}

document.getElementById('btn-snap-save').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    const kind = document.getElementById('snap-kind').value;
    const payload = await buildSnapshotPayload(kind);
    if (Object.keys(payload).length === 0 && kind !== 'manual') {
        setLoading(_btn, false);
        showToast('Nothing to snapshot yet — run an analysis, training, or prediction first.', 'warning');
        return;
    }
    try {
        await api('/api/snapshots', 'POST', {
            name: document.getElementById('snap-name').value.trim(),
            kind,
            payload,
        });
        showToast('Snapshot saved.', 'success');
        document.getElementById('snap-name').value = '';
        await refreshSnapshots();
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_btn, false); }
});

document.getElementById('btn-snap-refresh').addEventListener('click', () => refreshSnapshots());

document.getElementById('btn-snap-diff').addEventListener('click', async (e) => {
    const _btn = e.currentTarget; setLoading(_btn, true);
    const a = document.getElementById('snap-diff-a').value;
    const b = document.getElementById('snap-diff-b').value;
    if (!a || !b || a === b) { setLoading(_btn, false); showToast('Select two different snapshots.', 'warning'); return; }
    try {
        const res = await api('/api/snapshots/diff', 'POST', { a, b });
        document.getElementById('snapshot-diff-result').innerHTML = renderSnapshotDiff(res.diff);
    } catch (err) { showToast(err.message, 'error'); }
    finally { setLoading(_btn, false); }
});

// ============ Command palette (Tier 3 #12) ============

const COMMANDS = [
    { label: 'Switch to Data tab', tab: 'data' },
    { label: 'Switch to Preprocess tab', tab: 'preprocess' },
    { label: 'Switch to Analyze tab', tab: 'analyze' },
    { label: 'Switch to Train tab', tab: 'train' },
    { label: 'Switch to Predict tab', tab: 'predict' },
    { label: 'Switch to Schema tab', tab: 'schema' },
    { label: 'Switch to LLM Advisor tab', tab: 'llm' },
    { label: 'Load selected table', run: () => { switchTab('data'); clickIt('btn-load-table'); } },
    { label: 'Run custom SQL query', run: () => { switchTab('data'); return clickIt('btn-run-query'); } },
    { label: 'Database overview', run: () => { switchTab('data'); clickIt('btn-overview'); } },
    { label: 'Check LLM connection', run: () => { switchTab('data'); clickIt('btn-llm-check'); } },
    { label: 'Auto-train (suggest target → prepare → train)', run: () => { switchTab('train'); return clickIt('btn-auto-train'); } },
    { label: 'Train model', run: () => { switchTab('train'); return clickIt('btn-train'); } },
    { label: 'Check feature health', run: () => { switchTab('train'); clickIt('btn-feature-health'); } },
    { label: 'Save model', run: () => { switchTab('train'); return clickIt('btn-save-model'); } },
    { label: 'Export loaded data (CSV)', run: () => { switchTab('data'); clickIt('btn-export-data'); } },
    { label: 'Export preprocessed data (CSV)', run: () => { switchTab('preprocess'); clickIt('btn-export-preprocessed'); } },
    { label: 'Detect anomalies', run: () => { switchTab('schema'); clickIt('btn-anomaly-detect'); } },
    // ---- Tier D additions: reports, quick view, explain, monitor ----
    { label: 'Quick View (summary + charts)', run: () => { switchTab('data'); return runQuickView(); } },
    { label: 'Export analysis report (HTML)', tab: 'analyze', run: () => { switchTab('analyze'); return exportHtmlReport(); } },
    { label: 'Export training report (Markdown)', tab: 'train', run: () => { switchTab('train'); clickIt('btn-export-train-md'); } },
    { label: 'Explain first prediction row', tab: 'predict', run: () => { switchTab('predict'); clickIt('btn-explain-first'); } },
    { label: 'Capture drift reference (loaded data)', tab: 'schema', run: () => { switchTab('schema'); clickIt('btn-monitor-capture'); } },
    { label: 'Download all charts as PNG (scroll down)', run: () => { showToast('Click the ⤓ PNG button on any chart.', 'info'); } },
];

function clickIt(id) {
    const el = document.getElementById(id);
    if (el) { el.click(); return true; }
    return false;
}

let paletteActive = 0;
let paletteItems = [];

function renderPalette(filter) {
    const q = (filter || '').toLowerCase().trim();
    const match = c => !q || c.label.toLowerCase().includes(q) || ((c.tab || '').toLowerCase().includes(q));
    const filtered = COMMANDS.filter(match);
    paletteItems = filtered;
    const list = document.getElementById('command-palette-list');
    if (!filtered.length) {
        list.innerHTML = '<div class="command-palette-empty">No commands match "<strong>' + escapeHtml(filter || '') + '</strong>".</div>';
        return;
    }
    if (paletteActive > filtered.length - 1) paletteActive = filtered.length - 1;

    // Group results by tab/category for a scannable modal when no filter (Item 11).
    let html = '';
    const byGroup = {};
    for (const c of filtered) { const g = c.tab || 'Actions'; (byGroup[g] = byGroup[g] || []).push(c); }
    let idx = 0;
    for (const [group, items] of Object.entries(byGroup)) {
        html += `<div class="command-group">${escapeHtml(group)}</div>`;
        for (const c of items) {
            html += `<div class="command-item ${idx === paletteActive ? 'active' : ''}" data-idx="${idx}" role="option" ${idx === paletteActive ? 'aria-selected="true"' : ''}>` +
                `<span class="command-label">${escapeHtml(c.label)}</span>` +
                (c.tab ? `<span class="command-tab">${escapeHtml(c.tab)}</span>` : '') + `</div>`;
            idx++;
        }
    }
    list.innerHTML = html;
    list.querySelectorAll('.command-item').forEach(el => {
        el.addEventListener('mouseenter', () => { paletteActive = parseInt(el.dataset.idx, 10); renderPalette(document.getElementById('command-palette-input').value); });
        el.addEventListener('click', () => runPaletteCommand(parseInt(el.dataset.idx, 10)));
    });
    const activeEl = list.querySelector('.command-item.active');
    if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });
}

async function runPaletteCommand(idx) {
    const c = paletteItems[idx];
    if (!c) return;
    closePalette();
    if (c.tab) switchTab(c.tab);
    if (c.run) {
        try { await c.run(); }
        catch (e) { showToast(e.message || 'Command failed.', 'error'); }
    }
}

function openPalette() {
    const pal = document.getElementById('command-palette');
    pal.classList.remove('hidden');
    const input = document.getElementById('command-palette-input');
    paletteActive = 0;
    input.value = '';
    renderPalette('');
    input.focus();
}

function closePalette() {
    document.getElementById('command-palette').classList.add('hidden');
}

document.getElementById('command-palette-input').addEventListener('input', (e) => {
    paletteActive = 0;
    renderPalette(e.target.value);
});

document.getElementById('command-palette-input').addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
        e.preventDefault();
        paletteActive = Math.min(paletteActive + 1, paletteItems.length - 1);
        renderPalette(document.getElementById('command-palette-input').value);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        paletteActive = Math.max(paletteActive - 1, 0);
        renderPalette(document.getElementById('command-palette-input').value);
    } else if (e.key === 'Enter') {
        e.preventDefault();
        runPaletteCommand(paletteActive);
    } else if (e.key === 'Escape') {
        e.preventDefault();
        closePalette();
    }
});

document.getElementById('command-palette').addEventListener('click', (e) => {
    if (e.target.id === 'command-palette') closePalette();
});

document.getElementById('btn-open-palette').addEventListener('click', () => openPalette());

document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        const pal = document.getElementById('command-palette');
        if (pal.classList.contains('hidden')) openPalette();
        else closePalette();
    }
});

// ============ Tab switching ============

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// ============ Initial load ============

loadUIState(); // Tier D: restore the user's last tab & dashboard settings
refreshState();
refreshRecipes();
refreshSnapshots();
setInterval(pollNotifications, 4000);