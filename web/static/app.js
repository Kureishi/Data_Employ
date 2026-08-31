// ML Agent Web UI JavaScript

// ============ State ============
let currentTab = 'data';
let preprocessQueue = [];
let connectedTables = [];

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

function formatJson(val) {
    if (val === null || val === undefined) return 'N/A';
    if (typeof val === 'number') {
        if (Number.isInteger(val)) return val.toString();
        return val.toFixed(4);
    }
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
    let html = '<table class="dataframe"><thead><tr>';
    html += cols.map(c => `<th>${escapeHtml(c)}</th>`).join('');
    html += '</tr></thead><tbody>';
    for (const row of data) {
        html += '<tr>';
        html += cols.map(c => `<td>${escapeHtml(formatJson(row[c]))}</td>`).join('');
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

function renderTablesList(tables) {
    const list = document.getElementById('tables-list');
    if (!tables || tables.length === 0) {
        list.innerHTML = '<p class="hint">No tables found.</p>';
        return;
    }
    list.innerHTML = tables.map(t => `
        <div class="table-item" data-table="${escapeHtml(t)}">
            <span class="table-name">${escapeHtml(t)}</span>
            <span class="table-rows">click to load</span>
        </div>
    `).join('');

    // Attach click handlers to table items
    document.querySelectorAll('.table-item').forEach(el => {
        el.addEventListener('click', async () => {
            const table = el.dataset.table;
            try {
                const res = await api('/api/load', 'POST', { table });
                const info = document.getElementById('data-info');
                info.className = 'info-box success';
                info.innerHTML = `<strong>Loaded:</strong> ${escapeHtml(table)}<br><strong>Rows:</strong> ${res.rows}<br><strong>Columns:</strong> ${res.columns.join(', ')}`;
                document.getElementById('data-preview').innerHTML = renderTable(res.data, res.columns);
                showToast(`Loaded ${res.rows} rows from ${table}`, 'success');
                await refreshState();
            } catch (e) {
                showToast(e.message, 'error');
            }
        });
    });
}

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
    items.forEach((it, idx) => {
        const w = (Math.abs(it.value) / maxVal) * plotW;
        const y = padding.top + idx * (barH + gap);
        const isNeg = it.value < 0;
        const x = labelW + (isNeg ? plotW - w : 0);
        const label = String(it.label);
        const disp = (it.label || '').length > 18 ? it.label.slice(0, 16) + '…' : it.label;
        bars += `<text x="${labelW - 6}" y="${y + barH - 4}" text-anchor="end" font-size="11" fill="#6b7280">${escapeHtml(disp)}</text>`;
        bars += `<rect x="${x}" y="${y}" width="${Math.max(w, it.value === 0 ? 0 : 2)}" height="${barH}" fill="${color}" rx="2"></rect>`;
        bars += `<text x="${x + Math.max(w, 2) + 4}" y="${y + barH - 4}" font-size="11" fill="#1f2937">${formatJson(it.value)}</text>`;
    });

    return `<div class="chart"><div class="chart-title">${escapeHtml(opts.title || '')}</div>
            <svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${bars}</svg></div>`;
}

function renderConfusionMatrix(cm, classLabels) {
    if (!cm || !cm.length) return '';
    const n = cm.length;
    const labels = classLabels || cm.map((_, i) => i);
    let html = '<h4>Confusion Matrix</h4><table class="cm-table"><tr><th></th>';
    for (const l of labels) html += `<th>Pred ${escapeHtml(String(l))}</th>`;
    html += '</tr>';
    for (let i = 0; i < n; i++) {
        html += `<tr><th>Act ${escapeHtml(String(labels[i]))}</th>`;
        for (let j = 0; j < n; j++) {
            html += `<td>${cm[i][j]}</td>`;
        }
        html += '</tr>';
    }
    html += '</table>';
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
        const info = document.getElementById('data-info');
        info.className = 'info-box success';
        info.innerHTML = `<strong>Loaded table:</strong> ${escapeHtml(table)}<br><strong>Rows:</strong> ${res.rows}<br><strong>Columns:</strong> ${res.columns.join(', ')}`;
        document.getElementById('data-preview').innerHTML = renderTable(res.data, res.columns);
        showToast(`Loaded ${res.rows} rows from ${table}`, 'success');
        await refreshState();
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
        const info = document.getElementById('data-info');
        info.className = 'info-box success';
        info.innerHTML = `<strong>Query:</strong> <code>${escapeHtml(query)}</code><br><strong>Rows:</strong> ${res.rows}<br><strong>Columns:</strong> ${res.columns.join(', ')}`;
        document.getElementById('data-preview').innerHTML = renderTable(res.data, res.columns);
        showToast(`Loaded ${res.rows} rows from query`, 'success');
        await refreshState();
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
        const out = document.getElementById('analyze-result');
        if (type === 'health') {
            out.innerHTML = renderHealthReport(res.analysis);
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

async function startTraining(target, taskType, tuning) {
    const _btn = document.getElementById('btn-train');
    setLoading(_btn, true);
    document.getElementById('train-result').innerHTML = '';
    const progressBox = document.getElementById('train-progress');
    progressBox.innerHTML = '<div class="progress-track"><div class="progress-fill" style="width:0%"></div></div><p class="progress-label">Starting training…</p>';

    let jobId = null;
    try {
        const res = await api('/api/train', 'POST', { target_column: target, task_type: taskType, tuning });
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
    startTraining(target, taskType, tuning);
});

// ============ Prediction ============

let lastPredictions = null;
let lastPredictColumns = null;

function renderPredictions(res, containerId, filenameBase) {
    lastPredictions = res.predictions;
    lastPredictColumns = res.columns;
    const box = document.getElementById(containerId);
    box.innerHTML = renderTable(res.predictions, res.columns) +
        `<p class="hint">${res.rows} prediction(s). Use "Export Predictions (CSV)" below to download.</p>`;
    window._lastPredictFilename = filenameBase;
}

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

// ============ Tab switching ============

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// ============ Initial load ============

refreshState();