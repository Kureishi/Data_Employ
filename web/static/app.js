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
    el.querySelector('span').textContent = text;
    el.className = 'status-badge';
    if (state) el.classList.add(state);
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
}

// ============ Event handlers ============

// Connect
document.getElementById('btn-connect').addEventListener('click', async () => {
    const conn = document.getElementById('db-connection').value.trim();
    if (!conn) {
        showToast('Please enter a database connection string.', 'error');
        return;
    }
    try {
        const res = await api('/api/connect', 'POST', { connection: conn });
        showToast(res.message || 'Connected!', 'success');
        await refreshState();
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// Upload database file
document.getElementById('btn-upload-db').addEventListener('click', async () => {
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
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// Load table
document.getElementById('btn-load-table').addEventListener('click', async () => {
    const table = document.getElementById('data-table-select').value;
    const limit = document.getElementById('data-limit').value || null;
    if (!table) {
        showToast('Please select a table.', 'error');
        return;
    }
    try {
        const res = await api('/api/load', 'POST', { table, limit });
        const info = document.getElementById('data-info');
        info.className = 'info-box success';
        info.innerHTML = `<strong>Loaded table:</strong> ${escapeHtml(table)}<br><strong>Rows:</strong> ${res.rows}<br><strong>Columns:</strong> ${res.columns.join(', ')}`;
        document.getElementById('data-preview').innerHTML = renderTable(res.data, res.columns);
        showToast(`Loaded ${res.rows} rows from ${table}`, 'success');
        await refreshState();
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// Run query
document.getElementById('btn-run-query').addEventListener('click', async () => {
    const query = document.getElementById('data-query').value.trim();
    if (!query) {
        showToast('Please enter a SQL query.', 'error');
        return;
    }
    try {
        const res = await api('/api/load-query', 'POST', { query });
        const info = document.getElementById('data-info');
        info.className = 'info-box success';
        info.innerHTML = `<strong>Query:</strong> <code>${escapeHtml(query)}</code><br><strong>Rows:</strong> ${res.rows}<br><strong>Columns:</strong> ${res.columns.join(', ')}`;
        document.getElementById('data-preview').innerHTML = renderTable(res.data, res.columns);
        showToast(`Loaded ${res.rows} rows from query`, 'success');
        await refreshState();
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// Overview
document.getElementById('btn-overview').addEventListener('click', async () => {
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
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// LLM check
document.getElementById('btn-llm-check').addEventListener('click', async () => {
    try {
        const res = await api('/api/llm/check');
        if (res.status.available) {
            setStatus('llm-status', `LLM: ${res.status.detail}`, 'connected');
            showToast(`LLM available: ${res.status.detail}`, 'success');
        } else {
            setStatus('llm-status', 'LLM: Unavailable', 'error');
            showToast(`LLM unavailable: ${res.status.detail}`, 'warning');
        }
    } catch (e) {
        showToast(e.message, 'error');
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

document.getElementById('btn-apply-preprocess').addEventListener('click', async () => {
    if (preprocessQueue.length === 0) {
        showToast('No preprocessing operations queued.', 'warning');
        return;
    }
    try {
        const res = await api('/api/preprocess', 'POST', { operations: preprocessQueue });
        document.getElementById('preprocess-result').innerHTML = renderJson(res.summary);
        showToast(`Applied ${res.summary.total_operations} operations`, 'success');
        await refreshState();
    } catch (e) {
        showToast(e.message, 'error');
    }
});

document.getElementById('btn-clear-preprocess').addEventListener('click', () => {
    preprocessQueue = [];
    renderPreprocessQueue();
    document.getElementById('preprocess-result').innerHTML = '';
});

document.getElementById('btn-llm-suggest-preprocess').addEventListener('click', async () => {
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
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// ============ Analysis ============

document.getElementById('btn-analyze').addEventListener('click', async () => {
    const type = document.getElementById('analyze-type').value;
    const target = document.getElementById('analyze-target').value.trim() || null;
    try {
        const res = await api('/api/analyze', 'POST', { type, target_column: target });
        document.getElementById('analyze-result').innerHTML = renderJson(res.analysis);
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// ============ Training ============

document.getElementById('btn-train').addEventListener('click', async () => {
    const target = document.getElementById('train-target').value.trim();
    const taskType = document.getElementById('train-task-type').value || null;
    if (!target) {
        showToast('Please enter a target column.', 'error');
        return;
    }
    try {
        const res = await api('/api/train', 'POST', { target_column: target, task_type: taskType });
        const t = res.training;
        let html = `<strong>Task:</strong> ${escapeHtml(t.task_type)}<br>`;
        html += `<strong>Best Model:</strong> ${escapeHtml(t.best_model)}<br>`;
        html += `<strong>CV Score:</strong> ${formatJson(t.best_cv_score)}<br>`;
        html += renderMetrics(t.test_metrics);
        html += '<br><strong>Model Scores:</strong><br>';
        for (const [name, score] of Object.entries(t.model_scores)) {
            html += `&nbsp;&nbsp;${escapeHtml(name)}: ${formatJson(score)}<br>`;
        }
        if (t.feature_importance && t.feature_importance.length > 0) {
            html += '<br><strong>Top Features:</strong><br>';
            for (const fi of t.feature_importance.slice(0, 10)) {
                html += `&nbsp;&nbsp;${escapeHtml(fi.feature)}: ${formatJson(fi.importance)}<br>`;
            }
        }
        document.getElementById('train-result').innerHTML = html;
        showToast(`Trained ${t.best_model} (${t.task_type})`, 'success');
        await refreshState();
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// ============ Prediction ============

document.getElementById('btn-predict').addEventListener('click', async () => {
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
    try {
        const res = await api('/api/predict', 'POST', { data });
        document.getElementById('predict-result').innerHTML = renderTable(res.predictions, res.columns);
        showToast(`Made ${res.rows} prediction(s)`, 'success');
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// ============ LLM ============

document.getElementById('btn-llm-enable').addEventListener('click', async () => {
    const url = document.getElementById('llm-url').value.trim() || 'http://localhost:1234/v1';
    const model = document.getElementById('llm-model').value.trim() || null;
    try {
        const res = await api('/api/llm/enable', 'POST', { base_url: url, model });
        if (res.status.available) {
            setStatus('llm-status', `LLM: ${res.status.detail}`, 'connected');
            showToast(`LLM enabled: ${res.status.detail}`, 'success');
        } else {
            setStatus('llm-status', 'LLM: Unavailable', 'error');
            showToast(`LLM enabled but unavailable: ${res.status.detail}`, 'warning');
        }
    } catch (e) {
        showToast(e.message, 'error');
    }
});

document.getElementById('btn-llm-sql').addEventListener('click', async () => {
    const question = document.getElementById('llm-question').value.trim();
    if (!question) {
        showToast('Please enter a question.', 'error');
        return;
    }
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
    } catch (e) {
        showToast(e.message, 'error');
    }
});

document.getElementById('btn-llm-suggest-target').addEventListener('click', async () => {
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
    } catch (e) {
        showToast(e.message, 'error');
    }
});

document.getElementById('btn-llm-explain-results').addEventListener('click', async () => {
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
    } catch (e) {
        showToast(e.message, 'error');
    }
});

// ============ Tab switching ============

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// ============ Initial load ============

refreshState();