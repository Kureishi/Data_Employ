# ML Agent — SQL Database Machine Learning Agent

Automates end-to-end ML on SQL databases: discovers tables & relationships, auto-selects the best model via cross-validation, and produces predictions & analysis through a **CLI**, **Python API**, and **web dashboard**.

## Features
- **SQL**: multi-table schema discovery, semantic column typing, saved queries, auto-join builder, DB-native sampling, read-only safety + query timeouts, ER diagram.
- **Modeling**: auto task detection; selects from **9 regression / 7 classification** models by cross-validation with early-stop; optional hyperparameter tuning (`Quick`/`Full`); feature importance & engineering.
- **Explainability**: per-record feature contributions, what-if & batch what-if, confusion matrix / classification report, model-drift (PSI) & anomaly detection.
- **Workflow**: versioned experiments + champion/rollback, reusable pipeline recipes, result snapshots & diff, re-predict production tables, analysis/target caching, async jobs + toasts.
- **Web UI**: `⌘K` command palette, zero-dependency SVG charts (correlations, distributions, importance, confusion matrix), HTML/Markdown/PNG report export, drift/anomaly/leaderboard dashboards, Quick View, persistent dashboard state, table sparkline previews.
- **Production**: waitress/gunicorn serving, health/status endpoints, auth + rate limiting, durable job store, Redis-backed rate limiting & sessions, hard-killable jobs, session isolation.

## Install
```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python create_sample_db.py        # optional: generate a sample DB
```

## Quick Start (CLI)
```bash
python main.py -db sample_company.db -t employees -y salary                 # overview + train + report
python main.py -db sample_company.db -t employees -y salary --tuning quick  # hyperparameter tune
python main.py -db sample_company.db -t employees --predict '{"age":30,"years_experience":5,"education_level":"Bachelor"}'
python main.py -db sample_company.db -t employees -y salary --llm           # local LLM advisor (LM Studio)
python main.py --help
```

## Web Dashboard
```bash
python web_api.py --db sample_company.db                 # http://localhost:5000
python web_api.py --db sample_company.db --host 0.0.0.0 --port 8080   # served via waitress
python web_api.py --debug                                # force Flask dev server
```
On Linux/macOS use `gunicorn -c gunicorn.conf.py wsgi:app` for multi-process scaling.

## Programmatic Usage
```python
from ml_agent import MLAgent
agent = MLAgent("sample_company.db")                    # or a postgresql:// / mysql+pymysql:// URL
agent.list_tables()
df = agent.load_table("employees")
analysis = agent.analyze(target_column="salary", analysis_type="summary")
results = agent.train(target_column="salary", tuning="quick")  # auto best model
agent.predict_single({"age": 30, "years_experience": 5, "education_level": "Bachelor"})
agent.save_model("model.joblib"); agent.load_model("model.joblib")
agent.close()
```

## Training
- **Task type** auto-detected (regression / classification); `--task-type` overrides it.
- **Model selection**: cross-validated scoring (`StratifiedKFold` for classification), optional early-stop.
- **Tuning**: `off` / `quick` (`RandomizedSearchCV`) / `full` (`GridSearchCV`) — via `agent.train(..., tuning=)`, CLI `--tuning`, or the web UI's Tuning dropdown. Best params are shown in results and stored with the model.

## Preprocessing
Chain operations (applied in order) via the CLI, `agent.apply_preprocessing(...)`, or the web Preprocess tab: `--dropna`, `--fillna`, `--drop-columns`, `--keep-columns`, `--drop-duplicates`, `--filter`, `--scale standard|minmax`, `--encode label|onehot`, `--sample`, `--head`, and feature engineering (`--create-ratio|product|difference|bins`). Examples:
```bash
python main.py -db sample_company.db -t employees --drop-columns "name" --scale standard --encode label \
    --save-preprocessed-db preprocessed.db -y salary     # write cleaned data to a new DB (original untouched)
python main.py -db sample_company.db -t employees --fillna median -y salary
```
Feature-engineering ops are re-applied automatically to prediction data.

## SQL Queries
```python
result = agent.execute_query("SELECT * FROM employees WHERE salary > 80000")
agent.load_query_as_data("SELECT e.age, s.units_sold FROM employees e JOIN sales s ON e.emp_id = s.emp_id")
results = agent.train(target_column="units_sold")   # train on joined data
```

## Reference

### CLI Flags
| Flag | Description |
|------|-------------|
| `-db, --database` | SQL database connection string or SQLite file path |
| `-t, --table` | Table to use as the dataset |
| `--query` | Custom SQL query to use as the dataset |
| `--limit` | Max rows to load |
| `-y, --target` | Target column to predict |
| `--task-type` | `regression` or `classification` (auto-detected) |
| `--test-size` | Test-set fraction (default: 0.2) |
| `--cv-folds` | CV folds (default: 5) |
| `--random-state` | Random seed (default: 42) |
| `--dropna` / `--fillna [mean\|median\|mode\|zero\|constant\|ffill\|bfill]` | Handle missing values |
| `--fillna-value` | Value for `--fillna constant` |
| `--drop-columns` / `--keep-columns` | Comma-separated columns to drop / keep |
| `--drop-duplicates` / `--filter "expr"` | Dedupe / filter rows (pandas query) |
| `--scale standard\|minmax` | Scale numeric columns |
| `--encode label\|onehot` | Encode categorical columns |
| `--sample N` / `--head N` | Sample / keep first N rows |
| `--create-ratio` / `--create-product` / `--create-difference` | Create feature: `--create-X c1 c2 new_col` |
| `--create-bins col N` | Bin a numeric column |
| `--preprocess-summary` | Show preprocessing summary |
| `--save-preprocessed-db path` | Write cleaned data to a new SQLite DB (original untouched) |
| `--preprocessed-table` / `--include-original-tables` | Table name / also copy original tables |
| `--list-tables` / `--overview` | List tables / show DB overview |
| `--analyze summary\|correlations\|insights\|target\|health` | Run analysis |
| `--tuning off\|quick\|full` | Hyperparameter tune the best model |
| `--feature-importance` / `--model-info` | Show feature importance / model info |
| `--predict '{"col": val}'` | Predict a JSON record |
| `--predict-file file.csv` | Predict rows from a CSV |
| `--save-model path` / `--load-model path` | Save / load a model |
| `-o, --output file.csv` | Save predictions to CSV |
| `--json` / `-v, --verbose` | JSON output / verbose |
| `--llm` + `--llm-*` | LLM advisor flags (see below) |

### Analysis Types
- `summary` — basic stats, correlations, column insights, target analysis
- `correlations` — numeric correlation matrix
- `insights` — per-column stats & distributions
- `target` — target distribution + feature correlations
- `health` — quality report: missing values, duplicates, per-column health, class balance, warnings

## LLM Advisor (LM Studio)
Uses a **local** OpenAI-compatible LLM (LM Studio at `http://localhost:1234/v1`) — no data leaves the machine. If the server is offline, every method falls back to a built-in heuristic.

| Capability | CLI Flag |
|------------|----------|
| Availability check | `--llm-check` |
| Natural language → SQL (schema-validated) | `--llm-ask "..."` |
| Target recommendation | `--llm-suggest-target` |
| Preprocessing advice / apply | `--llm-suggest-preprocessing` / `--llm-apply-preprocessing` |
| Result interpretation | `--llm-explain-results` |
| Prediction narratives | `--llm-explain-predictions N` |

## MLAgent API (Python)
`list_tables()` `get_database_overview()` `get_table_summary()` `load_table(table, limit)` `execute_query(q)` `load_query_as_data(q)` `apply_preprocessing(ops)` `get_preprocessing_summary()` `save_preprocessed_db(path, table_name, include_original)` `analyze(target, type)` `train(target, task_type, tuning=, progress_callback=)` `predict(data)` `predict_single(row)` `get_model_info()` `get_feature_importance()` `save_model(path)` `load_model(path)` `format_results(results)` `enable_llm(base_url, model, timeout)` `llm_check()` `llm_generate_sql(q)` `llm_suggest_target()` `llm_suggest_preprocessing()` `llm_explain_results()` `llm_explain_predictions()`

## Web Dashboard
Tabs: **Data** (connect, browse, SQL, preview/export), **Preprocess**, **Analyze** (summary/correlations/insights/target/health + charts, snapshots & diff), **Train** (best model, tuning, live progress + cancel, feature-health), **Predict** (single/batch/table/CSV, explain, what-if), **Model Persistence**, **Schema** (ER diagram, auto-join, drift profiling, anomaly & drift monitoring, safety settings), **LLM Advisor**. Press **`⌘K`** for the command palette.

## REST API (abridged)
Core endpoints (all `/api/...`): `state`, `POST connect`, `POST disconnect`, `GET tables`, `GET tables/<t>/schema`, `GET tables/<t>/preview`, `GET overview`, `POST load`, `POST query`, `POST query/validate`, `POST load-query`, `GET columns/<t>/types`, `GET relationships`, `POST load-sample`, `POST load-join`, `POST query/save`, `GET query/list`, `POST profile/capture`, `GET profile/list`, `POST profile/compare`, `POST settings`, `POST synthesize`, `GET experiments`, `GET experiments/champion`, `POST experiments/<id>/promote`, `POST experiments/rollback`, `POST experiments/compare`, `POST preprocess`, `GET preprocess-summary`, `POST analyze`, `POST anomaly/detect`, `POST monitor/capture`, `POST monitor/check`, `GET suggest-targets`, `POST auto-prepare`, `POST feature-health`, `POST experiments/propose`, `POST train`, `GET train/status/<id>`, `POST train/cancel/<id>`, `GET model-info`, `POST predict`, `POST explain/prediction`, `POST explain/whatif`, `POST explain/batch-whatif`, `POST repredict`, `POST op/start`, `GET op/status/<id>`, `GET notifications`, `POST recipe/save`, `GET recipe/list`, `POST recipe/apply`, `GET snapshots`, `POST snapshots`, `POST snapshots/diff`, `POST predict/table`, `POST predict/current`, `POST predict/upload`, `GET export/csv`, `POST save-model`, `POST load-model`, `GET model/download`, `POST upload-model`, `POST llm/enable`, `GET llm/check`, `POST llm/sql`, `POST llm/suggest-target`, `POST llm/suggest-preprocessing`, `POST llm/apply-preprocessing`, `POST llm/explain-results`, `POST llm/explain-predictions`, plus ops `/health`, `/health/ready`, `/status`.

## Configuration (environment variables)
| Variable | Default | Purpose |
|----------|---------|---------|
| `MLAGENT_LOG_LEVEL` | `INFO` | Logging verbosity |
| `MLAGENT_WEB_HOST` / `MLAGENT_WEB_PORT` | `127.0.0.1` / `5000` | Bind address |
| `MLAGENT_WEB_MAX_CONTENT_LENGTH` | `50 MB` | Max upload/request body |
| `MLAGENT_WEB_UPLOAD_DIR` | `~/.mlagent/data/uploads` | Uploads + per-db stores |
| `MLAGENT_WEB_TRUST_PROXY` | `false` | Honor `X-Forwarded-*` behind a proxy |
| `MLAGENT_DB_POOL_SIZE` / `MLAGENT_DB_MAX_OVERFLOW` | `10` / `20` | Pool bounds (network DBs) |
| `MLAGENT_DB_POOL_RECYCLE` / `MLAGENT_DB_POOL_TIMEOUT` | `1800` / `30` | Rotate / reclaim idle connections |
| `MLAGENT_DB_POOL_PRE_PING` | `true` | Detect broken connections |
| `MLAGENT_DB_STMT_TIMEOUT` | unset | Native statement timeout (PG/MySQL) |
| `MLAGENT_SQLITE_BUSY_TIMEOUT` | `30000` ms | SQLite lock wait |
| `MLAGENT_WSGI_THREADS` | `8` | waitress worker threads |
| `MLAGENT_TRAIN_N_JOBS` | `1` | CV/tuning parallelism |
| `MLAGENT_MAX_CONCURRENT_JOBS` | unset | Async op concurrency cap (`429`) |
| `MLAGENT_API_TOKEN` | unset | Require bearer token (401) |
| `MLAGENT_RATE_LIMIT_PER_MINUTE` | `0` (off) | Per-IP request limit (429) |
| `MLAGENT_JOBS_DB_PATH` | unset | Durable job store (SQLite) |
| `MLAGENT_PROCESS_JOBS` | `false` | Run async jobs in a child process (hard-killable) |
| `MLAGENT_REDIS_URL` | unset | Shared Redis rate limiter / session registry |
| `MLAGENT_SESSIONS_DB_PATH` / `MLAGENT_SESSION_TTL` | unset / unset | Persistent session registry + TTL |

## Production & Scalability
- **Serving**: `python web_api.py` auto-serves via **waitress** (thread pool); `--debug` forces the dev server. Use `waitress-serve --threads=8 wsgi:app` or `gunicorn -c gunicorn.conf.py wsgi:app` (multi-process).
- **Scaling**: one session (loaded data + model) per process. Scale horizontally with multiple workers/instances; for parallel multi-session use an `X-Session-Id` header on `/api/connect` (isolated agents, backed by the shared session registry).
- **Observability**: `/health` (liveness), `/health/ready` (503 until DB connected), `/status` (request metrics, active jobs, sessions). Graceful SIGTERM/SIGINT drain. Optional bearer token + per-IP rate limiting.
- **Job safety**: UUID job IDs, bounded concurrency, training runs off the state lock, native DB timeouts, durable job store (jobs survive restarts), hard-killable subprocess jobs (`MLAGENT_PROCESS_JOBS=1`), atomic JSON stores, early-stop model search, analysis/target caching.
- **UI data**: charts/reports are dependency-free client-side SVG; PDF isn't produced but any chart exports as PNG.

## Project Structure
```
ml_agent/            orchestrator (agent.py), config.py, logging_utils.py, database.py,
                     model_selector.py, predictor.py, analyzer.py, preprocessor.py,
                     llm_advisor.py, job_store.py, ratelimit.py, session_store.py,
                     ops.py, workexec.py, cli.py, snapshots.py, recipes.py, relational.py,
                     anomaly.py, model_monitor.py, experiments.py, sql_validation.py
web/static/          index.html, style.css, app.js (dashboard)
web_api.py           Flask web API (waitress in production)
wsgi.py / gunicorn.conf.py   production WSGI entry + config
main.py / demo.py / create_sample_db.py   CLI entry, demo, sample-DB generator
requirements.txt     dependencies
```
