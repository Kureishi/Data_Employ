# ML Agent - SQL Database Machine Learning Agent

An intelligent agent that processes SQL databases, automatically determines the best machine learning model for the data, and produces predictions and analysis per user requests.

## Features

- **SQL Database Processing**: Connects to SQLite, PostgreSQL, MySQL, and other SQL databases via SQLAlchemy
- **Multi-Table Support**: Discovers all tables, schemas, primary/foreign keys, and relationships
- **Automatic Model Selection**: Evaluates 9 regression models and 7 classification models using cross-validation to find the best performer
- **Task Type Auto-Detection**: Automatically determines if the task is regression or classification based on data characteristics
- **Intelligent Feature Engineering**: 
  - Auto-detects and removes ID columns (auto-increment primary keys)
  - Handles categorical variables with one-hot encoding
  - Imputes missing values
  - Scales numeric features
- **Comprehensive Analysis**: Statistical summaries, correlations, column insights, and target analysis
- **Prediction Engine**: Make predictions on new data (single records or batches)
- **Model Persistence**: Save and load trained models
- **Feature Importance**: Identifies which features most influence predictions
- **Data Preprocessing**: Drop/fill missing values, filter rows, scale features, encode categories, and create new features

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Create a sample database

```bash
python create_sample_db.py
```

### 2. Run the demo

```bash
python demo.py
```

### 3. Use the Command-Line Interface

The CLI provides a full interface for interacting with the ML Agent:

```bash
# List tables in a database
python main.py -db sample_company.db --list-tables

# Get a comprehensive database overview
python main.py -db sample_company.db --overview

# Analyze data
python main.py -db sample_company.db -t employees --analyze summary
python main.py -db sample_company.db -t employees --analyze correlations
python main.py -db sample_company.db -t employees -y salary --analyze target

# Train a model (auto-detect task type)
python main.py -db sample_company.db -t employees -y salary

# Train a classification model explicitly
python main.py -db sample_company.db -t employees -y education_level --task-type classification

# Train and predict on new data (JSON)
python main.py -db sample_company.db -t employees -y salary \
    --predict '{"age": 30, "years_experience": 5.0, "education_level": "Bachelor", "dept_id": 1, "performance_score": 75.0, "satisfaction_score": 70.0}'

# Train and predict from a CSV file, save results
python main.py -db sample_company.db -t employees -y salary \
    --predict-file new_employees.csv --output predictions.csv

# Use a custom SQL query as the dataset
python main.py -db sample_company.db --query "SELECT * FROM employees" -y salary

# Save and reuse a trained model
python main.py -db sample_company.db -t employees -y salary --save-model model.joblib
python main.py --load-model model.joblib \
    --predict '{"age": 30, "years_experience": 5.0, "education_level": "Bachelor", "dept_id": 1, "performance_score": 75.0, "satisfaction_score": 70.0}'

# Show model info and feature importance
python main.py --load-model model.joblib --model-info --feature-importance

# Output results as JSON
python main.py -db sample_company.db -t employees -y salary --json

# Show help
python main.py --help
```

### 4. Use the agent programmatically

```python
from ml_agent import MLAgent

# Initialize agent with a SQL database
agent = MLAgent("sample_company.db")

# Explore the database
tables = agent.list_tables()
print(f"Tables: {tables}")

# Load a table
df = agent.load_table("employees")

# Analyze the data
analysis = agent.analyze(target_column="salary", analysis_type="summary")

# Train the best model automatically
results = agent.train(target_column="salary")
print(agent.format_results(results))

# Make predictions
new_employee = {
    "age": 30,
    "years_experience": 5.0,
    "education_level": "Bachelor",
    "dept_id": 1,
    "performance_score": 75.0,
    "satisfaction_score": 70.0,
}
prediction = agent.predict_single(new_employee)
print(f"Predicted salary: ${prediction['prediction']:,.2f}")

# Save and load models
agent.save_model("my_model.joblib")
agent.load_model("my_model.joblib")

# Close the connection
agent.close()
```

## Preprocessing

The CLI supports common data preprocessing operations applied before model training:

```bash
# Drop rows with missing values
python main.py -db sample_company.db -t employees --dropna -y salary

# Fill missing values
python main.py -db sample_company.db -t employees --fillna median -y salary
python main.py -db sample_company.db -t employees --fillna constant --fillna-value 0 -y salary

# Drop or keep columns
python main.py -db sample_company.db -t employees --drop-columns "name,emp_id" -y salary
python main.py -db sample_company.db -t employees --keep-columns "age,salary,education_level" -y salary

# Remove duplicates
python main.py -db sample_company.db -t employees --drop-duplicates -y salary

# Filter rows with a query expression
python main.py -db sample_company.db -t employees --filter "salary > 50000" -y salary

# Scale numeric features (target is excluded automatically)
python main.py -db sample_company.db -t employees --scale standard -y salary
python main.py -db sample_company.db -t employees --scale minmax -y salary

# Encode categorical columns
python main.py -db sample_company.db -t employees --encode label -y salary
python main.py -db sample_company.db -t employees --encode onehot -y salary

# Sample or limit rows
python main.py -db sample_company.db -t employees --sample 100 -y salary
python main.py -db sample_company.db -t employees --head 50 -y salary

# Feature engineering
python main.py -db sample_company.db -t employees --create-ratio salary years_experience salary_per_year -y salary
python main.py -db sample_company.db -t employees --create-product age years_experience experience_score -y salary
python main.py -db sample_company.db -t employees --create-difference performance_score satisfaction_score gap -y salary
python main.py -db sample_company.db -t employees --create-bins age 5 -y salary

# View preprocessing summary
python main.py -db sample_company.db -t employees --drop-columns "name" --scale standard --encode label --preprocess-summary -y salary
```

All preprocessing operations can be chained together and are applied in order. Feature engineering operations (like `--create-ratio`) are automatically applied to prediction data as well.

## Connecting to Different Databases

```python
# SQLite
agent = MLAgent("mydatabase.db")

# PostgreSQL
agent = MLAgent("postgresql://user:password@localhost/dbname")

# MySQL
agent = MLAgent("mysql+pymysql://user:password@localhost/dbname")

# SQL Server
agent = MLAgent("mssql+pyodbc://user:password@server/dbname")
```

## Using SQL Queries

```python
# Execute custom SQL queries
result = agent.execute_query("SELECT * FROM employees WHERE salary > 80000")

# Load query results as the working dataset
agent.load_query_as_data("""
    SELECT e.age, e.years_experience, e.education_level,
           s.product_category, s.units_sold
    FROM employees e
    JOIN sales s ON e.emp_id = s.emp_id
""")

# Train on the joined data
results = agent.train(target_column="units_sold")
```

## Model Selection

The agent automatically evaluates these models and selects the best one:

### Regression Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Extra Trees
- K-Neighbors
- SVR

### Classification Models
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Extra Trees
- K-Neighbors
- SVC

## CLI Reference

| Flag | Description |
|------|-------------|
| `-db, --database` | SQL database connection string or SQLite file path |
| `-t, --table` | Table name to use as the dataset |
| `--query` | Custom SQL query to use as the dataset |
| `--limit` | Maximum number of rows to load |
| `-y, --target` | Target column to predict |
| `--task-type` | Task type: `regression` or `classification` (auto-detected) |
| `--test-size` | Fraction of data for test set (default: 0.2) |
| `--cv-folds` | Number of cross-validation folds (default: 5) |
| `--random-state` | Random seed (default: 42) |
| `--dropna` | Drop rows with missing values |
| `--fillna` | Fill missing values: `mean`, `median`, `mode`, `zero`, `constant`, `ffill`, `bfill` |
| `--fillna-value` | Value to use with `--fillna constant` |
| `--drop-columns` | Comma-separated columns to drop |
| `--keep-columns` | Comma-separated columns to keep (drops all others) |
| `--drop-duplicates` | Drop duplicate rows |
| `--filter` | Filter rows using pandas query expression |
| `--scale` | Scale numeric columns: `standard` (z-score) or `minmax` |
| `--encode` | Encode categorical columns: `label` or `onehot` |
| `--sample` | Randomly sample N rows |
| `--head` | Keep only the first N rows |
| `--create-ratio` | Create ratio feature: `--create-ratio col1 col2 new_col` |
| `--create-product` | Create product feature: `--create-product col1 col2 new_col` |
| `--create-difference` | Create difference feature: `--create-difference col1 col2 new_col` |
| `--create-bins` | Bin a column: `--create-bins column 5` |
| `--preprocess-summary` | Show summary of preprocessing operations |
| `--list-tables` | List all tables in the database |
| `--overview` | Show comprehensive database overview |
| `--analyze` | Analysis type: `summary`, `correlations`, `insights`, `target` |
| `--feature-importance` | Show feature importance from trained model |
| `--model-info` | Show information about trained model |
| `--predict` | JSON string of data to predict on |
| `--predict-file` | CSV file containing data to predict on |
| `--save-model` | Save trained model to path |
| `--load-model` | Load a previously trained model |
| `-o, --output` | Save predictions to CSV file |
| `--json` | Output results as JSON |
| `-v, --verbose` | Show verbose output |

## API Reference

### MLAgent

| Method | Description |
|--------|-------------|
| `list_tables()` | List all tables in the database |
| `get_database_overview()` | Get comprehensive database metadata |
| `get_table_summary()` | Get summary of all tables |
| `load_table(table, limit)` | Load a table into memory |
| `execute_query(query)` | Execute a custom SQL query |
| `load_query_as_data(query)` | Load query results as working dataset |
| `preprocess(df)` | Get a DataPreprocessor for the dataset |
| `apply_preprocessing(ops)` | Apply preprocessing operations to the dataset |
| `get_preprocessing_summary()` | Get summary of preprocessing operations |
| `analyze(target, type)` | Perform data analysis |
| `train(target, task_type)` | Train the best model |
| `predict(data)` | Make predictions on new data |
| `predict_single(data)` | Predict a single record |
| `get_model_info()` | Get trained model information |
| `get_feature_importance()` | Get feature importance rankings |
| `save_model(path)` | Save trained model to disk |
| `load_model(path)` | Load a trained model |
| `format_results(results)` | Format results as readable text |

### Analysis Types
- `summary` - Basic stats, correlations, column insights, target analysis
- `correlations` - Correlation matrix for numeric columns
- `insights` - Per-column statistics and distributions
- `target` - Target column distribution and feature correlations

## Project Structure

```
├── ml_agent/
│   ├── __init__.py          # Package exports
│   ├── agent.py             # Main MLAgent orchestrator
│   ├── database.py          # SQL database processing
│   ├── model_selector.py    # Automatic model selection
│   ├── predictor.py         # Prediction engine
│   ├── analyzer.py          # Data analysis module
│   ├── preprocessor.py      # Data preprocessing module
│   └── cli.py               # Command-line interface
├── main.py                  # CLI entry point
├── create_sample_db.py      # Sample database generator
├── demo.py                  # Full workflow demo
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## License

MIT