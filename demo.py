"""
Demo script for the ML Agent.
Demonstrates the full workflow: database processing, model selection, and predictions.
"""
import json
import pandas as pd
from ml_agent import MLAgent


def main():
    print("=" * 70)
    print("ML AGENT DEMO - SQL Database Processing & Machine Learning")
    print("=" * 70)

    # 1. Initialize agent with SQLite database
    print("\n[1] Initializing ML Agent with sample_company.db...")
    agent = MLAgent("sample_company.db")

    # 2. Explore database structure
    print("\n[2] Database Overview:")
    tables = agent.list_tables()
    print(f"  Tables found: {tables}")

    summary = agent.get_table_summary()
    for t in summary:
        print(f"  - {t['table']}: {t['row_count']} rows, {t['column_count']} columns")

    # 3. Load data and perform analysis
    print("\n[3] Loading employees table and performing analysis...")
    df = agent.load_table("employees")
    print(f"  Loaded {len(df)} rows from employees table")

    # Basic analysis
    analysis = agent.analyze(target_column="salary", analysis_type="summary")
    print(f"  Data shape: {analysis['basic_stats']['rows']} rows x {analysis['basic_stats']['columns']} cols")
    print(f"  Numeric columns: {analysis['basic_stats']['numeric_columns']}")
    print(f"  Categorical columns: {analysis['basic_stats']['categorical_columns']}")

    # 4. Train model to predict salary (regression)
    print("\n[4] Training model to predict salary (regression task)...")
    results = agent.train(target_column="salary")
    print(agent.format_results(results))

    # 5. Make predictions
    print("\n[5] Making predictions on new employee data...")
    new_employees = [
        {"age": 30, "years_experience": 5.0, "education_level": "Bachelor",
         "dept_id": 1, "performance_score": 75.0, "satisfaction_score": 70.0},
        {"age": 45, "years_experience": 20.0, "education_level": "Master",
         "dept_id": 2, "performance_score": 85.0, "satisfaction_score": 80.0},
        {"age": 25, "years_experience": 1.0, "education_level": "High School",
         "dept_id": 4, "performance_score": 60.0, "satisfaction_score": 55.0},
    ]
    predictions = agent.predict(new_employees)
    print("\nPredicted salaries for new employees:")
    for i, (_, row) in enumerate(predictions.iterrows(), 1):
        print(f"  Employee {i}: Predicted salary = ${row['prediction']:,.2f}")

    # 6. Train classification model
    print("\n[6] Training model to predict education level (classification task)...")
    # Create a binary target for classification
    df_class = df.copy()
    df_class["high_performer"] = (df_class["performance_score"] >= 75).astype(int)
    agent.current_df = df_class
    results_class = agent.train(target_column="high_performer", task_type="classification")
    print(agent.format_results(results_class))

    # 7. Classification predictions
    print("\n[7] Predicting high performer status...")
    new_emp = {"age": 35, "years_experience": 10.0, "education_level": "Master",
               "dept_id": 1, "salary": 95000.0, "performance_score": 82.0,
               "satisfaction_score": 88.0}
    pred = agent.predict_single(new_emp)
    print(f"  Prediction: {pred}")

    # 8. Use SQL query to join tables and analyze sales
    print("\n[8] Using SQL query to join tables for sales analysis...")
    query = """
    SELECT e.emp_id, e.age, e.years_experience, e.education_level,
           e.performance_score, e.satisfaction_score,
           s.product_category, s.units_sold, s.unit_price, s.region
    FROM employees e
    JOIN sales s ON e.emp_id = s.emp_id
    """
    sales_df = agent.load_query_as_data(query)
    print(f"  Loaded {len(sales_df)} sales records with employee info")

    # Train model to predict units sold
    print("\n[9] Training model to predict units_sold (regression)...")
    results_sales = agent.train(target_column="units_sold")
    print(agent.format_results(results_sales))

    # 10. Feature importance
    print("\n[10] Feature Importance Analysis:")
    importance = agent.get_feature_importance()
    if importance is not None:
        print(importance.head(10).to_string(index=False))

    # 11. Save and load model
    print("\n[11] Saving and loading model...")
    agent.save_model("trained_model.joblib")
    agent2 = MLAgent("sample_company.db")
    agent2.load_model("trained_model.joblib")
    print("  Model saved and loaded successfully!")

    # 12. Cleanup
    agent.close()
    agent2.close()
    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()