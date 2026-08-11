"""
Command-line interface for the ML Agent.
Allows users to interact with the agent via CLI flags.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

from .agent import MLAgent


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="ml-agent",
        description="ML Agent - Process SQL databases, select the best ML model, and make predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List tables in a database
  python main.py -db sample_company.db --list-tables

  # Get database overview
  python main.py -db sample_company.db --overview

  # Analyze data
  python main.py -db sample_company.db -t employees --analyze summary
  python main.py -db sample_company.db -t employees --analyze correlations

  # Train a model (auto-detect task type)
  python main.py -db sample_company.db -t employees -y salary

  # Train a classification model explicitly
  python main.py -db sample_company.db -t employees -y high_performer --task-type classification

  # Train and predict on new data (JSON)
  python main.py -db sample_company.db -t employees -y salary \\
      --predict '{"age": 30, "years_experience": 5.0, "education_level": "Bachelor", "dept_id": 1, "performance_score": 75.0, "satisfaction_score": 70.0}'

  # Train and predict from a CSV file
  python main.py -db sample_company.db -t employees -y salary \\
      --predict-file new_employees.csv --output predictions.csv

  # Use a custom SQL query as the dataset
  python main.py -db sample_company.db --query "SELECT * FROM employees" -y salary

  # Save and reuse a trained model
  python main.py -db sample_company.db -t employees -y salary --save-model model.joblib
  python main.py -db sample_company.db --load-model model.joblib \\
      --predict '{"age": 30, "years_experience": 5.0, "education_level": "Bachelor", "dept_id": 1, "performance_score": 75.0, "satisfaction_score": 70.0}'
""",
    )

    # Database connection
    parser.add_argument(
        "-db", "--database",
        help="SQL database connection string or SQLite file path (e.g., sample_company.db)",
    )

    # Data source
    parser.add_argument(
        "-t", "--table",
        help="Table name to use as the dataset",
    )
    parser.add_argument(
        "--query",
        help="Custom SQL query to use as the dataset (overrides --table)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to load from the table",
    )

    # Target and task configuration
    parser.add_argument(
        "-y", "--target",
        help="Target column to predict",
    )
    parser.add_argument(
        "--task-type",
        choices=["regression", "classification"],
        default=None,
        help="Task type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Actions
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List all tables in the database",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Show a comprehensive database overview",
    )
    parser.add_argument(
        "--analyze",
        choices=["summary", "correlations", "insights", "target"],
        default=None,
        help="Perform data analysis on the loaded dataset",
    )
    parser.add_argument(
        "--feature-importance",
        action="store_true",
        help="Show feature importance from the trained model",
    )
    parser.add_argument(
        "--model-info",
        action="store_true",
        help="Show information about the trained model",
    )

    # Prediction inputs
    parser.add_argument(
        "--predict",
        help="JSON string of data to predict on (e.g., '{\"age\": 30, \"salary\": 50000}')",
    )
    parser.add_argument(
        "--predict-file",
        help="CSV file containing data to predict on",
    )

    # Model management
    parser.add_argument(
        "--save-model",
        metavar="PATH",
        help="Save the trained model to the given path",
    )
    parser.add_argument(
        "--load-model",
        metavar="PATH",
        help="Load a previously trained model from the given path",
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        help="Save predictions to a CSV file at the given path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output",
    )

    return parser


def _print_json(data: Any) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


def _print_table(headers: List[str], rows: List[List[Any]]) -> None:
    """Print data as a simple ASCII table."""
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in str_rows:
        print(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


def _load_prediction_data(predict_json: Optional[str], predict_file: Optional[str]) -> Optional[pd.DataFrame]:
    """Load prediction data from JSON string or CSV file."""
    if predict_json and predict_file:
        raise ValueError("Cannot use both --predict and --predict-file at the same time.")

    if predict_json:
        try:
            data = json.loads(predict_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --predict: {e}")
        if isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError("--predict must be a JSON object or array of objects")

    if predict_file:
        if not os.path.exists(predict_file):
            raise FileNotFoundError(f"Prediction file not found: {predict_file}")
        return pd.read_csv(predict_file)

    return None


def _format_predictions(predictions: pd.DataFrame) -> str:
    """Format predictions DataFrame as a readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("PREDICTIONS")
    lines.append("=" * 60)

    headers = ["#"] + list(predictions.columns)
    rows = []
    for i, (_, row) in enumerate(predictions.iterrows(), 1):
        formatted = [str(i)]
        for col in predictions.columns:
            val = row[col]
            if isinstance(val, float):
                formatted.append(f"{val:.4f}")
            else:
                formatted.append(str(val))
        rows.append(formatted)

    # Build table into the string (no direct printing)
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    lines.append(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * w for w in widths))
    for row in str_rows:
        lines.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))

    lines.append("=" * 60)
    return "\n".join(lines)


def run_cli(argv: Optional[List[str]] = None) -> int:
    """Run the ML Agent CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate required arguments
    if not args.database and not args.load_model:
        parser.error("--database is required (or use --load-model with a saved model)")

    try:
        # Initialize agent
        agent = MLAgent(
            args.database,
            task_type=args.task_type,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )

        # ========== Database exploration ==========
        if args.list_tables:
            tables = agent.list_tables()
            if args.json:
                _print_json({"tables": tables})
            else:
                print("Tables in database:")
                for t in tables:
                    print(f"  - {t}")
            agent.close()
            return 0

        if args.overview:
            overview = agent.get_database_overview()
            if args.json:
                _print_json(overview)
            else:
                print("=" * 60)
                print("DATABASE OVERVIEW")
                print("=" * 60)
                print(f"Connection: {overview['connection_string']}")
                print(f"Tables: {overview['table_count']}")
                for t in overview["tables"]:
                    print(f"\n  Table: {t['name']} ({t['row_count']} rows)")
                    print(f"  Columns:")
                    for col in t["columns"]:
                        pk = " [PK]" if col["primary_key"] else ""
                        print(f"    - {col['name']} ({col['type']}){pk}")
                    if t["foreign_keys"]:
                        print(f"  Foreign Keys:")
                        for fk in t["foreign_keys"]:
                            print(f"    - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
            agent.close()
            return 0

        # ========== Load model (skip data loading) ==========
        if args.load_model:
            if not os.path.exists(args.load_model):
                print(f"Error: Model file not found: {args.load_model}", file=sys.stderr)
                agent.close()
                return 1
            agent.load_model(args.load_model)
            print(f"Loaded model from {args.load_model}")
            if args.model_info:
                info = agent.get_model_info()
                if args.json:
                    _print_json(info)
                else:
                    print("=" * 60)
                    print("MODEL INFORMATION")
                    print("=" * 60)
                    for k, v in info.items():
                        if isinstance(v, dict):
                            fmt = {}
                            for mk, mv in v.items():
                                if isinstance(mv, (int, float)):
                                    fmt[mk] = round(float(mv), 4)
                                else:
                                    fmt[mk] = mv
                            print(f"  {k}: {fmt}")
                        else:
                            print(f"  {k}: {v}")
            if args.feature_importance:
                imp = agent.get_feature_importance()
                if imp is not None:
                    if args.json:
                        _print_json(imp.to_dict(orient="records"))
                    else:
                        print("\nFeature Importance:")
                        print(imp.to_string(index=False))
                else:
                    print("No feature importance available for this model.")

            # Make predictions with loaded model
            pred_df = _load_prediction_data(args.predict, args.predict_file)
            if pred_df is not None:
                predictions = agent.predict(pred_df)
                if args.output:
                    predictions.to_csv(args.output, index=False)
                    print(f"Predictions saved to {args.output}")
                if args.json:
                    _print_json(predictions.to_dict(orient="records"))
                else:
                    print(_format_predictions(predictions))

            agent.close()
            return 0

        # ========== Load dataset ==========
        if args.query:
            agent.load_query_as_data(args.query)
            if args.verbose:
                print(f"Loaded query results: {len(agent.current_df)} rows")
        elif args.table:
            agent.load_table(args.table, limit=args.limit)
            if args.verbose:
                print(f"Loaded table '{args.table}': {len(agent.current_df)} rows")
        else:
            # No data source specified - show help
            parser.error("Either --table, --query, or --load-model is required")

        # ========== Analysis ==========
        if args.analyze:
            analysis = agent.analyze(target_column=args.target, analysis_type=args.analyze)
            if args.json:
                _print_json(analysis)
            else:
                if args.analyze == "summary":
                    print("=" * 60)
                    print("DATA ANALYSIS SUMMARY")
                    print("=" * 60)
                    stats = analysis["basic_stats"]
                    print(f"Rows: {stats['rows']}, Columns: {stats['columns']}")
                    print(f"Numeric columns: {', '.join(stats['numeric_columns'])}")
                    print(f"Categorical columns: {', '.join(stats['categorical_columns'])}")
                    print(f"Missing values: {stats['missing_values']}")
                    print(f"Duplicate rows: {stats['duplicate_rows']}")
                    if "numeric_summary" in stats:
                        print("\nNumeric Summary:")
                        for col, desc in stats["numeric_summary"].items():
                            print(f"  {col}: mean={desc.get('mean', 0):.2f}, "
                                  f"std={desc.get('std', 0):.2f}, "
                                  f"min={desc.get('min', 0):.2f}, "
                                  f"max={desc.get('max', 0):.2f}")
                    if "target_analysis" in analysis and "error" not in analysis["target_analysis"]:
                        print("\nTarget Analysis:")
                        ta = analysis["target_analysis"]
                        print(f"  Target: {ta['target_column']} ({ta['dtype']})")
                        if "distribution" in ta:
                            print(f"  Distribution: {ta['distribution']}")
                        if "correlations_with_numeric_features" in ta:
                            print(f"  Top correlations: {ta['correlations_with_numeric_features']}")
                elif args.analyze == "correlations":
                    print("Correlation Matrix:")
                    for col, corrs in analysis.items():
                        print(f"  {col}: {corrs}")
                elif args.analyze == "insights":
                    print("Column Insights:")
                    for insight in analysis:
                        print(f"  {insight['column']} ({insight['dtype']}):")
                        for k, v in insight.items():
                            if k not in ("column", "dtype"):
                                print(f"    {k}: {v}")
                elif args.analyze == "target":
                    print("Target Analysis:")
                    for k, v in analysis.items():
                        print(f"  {k}: {v}")
            agent.close()
            return 0

        # ========== Training ==========
        if args.target:
            results = agent.train(
                target_column=args.target,
                task_type=args.task_type,
            )
            if args.json:
                # Convert feature_importance to serializable
                results_json = dict(results)
                if results_json.get("feature_importance") is not None:
                    results_json["feature_importance"] = results_json["feature_importance"].to_dict(orient="records")
                _print_json(results_json)
            else:
                print(agent.format_results(results))

            # Save model if requested
            if args.save_model:
                agent.save_model(args.save_model)
                print(f"\nModel saved to {args.save_model}")

            # Show feature importance if requested
            if args.feature_importance:
                imp = agent.get_feature_importance()
                if imp is not None:
                    print("\nFeature Importance:")
                    print(imp.to_string(index=False))

            # Show model info if requested
            if args.model_info:
                info = agent.get_model_info()
                print("\nModel Information:")
                for k, v in info.items():
                    if isinstance(v, dict):
                        fmt = {}
                        for mk, mv in v.items():
                            if isinstance(mv, (int, float)):
                                fmt[mk] = round(float(mv), 4)
                            else:
                                fmt[mk] = mv
                        print(f"  {k}: {fmt}")
                    else:
                        print(f"  {k}: {v}")

            # Make predictions
            pred_df = _load_prediction_data(args.predict, args.predict_file)
            if pred_df is not None:
                predictions = agent.predict(pred_df)
                if args.output:
                    predictions.to_csv(args.output, index=False)
                    print(f"\nPredictions saved to {args.output}")
                if args.json:
                    _print_json(predictions.to_dict(orient="records"))
                else:
                    print(_format_predictions(predictions))

            agent.close()
            return 0

        # ========== No action specified ==========
        if args.verbose:
            print(f"Loaded {len(agent.current_df)} rows from {agent.current_table}")
            print(f"Columns: {', '.join(agent.current_df.columns)}")
            print("\nUse --analyze to analyze data, or -y/--target to train a model.")
        else:
            parser.print_help()

        agent.close()
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """Entry point for the CLI."""
    sys.exit(run_cli())


if __name__ == "__main__":
    main()