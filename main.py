#!/usr/bin/env python3
"""
ML Agent Command-Line Interface Entry Point.

Usage:
    python main.py -db sample_company.db --list-tables
    python main.py -db sample_company.db -t employees -y salary
    python main.py -db sample_company.db -t employees -y salary --predict '{"age": 30}'
    python main.py -h

This entry point auto-installs any missing dependencies (pandas, numpy,
scikit-learn, SQLAlchemy, joblib, tabulate) into the active virtual
environment using pip before launching the CLI.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"

# Core packages needed before we can even import ml_agent.
# Keys are the pip package names; values are the import module names
# (they differ for scikit-learn -> sklearn).
CORE_DEPS = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "sqlalchemy": "sqlalchemy",
    "joblib": "joblib",
    "tabulate": "tabulate",
    "flask": "flask",
}


def _is_venv() -> bool:
    """Check if we're running inside a virtual environment."""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


def _ensure_dependencies() -> None:
    """Auto-install any missing packages from requirements.txt into the venv."""
    missing = []
    for pip_name, import_name in CORE_DEPS.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)

    if not missing:
        return

    if not _is_venv():
        print(
            "Warning: Not running inside a virtual environment. "
            "It is recommended to use a venv. Installing into the current "
            "Python environment instead.",
            file=sys.stderr,
        )

    print(
        f"Installing missing dependencies: {', '.join(missing)} ...",
        file=sys.stderr,
    )
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)]
        )
        print("Dependencies installed successfully.", file=sys.stderr)
    except Exception as e:
        print(
            f"Failed to auto-install dependencies: {e}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    _ensure_dependencies()
    from ml_agent.cli import main

    main()