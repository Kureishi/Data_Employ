#!/usr/bin/env python3
"""
ML Agent Command-Line Interface Entry Point.

Usage:
    python main.py -db sample_company.db --list-tables
    python main.py -db sample_company.db -t employees -y salary
    python main.py -db sample_company.db -t employees -y salary --predict '{"age": 30}'
    python main.py -h
"""

from ml_agent.cli import main

if __name__ == "__main__":
    main()