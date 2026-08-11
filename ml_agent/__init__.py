"""
ML Agent - An intelligent agent that processes SQL databases,
selects the best machine learning model, and makes predictions.
"""

from .agent import MLAgent
from .database import DatabaseProcessor
from .model_selector import ModelSelector
from .predictor import Predictor
from .analyzer import DataAnalyzer
from .cli import build_parser, run_cli, main

__version__ = "1.0.0"
__all__ = [
    "MLAgent", "DatabaseProcessor", "ModelSelector", "Predictor", "DataAnalyzer",
    "build_parser", "run_cli", "main",
]
