"""
ML Agent - An intelligent agent that processes SQL databases,
selects the best machine learning model, and makes predictions.
"""

from .agent import MLAgent
from .database import DatabaseProcessor
from .model_selector import ModelSelector
from .predictor import Predictor
from .analyzer import DataAnalyzer
from .preprocessor import DataPreprocessor
from .llm_advisor import LLMAdvisor, LLMAdvisorError, create_llm_advisor
from .relational import RelationalFeatureSynthesizer
from .experiments import ExperimentTracker
from .sql_validation import SQLValidator
from .anomaly import AnomalyDetector
from .model_monitor import ModelMonitor
from .recipes import RecipeStore
from .snapshots import SnapshotStore
from .cli import build_parser, run_cli, main

__version__ = "1.4.0"
__all__ = [
    "MLAgent", "DatabaseProcessor", "ModelSelector", "Predictor", "DataAnalyzer",
    "DataPreprocessor", "LLMAdvisor", "LLMAdvisorError", "create_llm_advisor",
    "RelationalFeatureSynthesizer", "ExperimentTracker", "SQLValidator",
    "AnomalyDetector", "ModelMonitor", "RecipeStore", "SnapshotStore",
    "build_parser", "run_cli", "main",
]
