"""Structured logging helpers.

Production readiness rule #2: use the ``logging`` module instead of bare
``print``/``sys.stderr`` so output is formatted, level-filterable and can be
routed to files / log shippers (JSON, syslog, CloudWatch, ...) in production.

Nothing in this module should raise on import -- logging setup is best-effort.
"""
from __future__ import annotations

import logging
from typing import Optional

from . import config


def configure_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure the root logger once and return the package logger.

    Safe to call multiple times; repeated calls only update the level of the
    root logger rather than re-adding duplicate handlers.
    """
    level = (level or config.LOG_LEVEL).strip().upper()
    format_string = format_string or config.LOG_FORMAT

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    # Add a console handler *once* so repeated config calls don't duplicate.
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_string))
        root.addHandler(handler)

    # Give the Flask/Werkzeug access logs a consistent, quieter level.
    logging.getLogger("werkzeug").setLevel(
        logging.WARNING if level == "INFO" else level
    )

    return logging.getLogger("ml_agent")