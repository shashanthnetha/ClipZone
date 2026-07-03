# -*- coding: utf-8 -*-
"""Centralized logging module for ClipPilot."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a logger with standard formatting and console/file support."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # We set base logger level to DEBUG so handlers can filter appropriately
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # Console Logging
    console_level_name = os.environ.get("CLIPPILOT_LOG_LEVEL", "INFO").upper()
    console_level = getattr(logging, console_level_name, logging.INFO)
    
    c_handler = logging.StreamHandler()
    c_handler.setLevel(console_level)
    
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    
    # Optional File Logging
    log_file_env = os.environ.get("CLIPPILOT_LOG_FILE")
    log_file_path: Optional[Path] = None
    
    if log_file_env:
        log_file_path = Path(log_file_env)
    elif os.environ.get("CLIPPILOT_FILE_LOGGING", "").lower() in ("true", "1", "yes"):
        from . import config as cfg
        try:
            cfg.ensure_dirs()
            log_file_path = cfg.LOG_DIR / "clippilot.log"
        except Exception:
            pass
            
    if log_file_path:
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            f_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            f_handler.setLevel(logging.DEBUG)
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
        except Exception:
            pass
            
    return logger
