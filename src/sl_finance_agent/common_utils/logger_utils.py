"""
Initial the logging handler and create the logger for all libs
Here are two different logger
1 Python built-in logging
2 Uru logger
"""

import logging
import os
from pathlib import Path
import sys
from loguru import logger

FILE_DIR = os.environ.get("FILE_DIR", "./tmp")
LOGGER_LEVEL = os.environ.get("LOGGER_LEVEL", "INFO")

# initial the logging
logging.basicConfig(
    level=getattr(logging, LOGGER_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(FILE_DIR) / Path("sl_finance_agent.log")),
        logging.StreamHandler()
    ]
)

def get_logger(logger_name: str) -> logging.Logger:
    return logging.getLogger(logger_name)

class CustomUruLog():

    def __init__(self) -> None:
        self._debug_log = []  # Collect debug messages for response
        self._rurlogger = logger

        # initial the uru logger
        # Remove the default logger
        logger.remove()

        # Console
        logger.add(
            sys.stdout,
            level="INFO",
            colorize=True,
            enqueue=True,          # Thread/process safe
            backtrace=False,
            diagnose=False,        # Don't expose local variables in production
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{process.name}:{thread.name}</cyan> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )

        # General log
        logger.add(
            Path(FILE_DIR) / Path("sl_finance_agent.log"),
            level="INFO",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            enqueue=True,
            encoding="utf-8",
        )

    def log_debug(self, message: str):
        """Log debug message and collect for response when DEBUG is enabled."""
        self._rurlogger.info(f"[DEBUG] {message}")
        self._debug_log.append(message)

    def clear_debug_log(self):
        """Clear debug log for new request."""
        self._debug_log = []

    def get_logger(self):
        return self._rurlogger

uru_logger = CustomUruLog()
