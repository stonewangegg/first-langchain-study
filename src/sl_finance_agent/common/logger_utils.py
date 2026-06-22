"""
Initial the logging handler and create the logger for all libs
"""

import logging
import os
from pathlib import Path

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
