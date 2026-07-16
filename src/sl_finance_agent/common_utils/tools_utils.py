"""
All tool utils for agent
"""

from datetime import datetime

from langchain.tools import tool

from .model_obj import FILE_DIR


# assistant function
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return "Current date and time is: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_file_dir() -> str:
    """Get the file working dir: FILE_DIR."""
    return FILE_DIR


