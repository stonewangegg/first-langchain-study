"""
Tools for file read and write 
"""

__version__ = "1.0.0"

from .tool_file_read import tool_custom_file_read
from .tool_file_write import tool_custom_file_write
from .tool_file_write import tool_generate_word_doc

__all__ = ["tool_custom_file_read", "tool_custom_file_write", "tool_generate_word_doc"]