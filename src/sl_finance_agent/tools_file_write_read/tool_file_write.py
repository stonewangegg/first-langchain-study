"""
This module provides LangChain-compatible tools for writing files to the local
filesystem from within an agent. It exposes two main tools:

1. ``tool_custom_file_write``
    A general-purpose plain-text file writer. It supports writing any text-based
    content (e.g. ``.txt``, ``.json``, ``.md``, ``.csv``) to a file located under
    a pre-configured root directory (``FILE_ROOT_DIR``). The tool performs path
    validation, content sanity checks, and delegates the actual write operation
    to LangChain's built-in :class:`WriteFileTool`.

2. ``tool_generate_word_doc``
    A specialized tool that converts Markdown-formatted text (headings, bold,
    and bullet lists) into a Microsoft Word document (``.docx``) using
    ``python-docx`` and saves it to a validated path under ``FILE_ROOT_DIR``.

Environment
-----------
- ``FILE_ROOT_DIR`` (optional)
    Environment variable that constrains where files may be written. If unset,
    the current working directory is used. Any write attempt targeting a path
    outside this root will be rejected (for ``tool_custom_file_write``) or
    fall back to this root (for ``tool_generate_word_doc``).

Typical use
-----------
These tools are decorated with ``@tool`` (or ``@tool(args_schema=...)``) so
they can be registered directly with a LangChain agent::

    from tools_file_write_read.tool_file_write import (
        tool_custom_file_write,
        tool_generate_word_doc,
    )
"""

import os
import re
import time

from docx import Document
from langchain.tools import tool
from langchain_community.tools import WriteFileTool
from pydantic import BaseModel, Field, field_validator

from ..common import get_logger
logger = get_logger(__name__)

FILE_ROOT_DIR = os.environ.get("FILE_ROOT_DIR", os.getcwd())

@tool
def tool_custom_file_write(file_path:str, content: str) -> str:
    """
    Write txt content to a file. Such as text, json, markdown.
    Args:
       file_path: The path save the file. e.g. path/to/your/file.md
       content: The text content to write to the file.
    Return: The file path if success, else return error message.
    """
    # Initialize the tool
    # Ensure the root_dir is set to where you want the file to be created
    file_full_path = file_path
    try:
        dir_name = os.path.dirname(file_full_path)
        if dir_name == FILE_ROOT_DIR:
            logger.info("Write file, file path is under current woring directory: %s", file_full_path)
        else:
            logger.error("⚠️ Write file: file path is not under the current file directory: %s", file_full_path)
            return f"""⚠️ Write file: file path is not under the current file directory: {file_full_path},
                                check the file path parameter"""
    except ValueError as e:
        logger.error("❌ Write file, Path is wrong: %s", file_full_path)
        return f"❌ Write file, {str(e)} Path is wrong: {file_full_path}, check the file path parameter"

    try:
        if content is None or len(content) == 0:
            logger.error("❌ Write file, content is empty")
            raise ValueError("❌ Write file, content is empty")

        write_tool = WriteFileTool(root_dir=file_full_path)

        # Write the file
        result = write_tool.run({
            "file_path": file_full_path,
            "text": content
        })
        
        logger.info("✅ Write file: %s, Result: %s", file_full_path, result)
        return str(result)

    except Exception as e:
        logger.error("❌ Write file error: %s", str(e))
        return f"Write file error: {str(e)}"

# --- MARKDOWN PARSER ---
def _add_markdown_content_to_docx(doc, content: str):
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip().replace("**", ""), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip().replace("**", ""), level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip().replace("**", ""), level=3)
        else:
            p = (
                doc.add_paragraph(style="List Bullet")
                if line.startswith(("- ", "* "))
                else doc.add_paragraph()
            )
            text = line[2:] if line.startswith(("- ", "* ")) else line
            parts = re.split(r"(\*\*.*?\*\*)", text)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    p.add_run(part[2:-2]).bold = True
                else:
                    p.add_run(part)

# Step 1: Define strict validation
class TargetPathValidation(BaseModel):
    """
    A strict validation for the file path of LLM call parameter
    """
    store_path: str = Field(description="The exact file path of LLM call")
    
    @field_validator('store_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """
        Validate the file path of LLM call parameter before tool call
        """
        v = v.strip().lower()
        if not v.startswith(FILE_ROOT_DIR):
            logger.error(" The file path is invalid: %s ... The correct should start with: %s", v, FILE_ROOT_DIR)
            raise ValueError(
                f"Invalid file path '{v}'. The correct should start with '{FILE_ROOT_DIR}'. Check the file path parameter and retry."
            )
        return v

# Step 2: Define tool with schema
@tool(args_schema=TargetPathValidation)
def tool_generate_word_doc(title: str, content: str, store_path:str) -> str:
    """
    Generate a Word document (.docx) containing the given title and content, and save to store_path.
    The content supports Markdown formatting (headings, bold text, bullet points).

    :param title: file title
    :param content: file content
    :paran store_path: file store path
    :return: file saved path
    """

    filename = f"{title.replace(' ', '_')}_{int(time.time())}.docx"
    logger.debug("Generating Word document: %s", filename)

    # 保存路径
    local_path_file = ""
    local_path = store_path
    if local_path.startswith(FILE_ROOT_DIR):
        if os.path.isdir(local_path):
            local_path_file = os.path.join(local_path, filename)
        elif local_path.endswith(".docx"):
            local_path_file = local_path
        else:
            logger.error("❌ Invalid Word document generate PATH: %s, STOP and check the path parameter!", local_path)
            return f"Invalid document generate PATH: {local_path}, STOP and check the path parameter!"

        logger.info("📌 Specify Word document generate PATH: %s", local_path_file)
    else:
        os.makedirs(FILE_ROOT_DIR, exist_ok=True)
        local_path_file = os.path.join(FILE_ROOT_DIR, filename)
        logger.warning("❌ Invalid Word document generate PATH: %s, 📌 use the default file root path : %s ", store_path, local_path_file)

    try:
        doc = Document()
        doc.add_heading(title, 0)
        _add_markdown_content_to_docx(doc, content)
        doc.save(local_path_file)

        # return the file down load link
        logger.info("✅ Taget document generated successfully: %s", local_path_file)
        return "✅ Target document generated successfully: {local_path_file}"
    except Exception as e:
        logger.error("❌ Word generated Error: %s", e)
        return f"❌ Word generated Error: {str(e)}"