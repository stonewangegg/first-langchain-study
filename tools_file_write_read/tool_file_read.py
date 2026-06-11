"""
Module providing a LangChain tool for reading files of various formats and returning their text content.

This module exposes :func:`tool_read_file`, a LangChain ``@tool``-decorated function that reads a
file from disk, selects an appropriate document loader based on the file extension, and returns the
extracted text content.

Supported file extensions / loaders:
    * ``.txt``, ``.md``, ``.json``, ``.yaml``, ``.yml``, ``.xml``, ``.html``,
      ``.js``, ``.log``, ``.cfg``, ``.ini`` -- :class:`langchain_community.document_loaders.TextLoader`
    * ``.docx`` -- :class:`langchain_community.document_loaders.Docx2txtLoader`
    * ``.pdf``  -- :class:`langchain_community.document_loaders.PDFPlumberLoader`
    * ``.xls``  -- :class:`langchain_community.document_loaders.UnstructuredExcelLoader`
    * ``.csv``  -- :class:`langchain_community.document_loaders.CSVLoader`

Environment variables:
    FILE_ROOT_DIR: Optional root directory used to validate that the provided ``file_path``
        resolves inside an allowed location. Defaults to the current working directory.

Logging:
    A module-level logger (``logging.getLogger(__name__)``) is used to report successful reads
    and any errors encountered while loading or parsing a file.
"""

# get the logger
import logging
import os
from pathlib import Path

from langchain.tools import tool
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader

logger = logging.getLogger(__name__)

FILE_ROOT_DIR = os.environ.get("FILE_ROOT_DIR", os.getcwd())

@tool
def tool_custom_file_read (file_path:str) -> str:
    """
    Reads a file and returns the extracted text content.
    Args: file_path (str): The name with full path of the document need to be read. e.g. path/to/your/document.pdf
    Returns: The extracted txt content from the file
    """
    # check the passed in file path
    file_full_path = file_path
    try:
        dir_name = os.path.dirname(file_full_path)
        if dir_name == FILE_ROOT_DIR:
            logger.info("File is under valid directory: %s", file_full_path)
        else:
            logger.error("⚠️ File path is not under the current file directory: %s", file_full_path)
            return f"File path is not under the current file directory: {file_full_path}, check the input file path"
    except ValueError as e:
        logger.error("❌ Paths are wrong: %s", file_full_path)
        return f"❌ Paths are wrong: {str(e)}, check the passed in parameter"

    # Get the file type
    ext = Path(file_full_path).suffix.lower()

    try:
        loader = None
        if ext in [".txt",".md",".json",".yaml",".xml",".html",".js",".log",".yml",".cfg",".ini"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext==".docx":
            loader = Docx2txtLoader(file_full_path)
        elif ext==".pdf":
            loader = PDFPlumberLoader(file_full_path)
        elif ext==".xls":
            loader = UnstructuredExcelLoader(file_full_path)
        elif ext==".csv":
            loader= CSVLoader(file_full_path)
        else:
            raise ValueError(f" ❌ Unsupported file type: {ext}")

        docs = loader.load()
        if not docs:
            logger.error("⚠️ Load file failed with file: %s", file_full_path)
            return f"⚠️ Load file failed with file: {file_full_path}, check the passed in parameters!"
        
        full_text = "\n\n".join([doc.page_content for doc in docs])

        logger.info("✅ Complete read the file: %s", file_full_path)
        return f"Complete read the file: {file_full_path}, content: " + full_text

    except Exception as e:
        logger.error("❌ Error reading file: %s, %s", file_full_path, str(e))
        return f"读取文件失败:{file_full_path}, {str(e)}"