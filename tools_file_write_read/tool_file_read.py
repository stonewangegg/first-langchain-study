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

Logging:
    A module-level logger (``logging.getLogger(__name__)``) is used to report successful reads
    and any errors encountered while loading or parsing a file.
"""

# get the logger
import logging
from pathlib import Path

from langchain.tools import tool
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader

logger = logging.getLogger(__name__)

@tool
def tool_custom_file_read (file_path:str) -> str:
    """
    Reads a file and returns the extracted text content.
    Args: file_path (str): The name with full path of the file need to be read. e.g. /path/to/your/document.pdf
    Returns: The extracted txt content from the file
    """
    # check the passed in file path
    file_full_path = Path(file_path)
    if file_full_path.is_file():
        logger.info("File is under valid directory and exist: %s", file_full_path)
    else:
        logger.error("⚠️ File path is wriong or not exist: %s", file_full_path)
        return f"File path is wriong or not exist: {file_full_path}, check the input file path, try again"


    # Get the file type
    ext = file_full_path.suffix.lower()

    try:
        loader = None
        if ext in [".txt",".md",".json",".yaml",".xml",".html",".js",".log",".yml",".cfg",".ini"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext==".docx":
            loader = Docx2txtLoader(file_path)
        elif ext==".pdf":
            loader = PDFPlumberLoader(file_path)
        elif ext==".xls":
            loader = UnstructuredExcelLoader(file_path)
        elif ext==".csv":
            loader= CSVLoader(file_path)
        else:
            raise ValueError(f" ❌ Unsupported file type: {ext}")

        docs = loader.load()
        if not docs:
            logger.error("⚠️ Load file failed with file: %s", file_full_path)
            return f"⚠️ Document load file failed with file: {file_full_path}, check the passed in parameters!"
        
        full_text = "\n\n".join([doc.page_content for doc in docs])

        logger.info("✅ Complete read the file: %s", file_full_path)
        return f"Complete read the file: {file_full_path}, content: " + full_text

    except Exception as e:
        logger.error("❌ Error reading file: %s, %s", file_full_path, str(e))
        return f"Read file faild: {file_full_path}, {str(e)}"