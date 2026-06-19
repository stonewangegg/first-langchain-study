"""LangChain tools for reading files and retrieving content from large PDFs.

Exposes three ``@tool``-decorated functions:

* :func:`tool_custom_file_read` -- reads a file from disk and returns its extracted text,
  selecting the loader by extension (``TextLoader`` / ``Docx2txtLoader`` / ``PDFPlumberLoader`` /
  ``UnstructuredExcelLoader`` / ``CSVLoader``).
* :func:`tool_lightRAG_large_file_read` -- chunks a large PDF page-by-page and persists the
  embeddings as a FAISS index at ``./faiss_index`` for later retrieval.
* :func:`tool_lightRAG_search_docs` -- queries that FAISS index with a natural-language
  question and returns the top-5 matching snippets.

A module-level logger (``logging.getLogger(__name__)``) is used to report successful reads
and any errors encountered while loading, parsing, or retrieving content.
"""

# get the logger
import logging
import os
from pathlib import Path

from langchain.tools import tool
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

FILE_DIR = os.environ.get("FILE_DIR", "./tmp")

@tool
def tool_custom_file_read (file_path:str) -> str:
    """
    Reads the target file and returns the extracted content.
    Args: file_path (str): The path of the file need to be read. e.g. /path/to/your/document.pdf
    Returns: The extracted txt content from the file
    """
    # check the passed in file path
    file_full_path = Path(file_path)
    if file_full_path.is_file():
        logger.info("File is under valid directory and exist: %s", file_full_path)
    else:
        # check the file with the actual physical path
        file_full_path = Path.cwd() / FILE_DIR / file_path.lstrip("/") 
        if(file_full_path.is_file()):
            logger.info("File is under valid directory and exist with actual physical path: %s", file_full_path)
        else:
            logger.error("⚠️ File path is wrong or not exist: %s", file_full_path)
            return f"File path is wriong or not exist: {file_full_path}, check the input file path, try again"

    # Get the file type
    ext = file_full_path.suffix.lower()
    try:
        loader = None
        if ext in [".txt",".md",".json",".yaml",".xml",".html",".js",".log",".yml",".cfg",".ini"]:
            loader = TextLoader(file_full_path, encoding="utf-8")
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

        full_text = ""
        if ext==".pdf":
            page_content = []
            for page in loader.lazy_load():
                if not page:
                    logger.error("⚠️ Load PDF file with lazy load failed with file: %s", file_full_path)
                    return f"⚠️ Load PDF file with lazy load failed: {file_full_path}, check the passed in file!"
                page_content.append(page.page_content)
            full_text = "\n\n".join(page_content)
        else:
            docs = loader.load()
            if not docs:
                logger.error("⚠️ Load file failed with file: %s", file_full_path)
                return f"⚠️ Document load file failed with file: {file_full_path}, check the passed in parameters!"
            full_text = "\n\n".join([page.page_content for page in docs])
        
        # check the final result
        if full_text:
            logger.info("✅ Complete read the file: %s", file_full_path)
            return f"Complete read the file: {file_full_path}, full content is: " + full_text
        else:
            logger.info("❌ Failed read the file content: %s", file_full_path)
            return f"Failed read the file content, nothing has been read, check: {file_full_path}"
        

    except Exception as e:
        logger.error("❌ Error reading file: %s, %s", file_full_path, str(e))
        return f"Read file faild: {file_full_path}, {str(e)}. Try to use the built-in `read_file` tool."
    
@tool
def tool_lightRAG_large_file_read(pdf_file_path:str) -> str:
    """
    Reads a large PDF file by chunking its content into a FAISS vector store for later retrieval.

    This tool is intended for PDF files that are too large to fit into the LLM context window
    directly. It lazily loads the PDF page by page, splits each page into smaller text chunks
    using a :class:`RecursiveCharacterTextSplitter` (chunk_size=1500, chunk_overlap=200), and
    embeds those chunks into a FAISS vector store using OpenAI embeddings. The resulting
    vector store is persisted to the local filesystem at ``./faiss_index`` so it can be
    queried by the companion tool :func:`tool_lightRAG_search_docs`.

    Side Effects:
        Persists a FAISS index to ``./faiss_index`` (containing ``index.faiss`` and
        ``index.pkl``). Any pre-existing content in that directory is overwritten.

    Args:
        pdf_file_path (str): The path of the PDF file to be read. May be an absolute path,
            a path relative to the current working directory. The file must exist and have a ``.pdf`` extension.

    Returns:
        str: A status message describing the outcome.
        * On success: a message confirming that the FAISS vector store was saved to
          ``./faiss_index`` and instructions to query it with
          :func:`tool_lightRAG_search_docs`.
        * On failure: an error message explaining why the file could not be processed
          (e.g. invalid/missing path, unsupported file type, or an empty/unreadable
          vector store).
    """

    # check the passed in file path
    file_full_path = Path(pdf_file_path)
    if file_full_path.is_file():
        logger.info("Large file is under valid directory and exist: %s", file_full_path)
    else:
        # check the file with the actual physical path
        file_full_path = Path.cwd() / FILE_DIR / pdf_file_path.lstrip("/") 
        if file_full_path.is_file():
            logger.info("Large file is under valid directory and exist with actual physical path: %s", file_full_path)
        else:
            logger.error("⚠️ large file path is wrong or not exist: %s", file_full_path)
            return f"Large file path is wriong or not exist: {file_full_path}, check the input file path, try again"
        
    ext = file_full_path.suffix.lower()
    if not ext == ".pdf":
        logger.error("❌ Unsupported file type: %s", ext)
        return f"❌ Unsupported file type: {ext}"
    
    loader = PDFPlumberLoader(file_full_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    # read the large PDF and chunk to FAISS vector store
    vectorstore = None
    buffer = []

    logger.info("🚀 Start reading PDF and writing result to the vector store")
    for page_doc in loader.lazy_load():

        chunks = splitter.split_documents(
            [page_doc]
        )

        buffer.extend(chunks)

        if len(buffer) >= 100:

            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    buffer,
                    OpenAIEmbeddings()
                )

            else:
                vectorstore.add_documents(
                    buffer
                )

            buffer.clear()

    # flush remaining
    if buffer:
        if vectorstore is None:
            vectorstore = FAISS.from_documents(
                buffer,
                OpenAIEmbeddings()
            )

        else:
            vectorstore.add_documents(
                buffer
            )

    if vectorstore:
        vectorstore.save_local(folder_path="./faiss_index")
        return """Tool call success, the result of this tool call that save as FAISS vector store: faiss_index was saved in the filesystem at this path: ./faiss_index
        with `vectorstore.save_local("./faiss_index")`
        directory:
        faiss_index/
        ├── index.faiss
        └── index.pkl

        You can read the result from the filesystem by using the `tool_lightRAG_search_docs` tool with any question string. 
        You should Retrieve the content you need with corresponding question use `tool_lightRAG_search_docs` tool.
        """
    else:
        logger.error("❌ Failed tool call, vector store is not available.")
        return "Failed tool call, vector store is not available. Check and try again."

@tool
def tool_lightRAG_search_docs(question: str) -> str:
    """
    Searches a previously built FAISS vector store for content relevant to a natural-language
    question and returns the matching document snippets.

    This tool is the read-side companion to :func:`tool_lightRAG_large_file_read`. It loads the
    FAISS index persisted at ``./faiss_index`` (created by the large-file read tool), builds a
    retriever configured to return the top ``k=5`` most relevant chunks, and invokes it with
    the supplied question using OpenAI embeddings for similarity search. The retrieved
    documents' ``page_content`` fields are then joined and returned as a single string.

    Note:
        The FAISS index at ``./faiss_index`` must already exist (typically created by first
        calling :func:`tool_lightRAG_large_file_read`). This tool does not create the index.
        ``allow_dangerous_deserialization=True`` is set on :meth:`FAISS.load_local`, which
        deserializes pickled data from disk and should only be used with trusted index files.

    Args:
        question (str): The natural-language question or query used to retrieve the most
            relevant chunks from the vector store.

    Returns:
        str: The concatenated ``page_content`` of the top retrieved documents. 
        If no documents are retrieved, an error message is returned that
        suggests checking the question and confirming that the ``./faiss_index`` vector
        store was built and persisted correctly.
    """
    vectorstore = FAISS.load_local(
        "./faiss_index",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(question)

    if docs:
        return "\n\n".join(
            d.page_content
            for d in docs
        )

    logger.error("Failed tool call, retrieve docs failed with question: %s, to vector store: './faiss_index'. Check and try again.", question)
    return f"Failed tool call, retrieve docs failed with question: {question}, to vector store: './faiss_index'. Check and try again."


if __name__ == "__main__":

    test_file_path = "/home/stonewang/study/langchain/first-start/first-start/src/tmp/reports/2023_annual/300475_2023年年度报告.pdf"
    full_text_content = []
    loader = PDFPlumberLoader(test_file_path)

    for page in loader.lazy_load():
        if not page:
            print(f"⚠️ Load PDF file with lazy load failed with file: {test_file_path}")
        
        print("Page metadata: " + str(page.metadata) + "\n")
        content = page.page_content
        full_text_content.append(content)
    
    print(f"FULL TEXT CONTENT: \n\n{"\n\n".join(full_text_content)}")