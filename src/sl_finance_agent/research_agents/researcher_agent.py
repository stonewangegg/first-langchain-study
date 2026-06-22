"""Researcher deep-agent for fetching CNINFO financial disclosure reports.

Builds a ``Researcher`` deep agent (on top of ``langchain-deepagents``) that
autonomously plans, downloads, and catalogs annual/quarterly PDF reports for
a listed company from the *CNINFO* (巨潮资讯) platform, then persists the
metadata of every retrieved file as a JSON manifest inside its virtual
workspace.

Environment variables
---------------------
* ``MAX_COMPLETION_TOKENS`` — LLM context window (default ``"16384"``).
* ``FILE_DIR`` — sandbox directory for the agent's ``FilesystemBackend``
  (default ``"./tmp"``).

Quickstart
----------
Call :func:`create_researcher_agent` with a ``ModelObj`` (see
``sl_finance_agent.agent_graph``) and invoke the returned agent, e.g.::

    agent = create_researcher_agent(model_obj)
    agent.invoke({"messages": [{
        "role": "user",
        "content": (
            'Download the 2023 annual report of "平安银行" and the Q2 2024 '
            'quarterly report of "000001", then write the metadata manifest '
            'to "manifest.json".'
        ),
    }]})

The agent plans the downloads, fetches them sequentially through the
``tool_cninfo_report_downloader`` skill, and finishes by writing a JSON
manifest whose entries contain at least ``title``, ``path`` and ``type`` for
every successfully retrieved PDF.
"""

# export MAX_COMPLETION_TOKENS as a passin varaible
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Literal, cast

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware, AgentState, ToolCallLimitMiddleware, hook_config
from langchain.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

import json

from pydantic import SecretStr

from ..cninfo_report_downloader import CNInfoReportDownloader
from ..common import ModelObj, SUPPORTED_LLM_TYPES, get_logger

MAX_COMPLETION_TOKENS = os.environ.get("MAX_COMPLETION_TOKENS", "16384")

# current working directory
CURRENT_WORKING_DIR = os.getcwd()

FILE_DIR = os.environ.get("FILE_DIR", "./tmp")

# export FILE_ROOT_DIR="your/file/root/dir"
FILE_ROOT_DIR = str(Path(CURRENT_WORKING_DIR) / Path(FILE_DIR))

# system prompt for research agent
RESEARCHER_SYSTEM_PROMPT = """
# You are an expert Web Researcher tasked with providing accurate, up-to-date, and well-sourced target files. 

## Your goal is search to download all target files of required information, and save meta data of the files to the meta data file.

## Core steps
1. Make Plan: Use `get_current_time` to get current time and make a plan to list all search target files.
2. Use `tool_cninfo_report_downloader` to download the target PDF files **One By One**, refer to the skill of "cninfo-report-downloader". 
3. You must generate a json file with the meta data of download PDF files.

## Constraints 
- Critical High Rule: Do not call concurrency download request with tool, You must wait the first query complete, then send the next query.
- **Strict Review**: After obtaining results, check each one to see if it is satisfied to the query, if yes then stop the query.
- The output meta data content must inculde title, path, type for each download file. 
- **If you already have task completed, STOP and Return the final results at once**.
"""

# get the logger
logger = get_logger(__name__)

# Tool of the special annual report pdf file search and download
@tool
def tool_cninfo_report_downloader(company: str, year:str, file_type:str, quarter:Literal[1,2,3,4], store_path:str) -> str:
    """
        下载报告
        :param company: 公司信息（股票代码）
        :param year: 年份
        :param type: 报告类型，'annual' 或 'quarterly'
        :param quarter:Literal[1,2,3,4] 季度，仅季度报告需要
        :paran store_path: 文件指定保存路径, 只能是路径，不能是文件名
        :return 完成下载的文件名称与完整保存路径
    """

    # Check if the string starts with the current working path
    logger.debug("The required file store path is: %s; file root dir is: %s", store_path, FILE_ROOT_DIR)
    local_store_path = store_path
    path_obj = Path(local_store_path)
    if local_store_path.endswith("/") or not path_obj.suffix:
        if local_store_path.startswith("/"):
            local_store_path = local_store_path.removeprefix("/")
        local_store_path = str(Path(FILE_ROOT_DIR) / local_store_path)
        os.makedirs(local_store_path, exist_ok=True)
        logger.info("The required store path is valid, the phisic file store path is: %s", local_store_path)
    elif path_obj.is_file():
        logger.error("The passed in store path is a file path: %s, it must be a folder path to store download file, check and retry", local_store_path)
        return f"The passed in store path is a file path: {local_store_path}, it must be a 'folder path' to store download file, check and retry"
    else:
        logger.error ("The passed in store path: '%s' is invalid, please check and retry!", local_store_path)
        return f"The passed in store path: {local_store_path} is invalid, it should be the folder directory for file download to, please check and retry!"
        
    # initial the CNInfoReportDownloader object
    downloader = CNInfoReportDownloader()
    
    # 搜索公司信息
    logger.info("正在搜索公司: %s ...", company)
    # fire the search and download
    company_info = downloader.search_company(company)
        
    if not company_info:
        logger.error("未找到公司: %s, 检查目标参数", company)
    
    logger.info("找到公司: %s (代码: %s)", company_info['name'], company_info['code'])
    
    # 下载报告
    result = downloader.download_report(company_info, year, file_type, quarter, local_store_path)

    if not result or result == "":
        logger.warning("⚠️ 未能下载 %s %s年的 %s 报告。请检查公司名称是否正确，或核实该公司是否发布了指定年份的报告。", company, year, file_type)
        return f"未能搜索到并下载 {company}, {year} 年的 {file_type} 报告，请检查公司名称是否正确，或核实该公司是否发布了指定年份的报告。"
        
    logger.info("ℹ️目标文件下载完成！文件保存路径: %s", result)
    return result

# inital the tool call limitation with 30
toolCallLimitMiddleware = cast(AgentMiddleware, ToolCallLimitMiddleware(tool_name="tool_cninfo_report_downloader", run_limit=30, exit_behavior="end"))

# assistant function
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return "Current date and time is: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def save_json_file(
    file_path: str,
    data: Any,
) -> str:
    """
    Save structured data as a JSON file.

    Args:
        file_path: Workspace-relative path.
        data: content (dict/list/object) to save.
    """

    content = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )

    result = fs_backend.write(
        file_path=file_path,
        content=content,
    )

    return str(result)

# Message limit middleware, to prevent the comtext overflow, initial the threshold = 100
class MessageLimitMiddleware(AgentMiddleware):
    def __init__(self, max_messages: int=100, agent_name: str=""):
        super().__init__()
        self.max_messages = max_messages
        self.agent_name = agent_name

    # jump t
    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:

        messages = state["messages"]

        length= len(messages)
        logger.info("[%s] New message is going to send to LLM again! Length = %d \n", self.agent_name, length)

        for i, msg in enumerate(messages):
            logger.debug("------> Message %d: [%s] Role=%s, Content='%s'\n", i, self.agent_name, msg.type, msg.content)
        if length >= self.max_messages:
            logger.warning("[%s] Message limit reached: %d", self.agent_name, len(state['messages']))
            return {
                "messages": [AIMessage("Conversation limit reached.")],
                "jump_to": "end"
            }
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_two_messages = state["messages"][-2:]
        
        for i, msg in enumerate(last_two_messages):
            logger.info("<------ [%s] The Model returned Message (last two) [%d]: Role=%s, Content='%s'\n", self.agent_name, i, msg.type, msg.content)

            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    logger.info(f"TOOL REQUEST ==> Name: f{tool_call['name']}, ARGS: {tool_call['args']}")

        # messages = state["messages"]
        # for i, msg in enumerate(messages):
        #     logger.info("<------ [%s] Model returned Message %d: Role=%s, Content='%s'\n", self.agent_name, i, msg.type, msg.content)
        
        return None

# initial the Message Middleware Object for manager agent
messageLimitMiddleware = MessageLimitMiddleware(max_messages=60, agent_name="Researcher")

# initial the cache backend for below cache=True
set_llm_cache(InMemoryCache())

# Configure the Built-in Filesystem Backend
fs_backend = FilesystemBackend(root_dir=FILE_DIR, virtual_mode=True)

# Supported LLM type identifiers for ``create_researcher_agent``.
def _resolve_llm(model_obj: ModelObj):
    """Return the chat model object that corresponds to ``llm_type``.

    Parameters
    ----------
    llm_type : str
        Identifier of the chat model to use. Must be one of
        ``SUPPORTED_LLM_TYPES`` (i.e. ``"ollama"`` or
        ``"vllm"``).

    Returns
    -------
    BaseChatModel
        The configured chat model instance (``ChatOllama`` or
        ``ChatOpenAI``) to plug into the deep agent.

    Raises
    ------
    ValueError
        If ``llm_type`` is not one of the supported identifiers.
    """
    if model_obj.llm_type == "ollama":
        # inital the model object of Ollama provider
        model_ollama = ChatOllama(
            model=model_obj.model_name,
            # validate_model_on_init=True,
            # num_thread=16,
            cache=True,
            verbose=True,     # Print additional LangChain logs.Useful for debugging: prompts, tool calls, intermediate chains
            reasoning=False,
            temperature=0.5,
            base_url=model_obj.model_base_url,
            repeat_penalty=1.05,
            num_ctx=int(MAX_COMPLETION_TOKENS),
            disable_streaming="tool_calling"
        )
        return model_ollama
    if model_obj.llm_type == "vllm":
        # initial the model object that vLLM provides an OpenAI-compatible API at localhost:8000
        model_vllm = ChatOpenAI(
            model=model_obj.model_name,                 # Model name (can be any vLLM-supported model)
            base_url=model_obj.model_base_url,          # vLLM server endpoint         
            api_key=SecretStr(model_obj.model_api_key), # vLLM uses a placeholder token
            temperature=0.4,
            top_p=0.9,
            max_completion_tokens=int(MAX_COMPLETION_TOKENS)
        )

        # disable the llm parallel tool calls
        model_vllm.bind(
            parallel_tool_calls=False
        )
        
        return model_vllm
    raise ValueError(
        f"Unsupported llm : {model_obj}. "
        f"Expected one of: {', '.join(SUPPORTED_LLM_TYPES)}"
    )


def create_researcher_agent(model_obj: ModelObj):
    """Create the senior Web Researcher deep agent.

    This factory wires a configured chat model (resolved from ``model_obj``)
    into the ``Researcher`` deep agent produced by ``create_deep_agent``. The
    agent is equipped with:

    * a sandboxed ``FilesystemBackend`` (``fs_backend``) rooted at
      ``FILE_DIR`` so the agent can read/write files inside its virtual
      workspace;
    * the tools ``get_current_time``, ``tool_cninfo_report_downloader`` and
      ``save_json_file`` for fetching the current date/time, downloading
      CNINFO disclosure PDFs and persisting structured metadata;
    * the ``RESEARCHER_SYSTEM_PROMPT`` instructing the agent to plan,
      download reports sequentially, and emit a JSON manifest;
    * two middlewares: ``messageLimitMiddleware`` (caps the conversation
      length to avoid context overflow) and ``toolCallLimitMiddleware``
      (caps the number of ``tool_cninfo_report_downloader`` calls per run
      to ``30``).

    Parameters
    ----------
    model_obj : ModelObj
        A ``ModelObj`` instance describing the underlying chat model. Its
        ``llm_type`` attribute must be one of ``SUPPORTED_LLM_TYPES``
        (i.e. ``"ollama"`` → ``ChatOllama``, or ``"vllm"`` →
        ``ChatOpenAI``). The ``model_name``, ``model_base_url`` and
        ``model_api_key`` fields are forwarded to the corresponding
        chat-model constructor.

    Returns
    -------
    CompiledStateGraph
        The deep agent instance produced by ``create_deep_agent``, ready
        to be invoked with a ``{"messages": [...]}`` input payload.

    Raises
    ------
    ValueError
        If ``model_obj.llm_type`` is not one of the identifiers listed in
        ``SUPPORTED_LLM_TYPES``.

    Examples
    --------
    >>> from sl_finance_agent.agent_graph import ModelObj
    >>> model_obj = ModelObj(
    ...     llm_type="ollama",
    ...     model_name="qwen3.5",
    ...     model_base_url="http://localhost:11434",
    ...     model_api_key="",
    ... )
    >>> agent = create_researcher_agent(model_obj)
    >>> result = agent.invoke({
    ...     "messages": [{
    ...         "role": "user",
    ...         "content": "Download the 2023 annual report of 平安银行.",
    ...     }],
    ... })
    """
    model = _resolve_llm(model_obj)
    logger.info("Creating Web Researcher deep agent with llm_type=%s", model_obj)

    return create_deep_agent(
        name="Researcher",
        model=model,
        backend=fs_backend,
        tools=[get_current_time, tool_cninfo_report_downloader, save_json_file],
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        middleware=[messageLimitMiddleware, toolCallLimitMiddleware]
    )
