"""
title: The agent1 is role "数据收集经理"
author: stone wang
description: the main agent that do the task: search and crewl all needed data from internet, for follow research. It must make sure all the data is correct and in time, then save data into Excel files. Automatically restore at required backend.
required_open_webui_version: 0.9.2+
requirements: deepagents, langchain_openai, langchain.tools, pydantic, tavily, typing
version: 0.1
"""
import os
from pathlib import Path
from pydantic import SecretStr
from typing import Literal, Any, cast
from datetime import datetime

from deepagents import create_deep_agent, SubAgent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.file_management.list_dir import ListDirectoryTool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ModelCallLimitMiddleware,
)

from ..cninfo_report_downloader import CNInfoReportDownloader
from ..tools_file_write_read import tool_custom_file_read, tool_custom_file_write, tool_generate_word_doc
from .agent_system_prompt import OFFICER_SYSTEM_PROMPT, RESEARCHER_SYSTEM_PROMPT, ANALYST_SYSTEM_PROMPT

from ..common_utils import get_logger
# get the logger
logger = get_logger(__name__)

# export MAX_COMPLETION_TOKENS as a passin varaible
MAX_COMPLETION_TOKENS = os.environ.get("MAX_COMPLETION_TOKENS", "16384")

# current working directory
CURRENT_WORKING_DIR = os.getcwd()

# export FILE_ROOT_DIR="your/file/root/dir"
FILE_ROOT_DIR = os.environ.get("FILE_ROOT_DIR", CURRENT_WORKING_DIR)



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
    local_store_path = store_path
    if local_store_path.startswith(FILE_ROOT_DIR):
        path_obj = Path(local_store_path)
        if path_obj.is_dir():
            os.makedirs(local_store_path, exist_ok=True)
            logger.info("The required store path is valid: %s", local_store_path)
        elif path_obj.is_file():
            logger.error("The passed in store path is a file path: %s, it must be the folder path to store download file, check and retry", local_store_path)
            return f"The passed in store path is a file path: {local_store_path}, it must be a 'folder path' to store download file, check and retry"
    else:
        logger.error("The required store path is invalid: %s, store download file to default folder under current working dir: %s", store_path, FILE_ROOT_DIR)
        return f"The required store path is invalid: {store_path}, check the passed in parameters"

        
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
        
    logger.info("\nℹ️目标文件下载完成！文件保存路径: %s", result)
    return result

# assistant function
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return "Current date and time is: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_current_working_path() -> str:
    """Returns the current working directory path."""
    return "Current working directory is: " + os.getcwd()

@tool
def create_directory(path: str) -> str:
    """
    Creates a new directory at the specified path.
    param: path The path of the directory to create.
    return: directory created success or not
    """
    try:
        os.makedirs(path, exist_ok=True)
        return f"Directory: '{path}' created successfully."
    except Exception as e:
        return f"Error creating directory: {str(e)}, check the passed in parameters"

LOCAL_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
LOCAL_BASEURL="http://192.168.8.50:8000/v1"

ONLINE_MODEL="minimax-m3:cloud"
ONLINE_BASEURL="http://172.30.0.1:11434"

# initial the model object that vLLM provides an OpenAI-compatible API at localhost:8000
model_vllm = ChatOpenAI(
    model=LOCAL_MODEL,                                # Model name (can be any vLLM-supported model)
    base_url=LOCAL_BASEURL,                           # vLLM server endpoint         
    api_key=SecretStr("EMPTY"),                 # vLLM uses a placeholder token
    temperature=0.4,
    top_p=0.9,
    max_completion_tokens=int(MAX_COMPLETION_TOKENS)
)

model_vllm_analyst = ChatOpenAI(
    model=LOCAL_MODEL,                                # Model name (can be any vLLM-supported model)
    base_url=LOCAL_BASEURL,                           # vLLM server endpoint         
    api_key=SecretStr("EMPTY"),                 # vLLM uses a placeholder token
    temperature=0.5,
    top_p=0.9,
    max_completion_tokens=int(MAX_COMPLETION_TOKENS)
)

model_vllm_researcher = ChatOpenAI(
    model=LOCAL_MODEL,                                # Model name (can be any vLLM-supported model)
    base_url=LOCAL_BASEURL,                           # vLLM server endpoint         
    api_key=SecretStr("EMPTY"),                 # vLLM uses a placeholder token
    temperature=0.3,
    top_p=0.9,
    max_completion_tokens=int(MAX_COMPLETION_TOKENS)
)

# inital the model object that Ollama provider
# model_ollama = ChatOllama(
#             model=ONLINE_MODEL,
#             validate_model_on_init=True,
#             reasoning=False,
#             # temperature=0.5,
#             base_url=ONLINE_BASEURL,
#             # repeat_penalty=1.5,
#             # num_ctx=16384
# )

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
        messages = state["messages"]
        for i, msg in enumerate(messages):
            logger.info("<------ Model returned Message %d: [%s] Role=%s, Content='%s'\n", i, self.agent_name,  msg.type, msg.content)
        return None

# initial the Message Middleware Object for manager agent
messageLimitMiddleware = MessageLimitMiddleware(max_messages=60, agent_name="Cordinator")

subMessageLimitMiddleware = MessageLimitMiddleware(max_messages=90, agent_name="Researcher")

# initial the critical tool call limitation middle ware object, Note the exit behavior is 'Continue'
toolCallLimitMiddleware_raw = ToolCallLimitMiddleware(
                                                        tool_name="tool_cninfo_report_downloader",
                                                        run_limit=30, exit_behavior="end"
                                                       )
toolCallLimitMiddleware = cast(
    AgentMiddleware,
    toolCallLimitMiddleware_raw
)

# Configure the Built-in Filesystem Backend
logger.info("current file directory for initial FilesystemBackend: %s", FILE_ROOT_DIR)
fs_backend = FilesystemBackend(root_dir=FILE_ROOT_DIR, virtual_mode=True)

# skills path: the parent directory of skills folder
SKILL_PATH = str(Path(CURRENT_WORKING_DIR) / "skills")
logger.info("skills_path: %s", SKILL_PATH)

# initial the langchain tavily sub agent
# 1. Researcher Agent: Finds raw information
agent_searcher: SubAgent  = {
    "name" : "Researcher",
    "description" : "Agent researcher: Follows user prompt from collaborator, searches and download target report PDF files",
    "model" : model_vllm_researcher,
    "skills" : [SKILL_PATH],
    "tools" : [get_current_time, tool_cninfo_report_downloader, tool_custom_file_write, ListDirectoryTool()],
    "middleware" : [toolCallLimitMiddleware, subMessageLimitMiddleware],
    "system_prompt" : RESEARCHER_SYSTEM_PROMPT
}

# initial the langchain report sub agent
# 2. Reporter Agent: Summarizes the report, writes and formats the final output.
# agent_analyst = SubAgent(
#     name = "Analyst",
#     description="Agent analyst: senior financial analyst of a listed company.",
#     model=model_vllm_analyst,
#     skills=[SKILL_PATH],
#     tools=[tool_read_file, tool_generate_word_doc, get_current_time, get_current_working_path],
#     system_prompt=ANALYST_SYSTEM_PROMPT
# )
agent_analyst: SubAgent  = {
    "name" : "Analyst",
    "description" : "Agent analyst: senior financial analyst of a listed company.",
    "model" : model_vllm_analyst,
    "skills" : [SKILL_PATH],
    "tools" : [tool_custom_file_read, tool_generate_word_doc, get_current_time, get_current_working_path, ListDirectoryTool()],
    "system_prompt" : ANALYST_SYSTEM_PROMPT
}

# 3. 配置 Checkpointer（真实生产环境建议用 PostgresSaver）
checkpointer_searcher = InMemorySaver()

# initial the collaborator agent object
# agent_collaborator = create_deep_agent(
#     name="collaborator",
#     model=model_vllm,
#     tools=[get_current_time, tool_generate_word_doc],
#     system_prompt=MAIN_SYSTEM_PROMPT,
#     subagents=[agent_searcher, agent_analyst],
#     middleware=[messageLimitMiddleware],
#     checkpointer=checkpointer_searcher
# )

agent_collaborator = create_deep_agent(
    name="collaborator",
    model=model_vllm,
    backend=fs_backend,
    tools=[get_current_time, get_current_working_path, create_directory, ListDirectoryTool()],
    system_prompt=OFFICER_SYSTEM_PROMPT,
    subagents=[agent_searcher, agent_analyst],
    middleware=[messageLimitMiddleware]
)