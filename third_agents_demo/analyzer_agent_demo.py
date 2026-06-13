"""
"""

from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, cast

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.tools import tool
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime, ToolCallLimitMiddleware, hook_config
from langchain.messages import AIMessage
from langchain_ollama import ChatOllama

from tools_file_write_read import tool_custom_file_read

MAX_COMPLETION_TOKENS = os.environ.get("MAX_COMPLETION_TOKENS", "16384")

# current working directory
CURRENT_WORKING_DIR = os.getcwd()

FILE_DIR = os.environ.get("FILE_DIR", "./tmp")

# export FILE_ROOT_DIR="your/file/root/dir"
FILE_ROOT_DIR = str(Path(CURRENT_WORKING_DIR) / Path(FILE_DIR))

# llm info
LOCAL_MODEL="qwen3.6:27b"
LOCAL_BASEURL="http://172.30.0.1:11434"
ONLINE_MODEL="minimax-m3:cloud"
ONLINE_BASEURL="http://172.30.0.1:11434"

ANALYST_SYSTEM_PROMPT = """
# You are a senior financial analyst of a listed company.

## Your goal is to review and analyze the target PDF files descriped in meta data json file.

## Core Steps
1. Firstly: You review the json file to make a plan with read target PDF file one by one.
2. Secondly: You use `tool_custom_file_read` to read the target PDF file follow the plan.
3. Thirdly: Analyze and summary the content with the skill 'senior-financial-dupont-analyst'.
4. Finally: Generate report file with markdown format.

## Core Principles
- You should anaylyze and summarize content base on corresponding skill of "senior-financial-dupont-analyst".
- **If you already have task completed, STOP and Return the final results at once**.
"""

# get the logger
logger = logging.getLogger(__name__)

# assistant function
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return "Current date and time is: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            logger.info("<------ [%s] Model returned Message %d: Role=%s, Content='%s'\n", self.agent_name, i, msg.type, msg.content)
        return None

# initial the Message Middleware Object for manager agent
messageLimitMiddleware = MessageLimitMiddleware(max_messages=60, agent_name="Researcher")

# inital the tool call limitation with 30
toolCallLimitMiddleware = cast(AgentMiddleware, ToolCallLimitMiddleware(tool_name="tool_cninfo_report_downloader", run_limit=30, exit_behavior="end"))

# inital the model object of Ollama provider
model_ollama = ChatOllama(
            model=ONLINE_MODEL,
            validate_model_on_init=True,
            # num_thread=16,
            verbose=True,                       # Print additional LangChain logs.Useful for debugging: prompts, tool calls, intermediate chains
            reasoning=False,
            temperature=0.5,
            base_url=ONLINE_BASEURL,
            repeat_penalty=1.05,
            num_ctx=int(MAX_COMPLETION_TOKENS),
            disable_streaming="tool_calling"
)

# Config the Built-in Filesystem Backend
fs_backend = FilesystemBackend(root_dir=FILE_DIR, virtual_mode=True)

# Mount/copy skills into virtual filesystem
with open(CURRENT_WORKING_DIR + "/skills/senior-financial-dupont-analyst/SKILL.md", "r", encoding="utf-8") as f:
    skill_content = f.read()

fs_backend.write(
    "/skills/senior-financial-dupont-analyst/SKILL.md",
    skill_content
)

# initial the main agent
agent_researcher = create_deep_agent(
    name="Researcher",
    model=model_ollama,
    skills=["/skills/"],
    backend=fs_backend,
    tools=[get_current_time, tool_custom_file_read],
    system_prompt=ANALYST_SYSTEM_PROMPT,
    middleware=[messageLimitMiddleware, toolCallLimitMiddleware]
)