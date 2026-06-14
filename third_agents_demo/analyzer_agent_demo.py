"""
Analyzer Agent Demo: Senior Financial Analyst Deep Agent

This module demonstrates how to build a **LangChain Deep Agent** that acts as a
senior financial analyst for a listed company. The agent is designed to review
and analyze target PDF files (e.g. annual / quarterly reports) described by a
metadata JSON file, and to produce a structured Markdown analysis report using
the *DuPont analysis* methodology.

Key Features
------------
* **Deep Agent orchestration** — built on top of ``deepagents.create_deep_agent``
  with a virtual ``FilesystemBackend`` for file I/O isolation.
* **Custom tooling** — exposes ``get_current_time`` and a project-specific
  ``tool_custom_file_read`` tool so the agent can inspect PDF documents.
* **Skill mounting** — copies the ``senior-financial-dupont-analyst`` skill
  (see ``skills/senior-financial-dupont-analyst/SKILL.md``) into the agent's
  virtual filesystem so it is available at runtime.
* **Safety guard-rails** — two middleware components are attached to the
  agent:
    - ``MessageLimitMiddleware`` — aborts the run with a clear message once
      the conversation history exceeds ``max_messages`` (default 50).
    - ``ToolCallLimitMiddleware`` — caps calls to ``tool_custom_file_read``
      at 30 invocations per run to prevent runaway tool usage.
* **Local LLM via Ollama** — configurable through the ``ONLINE_MODEL`` /
  ``LOCAL_MODEL`` and ``ONLINE_BASEURL`` / ``LOCAL_BASEURL`` constants; the
  context window size is controlled by the ``MAX_COMPLETION_TOKENS``
  environment variable (default ``"16384"``).

Configuration
-------------
The following environment variables are honored:

* ``MAX_COMPLETION_TOKENS`` — context window size for the chat model.
* ``FILE_DIR``              — directory (relative to the current working
                              directory) that will be exposed to the
                              ``FilesystemBackend`` (default ``"./tmp"``).

Usage
-----
This module is intended to be imported by test/entry-point scripts such as
``demo_agent_analyze_test.py``. It can also be run directly to perform a
smoke test, but typically the agent is invoked programmatically:

    from third_agents_demo.analyzer_agent_demo import agent_analyzer

    result = agent_analyzer.invoke({
        "messages": [{"role": "user", "content": "..."}],
    })

Notes
-----
* ``virtual_mode=True`` on the backend means the agent cannot escape the
  ``FILE_DIR`` sandbox; all file operations are translated to safe virtual
  paths.
* The DuPont skill content is read at import time from
  ``<cwd>/skills/senior-financial-dupont-analyst/SKILL.md`` and must exist
  for the module to import successfully.
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

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tools_file_write_read import tool_custom_file_read

MAX_COMPLETION_TOKENS = os.environ.get("MAX_COMPLETION_TOKENS", "16384")

# current working directory
CURRENT_WORKING_DIR = os.getcwd()

FILE_DIR = os.environ.get("FILE_DIR", "./tmp")

# export FILE_ROOT_DIR="/your/file/root/dir"
FILE_ROOT_DIR = str(Path(CURRENT_WORKING_DIR) / Path(FILE_DIR))

# llm info
LOCAL_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
LOCAL_BASEURL="http://192.168.8.50:8000/v1"
ONLINE_MODEL="minimax-m3:cloud"
ONLINE_BASEURL="http://172.30.0.1:11434"

ANALYST_SYSTEM_PROMPT = """
# You are a senior financial analyst of a listed company.

## Your goal is to review and analyze the target PDF files descriped in meta data json file.

## Core Steps
1. Firstly: You find and review the json file to make a plan for reading target PDF files.
2. Secondly: You use `tool_custom_file_read` to read the target PDF file **one bye one**.
3. Thirdly: Analyze and summary the content follow the skill 'senior-financial-dupont-analyst'.
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

    # jump to end if reach the limitation
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
messageLimitMiddleware = MessageLimitMiddleware(max_messages=100, agent_name="Analyzer")

# inital the tool call limitation with 30
toolCallLimitMiddleware = cast(AgentMiddleware, ToolCallLimitMiddleware(tool_name="tool_custom_file_read", run_limit=30, exit_behavior="end"))

# initial the cache backend for below cache=True
set_llm_cache(InMemoryCache())

# inital the model object of Ollama provider
model_ollama = ChatOllama(
            model=LOCAL_MODEL,
            # validate_model_on_init=True,
            # num_thread=16,
            cache=True,
            verbose=True,                       # Print additional LangChain logs.Useful for debugging: prompts, tool calls, intermediate chains
            reasoning=False,
            temperature=0.5,
            base_url=LOCAL_BASEURL,
            repeat_penalty=1.05,
            num_ctx=int(MAX_COMPLETION_TOKENS),
            disable_streaming="tool_calling"
)

# initial the model object that vLLM provides an OpenAI-compatible API at localhost:8000
model_vllm = ChatOpenAI(
    model=LOCAL_MODEL,                                # Model name (can be any vLLM-supported model)
    base_url=LOCAL_BASEURL,                           # vLLM server endpoint         
    api_key=SecretStr("EMPTY"),                 # vLLM uses a placeholder token
    temperature=0.7,
    top_p=0.9,
    max_completion_tokens=int(MAX_COMPLETION_TOKENS)
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
agent_analyzer = create_deep_agent(
    name="Analyzer",
    model=model_vllm,
    skills=["/skills/"],
    backend=fs_backend,
    tools=[get_current_time, tool_custom_file_read],
    system_prompt=ANALYST_SYSTEM_PROMPT,
    middleware=[messageLimitMiddleware, toolCallLimitMiddleware]
)