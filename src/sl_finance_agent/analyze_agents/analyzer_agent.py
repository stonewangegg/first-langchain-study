"""Analyzer Agent Demo: Senior Financial Analyst Deep Agent.

Builds a :func:`create_deep_agent`-backed deep agent that acts as a senior
financial analyst for a listed company: it reviews the target PDF files
described by a metadata JSON file and emits a Markdown report using the
``senior-financial-dupont-analyst`` skill (DuPont methodology).

Public surface
--------------
* :func:`create_analyzer_agent` -- factory that returns the compiled deep
  agent, configured with the :class:`FilesystemBackend` sandbox, tools,
  mounted skill and safety middleware.
* :func:`_resolve_llm`         -- maps a :class:`ModelObj` to a concrete
  ``ChatOllama`` or ``ChatOpenAI`` instance.
* :class:`MessageLimitMiddleware` -- guards against runaway conversation
  length by jumping to ``END`` once ``max_messages`` is reached.

Configuration
-------------
Environment variables (read at import time):

* ``MAX_COMPLETION_TOKENS`` -- context window size for the chat model
  (default ``"16384"``).
* ``FILE_DIR``              -- directory that backs the sandbox exposed to
  the agent via ``FilesystemBackend`` (default ``"./tmp"``).

Usage
-----
Import ``create_analyzer_agent`` and invoke the returned agent with a
``{"messages": [...]}`` payload::

    from sl_finance_agent.agent_graph import ModelObj
    from sl_finance_agent.analyze_agents.analyzer_agent_demo import (
        create_analyzer_agent,
    )

    agent = create_analyzer_agent(ModelObj(
        llm_type="ollama",
        model_name="llama3",
        model_base_url="http://localhost:11434",
    ))
    result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

Notes
-----
* ``FilesystemBackend`` is mounted with ``virtual_mode=True``, so all file
  operations are translated into safe virtual paths inside ``FILE_DIR``.
* The DuPont skill bundle is read at import time from
  ``<this-dir>/../skills/senior-financial-dupont-analyst/SKILL.md`` and
  must exist for the module to import successfully.
"""

from datetime import datetime
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

from ..tools_file_write_read import tool_custom_file_read
from ..common import SUPPORTED_LLM_TYPES, ModelObj, get_logger

MAX_COMPLETION_TOKENS = os.environ.get("MAX_COMPLETION_TOKENS", "16384")

# current working directory
CURRENT_WORKING_DIR = os.getcwd()

FILE_DIR = os.environ.get("FILE_DIR", "./tmp")

# export FILE_ROOT_DIR="/your/file/root/dir"
FILE_ROOT_DIR = str(Path(CURRENT_WORKING_DIR) / Path(FILE_DIR))

ANALYST_SYSTEM_PROMPT = """
# You are a senior financial analyst of a listed company.

## Your goal is to review and analyze the target PDF files descriped in meta data json file.

## Core Steps
1. Firstly: You find and review the json file to make a plan for reading target PDF files one by on.
2. Secondly: You use `tool_custom_file_read` to read one file finished and then read the next file, **Do Not allow Parallel reading**.
3. Thirdly: Analyze and summary the content follow the skill 'senior-financial-dupont-analyst'.
4. Finally: Generate report file with markdown format.

## Core Principles
- You should anaylyze and summarize content base on corresponding skill of "senior-financial-dupont-analyst".
- **If you already have task completed, STOP and Return the final results at once**.
"""

# get the logger
logger = get_logger(__name__)

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
        last_four_messages = state["messages"][-4:]
        
        for i, msg in enumerate(last_four_messages):
            logger.info("<------ [%s] The Model returned Message (last four) [%d]: Role=%s, Content='%s'\n", self.agent_name, i, msg.type, msg.content)

            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    logger.info(f"TOOL REQUEST ==> Name: {tool_call['name']}, ARGS: {tool_call['args']}")

        # messages = state["messages"]
        # for i, msg in enumerate(messages):
        #     if isinstance(msg, AIMessage):
        #         if msg.tool_calls:
        #             for tool_call in msg.tool_calls:
        #                 logger.info(f"TOOL REQUEST: f{tool_call['name']}, ARGS: {tool_call['args']}")
        #     logger.info("<------ [%s] Model returned Message %d: Role=%s, Content='%s'\n", self.agent_name, i, msg.type, msg.content)
        
        return None

# initial the Message Middleware Object for manager agent
messageLimitMiddleware = MessageLimitMiddleware(max_messages=100, agent_name="Analyzer")

# inital the tool call limitation with 30
toolCallLimitMiddleware = cast(AgentMiddleware, ToolCallLimitMiddleware(tool_name="tool_custom_file_read", run_limit=30, exit_behavior="end"))

# initial the cache backend for below cache=True
set_llm_cache(InMemoryCache())

# Config the Built-in Filesystem Backend
fs_backend = FilesystemBackend(root_dir=FILE_DIR, virtual_mode=True)

# Mount/copy skills into virtual filesystem
# 1. Get the absolute path of the directory where THIS tool script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. Build the target path relative to THIS directory
skill_target_path = os.path.join(CURRENT_DIR, "../skills/senior-financial-dupont-analyst/SKILL.md")
# 3. (Optional but recommended) Normalize the path to remove the "../"
skill_target_path = os.path.normpath(skill_target_path)
with open(skill_target_path, "r", encoding="utf-8") as f:
    skill_content = f.read()

fs_backend.write(
    "/skills/senior-financial-dupont-analyst/SKILL.md",
    skill_content
)

# Supported LLM type identifiers for ``create_analyzer_agent``.
def _resolve_llm(model_obj: ModelObj):
    """Resolve a ``ModelObj`` into a configured chat model.

    Maps ``model_obj.llm_type`` to a concrete chat model instance:

    * ``"ollama"`` -> :class:`ChatOllama` against ``model_base_url``.
    * ``"vllm"``   -> :class:`ChatOpenAI` against the vLLM
      OpenAI-compatible endpoint at ``model_base_url`` (with parallel
      tool calls disabled).

    The context window size for both providers is driven by the
    ``MAX_COMPLETION_TOKENS`` environment variable.

    Parameters
    ----------
    model_obj : ModelObj
        Dataclass specifying the provider (``llm_type``), ``model_name``,
        ``model_base_url`` and optional ``model_api_key``. See
        ``create_analyzer_agent`` for field-level documentation.

    Returns
    -------
    BaseChatModel
        The configured chat model (``ChatOllama`` or ``ChatOpenAI``)
        ready to plug into the deep agent.

    Raises
    ------
    ValueError
        If ``model_obj.llm_type`` is not one of ``SUPPORTED_LLM_TYPES``
        (i.e. not ``"ollama"`` or ``"vllm"``).
    """
    if model_obj.llm_type == "ollama":
        # inital the model object of Ollama provider
        model_ollama = ChatOllama(
            model=model_obj.model_name,
            # validate_model_on_init=True,
            # num_thread=16,
            cache=True,
            verbose=True,                       # Print additional LangChain logs.Useful for debugging: prompts, tool calls, intermediate chains
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
        f"Unsupported llm: {model_obj}. "
        f"Expected one of: {', '.join(SUPPORTED_LLM_TYPES)}"
    )


def create_analyzer_agent(model_obj: ModelObj):
    """Create the senior financial analyst deep agent.

    This factory builds a ``deepagents`` deep agent that reviews the
    target PDF files described by the metadata JSON file and emits a
    structured Markdown analysis report using the
    ``senior-financial-dupont-analyst`` skill. The agent is configured
    with a :class:`FilesystemBackend` sandbox rooted at ``FILE_DIR``,
    ``get_current_time`` and ``tool_custom_file_read`` tools, the
    mounted skill bundle, and the safety middleware
    (``MessageLimitMiddleware`` and ``ToolCallLimitMiddleware``).

    Parameters
    ----------
    model_obj : ModelObj
        Dataclass that specifies which chat model the agent should use.
        Must contain the following fields:

        * ``llm_type`` (``str``) -- identifier of the provider. Must be
          one of ``SUPPORTED_LLM_TYPES`` (``"ollama"`` or ``"vllm"``).
          When ``"ollama"``, a :class:`ChatOllama` instance is built
          against ``model_base_url``; when ``"vllm"``, a
          :class:`ChatOpenAI` instance is built against the vLLM
          OpenAI-compatible endpoint at ``model_base_url``.
        * ``model_name`` (``str``) -- the model name to instantiate
          (e.g. ``"llama3"`` for Ollama, ``"Qwen/Qwen2.5-..."`` for
          vLLM).
        * ``model_base_url`` (``str``) -- base URL of the chat model
          endpoint (``ChatOllama`` base URL or vLLM OpenAI-compatible
          endpoint).
        * ``model_api_key`` (``str``, optional) -- API key for the
          provider. Defaults to ``"Empty"``; vLLM uses a placeholder
          token.

    Returns
    -------
    CompiledStateGraph
        The deep agent instance produced by ``create_deep_agent``,
        ready to be invoked with a ``{"messages": [...]}`` input
        payload.

    Raises
    ------
    ValueError
        If ``model_obj.llm_type`` is not one of ``SUPPORTED_LLM_TYPES``
        (i.e. not ``"ollama"`` or ``"vllm"``).

    Examples
    --------
    >>> from sl_finance_agent.agent_graph import ModelObj
    >>> model_obj = ModelObj(
    ...     llm_type="ollama",
    ...     model_name="llama3",
    ...     model_base_url="http://localhost:11434",
    ... )
    >>> agent = create_analyzer_agent(model_obj)
    >>> result = agent.invoke({
    ...     "messages": [{"role": "user", "content": "Analyze ..."}],
    ... })
    """
    model = _resolve_llm(model_obj)
    logger.info("Creating Analyzer deep agent with llm_type=%s", model_obj)

    return create_deep_agent(
        name="Analyzer",
        model=model,
        skills=["/skills/"],
        backend=fs_backend,
        tools=[get_current_time, tool_custom_file_read],
        system_prompt=ANALYST_SYSTEM_PROMPT,
        middleware=[messageLimitMiddleware, toolCallLimitMiddleware],
    )
