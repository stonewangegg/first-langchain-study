"""Analyzer deep-agent for senior financial DuPont analysis of listed-company PDFs.

Builds a :func:`create_deep_agent`-backed deep agent that reviews the target
PDF files described by a metadata JSON file and emits a Markdown report using
the ``senior-financial-dupont-analyst`` skill (DuPont methodology).

Environment variables
---------------------
* ``MAX_COMPLETION_TOKENS`` — LLM context window (default ``"16384"``).
* ``FILE_DIR`` — sandbox directory for the agent's ``FilesystemBackend``
  (default ``"./tmp"``).

Quickstart
----------
Call :func:`create_analyzer_agent` with a ``ModelObj`` (see
``sl_finance_agent.agent_graph``) and invoke the returned agent, e.g.::

    agent = create_analyzer_agent(model_obj)
    agent.invoke({"messages": [{
        "role": "user",
        "content": (
            'Read the PDF files listed in "manifest.json" and produce a '
            'DuPont analysis report in Markdown.'
        ),
    }]})

The agent plans the reading order, fetches the PDFs sequentially through
``tool_custom_file_read`` (no parallel reads), and finishes by writing a
Markdown report that follows the ``senior-financial-dupont-analyst`` skill.

Notes
-----
* ``FilesystemBackend`` runs in ``virtual_mode=True``; all paths seen by the
  agent are translated into safe virtual paths inside ``FILE_DIR``.
* The DuPont ``SKILL.md`` is loaded at import time from
  ``<this-dir>/../skills/senior-financial-dupont-analyst/SKILL.md`` and must
  exist for the module to import successfully.
"""

from typing import Any, cast

from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime, ToolCallLimitMiddleware, hook_config
from langchain.messages import AIMessage
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from ..tools_file_write_read import tool_custom_file_read, copy_file_to_folder
from ..common_utils import ModelObj, get_logger, FS_BACKEND, get_current_time, get_file_dir, resolve_llm

ANALYST_SYSTEM_PROMPT = """
# You are a senior financial analyst of a listed company.

## Your goal is making a plan to review and analyze the target PDF files and finish all core steps below.

## Core Steps
1. Firstly: You find and review the json file for all target PDF files one by one.
2. Secondly: You use `tool_custom_file_read` to read target PDF file follow the plan, you must finish one and then the next one, **Do Not allow Parallel reading**.
3. Thirdly: Analyze and summary the content follow the skill 'senior-financial-dupont-analyst'. Then generate report file with markdown format and get the full path of the report file.
4. Finally: You must use `copy_file_to_folder` to copy the report file from current path into the destination directory:'/app/backend/shared-files'. **And return the report file name**.

## Core Principles
- You should anaylyze and summarize content base on corresponding skill of "senior-financial-dupont-analyst".
- **If you already have all steps completed, STOP and Return the final results at once**.
"""

# get the logger
logger = get_logger(__name__)

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

def create_analyzer_agent(model_obj: ModelObj):
    """Create the senior financial analyst deep agent.

    Factory that builds and returns a ``deepagents`` deep agent acting
    as a senior financial analyst for a listed company.

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
        (i.e. not ``"ollama"`` or ``"vllm"``). Propagated from
        :func:`resolve_llm`.

    Notes
    -----
    * The DuPont ``SKILL.md`` file is loaded at import time, so the
      module will fail to import if
      ``../skills/senior-financial-dupont-analyst/SKILL.md`` is missing.
    """
    model = resolve_llm(model_obj)
    logger.info("Creating Analyzer deep agent with llm_type=%s", model_obj)

    return create_deep_agent(
        name="Analyzer",
        model=model,
        skills=["/skills/"],
        backend=FS_BACKEND,
        tools=[get_current_time, tool_custom_file_read, copy_file_to_folder],
        system_prompt=ANALYST_SYSTEM_PROMPT,
        middleware=[messageLimitMiddleware, toolCallLimitMiddleware],
    )
