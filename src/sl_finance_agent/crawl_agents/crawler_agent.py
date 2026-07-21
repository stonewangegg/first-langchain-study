"""Crawler deep-agent for senior financial web search and page crawling.

Builds a :func:`create_deep_agent`-backed deep agent that searches the
web via Tavily, crawls the most relevant URLs, and emits a Markdown
report following the ``senior-financial-web-crawler`` skill.

Quickstart
----------
Call :func:`create_crawler_agent` with a ``ModelObj`` and invoke the
returned agent, e.g.::

    agent = create_crawler_agent(model_obj)
    agent.invoke({"messages": [{
        "role": "user",
        "content": (
            "Search and crawl the latest quarterly financial news for "
            "company XYZ and write a Markdown summary."
        ),
    }]})

The agent plans a search, calls ``tool_tavily_search`` and
``research_crawl`` **sequentially** (no parallel tool calls), then
writes a Markdown report via ``tool_custom_file_write`` that follows the
``senior-financial-web-crawler`` skill.

Notes
-----
* ``FilesystemBackend`` runs in ``virtual_mode=True``; all paths seen by
  the agent are translated into safe virtual paths inside ``FILE_DIR``.
* The ``SKILL.md`` is loaded at import time from
  ``<this-dir>/../skills/senior-financial-web-crawler/SKILL.md`` and must
  exist for the module to import successfully.
* ``MessageLimitMiddleware`` (default 100 messages) and
  ``ToolCallLimitMiddleware`` (default 30 calls to
  ``tool_custom_file_read``) protect against context overflow and
  runaway tool usage.
"""

from typing import Any, cast

from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, Runtime, ToolCallLimitMiddleware, hook_config
from langchain.messages import AIMessage

from ..common_utils import ModelObj, resolve_llm, uru_logger, get_current_time, FS_BACKEND
from ..tools_web_search_crawl import tool_tavily_search, tool_research_crawl
from ..tools_file_write_read import tool_custom_file_write, copy_file_to_folder


# system prompt for crawl agent
INDUSTRY_CRAWLER_SYSTEM_PROMPT = """
# You are an expert Web Crawler, tasked with crawling to provide accurate, up-to-date web page content for user query. 

## Your goal is searching and crawling target informaton and data of the query, then check, organize and summarize all content, 
# finally write int a file named with keyword of query and date suffix in markdown format.

## Core steps
1. Firstly: Use `get_current_time` to get the time, and make a todo plan for the user query and comply the skill. 
2. Secondly: You must make sure all web search and data crawl in time, use `tool_tavily_func` to search target page urls with the query in user prompt, 
stop when urls number is enough for query or reach 15.
3. Thirdly: Use `research_crawl` to crawl all searched target urls, with the query in user prompt as one parameter, 
use `tool_custom_file_write` write all return content to a raw content file in markdown format, with name is constructed as keywords as prefixes and timestamps as suffixes 
4. Fourthly: You must review all crawled content in the file from previous step, anaylze and summarize, 
generate the final report with `tool_custom_file_write` in markdown format.
5. Finally: Use `copy_file_to_folder` copy the report file into folder path '/app/backend/shared-files', and return the name of report file.

## Constraints 
- Critical High Rule: Do not call concurrency `tool_tavily_func` or `research_crawl` request, You must wait the first call complete, check the return results, 
then start next function call.
- Strict Review: 
1. After obtaining search results, check each one to see if it is satisfied to the query, for example: the urls return by `tool_tavily_func`, 
if any one of them is not satisify query enough, drop it. If urls is enough(no more than 15), then stop search immediately and go next step.
2. After obtaining crawl results, check and review all return content, if you adjust it is not enough to make the final report for user query, 
cache it (write it to a tmp file if needed), and back to step 2, then combine new return content, check and review again. 
**But the loop is certainly no more than 3 times**
- **If you already have enough content, STOP and Generate the final report at once**.
"""

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
        uru_logger.get_logger().info(f"{self.agent_name} New message is going to send to LLM again! Length = {length} \n")

        for i, msg in enumerate(messages):
            uru_logger.get_logger().debug(f"------> Message {i}: {self.agent_name} Role={msg.type}, Content={msg.content}\n")
        if length >= self.max_messages:
            uru_logger.get_logger().warning(f"{self.agent_name} Message limit reached: {len(state['messages'])}", self.agent_name)
            return {
                "messages": [AIMessage("Conversation limit reached.")],
                "jump_to": "end"
            }
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_four_messages = state["messages"][-4:]
        
        for i, msg in enumerate(last_four_messages):
            uru_logger.get_logger().info(f"<------ {self.agent_name} The Model returned Message (last four) {i}: Role={msg.type}, Content={msg.content}")

            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    uru_logger.get_logger().info(f"TOOL REQUEST ==> Name: {tool_call['name']}, ARGS: {tool_call['args']}")

        # messages = state["messages"]
        # for i, msg in enumerate(messages):
        #     if isinstance(msg, AIMessage):
        #         if msg.tool_calls:
        #             for tool_call in msg.tool_calls:
        #                 logger.info(f"TOOL REQUEST: f{tool_call['name']}, ARGS: {tool_call['args']}")
        #     logger.info("<------ [%s] Model returned Message %d: Role=%s, Content='%s'\n", self.agent_name, i, msg.type, msg.content)
        
        return None

class CrawlAgents:
    """
    """
    def __init__(self, agent_name: str) -> None:

        self.agent_name = agent_name

        # initial the Message Middleware Object for manager agent
        self.messageLimitMiddleware = MessageLimitMiddleware(max_messages=100, agent_name=self.agent_name)

        # inital the tool call limitation with 10
        self.toolCallLimitMiddleware = cast(AgentMiddleware, ToolCallLimitMiddleware(tool_name="tool_tavily_search", run_limit=10, exit_behavior="end"))

        pass

    def create_crawler_agent(self, model_obj: ModelObj):
        """
        Build and return the Crawler deep agent.

        Resolves the LLM from ``model_obj``, then assembles a deep agent named
        "Crawler" that searches the web via Tavily, crawls target pages, and
        writes markdown reports using the senior-financial-web-crawler skill.
        Uses a virtual filesystem backend and applies message/tool-call limits.

        Args:
            model_obj: Model specification used to resolve the underlying LLM.

        Returns:
            A configured ``create_deep_agent`` instance ready to be invoked.
        """
        model = resolve_llm(model_obj)
        uru_logger.get_logger().info(f"Creating Crawler deep agent with llm_type={model_obj}")

        return create_deep_agent(
            name=self.agent_name,
            model=model,
            skills=["/skills/"],
            backend=FS_BACKEND,
            tools=[get_current_time, tool_tavily_search, tool_research_crawl, tool_custom_file_write, copy_file_to_folder],
            system_prompt=INDUSTRY_CRAWLER_SYSTEM_PROMPT,
            middleware=[self.messageLimitMiddleware, self.toolCallLimitMiddleware],
        )



