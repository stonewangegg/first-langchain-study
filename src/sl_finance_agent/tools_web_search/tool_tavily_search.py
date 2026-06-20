"""Tavily web search tool for LangChain agents.

This module provides a LangChain-compatible tool that wraps the Tavily search
service, allowing agents to perform web searches and retrieve structured results
(query, answer, and a list of results with trimmed content).

Environment:
    export TAVILY_API_KEY=<your_tavily_api_key>  # required to authenticate.

Functions:
    tool_tavily_func(query: str) -> str: LangChain `@tool` that runs a Tavily
        web search and returns a pretty-printed JSON string of the trimmed
        results, or an error/no-results message on failure.
"""

# get the logger
import os
import json
import logging
from langchain.tools import tool

# export TAVILY_API_KEY=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
os.environ["TAVILY_API_KEY"] = "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"

from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)


# langchain tavily wrapper obejct via Tavily
tool_tavily = TavilySearch(
    max_results=3,
    include_answer=True,
    include_raw_content=False,
    include_images=False,
    include_image_descriptions=False,
    search_depth="basic",
    # time_range="day",
    # start_date=None,
    # end_date=None,
    # include_domains=None,
    # exclude_domains=None,
    # include_usage= False
)

@tool
def tool_tavily_func(query: str) -> str:
    """Run a web search via tavily langchain wrapper client, with all the passin parameters
    
    Args:
        query: The search keywords or question.
    """
    logger.info("Fire search via TavilySearchWrapper with query: %s\n\n", query)
    raw_results = tool_tavily.invoke(query)

    # drop no use items
    try:
        if raw_results and isinstance(raw_results, dict) and len(raw_results) > 0:
            trim_raw_results = {}
            if raw_results["query"] and isinstance(raw_results["query"], str):
                trim_raw_results["query"] = raw_results["query"]
            if raw_results["answer"] and isinstance(raw_results["answer"], str):
                trim_raw_results["answer"] = raw_results["answer"]
            if raw_results["results"] and isinstance(raw_results["results"], list):
                trim_raw_results["results"] = raw_results["results"]

            # shrink content of each result in results
            for item in trim_raw_results["results"]:
                if item["content"] and len(item["content"]) > 50:
                    item["content"] = item["content"][:50]

            # Serialize to JSON String
            # key settings:
            # - ensure_ascii=False: Keeps Chinese characters readable
            # - indent=2: Makes it pretty-printed (optional, helps debugging)
            json_string = json.dumps(trim_raw_results, ensure_ascii=False, indent=2)

            logger.info("Json Result of searching: %s\n\n", json_string)
            return json_string
        else:
            logger.warning("No results found.")
            return "No results found."
    except Exception as e:
        logger.error("Exception at connecting to Tavily or query processing: %s", e)
        return "Error occurred during search."