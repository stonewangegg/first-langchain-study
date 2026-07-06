"""Tavily-backed web search tool for LangChain agents.

This module exposes a LangChain-compatible ``@tool`` that wraps the Tavily
search service (``langchain_tavily.TavilySearch``) so that agents can run web
searches and consume the results as plain text/JSON. It is intended to be used
inside an agent's tool belt whenever the agent needs fresh, open-web context
(e.g. for financial news, company filings, or market commentary lookups).

Environment:
    export TAVILY_API_KEY=<your_tavily_api_key>   # required by TavilySearch.
    The module additionally hard-codes a development key via
    ``os.environ`` for local convenience; override it in production.

Functions:
    tool_tavily_func(query, max_results, include_domains=None) -> str:
        Public LangChain ``@tool``. Runs the search via ``_tool_tavily_func``,
        keeps only ``query`` / ``answer`` / ``results`` from the response, and
        returns one of:
            * ``"Get search results: {json_string}"`` on success (pretty JSON),
            * ``"No result found by this search."`` if Tavily yields nothing,
            * ``"Error occurred during search. {error}"`` on any exception.
"""

# get the logger
import os
import json
from typing import Dict
from langchain.tools import tool

# export TAVILY_API_KEY=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
os.environ["TAVILY_API_KEY"] = "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"

from langchain_tavily import TavilySearch

from ..common_utils import get_logger
logger = get_logger(__name__)


def _tool_tavily_func(query: str, max_results:int, include_domains: list|None = None, time_range: str = "month") -> Dict:
    """
        Example trimmed output::

            Get search results: {
            "query": "What happened at the last wimbledon",
            "answer": null,
            "results": [
                {
                "title": "Andy Murray pulls out of the men's singles draw at his last Wimbledon",
                "url": "https://www.nbcnews.com/news/sports/andy-murray-wimbledon-tennis-singles-draw-rcna159912",
                "content": "NBC News Now LONDON \u2014 Andy Murray, one of the last decade's most successful ...",
                "score": 0.6755297,
                "raw_content": null
                }
            ]
            }
    """

    # langchain tavily wrapper obejct via Tavily
    tool_tavily = TavilySearch(
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        include_image_descriptions=False,
        search_depth="basic",
        time_range=time_range,
        # start_date=None,
        # end_date=None,
        include_domains=include_domains,
        # exclude_domains=None,
        # include_usage= False
    )

    logger.info("Fire search via TavilySearchWrapper with query: %s\n\n", query)
    raw_results = tool_tavily.invoke(query)
    if raw_results:
        return raw_results
    else:
        return {}

@tool
def tool_tavily_search(query: str, max_results: int, time_range: str = "month", include_domains: list|None = None) -> str:
    """Run a web search via the Tavily LangChain wrapper and return the raw response.

    Delegates to the internal ``_tool_tavily_func`` helper, which calls
    ``langchain_tavily.TavilySearch`` with ``search_depth="basic"`` and
    ``include_answer=True``. No time-range filter is applied, so Tavily 
    searches the open web without date restrictions.

    From the response returned by ``_tool_tavily_func``, this tool keeps two
    top-level fields and pretty-prints them as JSON 

        * ``query``  -- the original query echoed back by Tavily.
        * ``results`` -- the list of result items **as returned by Tavily**
          (i.e. all original fields, e.g. ``title``, ``url``, ``content``,
          ``score``, ``raw_content``). The per-result ``score >= 4.0`` filter
          in the source builds a local ``shrink_results`` list that is
          currently never assigned, so the full unfiltered ``results`` list
          is what ends up in the output.

    Args:
        query: The search keywords or natural-language question to send to Tavily.
        max_results: Maximum number of results to retrieve from Tavily.
        time_range: The time range back from the current date to filter results.
        include_domains: Optional list of domains to restrict the search to
            (for example ``["reuters.com", "bloomberg.com"]``). ``None`` means
            no domain restriction and Tavily searches the open web.

    Returns:
        One of the following strings:

        * ``"Get search results: {json_string}"`` on success, where
          ``json_string`` is a pretty-printed JSON object containing the
          ``query`` and ``results`` fields described above.
        * ``"No result found by this search."`` if the raw response is empty
          (or the ``results`` list is empty / has no ``query`` to echo).
        * ``"Error occurred during this search. {str(e)}"`` if any exception
          is raised while calling Tavily or processing the response (the
          ``{str(e)}`` placeholder is replaced with ``str(exception)``).
    """
    
    raw_results = _tool_tavily_func(query=query, time_range=time_range, max_results=max_results, include_domains=include_domains)
    
    # drop no use items
    try:
        if raw_results and len(raw_results) > 0:
            shrink_raw_results = {}
            if raw_results["query"]:
                shrink_raw_results["query"] = raw_results["query"]
            # if raw_results["answer"]:
            #     trim_raw_results["answer"] = raw_results["answer"]
            if raw_results["results"] and len(raw_results["results"]) > 0:
                results = raw_results["results"]
                shrink_results = []
                for result in results:
                    if result["score"] >= 4.0: # will make this configure
                        shrink = {}
                        shrink["url"] = result["url"]
                        shrink["content"] = result["content"]
                        shrink["score"] = result["score"]
                        shrink_results.append(shrink)
                shrink_raw_results["results"] = raw_results["results"]

            # shrink content of each result in results
            # for item in trim_raw_results["results"]:
            #     if item["content"] and len(item["content"]) > 50:
            #         item["content"] = item["content"][:50]

            # Serialize to JSON String
            # key settings:
            # - ensure_ascii=False: Keeps Chinese characters readable
            # - indent=2: Makes it pretty-printed (optional, helps debugging)
            json_string = json.dumps(shrink_raw_results, ensure_ascii=False, indent=2)

            logger.info("Json Result of tavily searching: %s\n", json_string)
            return f"Get search results: {json_string}"
        else:
            logger.warning("No results found.")
            return "No result found by this search."
    except Exception as e:
        logger.error("Exception at connecting to Tavily or query processing: %s", e)
        return f"Error occurred during this search. {str(e)}"
    

if __name__ == "__main__":

    raw_results = _tool_tavily_func(query="新能源电动汽车近年行业整体营收、利润数据与增速数据", max_results=10)

    print(f"Searched raw results via Tavily:\n{raw_results}")

    pass