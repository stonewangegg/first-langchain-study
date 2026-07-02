"""LangChain tool wrapper around the SearxSearchWrapper.

This module exposes a single LangChain ``tool``, ``tool_searxng_search``, that
forwards search queries to a locally running SearXNG meta-search instance and
returns the results as a pretty-printed JSON string.

Environment variables
---------------------
SEARX_HOST
    Base URL of the SearXNG instance to query. Defaults to
    ``http://192.168.8.50:8080`` for local development.
"""

# get the logger
import json
from typing import Literal

from langchain.tools import tool
from langchain_community.utilities import SearxSearchWrapper
from sqlalchemy import literal

from .common_web_search_crawl import common_web_search_crawl

from ..common_utils import get_logger
logger = get_logger(__name__)

# Langchain SearXNG tool wrapper for search agent
@tool
def tool_searxng_search(query: str, engines: list, num_results: int=3) -> str:
    """
    A privacy-respecting meta-search engine. 
    Use this to find current news, facts, or real-time information.
    
    Args:
        query: The search keywords or question.
        egnings: the search engines list such as ['baidu', 'bing'].
        num_results: the result number of searching results.

    Returns:
        The web research results json string
    """

    raw_results = _tool_searxng_search(query=query, engines=engines, num_results=num_results)
    # Serialize to JSON String
    # key settings:
    # - ensure_ascii=False: Keeps Chinese characters readable
    # - indent=2: Makes it pretty-printed (optional, helps debugging)
    json_string = json.dumps(raw_results, ensure_ascii=False, indent=2)

    logger.info("Search results is: %s", json_string)
    return json_string
    
@tool
def tool_searxng_search_urls(query: str, engines: list = ['baidu', 'bing'], num_results: int=10) -> list[str]:
    """
    Search SearXNG and return a lightweight list of search hits (title, snippet, url)
    so the agent can quickly scan what is available and pick URLs to crawl next.

    This is a meta-search tool backed by a self-hosted SearXNG instance: the request
    is fanned out to the configured upstream ``engines`` (e.g. ``['baidu', 'bing']``)
    and the merged results are returned. Prefer this tool when the agent only needs
    to discover candidate links for a topic; use a follow-up fetch / parse tool
    downstream to read the page contents.

    Args:
        query: The search keywords or natural-language question to send to SearXNG.
        engines: The list of upstream search engines to query, e.g.
            ``['baidu', 'bing']``. It must be no-empty, default is ``['baidu', 'bing']``.
        num_results: The maximum number of results to retrieve per engine.
            Defaults to ``10``. Note that SearXNG may return fewer hits if a
            engine does not have enough matches for the query.

    Returns:
        list[dict]: A list of search-hit dictionaries. Each dictionary has the
        following keys:

            * ``title`` (str | None): The page title as reported by the engine.
            * ``snippet`` (str | None): A short text excerpt / summary of the
              page content.
            * ``url`` (str | None): The canonical URL of the result.

        Returns an empty list (``[]``) if the upstream SearXNG call fails, if no
        engine returns any result with a URL, or if an exception is raised while
        contacting SearXNG / processing the query.
    """

    # Extract URLs from results
    search_results = []

    raw_results = _tool_searxng_search(query=query, engines=engines, num_results=num_results)
    if raw_results:
        for result in raw_results:
            if result.get("link"):
                search_item = {
                    "title": result.get("title"),
                    "snippet": result.get("snippet"),
                    "url": result.get("snippet")
                }
                search_results.append(search_item)

    return search_results

def _tool_searxng_search(query: str, engines: list|None = [], num_results: int=10, time_range: Literal["day", "month", "week", "year"] = "month") -> list:
    """
    Do the urls search via searXNG service
    """

    searcher_searx = SearxSearchWrapper(searx_host=common_web_search_crawl.SEARX_HOST, 
                                        k=20,
                                        categories=["general"],
                                        engines=engines,
                                        params={
                                            "language": "zh",
                                            "safesearch": 2,
                                            # "pageno": 1,
                                            "time_range": time_range,
                                        },
    )

    try:
        logger.info("Fire search via SearxSearchWrapper with query: %s", query)
        raw_results = searcher_searx.results(query, num_results=num_results)
        # Serialize to JSON String
        # key settings:
        # - ensure_ascii=False: Keeps Chinese characters readable
        # - indent=2: Makes it pretty-printed (optional, helps debugging)
        json_string = json.dumps(raw_results, ensure_ascii=False, indent=2)
        logger.info("Search results is: %s", json_string)

        return raw_results
    
    except Exception as e:
        logger.warning("Error at connecting to SearXNG or query process: %s", str(e))
        return []
    

if __name__ == "__main__":

    # import logging

    # logging.basicConfig(level=logging.DEBUG)

    # logging.getLogger("urllib3").setLevel(logging.DEBUG)
    # logging.getLogger("requests").setLevel(logging.DEBUG)

    raw_results = _tool_searxng_search(query="新能源电动汽车近3年行业整体营收增速、利润增速官方数据", num_results=15, engines=['bing', 'baidu'])
     # Extract URLs from results
    search_results = []
    if raw_results:
        for result in raw_results:
            if result.get("link"):
                search_item = {
                    "title": result.get("title"),
                    "snippet": result.get("snippet"),
                    "url": result.get("link")
                }
                search_results.append(search_item)
    
    print (f"The urls result list is:\n{search_results}")

    pass