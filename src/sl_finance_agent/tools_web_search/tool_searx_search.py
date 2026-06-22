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
import os

from langchain.tools import tool
from langchain_community.utilities import SearxSearchWrapper

from ..common import get_logger
logger = get_logger(__name__)

# Configuration: export SEARX_HOST=http://192.168.8.50:8080
SEARX_HOST = os.environ.get("SEARX_HOST", "http://192.168.8.50:8080")
searcher_searx = SearxSearchWrapper(searx_host=SEARX_HOST)

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
    """
    try:
        logger.info("Fire search via SearxSearchWrapper with query: %s", query)
        raw_results = searcher_searx.results(query,
                                             num_results=num_results,
                                             engines=engines,
                                             timeout=8,
                                             safesearch=True,
                                             kwargs={
                                                "engine_params": {  # ← 重点：参数嵌套在此
                                                        "bing": {
                                                            "safesearch": 2,
                                                            "language": "zh-CN",
                                                            "region": "zh-CN",
                                                            "advanced_query": True
                                                            }
                                                    }
                                                }
                                             )
        # Serialize to JSON String
        # key settings:
        # - ensure_ascii=False: Keeps Chinese characters readable
        # - indent=2: Makes it pretty-printed (optional, helps debugging)
        json_string = json.dumps(raw_results, ensure_ascii=False, indent=2)

        logger.info("Search results is: %s", json_string)
        return json_string
    
    except Exception as e:
        logger.warning("Error at connecting to SearXNG or query process: %s", str(e))
        return f"Error at connecting to SearXNG or query process: {str(e)}"