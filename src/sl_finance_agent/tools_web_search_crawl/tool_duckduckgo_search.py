"""
LangChain tool wrapper around the DuckDuckGo search results component.

This module exposes a single LangChain ``tool``, ``tool_duckduckgo_search``,
that forwards search queries to the DuckDuckGo meta-search engine (via
:class:`~langchain_community.utilities.DuckDuckGoSearchAPIWrapper` and
:class:`~langchain_community.tools.DuckDuckGoSearchResults`) and returns the
raw JSON-encoded results. It is particularly useful for locating links to
specific file types (for example PDFs) by appending a ``filetype:`` hint to
the query.
"""

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from ..common_utils import get_logger
# get the logger
logger = get_logger(__name__)

# LangChain communication search tool
@tool
def tool_duckduckgo_search(query: str, num_results: int=10, file_type: str="") -> str:
    """
    A privacy-respecting build-in meta-search engine of LangChain Communication. 
    Use this to find current news, facts, or real-time information.

    Search the web for a specified topic and return relevant webpage links and summaries. It's especially useful for finding PDF download links.
    
    Args:
        query: The search keywords or question.
        num_results: the result number of searching results.
        file_type: the append search file type string, such as 'filetype: pdf'.
    """

    duck_duckGo_search_api_wrapper = DuckDuckGoSearchAPIWrapper(region="cn-zh", safesearch="on", max_results=10)

    try:
        
        # Force the addition of filetype:pdf to the end of search terms to increase the probability of matching PDFs.
        search_tool = DuckDuckGoSearchResults(api_wrapper=duck_duckGo_search_api_wrapper, num_results=num_results, return_direct=True, output_format="json")

        invoke_query = query
        if file_type:
            invoke_query = f"{query} {file_type}"

        logger.info("Fire search via duckduckgo search with query: %s\n\n", invoke_query)
        raw_results = search_tool.invoke(invoke_query)

        # json_string = json.dumps(raw_results, ensure_ascii=False, indent=2)

        logger.info("Search results is: %s \n", raw_results)
        return raw_results
    except Exception as e:
        logger.warning("Error at connect to SearXNG or query process: %s", str(e))
        return f"Error at connect to SearXNG or query process: {str(e)}"