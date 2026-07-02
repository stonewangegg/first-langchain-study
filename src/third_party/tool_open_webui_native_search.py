"""
"""

from typing import List, Optional

from fastapi import Request
from open_webui.main import app
from open_webui.models.users import Users
from open_webui.routers.retrieval import SearchForm, process_web_search

from ..sl_finance_agent.common_utils import uru_logger

async def tool_open_webui_search_native(
        query: str,
        __user__: Optional[dict] = None,
    ) -> List[str]:

    """
    Perform a web search using Open WebUI's native search backend.

    This tool invokes the built-in ``process_web_search`` endpoint exposed by
    Open WebUI (v0.9.5+), reusing the server's configured web search engine
    (SearXNG, Bing, Google, Tavily, etc.) and the requesting user's
    permissions/quotas. It is intended as a drop-in search backend for tools
    that need to follow up on a query with a crawl step.

    Args:
        query: The natural-language search query to submit to the native
            web search engine.
        __user__: The Open WebUI user context dict (must contain an ``id``
            key). This follows the Open WebUI tool convention so the
            function can authenticate against the configured search
            provider using the user's settings and quotas. If ``None``,
            the call is rejected.

    Returns:
        A list of result URLs (``str``) extracted from the search response.
        Returns an empty list if the user context is missing, the user
        cannot be resolved, or an exception is raised while calling the
        backend.
    """

    if __user__ is None:
        uru_logger.get_logger().error("User information required for native search")
        return []

    try:
        # v0.9.5: get_user_by_id is now async
        user = await Users.get_user_by_id(__user__["id"])
        if user is None:
            uru_logger.get_logger().error("User not found")
            return []

        # Use native search - SearchForm expects a list of queries
        form = SearchForm.model_validate({"queries": [query]})
        result = await process_web_search(
            request=Request(scope={"type": "http", "app": app}),
            form_data=form,
            user=user,
        )

        # recode the debug message
        uru_logger.log_debug(f"Native search for '{query}' returned {result}")

        # Extract URLs from result items - v0.9.5 returns items with 'link' field
        urls = []
        items = result.get("items", [])
        for item in items:
            # Handle both dict and object formats
            if isinstance(item, dict):
                link = item.get("link")
            elif hasattr(item, "link"):
                link = item.link
            else:
                continue
            if link:
                urls.append(link)

        uru_logger.log_debug(f"Native search for '{query}' returned {len(urls)} URLs")

        # return the urls list for follow crawl
        return urls

    except Exception as e:
        uru_logger.get_logger().exception(f"Error in native search: {str(e)}")
        return []