"""
The tools for crawler via Crawl4AI REST API

Crawl target urls with query, orgnize all crawled content and return in json

"""

import asyncio
import json
import re
import traceback
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from langchain.tools import tool
import requests
from urllib.parse import parse_qs, urlparse, quote
from rapidfuzz import fuzz

from ..common_utils import uru_logger
from .common_web_search_crawl import common_web_search_crawl

LLM_INSTRUCTION: str = """Focus on extracting the core content. Summarize lengthy sections into concise points
            Include:
            - Key concepts and explanations
            - Important examples
            - Critical details that enhance understanding
            - Data from tables that support the main content
            - Any relevant data snippets
            Exclude:
            - Navigation elements
            - Sidebars
            - Footer content
            - Marketing or promotional material
            - Advertisements
            - User comments
            - Any other non-essential information
            Format the output as clean markdown with proper code blocks and headers.
            """

class ResearchCrawlMode:
    """Enumeration of research crawling modes."""
    PSEUDO_ADAPTIVE = "pseudo_adaptive"
    LLM_GUIDED = "llm_guided"
    BFS_DEEP = "bfs_deep"
    RESEARCH_FILTER = "research_filter"

class CrawlSettings(BaseModel):
    CRAWL4AI_EXCLUDE_IMAGES: Literal["None", "External", "All"] = Field( 
        title="Exclude Images", 
        default="None", 
        description="Exclude images from crawling (None, External, All).", 
    )

    CRAWL4AI_MIN_IMAGE_SCORE: int = Field(
        title="Min Image Score To Include",
        default=5,
        ge=0,
        le=10,
        description="Minimum image score from Crawl4AI to consider including in the response. Min 0, Max 10.",
    )

    RESEARCH_LLM_LINK_SELECTION: bool = Field(
        default=True,
        description="Use LLM to select next links when in llm_guided mode.",
    )

# =============================================================================
# Crawl4AI Configuration Stubs
# These classes mirror the crawl4ai library's config classes but are lightweight
# stubs that just serialize to JSON for the remote Crawl4AI API. This avoids
# requiring the full crawl4ai package (which has heavy browser dependencies).
# =============================================================================
# Crawl4AI 0.9.0
"""
"BrowserConfig": {
        "browser_type", "headless", "browser_mode", "viewport_width",
        "viewport_height", "viewport", "device_scale_factor", "accept_downloads",
        "java_script_enabled", "text_mode", "light_mode", "enable_stealth",
        "avoid_ads", "avoid_css", "user_agent", "user_agent_mode",
        "user_agent_generator_config", "verbose", "memory_saving_mode",
        "max_pages_before_recycle",
    },
    "CrawlerRunConfig": {
        # content selection / cleaning
        "word_count_threshold", "only_text", "css_selector", "target_elements",
        "excluded_tags", "excluded_selector", "keep_data_attributes", "keep_attrs",
        "remove_forms", "prettiify", "parser_type",
        # strategy objects (nested type is gated by the recursion)
        "extraction_strategy", "chunking_strategy", "markdown_generator",
        "scraping_strategy", "table_extraction",
        # locale / geo
        "locale", "timezone_id", "geolocation",
        # cache
        "cache_mode", "bypass_cache", "disable_cache", "no_cache_read",
        "no_cache_write", "check_cache_freshness", "cache_validation_timeout",
        "fetch_ssl_certificate",
        # timing / waiting
        "wait_until", "page_timeout", "wait_for", "wait_for_timeout",
        "wait_for_images", "delay_before_return_html", "mean_delay", "max_range",
        # scrolling / rendering
        "ignore_body_visibility", "scan_full_page", "scroll_delay",
        "max_scroll_steps", "process_iframes", "flatten_shadow_dom",
        "remove_overlay_elements", "remove_consent_popups",
        "adjust_viewport_to_content", "virtual_scroll_config",
        # media / capture
        "screenshot", "screenshot_wait_for", "screenshot_height_threshold",
        "force_viewport_screenshot", "pdf", "capture_mhtml",
        "image_description_min_word_threshold", "image_score_threshold",
        "table_score_threshold",
        # links / images filtering
        "exclude_external_images", "exclude_all_images",
        "exclude_social_media_domains", "exclude_external_links",
        "exclude_social_media_links", "exclude_domains", "exclude_internal_links",
        "score_links", "preserve_https_for_internal_links", "link_preview_config",
        # misc safe knobs
        "verbose", "log_console", "capture_network_requests",
        "capture_console_messages", "method", "stream", "prefetch", "url",
        "check_robots_txt", "user_agent", "user_agent_mode",
        "user_agent_generator_config", "url_matcher", "match_mode", "max_retries",
    }
""" 
# =============================================================================

class CacheMode:
    """Cache mode enumeration for Crawl4AI."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    BYPASS = "bypass"
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"


class LLMConfig:
    """LLM configuration for Crawl4AI extraction strategies."""

    def __init__(
        self,
        provider: str = "openai/gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key_token: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ):
        self.provider = provider
        self.base_url = base_url
        self.api_token = api_key_token
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def dump(self) -> dict:
        """Serialize to Crawl4AI API format: {"type": "ClassName", "params": {...}}"""
        params = {"provider": self.provider, "temperature": self.temperature}
        if self.base_url:
            params["base_url"] = self.base_url
        if self.api_token:
            params["api_token"] = self.api_token
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        return {"type": "LLMConfig", "params": params}

class LLMExtractionStrategy:
    """LLM-based extraction strategy for Crawl4AI."""

    def __init__(
        self,
        llm_config: LLMConfig,
        instruction: str = "",
        input_format: str = "fit_markdown",
        schema: Optional[dict] = None,
    ):
        self.llm_config = llm_config
        self.instruction = instruction
        self.input_format = input_format
        self.schema = schema

    def dump(self) -> dict:
        """Serialize to Crawl4AI API format: {"type": "ClassName", "params": {...}}"""
        params = {
            "llm_config": self.llm_config.dump(),
            "instruction": self.instruction,
            "input_format": self.input_format,
        }
        if self.schema:
            params["schema"] = {"type": "dict", "value": self.schema}
        return {"type": "LLMExtractionStrategy", "params": params}

class JsonCssExtractionStrategy:
    """Tranditional extraction strategy for Crawl4AI"""

    def __init__(self):
        self.schema = {
            "name": "article_data",
            "baseSelector": "article",
            "fields": [
                {"name": "title", "selector": "h1", "type": "text"},
                {"name": "author", "selector": ".author-name", "type": "text"},
                {"name": "publish_date", "selector": "time", "type": "text", "attribute": "datetime"},
                {"name": "tags", "selector": ".tag", "type": "text", "multiple": True}
            ]
        }

    def dump(self) -> dict:
        params = {
            "schema": self.schema
        }
        return{"type": "JsonCssExtractionStrategy", "params": params}

class PruningContentFilter:
    """Content filter that prunes irrelevant content."""

    def __init__(
        self,
        threshold: float = 0.5,
        threshold_type: str = "fixed",
        min_word_threshold: int = 50,
    ):
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.min_word_threshold = min_word_threshold

    def dump(self) -> dict:
        """Serialize to Crawl4AI API format: {"type": "ClassName", "params": {...}}"""
        return {
            "type": "PruningContentFilter",
            "params": {
                "threshold": self.threshold,
                "threshold_type": self.threshold_type,
                "min_word_threshold": self.min_word_threshold,
            },
        }

class DefaultMarkdownGenerator:
    """Default markdown generator for Crawl4AI."""

    def __init__(
        self,
        content_filter: Optional[PruningContentFilter] = None,
        options: Optional[dict] = None,
    ):
        self.content_filter = content_filter
        self.options = options or {}

    def dump(self) -> dict:
        """Serialize to Crawl4AI API format: {"type": "ClassName", "params": {...}}"""
        params = {"options": {"type": "dict", "value": self.options}}
        if self.content_filter:
            params["content_filter"] = self.content_filter.dump()
        return {"type": "DefaultMarkdownGenerator", "params": params}

class DefaultTableExtraction:
    """Default table extraction strategy."""

    def __init__(self):
        pass

    def dump(self) -> dict:
        """Serialize to Crawl4AI API format: {"type": "ClassName", "params": {...}}"""
        return {"type": "DefaultTableExtraction", "params": {}}
    
class LinkPreviewConfig:
    """Enable link head extraction with detailed configuration"""

    def __init__(
        self,
        include_internal: bool = True,           # Extract from internal links
        include_external: bool = False,          # Skip external links for this example
        max_links: int = 10,                     # Limit to 10 links for demo
        concurrency: int = 5,                    # Process 5 links simultaneously
        timeout: int = 10,                       # 10 second timeout per link
        link_query: str | None = "",             # Query for contextual scoring
        score_threshold: float = 0.3,            # Only include links scoring above 0.3
        verbose: bool = True                     # Show detailed progress
    ):
        self.include_internal = include_internal
        self.include_external = include_external
        self.max_links = max_links
        self.concurrency = concurrency
        self.timeout = timeout
        self.query = link_query
        self.score_threshold = score_threshold
        self.verbose = verbose

    def dump(self) -> dict:
        """Serialize link extract configure for Crawl4AI."""
        params = {
            "include_internal": self.include_internal,
            "include_external": self.include_external,
            "max_links": self.max_links,
            "concurrency": self.concurrency,
            "timeout": self.timeout,
            "score_threshold": self.score_threshold,
            "verbose": self.verbose,
        }
        if self.query:
            params["query"] = self.query
        return {"type": "LinkPreviewConfig", "params": params}

class BrowserConfig:
    """Browser configuration for Crawl4AI."""

    def __init__(
        self,
        headless: bool = True,
        light_mode: bool = False,
        verbose: bool = False,
        user_agent: Optional[str] = None,
        browser_type: Optional[str]= "chromium",
    ):
        self.headless = headless
        self.light_mode = light_mode
        self.verbose = verbose
        self.user_agent = user_agent
        self.browser_type = browser_type

    def dump(self) -> dict:
        """Serialize to Crawl4AI API format: {"type": "ClassName", "params": {...}}"""
        params = {
            "headless": self.headless,
            "light_mode": self.light_mode,
            "verbose": self.verbose,
            "browser_type": self.browser_type,
        }
        if self.user_agent:
            params["user_agent"] = self.user_agent
        return {"type": "BrowserConfig", "params": params}

class CrawlerRunConfig:
    """Crawler run configuration for Crawl4AI."""

    def __init__(
        self,
        markdown_generator: Optional[DefaultMarkdownGenerator] = None,
        extraction_strategy: Optional[LLMExtractionStrategy|JsonCssExtractionStrategy] = None,
        table_extraction: Optional[DefaultTableExtraction] = None,
        link_preview_config: Optional[LinkPreviewConfig] = None,
        exclude_external_links: bool = False,
        exclude_social_media_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        user_agent: Optional[str] = None,
        stream: bool = False,
        cache_mode: str = CacheMode.BYPASS,
        page_timeout: int = 60000,
        only_text: bool = False,
        word_count_threshold: int = 200,
        exclude_all_images: bool = False,
        exclude_external_images: bool = True,
        remove_overlay_elements: bool = True,
        score_links: bool = True,
    ):
        self.markdown_generator = markdown_generator
        self.extraction_strategy = extraction_strategy
        self.table_extraction = table_extraction
        self.link_preview_config = link_preview_config
        self.exclude_external_links = exclude_external_links
        self.exclude_social_media_domains = exclude_social_media_domains or []
        self.exclude_domains = exclude_domains or []
        self.user_agent = user_agent
        self.stream = stream
        self.cache_mode = cache_mode
        self.page_timeout = page_timeout
        self.only_text = only_text
        self.word_count_threshold = word_count_threshold
        self.exclude_all_images = exclude_all_images
        self.exclude_external_images = exclude_external_images
        self.remove_overlay_elements = remove_overlay_elements
        self.score_links = score_links

    def dump(self) -> dict:
        """Serialize to Crawl4AI API format: {"type": "ClassName", "params": {...}}"""
        params = {
            "exclude_external_links": self.exclude_external_links,
            "exclude_social_media_domains": self.exclude_social_media_domains,
            "exclude_domains": self.exclude_domains,
            "stream": self.stream,
            "cache_mode": self.cache_mode,
            "page_timeout": self.page_timeout,
            "only_text": self.only_text,
            "word_count_threshold": self.word_count_threshold,
            "exclude_all_images": self.exclude_all_images,
            "exclude_external_images": self.exclude_external_images,
            "remove_overlay_elements": self.remove_overlay_elements,
            "score_links": self.score_links,
        }
        if self.markdown_generator:
            params["markdown_generator"] = self.markdown_generator.dump()
        if self.extraction_strategy:
            params["extraction_strategy"] = self.extraction_strategy.dump()
        if self.table_extraction:
            params["table_extraction"] = self.table_extraction.dump()
        if self.link_preview_config:
            params["link_preview_config"] = self.link_preview_config.dump()
        if self.user_agent:
            params["user_agent"] = self.user_agent
        return {"type": "CrawlerRunConfig", "params": params}

# Unused stubs kept for potential future use
class BestFirstCrawlingStrategy:
    """Best-first crawling strategy stub."""

    pass

class KeywordRelevanceScorer:
    """Keyword relevance scorer stub."""

    pass

# ============================================================================
# End of Crawl4AI Configuration Stubs
# ============================================================================

class ArticleData(BaseModel):
    topic: str
    summary: str

@tool
async def research_crawl(
        urls: List[str],
        query: str,
        mode: str = "pseudo_adaptive"
    ) -> dict | None:

    """Crawl a list of URLs in research mode and return content relevant to ``query``.

    This is the public LangChain ``@tool`` entry point that an agent calls when
    it wants to perform a multi-page research crawl (as opposed to a single
    one-shot crawl via ``_crawl_url``). Given a small set of starting URLs and a
    natural-language ``query``, it delegates to one of the implemented crawling
    strategies based on ``mode``, each of which explores beyond the initial URLs
    to surface additional pages relevant to the research topic.

    The tool is async because each underlying crawl strategy in turn calls the
    asynchronous Crawl4AI HTTP endpoint via ``requests`` wrapped in the
    ``asyncio`` event loop. All strategies ultimately return a dict of the same
    shape, e.g.:

        {
            "content":    [{"url": ..., "title": ..., "content": [...], ...}, ...],
            "images":     [...],   # absolute image URLs across all crawled pages
            "videos":     [...],   # absolute video URLs across all crawled pages
            "links":      [...],   # discovered links (research mode only)
            "pages_crawled": <int>,
        }

    or ``None`` if a strategy returns nothing.

    Currently supported modes (see ``ResearchCrawlMode``):
        * ``"pseudo_adaptive"`` (default) -- ``_pseudo_adaptive_crawl``:
          bounded breadth-first crawl that starts from ``urls`` (up to 5), scores
          discovered links by keyword relevance to ``query``, and follows the
          highest-scoring links up to ``max_depth``. Stays within the same
          domain by default. Cheap, deterministic, no extra LLM calls for
          link selection.
        * ``"llm_guided"`` -- ``_llm_guided_crawl``: similar BFS but uses an
          LLM-extraction-style configuration for link evaluation, with a
          keyword-scoring fallback when LLM selection is unavailable or
          fails. Better link quality at the cost of additional latency.

    Args:
        urls: A string list of Seed URLs to begin crawling from. 
        query: A string that Free-form research question / topic. Used both as the keyword
            source for scoring discovered links and as the semantic anchor for
            ``LLM_INSTRUCTION``-style extraction inside each page.
        mode: Optional, default is ``PSEUDO_ADAPTIVE``. Which crawling strategy to use. Should be one of
            ``ResearchCrawlMode.PSEUDO_ADAPTIVE`` or
            ``ResearchCrawlMode.LLM_GUIDED``. Any unknown value is logged and
            falls back to ``PSEUDO_ADAPTIVE`` so the agent never gets a hard
            error for a typo.

    Returns:
        A dict shaped as described above, or ``None`` if the chosen strategy
        returns no payload. On network / Crawl4AI errors, strategies may
        return a dict with an ``"error"`` and ``"details"`` keys instead of
        raising, so that the calling agent always receives a serializable
        result.
    """
    
    if mode == ResearchCrawlMode.PSEUDO_ADAPTIVE:
        return await _pseudo_adaptive_crawl(urls, query)
    elif mode == ResearchCrawlMode.LLM_GUIDED:
        return await _llm_guided_crawl(urls, query)
    # elif mode == ResearchCrawlMode.BFS_DEEP:
    #     return await _bfs_deep_crawl(urls, query)
    # elif mode == ResearchCrawlMode.RESEARCH_FILTER:
    #     return await _research_filter_crawl(urls, query)
    else:
        # Default to pseudo_adaptive for unknown modes
        uru_logger.get_logger().warning(
            f"Unknown research crawl mode: {mode}, defaulting to pseudo_adaptive"
        )
        return await _pseudo_adaptive_crawl(urls, query)
    
async def _pseudo_adaptive_crawl(
        start_urls: List[str],
        query: str,
        max_pages: int=15,
        max_depth: int=2,
        batch_size: int=5,
        include_external: bool=False
    ) -> dict:
    """
    Implement a simplified adaptive-like crawling using:
    1. Initial crawl to discover links
    2. Keyword-based filtering of discovered links
    3. Iterative crawling with priority scoring
    """
    from collections import deque

    crawlSettings = CrawlSettings()

    keywords = query.lower().split()

    # Track crawled pages and discovered links
    crawled_pages = set()
    crawled_results = []
    all_images = []
    all_videos = []

    # Queue of (url, depth, initial_score)
    queue = deque()

    # this is the key point of target urls prepare to crawl
    # Add initial URLs with base score
    for url in start_urls[:5]:  # Limit starting URLs
        if url not in crawled_pages:
            # score = sum(1 for kw in keywords if kw in url.lower())
            score = sum(1 for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', url.lower()))
            queue.append((url, 0, score))

    # set the total urls to singleton
    common_web_search_crawl.set_total_urls(max_pages)

    # start main loop
    while queue and len(crawled_pages) < max_pages:
        
        # Get batch of URLs to process
        batch = []
        for _ in range(min(batch_size, len(queue))):
            if queue:
                batch.append(queue.popleft())

        # Sort batch by score (highest first)
        batch.sort(key=lambda x: x[2], reverse=True)

        for url, depth, score in batch:
            if len(crawled_pages) >= max_pages or depth > max_depth:
                continue

            if url in crawled_pages:
                continue

            crawled_pages.add(url)

            # Crawl the page content with link extraction
            result = await _crawl_url(
                urls=[url],
                crawlSettings=crawlSettings,
                link_query=query,
                extract_links=True
            )

            # check the crawled result form _crawl_url
            if result.get("content"):
                crawled_results.extend(result["content"])

            if result.get("images"):
                all_images.extend(result["images"])

            if result.get("videos"):
                all_videos.extend(result["videos"])

            # If we haven't reached max depth, discover and score new links
            if depth < max_depth:
                # going down deep
                discovered_links = result.get("links", [])

                for link in discovered_links:
                    if link in crawled_pages:
                        continue

                    parsed_link = urlparse(link)
                    parsed_url = urlparse(url)

                    # Check domain restrictions
                    if not include_external:
                        if (
                            parsed_link.netloc
                            and parsed_link.netloc != parsed_url.netloc
                        ):
                            continue

                    # Score the link
                    link_lower = link.lower()
                    link_score = sum(1 for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', link_lower))

                    if link_score > 0:  # Only follow relevant links
                        queue.append((link, depth + 1, link_score))

    uru_logger.get_logger().info(f"[Pseudo-Adaptive] Crawled {len(crawled_pages)} pages")

    return {
        "content": crawled_results,
        "images": all_images,
        "videos": all_videos,
        "pages_crawled": len(crawled_pages),
    }

# LLM Guider Crawl
async def _llm_guided_crawl(
        start_urls: List[str],
        query: str
    ) -> dict:
    """
    Use LLM to intelligently select which links to crawl next.
    This provides a form of "intelligent" crawling via the API.
    """

    crawlSettings = CrawlSettings()

    max_pages = 15
    use_llm_selection = crawlSettings.RESEARCH_LLM_LINK_SELECTION
    include_external = False

    crawled_pages = set()
    crawled_results = []
    all_images = []
    all_videos = []
    
    # Process starting URLs
    urls_to_process = list(start_urls[:5])

    while urls_to_process and len(crawled_pages) < max_pages:
        current_url = urls_to_process.pop(0)

        if current_url in crawled_pages:
            continue

        crawled_pages.add(current_url)

        # Crawl page with link extraction
        result = await _crawl_url(
            urls=[current_url],
            crawlSettings=crawlSettings,
            link_query=query,
            extract_links=True
        )

        if result.get("content"):
            crawled_results.extend(result["content"])

        if result.get("images"):
            all_images.extend(result["images"])

        if result.get("videos"):
            all_videos.extend(result["videos"])

        # Get discovered links
        discovered_links = result.get("links", [])[:15]

        if not discovered_links:
            continue

        # Filter links by domain
        if not include_external:
            parsed_current = urlparse(current_url)
            filtered_links = []
            for link in discovered_links:
                parsed_link = urlparse(link)
                if (
                    not parsed_link.netloc
                    or parsed_link.netloc == parsed_current.netloc
                ):
                    filtered_links.append(link)
            discovered_links = filtered_links

        if not discovered_links:
            continue

        # send the crawled results to LLM for evaluation and selection
        if use_llm_selection:
            # Use LLM to select next links
            link_selection_prompt = f"""
                Given the research query: "{query}"

                These links were discovered from the current page:
                {chr(10).join([f"{i+1}. {link}" for i, link in enumerate(discovered_links[:10])])}

                Select the 3 most relevant links to explore next that would best help answer the research query.
                Return only the link numbers (e.g., "1, 3, 5"), nothing else.
            """

            try:
                # Fire an agent call

                # Make API call for LLM-based link selection
                # For now, we'll fall back to keyword scoring if this fails
                selected_indices = []
                try:
                    # This would require a separate API call - simplify for now
                    pass
                except:
                    pass
            except Exception as e:
                uru_logger.get_logger().warning(f"LLM link selection failed: {e}")

        # Fallback: Add high-scoring links based on keyword relevance
        keywords = query.lower().split()
        scored_links = []
        for link in discovered_links:
            if link in crawled_pages or link in urls_to_process:
                continue
            link_lower = link.lower()
            # score = sum(1 for kw in keywords if kw in link_lower)
            score = sum(1 for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', link_lower))
            if score > 0:
                scored_links.append((link, score))

        scored_links.sort(key=lambda x: x[1], reverse=True)

        # Add top links to processing queue
        for link, score in scored_links[:3]:
            if link not in urls_to_process and link not in crawled_pages:
                urls_to_process.append(link)

    uru_logger.get_logger().info(f"[LLM-Guided] Crawled {len(crawled_pages)} pages")

    return {
        "content": crawled_results,
        "images": all_images,
        "videos": all_videos,
        "pages_crawled": len(crawled_pages),
    }

def _send_crawl_request(urls: list, browser_config: BrowserConfig, crawler_config: CrawlerRunConfig) -> Any | None:

    # the base url of crawl4ai service
    endpoint = f"{common_web_search_crawl.CRAWL4AI_BASE_URL}/crawl"
    uru_logger.get_logger().info(f"Contacting Crawl4AI at {endpoint} for URLs: {urls}")

    headers = {"Content-Type": "application/json"}
    # Add API token if configured
    headers["Authorization"] = f"Bearer {common_web_search_crawl.CRAWL4AI_API_TOKEN}"

    # key point the payload sending to crawl4ai service with all settings
    payload = {
        "urls": urls,
        "screenshot": False,
        "browser_config": browser_config.dump(),
        "crawler_config": crawler_config.dump(),
    }

    uru_logger.log_debug(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        # Using a timeout to prevent the UI from hanging
        timeout = 60 * len(urls) + 60

        # send the crawl request with all configure and query message and URLs
        response = requests.post(
            endpoint, json=payload, headers=headers, timeout=timeout
        )
        
        if response.status_code != 200:
            uru_logger.get_logger().error(response.text)
        
        response.raise_for_status()

        return response.json()
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error connecting to Crawl4AI: {str(e)}. Check if the URL {common_web_search_crawl.CRAWL4AI_BASE_URL} is accessible."
        uru_logger.get_logger().error(error_msg)
        return None
    

# fire the crawl with crawl4ai JsonCssExtractionStrategy
async def _crawl_url(
    urls: list[str] | str,
    crawlSettings: CrawlSettings,
    link_query: Optional[str] = None,
    extract_links: bool = False
) -> dict:
    """
    Internal function to crawl URLs and extract content.
    This tool converts any webpage into clean content and extracts images and videos.

    :param urls: The exact web URL(s) to extract data from.
    :param query: Optional search query for research mode.
    :param extract_links: Whether to extract and return discovered links for research mode.
    """
    content_counter:int =0

    if isinstance(urls, str):
        urls = [urls]

    for idx, url in enumerate(urls):
        # Ensure URL starts with http
        if not url.startswith("http"):
            urls[idx] = f"https://{url}"

    # Building configs
    browser_config = BrowserConfig(
        headless=True,
        light_mode=True,
        verbose=True,
    )

    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(),
        options={"ignore_links": True, "escape_html": False, "body_width": 80},
    )

    # link extract configure
    link_preview_config=LinkPreviewConfig(
        include_internal=True,           # Extract from internal links
        include_external=False,          # Skip external links for this example
        max_links=10,                    # Limit to 10 links for demo
        concurrency=5,                   # Process 5 links simultaneously
        timeout=10,                      # 10 second timeout per link
        link_query=link_query,                     # Link contextual query to crawl4ai
        score_threshold=0.3,             # Only include links scoring above 0.3
        verbose=True                     # Show detailed progress
    )


    # Fall back to tranditinal Json CSS extraction strategy with out LLM
    # On the server side it configure the LLM provider, test to see if it works on the server side
    jsonCssExtractionStrategy = JsonCssExtractionStrategy()

    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        extraction_strategy=jsonCssExtractionStrategy,
        table_extraction=DefaultTableExtraction(),
        link_preview_config=link_preview_config,
        exclude_external_links=True,
        exclude_social_media_domains=[
            d.strip()
            for d in "facebook.com,twitter.com,x.com,linkedin.com,instagram.com,pinterest.com,tiktok.com,snapchat.com,reddit.com".split(",")
        ],
        exclude_domains=[],
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.1.2.3 Safari/537.36",
        stream=False,
        cache_mode=CacheMode.BYPASS,
        page_timeout=60 * 1000,  # Convert to milliseconds
        only_text=False,         # Notice watching this effort in testing and decide if make it be True
        word_count_threshold=200,
        exclude_all_images=crawlSettings.CRAWL4AI_EXCLUDE_IMAGES == "All",
        exclude_external_images=crawlSettings.CRAWL4AI_EXCLUDE_IMAGES == "External",
    )

    crawl_counter:int = len(urls) # refact to be a global counter later

    # fire the crawl
    try:
        
        raw_crawled_data = _send_crawl_request(urls=urls, browser_config=browser_config, crawler_config=crawler_config)
        if not raw_crawled_data:
            uru_logger.get_logger().error(f"Crawl failed with URLs, the response is:\n {str(raw_crawled_data)}")
            return {}
            
        uru_logger.log_debug(f"Crawled raw crawled data: \n{json.dumps(raw_crawled_data, ensure_ascii=False, indent=2)}.")

        # Crawl4AI returns content in the 'results' key as a list
        results = []
        seen_images = set()
        seen_videos = set()
        all_images = []
        all_videos = []
        all_links = {}  # For research mode link extraction

        result_data_list = raw_crawled_data.get("results", [])
        uru_logger.get_logger().info(f"Crawled raw result length: {len(result_data_list)}.")
        for data_item in result_data_list:

            # check crawed data in this item is success or not
            if data_item.get("success") is not True:
                uru_logger.get_logger().warning(f"Can not find the success flag, pass: {data_item}")
                continue

            # url of this one crawled page
            url = data_item.get("url", "")
            parsed_url = urlparse(url)

            uru_logger.log_debug(f"Craweld page url parsed: {parsed_url}")

            # Extract media - use consistent min score valve for both images and videos
            # ----------------------------------------------------------------------------------------------------
            # images
            image_list = []
            min_score = crawlSettings.CRAWL4AI_MIN_IMAGE_SCORE
            found_images = list(
                filter(
                    lambda x: x.get("score", 0) >= min_score,
                    data_item.get("media", {}).get("images", []),
                )
            )
            for img in found_images:
                src = img.get("src")
                if src:
                    # Fix protocol-relative URLs
                    if src.startswith("//"):
                        src = f"https:{src}"
                    elif not src.startswith("http"):
                        # Handle relative URLs
                        src = f"{parsed_url.scheme}://{parsed_url.netloc}/{src.lstrip('/')}"

                    parsed_image = urlparse(src)

                    if (
                        f"{parsed_image.scheme}://{parsed_image.netloc}/{parsed_image.path}"
                        not in seen_images
                    ):
                        seen_images.add(
                            f"{parsed_image.scheme}://{parsed_image.netloc}/{parsed_image.path}"
                        )
                        image_list.append(src)
            
            # videos
            video_list = []
            found_videos = list(
                filter(
                    lambda x: x.get("score", 0) >= min_score,
                    data_item.get("media", {}).get("videos", []),
                )
            )
            for vid in found_videos:

                src = vid.get("src")
                if src:

                    # Fix protocol-relative URLs
                    if src.startswith("//"):
                        src = f"https:{src}"
                    elif not src.startswith("http"):
                        # Handle relative URLs
                        src = f"{parsed_url.scheme}://{parsed_url.netloc}/{src.lstrip('/')}"

                    parsed_video = urlparse(src)

                    if (
                        f"{parsed_video.scheme}://{parsed_video.netloc}/{parsed_video.path}"
                        not in seen_images
                    ):
                        seen_videos.add(
                            f"{parsed_video.scheme}://{parsed_video.netloc}/{parsed_video.path}"
                        )
                        video_list.append(src)
            # ----------------------------------------------------------------------------------------------------

            # Extract links for research mode
            if extract_links and link_query:
                internal_list = []
                external_list = []

                links = data_item.get("links", {})
                if links and len(links) >= 1:

                    for internal in links.get("internal", []):
                        if (internal.get("text") 
                            and internal.get("contextual_score") 
                            and internal.get("contextual_score") >= 0.5
                            and _fuzzy_similarity(internal.get("text"), link_query) >= 0.45):
                            internal_list.append({"href": internal.get("href"), "text": internal.get("text")})

                    for external in links.get("external", []):
                        if (external.get("text") 
                            and external.get("contextual_score") 
                            and external.get("contextual_score") >= 0.5
                            and _fuzzy_similarity(external.get("text"), link_query) >= 0.45):
                            external_list.append({"href": external.get("href"), "text": external.get("text")})
                
                all_links["internal"] = internal_list
                all_links["external"] = external_list

            # ----------------------------------------------------------------------------------------------------

            # Now get the page content
            extracted_markdown = data_item.get("markdown", "{}").get("markdown_with_citations", "")
            if not extracted_markdown:
                uru_logger.get_logger().warning(f"The extracted markdown content is 'empty', check the url: {url}")

            # Build result with URL included
            results.append(
                {
                    "url": url,
                    "metadata": data_item.get("metadata", {}),
                    "content": extracted_markdown,
                    "images": image_list,
                    "videos": video_list,
                }
            )

            # the final all images and videos
            all_images.extend(image_list)
            all_videos.extend(video_list)

        content_counter += len(results) # refact to be a global counter later

        response = {
            "content": results,
            "images": all_images or [],
            "videos": all_videos or [],
            "links": all_links if extract_links else [],
        }

        uru_logger.get_logger().info(f"Successfully crawled {len(results)} URLs, return the final response.")

        return response

    except requests.exceptions.RequestException as e:
        error_msg = f"Network error connecting to Crawl4AI: {str(e)}. Check if the URL {common_web_search_crawl.CRAWL4AI_BASE_URL} is accessible."
        uru_logger.get_logger().error(error_msg)
        return {"error": error_msg, "details": e}
    except Exception as e:
        error_msg = (
            f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
        )
        uru_logger.get_logger().error(error_msg)
        return {"error": error_msg, "details": e}

# fuzzy matching (best for search)
def _fuzzy_similarity(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100


if __name__ == "__main__":

    crawlSettings = CrawlSettings()

    response = asyncio.run(_crawl_url(
                                urls="http://www.cnautonews.com/chengyongcar/2026/05/11/detail_20260511389571.html", 
                                crawlSettings=crawlSettings, 
                                link_query="新能源电动汽车近年行业整体营收、利润数据与增速数据", 
                                extract_links=True
                            )
                        )

    if isinstance(response, dict):
        print(f"Crawl results: \n{response}")

    pass