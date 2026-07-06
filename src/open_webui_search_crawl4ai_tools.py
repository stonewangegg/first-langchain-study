"""
"""

import asyncio
from typing import Literal
from urllib.parse import parse_qs, urlparse

from jinja2 import Template
from pydantic import BaseModel, Field, model_validator

from sl_finance_agent import common_web_search_crawl, uru_logger, SUPPORTED_LLM_TYPES, model_factory, create_crawler_agent


class Tools:

    class Valves(BaseModel):
        INITIAL_RESPONSE: str = Field(
            title="Initial delta response",
            default="One moment while I crawl the internet...",
            description="The response the tool will post in the chat window when it starts its search and crawl. Set as blank for no response.",
        )
        USE_NATIVE_SEARCH: bool = Field(
            title="Use Native Search",
            default=True,
            description="Use OpenWebUI's native web search (in addition to or instead of SearXNG).",
        )
        SEARCH_WITH_SEARXNG: bool = Field(
            title="Search with SearXNG",
            default=False,
            description="Use SearXNG for gathering additional URLs for crawling.",
        )
        SEARXNG_BASE_URL: str = Field(
            title="SearXNG Search URL",
            default=f"{common_web_search_crawl.SEARX_HOST}/search?format=json&q=<query>",
            description="The full URL for your SearXNG API instance. Insert <query> where the search terms should go.",
        )
        SEARXNG_API_TOKEN: str = Field(
            title="SearXNG API Token",
            default="",
            description="The API token or Secret for your SearXNG instance.",
        )
        SEARXNG_METHOD: Literal["GET", "POST"] = Field(
            title="SearXNG HTTP Method",
            default="GET",
            description="HTTP method to use for SearXNG API calls (GET or POST).",
        )
        SEARXNG_TIMEOUT: int = Field(
            title="SearXNG Timeout",
            default=30,
            description="The timeout (in seconds) for SearXNG API requests.",
        )
        SEARXNG_MAX_RESULTS: int = Field(
            title="SearXNG Max Results",
            default=10,
            description="The maximum number of results to return from SearXNG.",
        )
        CRAWL4AI_BASE_URL: str = Field(
            title="Crawl4AI Base URL",
            default=common_web_search_crawl.CRAWL4AI_BASE_URL,
            description="The base URL for your Crawl4AI instance.",
        )
        CRAWL4AI_API_TOKEN: str = Field(
            title="Crawl4AI API Token",
            default=common_web_search_crawl.CRAWL4AI_API_TOKEN,
            description="API token for authenticating with your Crawl4AI instance. Leave empty if no authentication is required.",
        )
        CRAWL4AI_USER_AGENT: str = Field(
            title="Crawl4AI User Agent",
            default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.1.2.3 Safari/537.36",
            description="Custom User-Agent string for Crawl4AI.",
        )
        CRAWL4AI_TIMEOUT: int = Field(
            title="Crawl4AI Timeout",
            default=60,
            description="The timeout (in seconds) for Crawl4AI requests.",
        )
        CRAWL4AI_BATCH: int = Field(
            title="Crawl4AI Batch",
            default=5,
            description="The number of URLs to send to Crawl4AI per batch. If more than this number of URLs are found in total, the tool will send them to Crawl4AI in batches of this number. Useful for reducing the tokens used by the LLM per crawl.",
        )
        CRAWL4AI_MAX_URLS: int = Field(
            title="Crawl4AI Maximum URLs to crawl",
            default=20,
            description="The maximum number of URLs to crawl with Crawl4AI.",
        )
        CRAWL4AI_EXTERNAL_DOMAINS: bool = Field(
            title="Crawl External Domains",
            default=False,
            description="Allow Crawl4AI to crawl external/additional URL domains.",
        )
        CRAWL4AI_EXCLUDE_DOMAINS: str = Field(
            title="Excluded Domains",
            default="",
            description="Comma-separated list of external domains to exclude from crawling.",
        )
        CRAWL4AI_EXCLUDE_SOCIAL_MEDIA_DOMAINS: str = Field(
            title="Excluded Social Media Domains",
            default="facebook.com,twitter.com,x.com,linkedin.com,instagram.com,pinterest.com,tiktok.com,snapchat.com,reddit.com",
            description="Comma-separated list of social media domains to exclude from crawling.",
        )
        CRAWL4AI_EXCLUDE_IMAGES: Literal["None", "External", "All"] = Field(
            title="Exclude Images",
            default="None",
            description="Exclude images from crawling (None, External, All).",
        )
        CRAWL4AI_WORD_COUNT_THRESHOLD: int = Field(
            title="Word Count Threshold",
            default=200,
            description="The minimum word count threshold for content to be included.",
        )
        CRAWL4AI_TEXT_ONLY: bool = Field(
            title="Text Only",
            default=False,
            description="Only extract text content, excluding images and other media.",
        )
        CRAWL4AI_MAX_IMAGES: int = Field(
            title="Max Images",
            default=8,
            description="Maximum number of images to include in results. Set to 0 for no images.",
        )
        CRAWL4AI_MIN_IMAGE_SCORE: int = Field(
            title="Min Image Score To Include",
            default=6,
            ge=0,
            le=10,
            description="Minimum image score from Crawl4AI to consider including in the response. Min 0, Max 10.",
        )
        CRAWL4AI_VALIDATE_IMAGES: bool = Field(
            title="Validate Image Links",
            default=True,
            description="Validate any image links to make sure they are accessible.",
        )
        CRAWL4AI_MAX_TOKENS: int = Field(
            title="Max Tokens used by web content",
            default=0,
            description="Maximum tokens to use for the web search content response. Set to 0 for unlimited.",
        )
        # Output Settings
        RESPONSE_MODE: Literal["rich", "text"] = Field(
            title="Response Mode",
            default="rich",
            description="'rich' = Interactive HTML gallery with images (recommended). 'text' = Plain markdown with cited sources.",
        )
        OUTPUT_MAX_CHARS: int = Field(
            title="Max Output Characters",
            default=50000,
            description="Maximum characters in the final output to prevent response crashes. Set to 0 for unlimited (not recommended).",
        )
        LLM_BASE_URL: str = Field(
            title="LLM Base URL",
            default="https://openrouter.ai/api/v1",
            description="The base URL for your preferred OpenAI-compatible LLM.",
        )
        LLM_API_TOKEN: str = Field(
            title="LLM API Token",
            default="empty",
            description="Optional API Token for your preferred OpenAI-compatible LLM.",
        )
        LLM_PROVIDER: str = Field(
            title="LLM Provider and model",
            default="openrouter/auto",
            description="The LLM provider and model to use (see https://docs.crawl4ai.com/core/browser-crawler-config/#3-llmconfig-essentials).",
            examples=[
                "openai/gpt-4o",
                "ollama/llama-3-70b",
                "openrouter/auto",
                "azure/gpt-4o",
                "anthropic/claude-2",
            ],
        )
        LLM_TEMPERATURE: float = Field(
            title="LLM Temperature",
            default=0.3,
            description="The temperature to use for the LLM.",
        )
        LLM_INSTRUCTION: str = Field(
            title="LLM Extraction Instruction",
            default="""Focus on extracting the core content. Summarize lengthy sections into concise points
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
            """,
            description="The instruction to use for the LLM when extracting from the webpage.",
        )
        LLM_MAX_TOKENS: int = Field(
            title="LLM Max Tokens",
            default=4096,
            description="The maximum number of tokens to use for the LLM.",
        )
        LLM_TOP_P: float | None = Field(
            title="LLM Top P",
            default=None,
            description="The top_p value to use for the LLM.",
        )
        LLM_FREQUENCY_PENALTY: float | None = Field(
            title="LLM Frequency Penalty",
            default=None,
            description="The frequency penalty to use for the LLM.",
        )
        LLM_PRESENCE_PENALTY: float | None = Field(
            title="LLM Presence Penalty",
            default=None,
            description="The presence penalty to use for the LLM.",
        )
        MORE_STATUS: bool = Field(
            title="More status updates",
            default=False,
            description="Show more status updates during web search and crawl",
        )
        DEBUG: bool = Field(
            title="Debug logging",
            default=False,
            description="Enable detailed debug logging",
        )

        @model_validator(mode="after")
        def validate_settings(self):
            """Validate the conditional settings."""

            # USE_NATIVE_SEARCH or SEARCH_WITH_SEARXNG must be selected
            if not self.USE_NATIVE_SEARCH and not self.SEARCH_WITH_SEARXNG:
                raise ValueError(
                    "Either 'Use Native Search' or 'Search with SearXNG' must be enabled"
                )

            # SEARXNG_BASE_URL is required only when SEARCH_WITH_SEARXNG is True
            if self.SEARCH_WITH_SEARXNG and (
                not self.SEARXNG_BASE_URL or not self.SEARXNG_BASE_URL.strip()
            ):
                raise ValueError(
                    "'SearXNG Search URL' is required when 'Search with SearXNG' is enabled. "
                    "Please provide the URL for your SearXNG instance."
                )
            return self
        
    class UserValves(BaseModel):
        """Per-user configurable options for Research Mode and crawling strategies."""

        SEARXNG_MAX_RESULTS: int | None = Field(
            title="SearXNG Max Results",
            default=None,
            description="The maximum number of results to return from SearXNG.",
        )
        CRAWL4AI_MAX_URLS: int | None = Field(
            title="Crawl4AI Maximum URLs to crawl",
            default=None,
            description="The maximum number of URLs to crawl with Crawl4AI.",
        )
        CRAWL4AI_MAX_IMAGES: int | None = Field(
            title="Max Images",
            default=None,
            description="Maximum number of images to include (overrides admin setting).",
        )
        RESEARCH_MODE: bool = Field(
            default=False,
            description="Enable research mode using Crawl4AI with Deep Crawling.",
        )
        RESEARCH_CRAWL_MODE: Literal[
            "pseudo_adaptive", "llm_guided", "bfs_deep", "research_filter"
        ] = Field(
            default="pseudo_adaptive",
            description="""The crawling strategy to use in Research Mode:
            - pseudo_adaptive: Keyword-based URL scoring and iterative crawling
            - llm_guided: Use LLM to intelligently select which links to crawl next
            - bfs_deep: Breadth-first search style deep crawling
            - research_filter: Research mode with URL filtering and relevance scoring""",
        )
        RESEARCH_KEYWORD_WEIGHT: float = Field(
            default=0.7,
            description="The keyword relevance weight when using Research mode.",
        )
        RESEARCH_MAX_DEPTH: int = Field(
            default=2,
            le=10,
            description="The maximum depth of links to follow for the Research mode. CAUTION: Too high a value may cause excessive crawling.",
        )
        RESEARCH_MAX_PAGES: int = Field(
            default=15,
            le=25,
            description="The maximum number of pages to crawl in Research mode. CAUTION: Too high a value may cause excessive crawling.",
        )
        RESEARCH_BATCH_SIZE: int = Field(
            default=5,
            description="Number of URLs to process per batch during research crawling.",
        )
        RESEARCH_LLM_LINK_SELECTION: bool = Field(
            default=True,
            description="Use LLM to select next links when in llm_guided mode.",
        )
        RESEARCH_INCLUDE_EXTERNAL: bool = Field(
            default=False,
            description="Allow following external domains during research crawling.",
        )

    def __init__(self):
        
        # get all parameters settings
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        # reconstruct the searXNG searching url to ensure it is in correct format
        if self.valves.SEARCH_WITH_SEARXNG and self.valves.SEARXNG_BASE_URL:
            # Ensure SearXNG URL is properly formatted
            searxng_parsed_url = urlparse(self.valves.SEARXNG_BASE_URL)
            searxng_parsed_url_query = parse_qs(searxng_parsed_url.query)
            if "q" not in searxng_parsed_url_query:
                searxng_parsed_url_query["q"] = ["<query>"]
            if "format" in searxng_parsed_url_query:
                if searxng_parsed_url_query["format"][0] != "json":
                    searxng_parsed_url_query["format"][0] = "json"
            reconstructed_query = "&".join(
                [f"{key}={value[0]}" for key, value in searxng_parsed_url_query.items()]
            )
            self.valves.SEARXNG_BASE_URL = f"{searxng_parsed_url.scheme}://{searxng_parsed_url.netloc}{searxng_parsed_url.path}?{reconstructed_query}"

        self.crawl_counter = 0
        self.content_counter = 0
        self._debug_log = []  # Collect debug messages for response
        self._seen_images = set()  # Track unique images to avoid duplicates
        self.total_urls = 0

        uru_logger.get_logger().info("Web Search and Crawl tool initialized")

    async def search_crawl(self, user_prompt: str, mode_str: str) -> str:
        """
        """

        model_obj = model_factory(mode_str)

        final_content = None

        # Invoke the agent
        if model_obj:
            # Invoke the agent
            final_response = await create_crawler_agent(model_obj).ainvoke(
                                                                        {
                                                                            "messages": [
                                                                                {
                                                                                    "role": "user",
                                                                                    "content": user_prompt,
                                                                                }
                                                                            ]
                                                                        }
                                                                    )

            final_content = final_response["messages"][-1].content
            # Print the agent's response
            uru_logger.get_logger().info("\n**Final response**: \n" + final_content)

        return final_content or ""


if __name__ == "__main__":

    user_prompt_test = """
    # 目标上市公司: "{{company_name}}"

    ## 搜索获取最近一'{{time_range}}'，网络上发布的目标上市公司所属行业信息与数据, 根据用户要求: '{{query_str}}', 进行分析、总结, 生成报告
    """

    # get the user input parameters value
    company_name, time_range, query_str, model_str = map(str, input(f"Enter your target company name, time range, and query string, model[{SUPPORTED_LLM_TYPES}] separated by space: ").split())

    user_prompt_template = Template(user_prompt_test)

    user_prompt_final = user_prompt_template.render(company_name=company_name, time_range=time_range, query_str=query_str)

    uru_logger.get_logger().info("🚀 Starting the Main Agent workflow for: '%s'...\n", user_prompt_final)

    tools = Tools()

    asyncio.run(tools.search_crawl(user_prompt=user_prompt_final, mode_str=model_str))


    
        