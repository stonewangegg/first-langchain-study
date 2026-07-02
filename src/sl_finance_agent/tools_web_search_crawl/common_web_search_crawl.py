"""
"""
import os


class Common_web_search_crawl:

    # crawl4ai
    CRAWL4AI_BASE_URL = "http://192.168.8.50:11235"
    CRAWL4AI_API_TOKEN = "d6e20d9d18c470e8d62c411b61f2edc17003f0f38544c4f4bdd4956f21c1beed"

    # Configuration: export SEARX_HOST=http://192.168.8.50:8080
    SEARX_HOST = os.environ.get("SEARX_HOST", "http://192.168.8.50:8080")

    def __init__(self) -> None:

        self._total_urls: int = 0
        pass

    def set_total_urls(self, urls:int):
        self._total_urls = urls

common_web_search_crawl = Common_web_search_crawl()