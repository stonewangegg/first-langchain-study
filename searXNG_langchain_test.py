"""
The unit test for searXNG langchain wrapper

"""

import os
import json
import requests

from urllib.parse import urlencode

from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Configuration: export SEARX_HOST=http://192.168.8.50:8080
searx_host = os.environ.get("SEARX_HOST", "http://192.168.8.50:8080")
searcher_searx = SearxSearchWrapper(searx_host=searx_host, categories=["general"])

# Langchain SearXNG tool wrapper for search agent
def tool_searxng(query: str) -> str:
    """
    A privacy-respecting meta-search engine. 
    Use this to find current news, facts, or real-time information.
    
    Args:
        query: The search keywords or question.
    """
    try: 
        print(f"Fire search via SearxSearchWrapper with query: {query}\n\n")
        raw_results = searcher_searx.results(query,
                                             num_results=10,
                                             engines=["bing"],
                                             timeout=8,
                                             kwargs={
                                                "engine_params": {  # ← 重点：参数嵌套在此
                                                        "bing": {
                                                            "safesearch": 2,
                                                            "nocache": 1,
                                                            "language": "zh-CN",
                                                            "region": "zh-CN",
                                                            "pageno": 1,
                                                            "format": "json"
                                                            }
                                                    }
                                                }
                                             )
        # Serialize to JSON String
        # key settings:
        # - ensure_ascii=False: Keeps Chinese characters readable
        # - indent=2: Makes it pretty-printed (optional, helps debugging)
        json_string = json.dumps(raw_results, ensure_ascii=False, indent=2)

        print(f"Search raw results is: {raw_results} \n\n")
        return json_string
    
    except Exception as e:
        return f"Error at connecting to SearXNG or query process: {str(e)}"
    

def searxng_search(
    query: str,
    searxng_host: str = "http://192.168.8.50:8080",  
    engines: list = ["bing"],  # 默认仅用Bing
    language: str = "zh-CN",   
    safesearch: int = 2,        # 0=关闭 1=中等 2=严格
    num_results: int = 6,
    timeout: int = 10
) -> str:
    """
    直接调用 SearXNG 原生 API（无 LangChain 封装）
    解决方案核心：
    1. 优先指定 engines=bing
    2. 使用百度专用参数语法
    3. 添加年报搜索关键词优化
    """
    # === 关键修正 1：适配百度引擎的参数规则 ===
    # 百度要求：language=zh 且需添加专业搜索语法
    params = {
        "q": f"{query} filetype:pdf",
        "engines": ",".join(engines),  # 多引擎用逗号分隔
        "language": language,          # 百度仅接受 'zh'
        "safesearch": safesearch,
        "format": "json",
        "pageno": 1,
        "time_range": "",             # 清空时间范围避免干扰
        "categories": "general",      # 仅通用搜索（避免新闻/图片污染）
        "nocache": 1
    }

    # === 关键修正 2：绕过 SearXNG 参数解析缺陷 ===
    # 直接拼接百度原生参数（SearXNG 不会自动转换）
    if "baidu" in engines:
        params["q"] += " &lm=1"  # 百度参数：按时间倒序 (lm=1)
        params["q"] = params["q"].replace("site:", "site%3A")  # 手动转义
        params["language"] = "zh"

    try:
        url = f"{searxng_host}/search?{urlencode(params)}"
        print(f"🔍 调用 API: {url}\n")  # 调试用：打印实际请求URL
        
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}  # 模拟浏览器
        )
        response.raise_for_status()
        
        results = response.json()["results"]
        
        return json.dumps(results[:num_results], ensure_ascii=False, indent=2)  # 返回原始 top number results
    
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")
        return ""



def tool_duckduckgo_search(query: str) -> str:
    """
    A privacy-respecting build-in meta-search engine of LangChain Communication. 
    Use this to find current news, facts, or real-time information.

    Search the web for a specified topic and return relevant webpage links and summaries. It's especially useful for finding PDF download links.
    
    Args:
        query: The search keywords or question.
    """

    duck_duckGo_search_api_wrapper = DuckDuckGoSearchAPIWrapper(region="cn-zh", safesearch="on", max_results=10)

    try:
        
        # Force the addition of filetype:pdf to the end of search terms to increase the probability of matching PDFs.
        search_tool = DuckDuckGoSearchResults(api_wrapper=duck_duckGo_search_api_wrapper, num_results=10, return_direct=True, output_format="json")

        invoke_query = f"{query} filetype:pdf"

        print(f"Fire search via duckduckgo search with query: {invoke_query}\n\n")
        raw_results = search_tool.invoke(invoke_query)

        # json_string = json.dumps(raw_results, ensure_ascii=False, indent=2)

        print(f"Search results is: {raw_results} \n")
        return raw_results
    except Exception as e:
        return f"Error at connecting to SearXNG or query process: {str(e)}"



if __name__ == "__main__":

    test_url = "贵州茅台2023年年报"

    results = tool_duckduckgo_search(test_url)