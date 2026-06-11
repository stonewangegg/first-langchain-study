"""
Test using WebBaseLoader to load web pages and verify if target content matches requirements.
"""
import os

MY_UA = "Mozilla/5.0 (Macintosh, Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
os.environ['USER_AGENT'] = MY_UA

import re, httpx, pdfplumber, io
from langchain_community.document_loaders import WebBaseLoader
from bs4.filter import SoupStrainer
from langchain.tools import tool

@tool
def tool_fetch_url_info_with_loader(url: str, target_year: int = 2025, target_type: str = "年报年度报告") -> dict:
    """
    PDF direct link: Check compliance via httpx.
    HTML pages: Use WebBaseLoader combined with BeautifulSoup pre-filtering to accurately extract key web information. Core advantage: Only parses HTML fragments containing target keywords, never loads full web page content.
    Finally, output the verification result data object.
    """
    result = {
        "url": url,
        "is_valid": False,
        "file_type": "unknown",
        "detected_year": None,
        "detected_type": None,
        "is_official_source": False,
        "download_link": url,
        "verification_status": "失败",
        "details": ""
    }

    # Set User-Agent to mimic a browser to prevent interception by some websites
    headers = {"User-Agent": MY_UA}

    try:

        # 0. Use httpx to check the URL
        response = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "").lower()

        # 1. Pre-check: If it is a PDF direct link, perform lightweight metadata check directly (WebBaseLoader cannot handle PDF binaries)
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            result["file_type"] = "PDF"

            # 2. Verify legal disclosure source (Check if PDF contains watermarks or footers from CNINFO, etc.)
            official_domains = ["cninfo.com.cn", "sse.com.cn", "szse.cn", "hkexnews.hk"]
            if any(domain in url for domain in official_domains):
                result["is_official_source"] = True
            else:
                result["is_official_source"] = False
                result["details"] = "非官方信披源"
                return result
            
            if str(target_year) in url and target_type in url:
                # Simple URL check
                result["detected_year"] = target_year
                result["detected_type"] = target_type
            else:
                # Use pdfplumber to read the content of the first 3 pages of the PDF (key info is usually on the cover or table of contents)
                with pdfplumber.open(io.BytesIO(response.content)) as pdf:

                    pdf_text = ""
                    for page_num in range(min(3, len(pdf.pages))):
                        pdf_text += pdf.pages[page_num].extract_text()
                    
                    # 3. Extract year (Match patterns like 2024年, 2024 Annual Report)
                    year_match = re.search(r"(20\d{2})\s*年", pdf_text)
                    if year_match:
                        result["detected_year"] = int(year_match.group(1))
                    
                    # 4. Extract report type (Match Annual Report, Annual Report, ESG Report, Audit Report)
                    type_match = re.search(r"(年度报[告表]|ESG报[告表]|社会责任报[告表]|审计报告)", pdf_text)
                    if type_match:
                        result["detected_type"] = type_match.group(1)
                    
                # 5. Final check to see if target year and report type are met
                if (result["detected_year"] and result["detected_type"]):
                    target_year_ok = result["detected_year"] == target_year
                    target_type_ok = result["detected_type"] in target_type
                    if (target_year_ok and target_type_ok):
                        result["verification_status"] = "获取到目标年份和报告类型的PDF直链，验证通过"
                        result["is_valid"] = True
                    else:
                        result["verification_status"] = "验证失败"
                        result["details"] = "不符合年度与报告类型要求"
                        result["is_valid"] = False
        # 6. HTML files: Use WebBaseLoader
        else:
            # 7. Core: Configure SoupStrainer for "targeted" content extraction
            # We only let the loader parse tags that might contain years (e.g., 2024, 2025) and report types (e.g., Annual Report, Announcement)
            # This directly skips over 90% of irrelevant HTML structures in the web page
            target_tags = SoupStrainer(text=re.compile(r"(202[3-5]|2026|年报|年度报告|公告|摘要)"))

            loader = WebBaseLoader(
                web_paths=url,
                # Pass bs_kwargs to implement pre-filtering, greatly reducing memory usage and parsing time
                bs_kwargs={"parse_only": target_tags},
                # Keep UA masking to prevent basic anti-scraping interception
                header_template=headers,
                requests_per_second=1, # Limit requests per second to avoid triggering anti-scraping measures
                raise_for_status=True
            )
                    
            # 8. Load document content after "targeted filtering"
            docs = loader.load()
            if not docs:
                result["details"] = "网页加载失败或未找到与目标报告相关的关键信息"
                return result
                
            # At this point, page_content only contains the small amount of core text we care about, very clean
            extracted_text = docs[0].page_content
            result["file_type"] = "HTML"
            result["is_valid"] = True

            # 9. Perform precise regex matching on the very short text
            year_match = re.search(r"(20\d{2})\s*年", extracted_text)
            if year_match:
                result["detected_year"] = int(year_match.group(1))
            else:
                result["details"] = "year match failed"
                return result
                
            type_match = re.search(r"(年度报[告表]|ESG报[告表]|社会责任报[告表]|审计报告)", extracted_text)
            if type_match:
                result["detected_type"] = type_match.group(1)
            else:
                result["details"] = "type match failed"
                return result

            # 10. Verify official disclosure source
            official_domains = ["cninfo.com.cn", "sse.com.cn", "szse.cn", "hkexnews.hk"]
            if any(domain in url for domain in official_domains):
                result["is_official_source"] = True
            else:
                result["is_official_source"] = False
                result["details"] = "非官方信披源"
                return result
                
            # 11. Comprehensive verification logic
            year_ok = result["detected_year"] == target_year
            type_ok = result["detected_type"] in target_type
            
            if result["is_valid"] and year_ok and type_ok:
                result["verification_status"] = "完美匹配，验证通过"
                result["details"] = f"已确认为 {result['detected_year']} 年 {result['detected_type']}。"
            elif result["is_valid"]:
                result["verification_status"] = "信息不符"
                result["details"] = f"年份或类型不匹配 (检测到: {result['detected_year']} {result['detected_type']})"

    except Exception as e:
        result["details"] = f"抓取或解析失败: {str(e)}"

    print(f"The searched link: {url}: \n Verification result: {result} \n\n")
    return result


# 简单测试
if __name__ == "__main__":
    # 假设这是一个包含大量无关信息的财经新闻网页
    test_url = 'https://www.cninfo.com.cn/new/disclosure?stockCode=600519&orgId=9900024238'

    # info = tool_fetch_url_info_with_loader(test_url, target_year=2024, target_type="年报")
