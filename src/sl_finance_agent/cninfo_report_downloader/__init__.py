"""
cninfo-report-downloader
从巨潮网站下载指定公司、指定年份的年报和季度报告
"""

__version__ = "1.0.0"

from .cninfo_report_download import CNInfoReportDownloader

__all__ = ["CNInfoReportDownloader"]
