"""
Finance Assistant Tools for Open-WebUI
"""
import os
import sys

# add the sl_finance_agent directory to system path on runtime
# NOTE: change it when install this tool into open-webui, get the sl_finance_agent abs path and replace below
sys.path.append("/home/hzsto/study/langchain/first-start/src")

import logging

from pathlib import Path
from typing import Any
from jinja2 import Template

from sl_finance_agent import CustomWorkflowState, graph_one, ModelObj

class Tools:

    # the logging initialization flag
    __logging_initialized = False

    def __init__(self):

        # llm configure
        self.model_obj = ModelObj(
            llm_type = "vllm",
            model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B-FP8"),
            model_base_url =  os.environ.get("MODEL_BASE_URL", "http://192.168.8.50:8000/v1"),
            model_api_key = os.environ.get("MODEL_API_KEY", "local_empty")
        )

        # make the file working dir if needed
        self.file_dir = Path(os.getcwd()) / Path("./tmp")
        if not self.file_dir.exists():
            os.makedirs(self.file_dir, exist_ok=True)
            self.logger.info(f"Create file work space directory: {self.file_dir}")

        # check and initial logging if needed
        if not Tools.__logging_initialized:

            self.logger = logging.getLogger("tools_finance_assistant")

            if not self.logger.handlers:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler = logging.FileHandler("./tmp/tools_finance_assistant.log")
                file_handler.setFormatter(formatter)
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                
                self.logger.addHandler(file_handler)
                self.logger.addHandler(stream_handler)

                self.logger.setLevel(logging.INFO)

            Tools.__logging_initialized = True

    def run_agent_analyzewithDupont(self, user_prompt) -> (dict[str, Any] | Any):
        """
        Run the full Research -> Analyzer workflow on the user's prompt.

        This is the main entry point exposed to Open-WebUI. It builds an
        initial :class:`CustomWorkflowState` from the incoming ``user_prompt``
        and the configured LLM ``self.model_obj``, then invokes
        :data:`sl_finance_agent.graph_one` to execute the LangGraph workflow:

        Parameters
        ----------
        user_prompt : str
            The natural-language request from the end user describing
            the financial analysis to perform.

        Returns
        -------
        dict[str, Any] | Any
            The ``analysis_result`` field of the final graph state,
            typically a markdown-formatted report produced by the
            Analyzer node.
        """
        self.logger.info("🚀 Starting the **Main Graph** workflow for: '%s'\n", user_prompt)

        initial_state: CustomWorkflowState = {
            "user_query": user_prompt,
            "model_obj": self.model_obj,
            "research_result": "",
            "analysis_result": ""
        }

        # here invoke with the graph_one
        result = graph_one.invoke(initial_state)

        return result["analysis_result"]


if __name__ == "__main__":

    user_prompt = """
    ## 目标上市公司: "{{company_name}}"

    ## 搜索内容要求

    - 首先获取当前日期
    - 搜集周期为: "{{year_start_date}}" 至 "{{year_end_date}}"
    - 搜索网站为：巨潮网站(www.cninfo.com.cn)
    - 搜索并下载指定上市公司指定期间已经披露的各年年报的PDF文件
    - 搜索并下载指定上市公司本年度最新一个季度的季报的PDF文件

    ## 报告文件下载完成后，生成所有文件的 meta data 信息文件
    """

    # get the user input parameters value
    company_name, year_start_date, year_end_date = map(
        str, 
        input(
            "Enter your target company, year_start_date, year_end_date, separated by space: "
        ).split()
    )

    user_prompt_template = Template(user_prompt)

    user_prompt_final = user_prompt_template.render(company_name=company_name, year_start_date=year_start_date, year_end_date=year_end_date)

    tools = Tools()

    print("🚀 Starting the **Main Graph** workflow for: '%s'\n", user_prompt_final)

    result = tools.run_agent_analyzewithDupont(user_prompt_final)

    print("Final result: %s", result["analysis_result"])
