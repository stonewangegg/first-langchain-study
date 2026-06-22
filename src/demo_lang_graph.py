"""
The Entry point of agent graph
"""

import logging

from jinja2 import Template

from sl_finance_agent import CustomWorkflowState, graph_one, model_factory, SUPPORTED_LLM_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./tmp/demo_lang_graph_test.log"),
        logging.StreamHandler()
    ]
)

# get the logger
logger = logging.getLogger(__name__) 

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
    company_name, year_start_date, year_end_date, model_str = map(
        str, 
        input(
            f"Enter your target company, year_start_date, year_end_date, model[{SUPPORTED_LLM_TYPES}]: separated by space: "
        ).split()
    )

    user_prompt_template = Template(user_prompt)

    user_prompt_final = user_prompt_template.render(company_name=company_name, year_start_date=year_start_date, year_end_date=year_end_date)

    
    
    model_obj = model_factory(model_str)
    if model_obj is not None:

        logger.info("🚀 Starting the **Main Graph** workflow for: '%s'\n", user_prompt_final)
        
        initial_state: CustomWorkflowState = {
            "user_query": user_prompt_final,
            "model_obj": model_obj,
            "research_result": "",
            "analysis_result": ""
        }

        result = graph_one.invoke(initial_state)

        logger.info("Final result: %s", result["analysis_result"])