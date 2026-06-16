"""
Agent researcher testing
"""

import logging
from jinja2 import Template
from sl_finance_agent import create_researcher_agent, SUPPORTED_LLM_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("demo_agent_research_test.log"),
        logging.StreamHandler()
    ]
)

# get the logger
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    user_prompt_test = """
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
    company_name, year_start_date, year_end_date, model_str = map(str, input(f"Enter your target company, year_start_date, year_end_date, model[{SUPPORTED_LLM_TYPES}] separated by space: ").split())

    user_prompt_template = Template(user_prompt_test)

    user_prompt_final = user_prompt_template.render(company_name=company_name, year_start_date=year_start_date, year_end_date=year_end_date)

    logger.info("🚀 Starting the Main Agent workflow for: '%s'...\n", user_prompt_final)


    # Invoke the agent
    final_response = create_researcher_agent(model_str).invoke({"messages": [{"role": "user", "content": user_prompt_final}]})

    # Print the agent's response
    logger.info("\n**Final response**: \n" + final_response["messages"][-1].content)