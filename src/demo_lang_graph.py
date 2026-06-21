"""
The Entry point of agent graph
"""

import logging

from jinja2 import Template

from sl_finance_agent import CustomWorkflowState, graph_one, ModelObj, SUPPORTED_LLM_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("demo_lang_graph_test.log"),
        logging.StreamHandler()
    ]
)

# get the logger
logger = logging.getLogger(__name__)

# llm info
LOCAL_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
LOCAL_BASEURL="http://192.168.8.50:8000/v1"
ONLINE_MODEL="minimax-m3:cloud"
ONLINE_BASEURL="http://172.30.0.1:11434"

def model_factory(llm_type: str) -> ModelObj | None:
        
        if llm_type not in SUPPORTED_LLM_TYPES:
            logger.error("Unsupported LLM type: %s", llm_type)
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        if llm_type ==SUPPORTED_LLM_TYPES[0]:
            return ModelObj(llm_type = llm_type, model_name=ONLINE_MODEL, model_base_url=ONLINE_BASEURL, model_api_key = "")
        elif llm_type ==SUPPORTED_LLM_TYPES[1]:
            return ModelObj(llm_type = llm_type, model_name=LOCAL_MODEL, model_base_url=LOCAL_BASEURL, model_api_key = "empty")
        
        return None
        

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
            "Enter your target company, year_start_date, year_end_date, separated by space: "
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