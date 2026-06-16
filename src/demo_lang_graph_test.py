"""

"""

import logging
from typing import TypedDict

from jinja2 import Template
from langgraph.graph import END, START, StateGraph

from sl_finance_agent import create_researcher_agent, SUPPORTED_LLM_TYPES
from sl_finance_agent import create_analyzer_agent

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

class CustomWorkflowState(TypedDict):
    """
    For the parent graph, define custom shared State
    """
    user_query: str
    model_str: str
    research_result: str
    analysis_result: str

def researcher_node(state: CustomWorkflowState):

    result = create_researcher_agent(state["model_str"]).invoke({
        "messages": [
            {
                "role": "user",
                "content": state["user_query"]
            }
        ]
    })

    return {
        "research_result": result["messages"][-1].content
    }

def analyzer_node(state: CustomWorkflowState):
    
    prompt = f"""
    Find and read the meta data json file, then review all target files with the meta data in the json file. Analyze and generate the report, write down in markdown format.
    Reference the Research Result:
    {state['research_result']}
    """

    result = create_analyzer_agent(state["model_str"]).invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    return {
        "analysis_result": result["messages"][-1].content
    }


builder = StateGraph(CustomWorkflowState)

builder.add_node("Researcher", researcher_node)
builder.add_node("Analyzer", analyzer_node)

builder.add_edge(START, "Researcher")
builder.add_edge("Researcher", "Analyzer")
builder.add_edge("Analyzer", END)

graph = builder.compile()



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
            f"Enter your target company, year_start_date, year_end_date, model[{SUPPORTED_LLM_TYPES}] separated by space: "
        ).split()
    )

    user_prompt_template = Template(user_prompt)

    user_prompt_final = user_prompt_template.render(company_name=company_name, year_start_date=year_start_date, year_end_date=year_end_date)

    logger.info("🚀 Starting the **Main Graph** workflow for: '%s'\n", user_prompt_final)

    initial_state: CustomWorkflowState = {
        "user_query": user_prompt_final,
        "model_str": model_str,
        "research_result": "",
        "analysis_result": ""
    }

    result = graph.invoke(initial_state)

    logger.info("Final result: %s", result["analysis_result"])