"""
Agent analyzer testing
"""

from sl_finance_agent import create_analyzer_agent, SUPPORTED_LLM_TYPES, get_logger, model_factory

# llm info
LOCAL_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
LOCAL_BASEURL="http://192.168.8.50:8000/v1"
ONLINE_MODEL="minimax-m3:cloud"
ONLINE_BASEURL="http://172.30.0.1:11434"


# get the logger
logger = get_logger(__name__)

if __name__ == "__main__":

    model_str = input(f"Please enter the model name [supported: {SUPPORTED_LLM_TYPES}] used in this query ....: ")

    user_prompt_test = """
    Find and read the meta data json file, then review all target files with the meta data in the json file. Analyze and generate the report, write down in markdown format. 
    """
    logger.info("🚀 Starting the Main Agent workflow for: '%s'...\n", user_prompt_test)

    model_obj = model_factory(model_str)

    # Invoke the agent
    if model_obj:
        final_response = create_analyzer_agent(model_obj).invoke({"messages": [{"role": "user", "content": user_prompt_test}]})

        # Print the agent's response
        logger.info("\n**Final response**: \n" + final_response["messages"][-1].content)
