"""
Agent analyzer testing
"""

import logging

from third_agents_demo import agent_analyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("demo_agent_analyze_test.log"),
        logging.StreamHandler()
    ]
)

# get the logger
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    user_prompt_test = """
    Find and read the meta data json file, then review all target files with the meta data in the json file. Analyze and generate the report, write down in markdown format.  
    """
    logger.info("🚀 Starting the Main Agent workflow for: '%s'...\n", user_prompt_test)

    # Invoke the agent
    final_response = agent_analyzer.invoke({"messages": [{"role": "user", "content": user_prompt_test}]})

    # Print the agent's response
    logger.info("\n**Final response**: \n" + final_response["messages"][-1].content)
