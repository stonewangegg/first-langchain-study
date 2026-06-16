"""
The testing suite for fisrt demo agents
"""

import logging

from langchain_core.runnables import RunnableConfig

from jinja2 import Template

from sl_finance_agent import agent_collaborator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("first_demo_agent.log"),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":

    # read agent system prompt from file
    with open('./first_agents_demo/user_prompt_test.md', 'r', encoding='utf-8') as file:
        user_prompt_test = file.read()

    # get the user input parameters value
    company_name, year_start_date, year_end_date = map(str, input("Enter your target company, year_start_date, year_end_date separated by space: ").split())

    user_prompt_template = Template(user_prompt_test)

    user_prompt_final = user_prompt_template.render(company_name=company_name, year_start_date=year_start_date, year_end_date=year_end_date)

    print(f"🚀 Starting the Main Agent workflow for: '{user_prompt_final}'...\n")

    # define the thread id for whole task's check point
    runnable_config = RunnableConfig({"configurable": {"thread_id": "deep-researcher-01"}})


    # Invoke the agent
    # final_response = agent_collaborator.invoke({"messages": [{"role": "user", "content": custom_internet_search_prompt}]}, config=runnable_config)
    final_response = agent_collaborator.invoke({"messages": [{"role": "user", "content": user_prompt_final}]})

    # Print the agent's response
    print("\n**Final response**: \n" + final_response["messages"][-1].content)