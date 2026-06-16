# Step 1: Install dependencies
# pip install deepagents langchain-openai langchain-core

import os

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from pydantic import SecretStr

from tavily import TavilyClient
from typing import Literal

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# System prompt to steer the agent to be an expert researcher
RESEARCH_INSTRUCTIONS = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

# Step 2: Define custom tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is 22°C and sunny."

@tool
def search_documents(query: str) -> str:
    """Search internal documents for relevant information."""
    return f"Search results for: {query}"

@tool
def internet_search(
        query: str,
        max_result: int =5,
        topci: Literal["general", "news", "finance"] = "general",
        incclude_raw_content: bool = False
):
    """Run a web search via tavily client."""
    return tavily_client.search(
        query,
        max_results = max_result,
        include_raw_content = incclude_raw_content,
        topic = topci
    )

# Step 3: Configure vLLM as the model backend
# vLLM provides an OpenAI-compatible API at localhost:8000
llm = ChatOpenAI(
    model="Qwen/Qwen3.6-35B-A3B-FP8",          # Model name (can be any vLLM-supported model)
    base_url = "http://192.168.8.50:8000/v1",  # vLLM server endpoint         
    api_key=SecretStr("EMPTY"),                # vLLM uses a placeholder token
    temperature=0.7,
    max_completion_tokens=8192
)

# Step 4: Create the deep agent
agent = create_deep_agent(
    model=llm,
    tools=[get_weather, search_documents],
    system_prompt="""You are a helpful research assistant with access to tools.
    - Plan tasks using write_todos
    - Use tools when needed
    - Manage context via file system
    - Spawn sub-agents for complex tasks"""
)

agent2 = create_deep_agent(
    model=llm,
    tools=[internet_search],
    system_prompt=RESEARCH_INSTRUCTIONS
)

# Step 5: Run the agent
# response = agent.invoke(
#     {"messages": [{"role": "user", "content": "Research the latest AI trends and write a summary"}]}
# )

# print(response["messages"][-1].content)

result = agent2.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})

# Print the agent's response
print(result["messages"][-1].content)
