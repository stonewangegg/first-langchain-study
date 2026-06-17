"""
"""

# get the logger
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from research_agents import create_researcher_agent
from analyze_agents import create_analyzer_agent


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

graph_one = builder.compile()