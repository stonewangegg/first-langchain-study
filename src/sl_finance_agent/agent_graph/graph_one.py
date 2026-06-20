"""
Graph One: Two-Stage Research and Analysis Workflow.

This module defines a LangGraph :class:`StateGraph` that orchestrates a
sequential two-agent workflow for financial analysis::

    START -> Researcher -> Analyzer -> END

Workflow
--------
1. **Researcher** node invokes the researcher agent with the user's
   query and stores the final response in ``research_result``.
2. **Analyzer** node receives the research output and instructs the
   analyzer agent to locate the metadata JSON file, review the listed
   target files, and produce a markdown-formatted analysis report,
   which is stored in ``analysis_result``.

The shared :class:`CustomWorkflowState` flows through both nodes and
carries the user query, the model configuration, and the intermediate
results produced by each step.

The compiled graph is exposed as :data:`graph_one` and can be invoked
directly with an initial state payload::

    graph_one.invoke({
        "user_query": "...",
        "model_obj": {"model_name": "..."},
    })
"""

# get the logger
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..research_agents import create_researcher_agent
from ..analyze_agents import create_analyzer_agent


logger = logging.getLogger(__name__)

class CustomWorkflowState(TypedDict):
    """
    For the parent graph, define custom shared State
    """
    user_query: str
    model_obj: dict
    research_result: str
    analysis_result: str

def researcher_node(state: CustomWorkflowState):

    result = create_researcher_agent(state["model_obj"]["model_name"]).invoke({
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

    result = create_analyzer_agent(state["model_obj"]["model_name"]).invoke({
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

# initial the lang graph builder with the CustomWorkflowState
builder = StateGraph(CustomWorkflowState)

builder.add_node("Researcher", researcher_node)
builder.add_node("Analyzer", analyzer_node)

builder.add_edge(START, "Researcher")
builder.add_edge("Researcher", "Analyzer")
builder.add_edge("Analyzer", END)

graph_one = builder.compile()