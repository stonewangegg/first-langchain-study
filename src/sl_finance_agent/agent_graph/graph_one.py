"""
Graph One: Two-Stage Research and Analysis Workflow.

Defines a LangGraph :class:`StateGraph` that chains a Researcher agent
with an Analyzer agent to produce a financial analysis report from a
user query::

    START -> Researcher -> Analyzer -> END

The Researcher node runs the query through the researcher agent and
stores the reply in ``research_result``. The Analyzer node then reads
the metadata JSON plus its target files, using ``research_result`` as
context, and writes a markdown report into ``analysis_result``.

:class:`CustomWorkflowState` carries the user query, model config, and
intermediate results across nodes. The compiled graph is exposed as
:data:`graph_one` ::

    graph_one.invoke({
        "user_query": "...",
        "model_obj": {"model_name": "..."},
    })
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..research_agents import create_researcher_agent
from ..analyze_agents import create_analyzer_agent

from ..common_utils import ModelObj, get_logger


logger = get_logger(__name__)

# define the custom work flow state used in the Graph
class CustomWorkflowState(TypedDict):
    """
    For the parent graph, define custom shared State
    """
    user_query: str
    model_obj: ModelObj
    research_result: str
    analysis_result: str

def researcher_node(state: CustomWorkflowState):

    result = create_researcher_agent(state["model_obj"]).invoke({
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

    result = create_analyzer_agent(state["model_obj"]).invoke({
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