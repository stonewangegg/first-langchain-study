"""
Graph One: Two-Stage Research and Analysis Workflow.

LangGraph :class:`StateGraph` chaining Researcher and Analyzer agents
to produce a financial analysis report from a user query::

    START -> Researcher -> Analyzer -> Cleaner -> END

- Researcher: runs ``user_query`` via the researcher agent; stores the
  reply in ``research_result``.
- Analyzer: reads the metadata JSON and its target files, using
  ``research_result`` as context; writes a markdown report into
  ``analysis_result``.
- Cleaner: removes non-essential files from ``FILE_DIR`` (keeps
  ``sl_finance_agent.log`` and ``skills``).

:class:`CustomWorkflowState` carries ``user_query``, ``model_obj``, and
intermediate results across nodes. The compiled graph is exposed as
:data:`graph_one``.
"""

from pathlib import Path
import shutil
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..research_agents import create_researcher_agent
from ..analyze_agents import create_analyzer_agent
from ..common_utils import ModelObj, uru_logger, FILE_DIR

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
    Find and read the meta data json file, then review all target files with the meta data in the json file. Analyze and generate the report base on requirement. 
    Reference the Research Result below:
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

def clean_up_node(state: CustomWorkflowState):

    # do the clean up
    KEEP = {
        "sl_finance_agent.log",
        "skills",
    }

    folder = Path(FILE_DIR)

    for item in folder.iterdir():
        if item.name in KEEP:
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    uru_logger.get_logger().info("Folder 'tmp' has been clean up !")
    return state

# initial the lang graph builder with the CustomWorkflowState
builder = StateGraph(CustomWorkflowState)

builder.add_node("Researcher", researcher_node)
builder.add_node("Analyzer", analyzer_node)
builder.add_node("Cleaner", clean_up_node)

builder.add_edge(START, "Researcher")
builder.add_edge("Researcher", "Analyzer")
builder.add_edge("Analyzer", "Cleaner")
builder.add_edge("Cleaner", END)

graph_one = builder.compile()