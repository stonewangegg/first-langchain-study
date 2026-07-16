"""
Finance Assistant Tools for Open-WebUI
"""
import os
import sys

# add the sl_finance_agent directory to system path on runtime
# NOTE: change it when install this tool into open-webui, get the sl_finance_agent abs path and replace below
sys.path.append("/home/hzsto/study/langchain/first-start/src")

# set the file working dir before all initialize
os.environ["FILE_DIR"] = "/home/hzsto/study/langchain/first-start/src/tmp"

from pathlib import Path
from typing import Any
from jinja2 import Template

from sl_finance_agent import CustomWorkflowState, graph_one, ModelObj, get_logger, FILE_DIR

class Tools:

    def __init__(self):

        self.logger = get_logger(__name__)

        self.download_url = "http://192.168.8.50:8082/"

        # llm configure
        self.model_obj = ModelObj(
            llm_type = "vllm",
            model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B-FP8"),
            model_base_url =  os.environ.get("MODEL_BASE_URL", "http://192.168.8.50:8000/v1"),
            model_api_key = os.environ.get("MODEL_API_KEY", "empty")
        )

        # make the file working dir if needed
        self.file_dir = Path(FILE_DIR)
        if not self.file_dir.exists():
            os.makedirs(self.file_dir, exist_ok=True)
            self.logger.info(f"Create file work space directory: {self.file_dir}")

    async def run_graph_agent_finance_analyze(
            self, 
            user_prompt,
            __event_emitter__=None
        ) -> (dict[str, Any] | Any):
        """
        Execute the Research -> Analyzer LangGraph workflow on the user's prompt.

        This is the main entry point exposed to Open-WebUI. It builds an
        initial :class:`CustomWorkflowState` from the incoming ``user_prompt``
        and the configured LLM ``self.model_obj``, then drives
        :data:`sl_finance_agent.graph_one` via ``astream_events`` (protocol
        ``v2``) so that intermediate progress can be observed and forwarded
        back to the host UI through ``__event_emitter__``.

        Once streaming completes, return ``final_state["analysis_result"]``
           (a markdown-formatted report produced by the Analyzer node) and
           emit a ``"✅ Analysis completed"`` status event.

        Parameters
        ----------
        user_prompt : str
            The natural-language request from the end user describing
            the financial analysis to perform.
        __event_emitter__ : Awaitable, optional
            Async callback supplied by Open-WebUI used to push status
            updates back to the chat UI. When ``None`` — for example when
            running this tool directly from a plain Python script —
            status events are skipped but the workflow still runs to
            completion and its result is still returned.

        Returns
        -------
        dict[str, Any] | Any
            The value of the ``analysis_result`` field in the final graph
            state — typically a markdown-formatted report produced by the
            Analyzer node of the workflow.

        Raises
        ------
        RuntimeError
            If the LangGraph workflow finishes without producing a state
            payload (i.e. ``final_state`` is still ``None`` after streaming),
            the user is asked to check the graph configuration and try again.
        Exception
            Any exception raised while streaming events is logged, surfaced
            to the UI as a ``"❌ Graph Workflow execution Failed: ..."``
            status event, and then re-raised to the caller.
        """

        initial_state: CustomWorkflowState = {
            "user_query": user_prompt,
            "model_obj": self.model_obj,
            "research_result": "",
            "analysis_result": ""
        }

        self.logger.info("🚀 Starting the **Main Graph** workflow for: '%s'\n\nTo the model: '%s'", user_prompt, self.model_obj)

        # here invoke with the graph_one
        # result = graph_one.invoke(initial_state)
        # return result["analysis_result"]

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "🚀 Starting agent graph one financial analysis...",
                        "done": False,
                    },
                }
            )

        final_state = None

        try:
            # invoke the agent graph one via astream_events 
            async for event in graph_one.astream_events(
                initial_state,
                version="v2",
            ):

                event_type = event.get("event", "")

                # event tool start 
                if event_type == "on_tool_start":

                    tool_name = event.get("name", "unknown_tool")

                    self.logger.info("Tool started: %s",tool_name)

                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"🔧 Running: {tool_name}",
                                    "done": False,
                                },
                            }
                        )

                # Tool finished
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown_tool")
                    self.logger.info( "Tool completed: %s", tool_name)

                # Node started
                elif event_type == "on_chain_start":
                    node_name = event.get("name", "")
                    if node_name:
                        self.logger.info("Node started: %s",node_name)

                # Graph completed
                elif event_type == "on_chain_end":
                    # get the agent final result data
                    data = event.get("data", {})
                    output = data.get("output")
                    if isinstance(output, dict):
                        final_state = output

        except Exception as ex:

            self.logger.exception("Workflow execution failed: %s", str(ex))

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"❌ Graph Workflow execution Failed: {str(ex)}",
                            "done": True,
                        },
                    }
                )

            raise

        if final_state is None:
            raise RuntimeError("❌ Lang Graph one completed without returning a state. Check and try again.")

        final_answer = final_state.get("analysis_result","")

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "✅ Analysis completed",
                        "done": True,
                    },
                }
            )

        return final_answer




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
    company_name, year_start_date, year_end_date = map(
        str, 
        input(
            "Enter your target company, year_start_date, year_end_date, separated by space: "
        ).split()
    )

    user_prompt_template = Template(user_prompt)

    user_prompt_final = user_prompt_template.render(company_name=company_name, year_start_date=year_start_date, year_end_date=year_end_date)

    tools = Tools()

    result = tools.run_graph_agent_finance_analyze(user_prompt_final)

    print("Final result: %s", result)
