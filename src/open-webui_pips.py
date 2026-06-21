"""
"""

import logging
import os
from pathlib import Path

from pydantic import BaseModel
from sl_finance_agent import CustomWorkflowState, graph_one, ModelObj


class Pipe:

    # the logging initialization flag
    __logging_initialized = False

    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

        self.file_dir = Path(os.getcwd()) / Path("./tmp")

        # llm configure
        self.model_obj = ModelObj(
            llm_type = "vllm",
            model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B-FP8"),
            model_base_url =  os.environ.get("MODEL_BASE_URL", "http://192.168.8.50:8000/v1"),
            model_api_key = os.environ.get("MODEL_API_KEY", "local_empty")
        )


        # check and initial logging if needed
        if not Pipe.__logging_initialized:

            self.logger = logging.getLogger("tools_finance_assistant")

            if not self.logger.handlers:
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler = logging.FileHandler("./tmp/tools_finance_assistant.log")
                file_handler.setFormatter(formatter)
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                
                self.logger.addHandler(file_handler)
                self.logger.addHandler(stream_handler)

                self.logger.setLevel(logging.INFO)

            Pipe.__logging_initialized = True

        if not self.file_dir.exists():
            os.makedirs(self.file_dir, exist_ok=True)
            self.logger.info(f"Create file work space directory: {self.file_dir}")

    async def pipe(
        self,
        body: dict,
        __user__: dict = {},
        __event_emitter__=None,
        __event_call__=None,
    ):

        messages = body.get("messages", [])

        if not messages:
            yield "No messages, please check and try again."
            return

        user_query = messages[-1]["content"]

        self.logger.info("🚀 Starting the **Main Graph** workflow for: '%s'\n", user_query)

        initial_state: CustomWorkflowState = {
            "user_query": user_query,
            "model_obj": self.model_obj,
            "research_result": "",
            "analysis_result": ""
        }

        # here invoke with the graph_one
        # result = graph_one.invoke(initial_state)

        async for event in graph_one.astream_events(initial_state, version="v2"):

            if event["event"] == "on_tool_start":

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Running {event['name']}",
                                "done": False
                            }
                        }
                    )

            elif event["event"] == "on_chat_model_stream":

                chunk = event["data"].get("chunk")
                content = getattr(chunk, "content", None) if chunk else None

                if content:
                    yield content


