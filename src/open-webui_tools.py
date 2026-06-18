"""
Finance Assistant Tools for Open-WebUI
"""

import logging
import os
from pathlib import Path
from typing import Any

from sl_finance_agent import CustomWorkflowState, graph_one

class Tools:

    # the logging initialization flag
    __logging_initialized = False

    def __init__(self):

        self.file_dir = Path(os.getcwd()) / Path("./tmp")

        # llm configure
        self.model_obj = {
            "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B-FP8"),
            "model_base_url": os.environ.get("MODEL_BASE_URL", "http://192.168.8.50:8000/v1"),
            "model_api_key": os.environ.get("MODEL_API_KEY", "local_empty")
        }


        # check and initial logging if needed
        if not Tools.__logging_initialized:

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

            Tools.__logging_initialized = True

        if not self.file_dir.exists():
            os.makedirs(self.file_dir, exist_ok=True)
            self.logger.info(f"Create file work space directory: {self.file_dir}")

    def analyzewithDupont(self, user_prompt) -> (dict[str, Any] | Any):
        """
        Run the full Research -> Analyzer workflow on the user's prompt.

        This is the main entry point exposed to Open-WebUI. It builds an
        initial :class:`CustomWorkflowState` from the incoming ``user_prompt``
        and the configured LLM ``self.model_obj``, then invokes
        :data:`sl_finance_agent.graph_one` to execute the LangGraph workflow:

        Parameters
        ----------
        user_prompt : str
            The natural-language request from the end user describing
            the financial analysis to perform.

        Returns
        -------
        dict[str, Any] | Any
            The ``analysis_result`` field of the final graph state,
            typically a markdown-formatted report produced by the
            Analyzer node.
        """

        self.logger.info("🚀 Starting the **Main Graph** workflow for: '%s'\n", user_prompt)

        initial_state: CustomWorkflowState = {
            "user_query": user_prompt,
            "model_obj": self.model_obj,
            "research_result": "",
            "analysis_result": ""
        }

        # here invoke with the graph_one
        result = graph_one.invoke(initial_state)

        return result["analysis_result"]


