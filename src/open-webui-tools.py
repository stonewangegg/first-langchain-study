"""
Finance Assistant Tools
"""

import logging
import os
from pathlib import Path


class Tools:

    # the logging initialization flag
    _logging_initialized = False


    def __init__(self):

        self.file_dir = Path(os.getcwd()) / Path("./tmp")

        # llm configure
        self.model_name = ""
        self.model_baseurl = ""
        self.model_api_key = ""


        # check and initial logging if needed
        if not Tools._logging_initialized:

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

            Tools._logging_initialized = True

        if not self.file_dir.exists():
            os.makedirs(self.file_dir, exist_ok=True)
            self.logger.info(f"Create file work space directory: {self.file_dir}")


