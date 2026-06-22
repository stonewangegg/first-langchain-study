"""
Define the model object class
"""

# define the support model types
from dataclasses import dataclass

from logger_utils import get_logger
logger = get_logger(__name__)

# llm info
LOCAL_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
LOCAL_BASEURL="http://192.168.8.50:8000/v1"
ONLINE_MODEL="minimax-m3:cloud"
ONLINE_BASEURL="http://172.30.0.1:11434"

SUPPORTED_LLM_TYPES = ("ollama", "vllm")

@dataclass
class ModelObj:
    llm_type: str
    model_name: str
    model_base_url: str
    model_api_key: str = "Empty"

def model_factory(llm_type: str) -> ModelObj | None:
        
        if llm_type not in SUPPORTED_LLM_TYPES:
            logger.error("Unsupported LLM type: %s", llm_type)
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        if llm_type ==SUPPORTED_LLM_TYPES[0]:
            return ModelObj(llm_type = llm_type, model_name=ONLINE_MODEL, model_base_url=ONLINE_BASEURL, model_api_key = "")
        elif llm_type ==SUPPORTED_LLM_TYPES[1]:
            return ModelObj(llm_type = llm_type, model_name=LOCAL_MODEL, model_base_url=LOCAL_BASEURL, model_api_key = "empty")
        
        return None