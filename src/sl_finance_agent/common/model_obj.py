"""
Define the model object class
"""

# define the support model types
from dataclasses import dataclass

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