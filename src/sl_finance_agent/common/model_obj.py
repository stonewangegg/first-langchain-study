"""
Define the model object class
"""

# define the support model types
from dataclasses import dataclass


SUPPORTED_LLM_TYPES = ("ollama", "vllm")

@dataclass
class ModelObj:
    llm_type: str
    model_name: str
    model_base_url: str
    model_api_key: str = "Empty"