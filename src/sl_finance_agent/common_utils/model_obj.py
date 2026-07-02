"""
Define the model object class
"""

import os

# define the support model types
from dataclasses import dataclass
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .logger_utils import get_logger
logger = get_logger(__name__)

# llm info
LOCAL_MODEL="Qwen/Qwen3.6-35B-A3B-FP8"
LOCAL_BASEURL="http://192.168.8.50:8000/v1"
LOCAL_API_KEY="empty"
ONLINE_MODEL="minimax-m3:cloud"
ONLINE_BASEURL="http://172.30.0.1:11434"
ONLINE_API_KEY=os.environ.get("ONLINE_API_KEY", "")

SUPPORTED_LLM_TYPES = ("ollama", "vllm")

# model default parameters
MAX_COMPLETION_TOKENS = os.environ.get("MAX_COMPLETION_TOKENS", "16384")

# current working directory
CURRENT_WORKING_DIR = os.getcwd()
FILE_DIR = os.environ.get("FILE_DIR", "./tmp")
# export FILE_ROOT_DIR="/your/file/root/dir"
FILE_ROOT_DIR = str(Path(CURRENT_WORKING_DIR) / Path(FILE_DIR))

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

# Supported LLM type identifiers for ``create_analyzer_agent``.
def resolve_llm(model_obj: ModelObj) -> BaseChatModel:
    """Resolve a ``ModelObj`` into a configured chat model.

    Maps ``model_obj.llm_type`` to a concrete chat model instance:

    * ``"ollama"`` -> :class:`ChatOllama` against ``model_base_url``.
    * ``"vllm"``   -> :class:`ChatOpenAI` against the vLLM
      OpenAI-compatible endpoint at ``model_base_url`` (with parallel
      tool calls disabled).

    The context window size for both providers is driven by the
    ``MAX_COMPLETION_TOKENS`` environment variable.

    Parameters
    ----------
    model_obj : ModelObj
        Dataclass specifying the provider (``llm_type``), ``model_name``,
        ``model_base_url`` and optional ``model_api_key``. See
        ``create_analyzer_agent`` for field-level documentation.

    Returns
    -------
    BaseChatModel
        The configured chat model (``ChatOllama`` or ``ChatOpenAI``)
        ready to plug into the deep agent.

    Raises
    ------
    ValueError
        If ``model_obj.llm_type`` is not one of ``SUPPORTED_LLM_TYPES``
        (i.e. not ``"ollama"`` or ``"vllm"``).
    """
    if model_obj.llm_type == "ollama":
        # inital the model object of Ollama provider
        model_ollama = ChatOllama(
            model=model_obj.model_name,
            # validate_model_on_init=True,
            # num_thread=16,
            cache=True,
            verbose=True,                       # Print additional LangChain logs.Useful for debugging: prompts, tool calls, intermediate chains
            reasoning=False,
            temperature=0.5,
            base_url=model_obj.model_base_url,
            repeat_penalty=1.05,
            num_ctx=int(MAX_COMPLETION_TOKENS),
            disable_streaming="tool_calling"
        )
        return model_ollama
    if model_obj.llm_type == "vllm":
        # initial the model object that vLLM provides an OpenAI-compatible API at localhost:8000
        model_vllm = ChatOpenAI(
            model=model_obj.model_name,                 # Model name (can be any vLLM-supported model)
            base_url=model_obj.model_base_url,          # vLLM server endpoint         
            api_key=SecretStr(model_obj.model_api_key), # vLLM uses a placeholder token
            temperature=0.4,
            top_p=0.9,
            max_completion_tokens=int(MAX_COMPLETION_TOKENS)
        )

        # disable the llm parallel tool calls
        model_vllm.bind(
            parallel_tool_calls=False
        )
        return model_vllm
    raise ValueError(
        f"Unsupported llm: {model_obj}. "
        f"Expected one of: {', '.join(SUPPORTED_LLM_TYPES)}"
    )