""" First Deep agents """

__version__ = "1.0.0"

from .demo_agent1 import agent_collaborator
from .agent_system_prompt import OFFICER_SYSTEM_PROMPT, RESEARCHER_SYSTEM_PROMPT, ANALYST_SYSTEM_PROMPT

__all__ = ["agent_collaborator", "OFFICER_SYSTEM_PROMPT", "RESEARCHER_SYSTEM_PROMPT", "ANALYST_SYSTEM_PROMPT"]