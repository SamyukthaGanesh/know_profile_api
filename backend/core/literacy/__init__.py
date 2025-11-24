"""
AI Literacy Module
Generates prompts for explaining AI decisions to different audiences.
"""

from .prompt_generator import (
    PromptGenerator,
    ExplanationContext,
    AudienceType,
    ExplanationType,
    create_prompt_generator
)
from .user_prompts import (
    UserPromptGenerator,
    UserProfile,
    create_user_prompt_generator
)
from .banker_prompts import (
    BankerPromptGenerator,
    BankerProfile,
    create_banker_prompt_generator
)

__all__ = [
    # Prompt generator base
    'PromptGenerator',
    'ExplanationContext',
    'AudienceType',
    'ExplanationType',
    'create_prompt_generator',
    
    # User prompts
    'UserPromptGenerator',
    'UserProfile',
    'create_user_prompt_generator',
    
    # Banker prompts
    'BankerPromptGenerator',
    'BankerProfile',
    'create_banker_prompt_generator',
]