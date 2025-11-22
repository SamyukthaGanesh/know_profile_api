from typing import Dict
from pydantic import BaseModel

class ConsentState(BaseModel):
    fraud: bool = True
    personalization: bool = True
    training: bool = True

CONSENT_REGISTRY: Dict[str, ConsentState] = {}

def get_consent(user_id: str) -> ConsentState:
    return CONSENT_REGISTRY.get(user_id, ConsentState())

def set_consent(user_id: str, new_state: ConsentState) -> ConsentState:
    CONSENT_REGISTRY[user_id] = new_state
    return new_state
