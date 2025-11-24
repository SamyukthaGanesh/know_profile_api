import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("/Users/ssunehra/ghci/ai_governance_chatbot/configs/.env")

OPENAI_API_BASE = os.getenv("BASE_URL", "")
OPENAI_API_KEY = os.getenv("API_KEY", "")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")