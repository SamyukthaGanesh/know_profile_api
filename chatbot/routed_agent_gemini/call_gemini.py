
from openai import OpenAI
from ai_governance_chatbot.routed_agent_gemini.config import OPENAI_API_BASE, OPENAI_API_KEY

def call_llm_gemini(messages,
                    tools,
                    max_completion_tokens=10000,
                    temperature=0,
                    model_name='gemini-2.5-pro'):
       try:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
            kwargs = {
                "model": model_name,
                "messages": messages,
                "tools": tools,
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "tool_choice": "auto",
            }
            
            response = client.chat.completions.create(**kwargs)
            print(response)
            return response.choices[0].message
       
       except Exception as e:
            print(f"Error during LLM processing: {e}")
            # Create a mock message object instead of returning None
            from types import SimpleNamespace
            return SimpleNamespace(
                content="Sorry, an error occurred while processing your request: " + str(e),
                tool_calls=[],  # Empty list instead of None
            )

if __name__ == "__main__":
    print(call_llm_gemini("Hello, world!"))