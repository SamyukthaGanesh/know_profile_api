from ollama import chat, ChatResponse
from typing import Iterator, Optional

def call_llm(
        user_content: str,
        sys_content: Optional[str] = None,
        model: str = 'deepseek-r1:14b',
        is_stream: bool = False,
        n_ctx: int = 10000,
        temperature: float = 0.0,
) -> str:
    """Call the LLM model.

    Args:
        user_content: User input content
        sys_content: System context content
        model: LLM model name
        is_stream: Whether to stream the response
        n_ctx: Context window size
        temperature: Temperature parameter
        relevant_rules: Relevant rules to include in context

    Returns:
        Generated text response
    """
    response: ChatResponse = chat(model='deepseek-r1:14b', messages=[
        {
            'role': 'user',
            'content': user_content
        }
    ], stream=is_stream, options={'num_ctx': n_ctx, 'temperature': temperature})

    return response


def call_llm_stream(
        user_content: str,
        sys_content: Optional[str] = None,
        model: str = 'deepseek-r1:70b',
        n_ctx: int = 10000,
        temperature: float = 0.0,
        relevant_rules: Optional[str] = None
) -> Iterator[str]:
    """Stream responses from the LLM model.

    Args:
        user_content: User input content
        sys_content: System context content
        model: LLM model name
        n_ctx: Context window size
        temperature: Temperature parameter
        relevant_rules: Relevant rules to include in context

    Returns:
        Iterator yielding generated text chunks
    """
    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'user',
            'content': user_content
        }
    ], stream=True, options={'num_ctx': n_ctx, 'temperature': temperature})

    for chunk in response:
        yield chunk['message']['content']


if __name__ == "__main__":
    transactions = {

        "transactions":[
            {
            "id": "T001", "customer_id": "CUST001",
            "ts": "2025-07-10T13:00:00Z", "amount": 150.00, "category": "Groceries", "merchant": "BigBazaar"
            },
            {
            "id": "T002", "customer_id": "CUST001",
            "ts": "2025-08-12T14:00:00Z", "amount": 80.00, "category": "Fuel", "merchant": "HP Petrol"
            },
            {
            "id": "T003", "customer_id": "CUST001",
            "ts": "2025-09-05T09:10:00Z", "amount": 1200.00, "category": "Electronics", "merchant": "Flipkart"
            },
            {
            "id": "T004", "customer_id": "CUST001",
            "ts": "2025-10-01T10:00:00Z", "amount": 500.00, "category": "Loan Interest", "merchant": "Bank"
            }
        ],
        "coupons": [
            {
            "id": "CPN001", "customer_id": "CUST001",
            "description": "10 percent off on groceries above $100", "valid_till": "2025-12-31"
            },
            {
            "id": "CPN002", "customer_id": "CUST001",
            "description": "5 percent cashback on fuel purchases", "valid_till": "2025-11-30"
            }
        ]
    }

    print(call_llm(f"Give an analysis of my overall spending and how I can improve it: {transactions}"))