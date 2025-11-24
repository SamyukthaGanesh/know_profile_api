from ai_governance_chatbot.routed_agent_gemini.fetch_tools import tool_get_decision_log, tool_get_customer, tool_list_transactions
import json
from ai_governance_chatbot.routed_agent_gemini.call_gemini import call_llm_gemini

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Fetch customer profile by ID",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "List transactions for a customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                },
                "required": ["customer_id"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_decision_log",
            "description": "Get natural-language decision log by customer_id",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"}
                },
                "required": ["customer_id"]
            },
        },
    },
]

DISPATCH = {
    "get_customer": tool_get_customer,
    "list_transactions": tool_list_transactions,
    "get_decision_log": tool_get_decision_log,
}

def chat_with_tools(
        user_id: str,
        user_query: str,
        system_prompt: str | None = None,
        messages: list[dict] | None = None
    ) -> str:
    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": f"UserId: {user_id}\nQuery: {user_query}"
    })
    while True:
        msg = call_llm_gemini(
            messages=messages,
            tools=TOOLS
        )

        # Add a null check as a safety measure
        if msg is None:
            return "Sorry, there was an error processing your request. Please try again."

        # If the model asked to call tool(s)...
        if msg.tool_calls:
            # Append the assistant "tool call" message to history
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            print("Tool calls requested:", [tc.function.name for tc in msg.tool_calls])

            # For each tool call, execute locally and add a tool response message
            for tc in msg.tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                    print(f"Calling tool {name} with args: {args}")
                except json.JSONDecodeError:
                    args = {}

                print(name)
                # Route to your Python function
                func = DISPATCH.get(name)
                if func is None:
                    print(f"Unknown tool requested: {name}")
                    tool_result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        tool_result = func(**args)
                    except Exception as e:
                        print(f"Error calling tool {name}: {e}")
                        tool_result = {"error": str(e)}

                # Attach tool response, referencing tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps({"result": tool_result}, ensure_ascii=False)
                })

            # loop continues; model will see tool outputs and (usually) produce a final answer
            continue

        # No tool calls -> return final text
        return msg.content or "Sorry, I couldn't generate a response."


if __name__ == "__main__":
    print("Ask me something (type 'exit' to quit).")
    messages = []
    system_prompt = (
        "You are a financial assistant. Use only tool data to answer. "
        "Show dates explicitly and be concise."
    )
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        ans = chat_with_tools(
            user_id="U1000",
            user_query=q,
            system_prompt=system_prompt,
            messages=messages  # Pass the same messages list each time
        )
        print("\nAssistant:", ans, "\n")