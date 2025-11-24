from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ai_governance_chatbot.routed_agent_gemini.bot import chat_with_tools
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from ai_governance_chatbot.routed_agent_gemini.fetch_tools import get_db_connection
from ai_governance_chatbot.routed_agent_gemini.generate_report import fetch_decision_prompt_request
from ai_governance_chatbot.routed_agent_gemini.call_gemini_report import Report
from ai_governance_chatbot.regulatory_companion.rag_regulatory_companion_copilot import generate_answer, call_llm_gemini

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only; restrict in prod!
    allow_methods=["*"],
    allow_headers=["*"],
)

class RegulationQuery(BaseModel):
    query: str

class RegulationResponse(BaseModel):
    answer: str

@app.post("/regulation_chat", response_model=RegulationResponse)
def regulation_chat(req: RegulationQuery):
    """Answer regulatory questions using RAG and LLM."""
    context = generate_answer(req.query)
    answer = call_llm_gemini(
        user_content=context,
        system_content="You are a regulatory compliance assistant. Based on the provided regulation details, answer the user's query clearly and concisely."
    )
    return {"answer": answer}

# In-memory conversation history store: {user_id: [messages]}
conversation_histories: Dict[str, List[Dict[str, Any]]] = {}

class ChatRequest(BaseModel):
    user_id: str
    user_query: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Retrieve or initialize conversation history for this user
    messages = conversation_histories.get(req.user_id)
    if messages is None:
        messages = []
        # Add system prompt at the start of a new conversation
        messages.append({
            "role": "system",
            "content": (
                "You are a financial assistant. Use only tool data to answer. "
                "Show dates explicitly and be concise."
            )
        })
    # Call chat_with_tools with the user's history
    answer = chat_with_tools(
        user_id=req.user_id,
        user_query=req.user_query,
        messages=messages
    )
    # Save updated history
    conversation_histories[req.user_id] = messages
    return {"answer": answer}

@app.get("/decisions/{user_id}")
def list_decision_ids(user_id: str):
    """Fetch all decision IDs for a given user."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT decision_id FROM decision_logs WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        return {"decision_ids": [row["decision_id"] for row in rows]}
    finally:
        conn.close()

@app.get("/decision_report/{decision_id}")
def get_decision_report(decision_id: str):
    """Fetch the full JSON report for a given decision ID."""
    try: 
        report = fetch_decision_prompt_request(decision_id)
        if not report:
            raise HTTPException(status_code=404, detail="Decision not found")
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))