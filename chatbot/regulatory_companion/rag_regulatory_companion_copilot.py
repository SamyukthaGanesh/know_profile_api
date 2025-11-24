import json
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

from ai_governance_chatbot.routed_agent_gemini.config import OPENAI_API_BASE, OPENAI_API_KEY


def call_llm_gemini(user_content,
                    system_content,
                    max_completion_tokens=10000,
                    temperature=0,
                    client=None,
                    model_name='gemini-2.5-pro'):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that understands the regulations of a bank for taking various decisions. Answer the questions about it."},
                {"role": "user", "content": user_content}
            ],
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature
        }
        response = client.chat.completions.parse(**kwargs)
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error during LLM processing: {e}")
        return None


# Load regulations
with open('ai_governance_framework/core/compliance/regulations/regulations_db.json') as f:
    db = json.load(f)
policies = db['policies']

def condition_to_str(cond):
    """Convert the condition dict to a readable string."""
    if not cond:
        return ""
    if 'logical_operator' in cond and 'sub_conditions' in cond:
        subs = [
            f"{sc.get('feature', '')} {sc.get('operator', '')} {sc.get('value', '')}"
            for sc in cond['sub_conditions']
        ]
        return f"Condition: {cond['logical_operator']} of [{'; '.join(subs)}]"
    elif 'feature' in cond:
        return f"Condition: {cond.get('feature', '')} {cond.get('operator', '')} {cond.get('value', '')}"
    return str(cond)

# Prepare texts for embedding (include all relevant fields)
texts = [
    (
        f"Name: {p['name']}. "
        f"Description: {p['description']}. "
        f"Tags: {', '.join(p.get('tags', []))}. "
        f"Type: {p.get('policy_type', '')}. "
        f"Source: {p.get('regulation_source', '')}. "
        f"Action: {p.get('action', '')}. "
        f"Rationale: {p.get('rationale', '')}. "
        f"{condition_to_str(p.get('condition', {}))}"
    )
    for p in policies
]

# Compute embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_policy(query, top_k=3):
    """Retrieve top_k most relevant policies for a query string."""
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = []
    for idx in I[0]:
        results.append(policies[idx])
    return results

def generate_answer(query):
    results = retrieve_policy(query, top_k=3)
    if not results:
        return "Sorry, I couldn't find a relevant regulation."
    p = results[0]
    return (
        f"**{p['name']}** ({p['regulation_source']}):\n"
        f"{p['description']}\n\n"
        f"Condition: {json.dumps(p.get('condition', {}), indent=2)}\n"
        f"Action: {p['action']}\n"
        f"Rationale: {p['rationale']}\n"
        f"References: {', '.join(p.get('references', [])) if p.get('references') else 'N/A'}"
    )

if __name__ == "__main__":
    user_query = "What are the age related policies"
    print(call_llm_gemini(
        user_content=generate_answer(user_query),
        system_content="You are a regulatory compliance assistant. Based on the provided regulation details, answer the user's query clearly and concisely."
    ))