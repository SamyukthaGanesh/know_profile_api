"""
Regulations RAG System - Retrieval-Augmented Generation for AI Governance Regulations

This module provides a FAISS-based retrieval system for the regulations database,
enabling context-aware responses about compliance policies using Gemini.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import sys

# Add the parent directory to sys.path to import the Gemini module
sys.path.append('/Users/ssunehra/ghci')

# Import the Gemini API handler
from ai_governance_chatbot.routed_agent_gemini.call_gemini import call_llm_gemini

# Path to the regulations database
REGULATIONS_DB_PATH = "/Users/ssunehra/ghci/ai_governance_framework/core/compliance/regulations/regulations_db.json"

class RegulationsRAG:
    """
    A class that provides Retrieval-Augmented Generation for the regulations database.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 3):
        """
        Initialize the RAG system.

        Args:
            model_name: The sentence-transformer model to use for embeddings
            top_k: Number of regulations to retrieve for each query
        """
        self.top_k = top_k
        self.regulations = self._load_regulations()

        # Initialize the embedding model
        self.embed_model = SentenceTransformer(model_name)
        self.embedding_size = self.embed_model.get_sentence_embedding_dimension()

        # Create and populate FAISS index
        self.index = None
        self.regulation_texts = []
        self.regulation_objects = []
        self._build_index()

    def _load_regulations(self) -> Dict:
        """Load the regulations database from the JSON file."""
        try:
            with open(REGULATIONS_DB_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading regulations database: {e}")
            return {"policies": []}

    def _build_index(self):
        """Build the FAISS index from the regulations."""
        self.regulation_texts = []
        self.regulation_objects = []

        # Extract texts for embedding
        for policy in self.regulations.get("policies", []):
            # Create a text representation of the policy for embedding
            policy_text = f"{policy['policy_id']} - {policy['name']}: {policy['description']}"

            # Add metadata for richer context
            if policy.get('rationale'):
                policy_text += f" Rationale: {policy['rationale']}"

            if policy.get('tags'):
                policy_text += f" Tags: {', '.join(policy['tags'])}"

            self.regulation_texts.append(policy_text)
            self.regulation_objects.append(policy)

        # Skip index creation if no regulations
        if not self.regulation_texts:
            print("Warning: No regulations found to index")
            return

        # Generate embeddings for all regulations
        embeddings = self.embed_model.encode(self.regulation_texts)

        # Normalize the embeddings (important for cosine similarity in FAISS)
        faiss.normalize_L2(embeddings)

        # Create the FAISS index - IndexFlatIP is for inner product which works for normalized vectors (cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(embeddings.astype(np.float32))

        print(f"FAISS index built with {len(self.regulation_texts)} regulations")

    def retrieve(self, query: str) -> List[Tuple[float, Dict]]:
        """
        Retrieve relevant regulations for a query.

        Args:
            query: The user query to retrieve regulations for

        Returns:
            List of (score, regulation) tuples sorted by relevance
        """
        if not self.index or not self.regulation_texts:
            print("Index not built or no regulations available")
            return []

        # Generate embedding for the query
        query_embedding = self.embed_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search the FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), min(self.top_k, len(self.regulation_texts)))

        # Return the top regulations with their scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append((float(score), self.regulation_objects[idx]))

        return results

    def generate_response(self, query: str, chat_history: Optional[List] = None) -> str:
        """
        Generate a response to a user query using RAG.

        Args:
            query: The user query
            chat_history: Optional chat history for context

        Returns:
            The generated response
        """
        # Retrieve relevant regulations
        retrieved_regs = self.retrieve(query)

        if not retrieved_regs:
            return "I couldn't find any relevant regulations for your query. Please try a different question."

        # Format retrieved regulations for the context
        context = "Here are the relevant regulations:\n\n"
        for i, (score, reg) in enumerate(retrieved_regs, 1):
            context += f"{i}. Policy ID: {reg['policy_id']}\n"
            context += f"   Name: {reg['name']}\n"
            context += f"   Description: {reg['description']}\n"

            if reg.get('regulation_source'):
                context += f"   Source: {reg['regulation_source']}\n"

            if reg.get('rationale'):
                context += f"   Rationale: {reg['rationale']}\n"

            context += f"   Relevance Score: {score:.2f}\n\n"

        # Prepare messages for Gemini
        messages = []

        # Add system message with context and instructions
        system_message = {
            "role": "system",
            "content": f"You are an AI Governance Assistant that helps with compliance questions. Answer the user's query based on the following regulations:\n\n{context}"
        }
        messages.append(system_message)

        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)

        # Add the current user query
        messages.append({"role": "user", "content": query})

        # Define function calling schema for the LLM
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_regulation_details",
                    "description": "Get detailed information about a specific regulation policy",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "policy_id": {
                                "type": "string",
                                "description": "The ID of the regulation policy to get details for"
                            }
                        },
                        "required": ["policy_id"]
                    }
                }
            }
        ]

        # Call Gemini to generate a response
        response = call_llm_gemini(
            messages=messages,
            tools=tools,
            temperature=0.2
        )

        if not response:
            return "I'm sorry, I encountered an error processing your request. Please try again later."

        # Return the content of the response
        return response.content