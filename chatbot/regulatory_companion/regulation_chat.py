#!/usr/bin/env python3
"""
Regulations Chat Interface

A simple command-line chat interface for interacting with the AI Governance Regulations
using Retrieval-Augmented Generation (RAG) with FAISS and Gemini.
"""

import sys
import argparse
from typing import List, Dict, Any
import json

try:
    from regulations_rag import RegulationsRAG
except ImportError:
    print("Error: Could not import RegulationsRAG. Make sure you're in the correct directory.")
    sys.exit(1)

def print_header():
    """Print welcome header for the chat interface"""
    print("\n" + "=" * 80)
    print("AI Governance Regulations Assistant".center(80))
    print("Type 'quit', 'exit', or Ctrl+C to end the conversation".center(80))
    print("=" * 80 + "\n")

def validate_env():
    """Check if required packages are installed"""
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Error: Required package missing - {e}")
        print("\nPlease install required packages with:")
        print("pip install faiss-cpu sentence-transformers numpy\n")
        return False

    return True

def handle_function_call(message: Dict[str, Any], regulations_rag) -> str:
    """Handle LLM function calls"""
    if message.tool_calls:
        responses = []
        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_regulation_details":
                args = json.loads(tool_call.function.arguments)
                policy_id = args.get("policy_id")

                # Find the regulation with matching policy_id
                for policy in regulations_rag.regulations.get("policies", []):
                    if policy.get("policy_id") == policy_id:
                        # Format regulation details
                        details = json.dumps(policy, indent=2)
                        responses.append(f"Details for {policy_id}:\n{details}")
                        break
                else:
                    responses.append(f"Could not find regulation with policy ID: {policy_id}")

        if responses:
            return "\n\n".join(responses)

    return message.content or "No response generated."

def chat_loop(regulations_rag: RegulationsRAG):
    """Main chat interaction loop"""
    print_header()

    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the AI Governance Regulations Assistant. Goodbye!")
                break

            if not user_input:
                continue

            # Add user message to history
            chat_history.append({"role": "user", "content": user_input})

            # Generate response
            print("\nAssistant: ", end="", flush=True)

            # Generate response using RAG
            response = regulations_rag.generate_response(user_input, chat_history)

            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Handle function calling
                print(handle_function_call(response, regulations_rag))
            else:
                # Regular response
                print(response)

            # Add assistant response to history
            chat_history.append({"role": "assistant", "content": response if isinstance(response, str) else response.content})

            # Keep history manageable by limiting to last 10 messages
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            print("\n\nExiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

def main():
    """Main function to run the chat interface"""
    parser = argparse.ArgumentParser(description="AI Governance Regulations Chat")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Embedding model to use (default: all-MiniLM-L6-v2)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of regulations to retrieve for each query (default: 3)")
    args = parser.parse_args()

    # Validate required packages
    if not validate_env():
        return

    print("Initializing AI Governance Regulations Assistant...")
    print("Loading regulations database and building search index...")

    # Initialize RAG system
    try:
        regulations_rag = RegulationsRAG(
            model_name=args.model,
            top_k=args.top_k
        )

        # Start chat loop
        chat_loop(regulations_rag)

    except Exception as e:
        print(f"Error initializing the regulations assistant: {e}")

if __name__ == "__main__":
    main()