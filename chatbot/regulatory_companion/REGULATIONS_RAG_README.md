# AI Governance Regulations RAG System

A Retrieval-Augmented Generation (RAG) system for AI Governance Regulations using FAISS for efficient semantic search and Gemini for generating context-aware responses.

## Overview

This system provides an intelligent chat interface for querying and understanding AI governance regulations. It uses:

- **FAISS** for efficient vector similarity search
- **Sentence Transformers** for embedding text into vectors
- **Gemini LLM** (via OpenAI-compatible API) for generating natural language responses

The RAG approach retrieves the most relevant regulations for a query and provides them as context to the LLM, resulting in more accurate and grounded responses.

## Installation

1. Clone the repository (if not already done)
2. Navigate to the project directory
3. Run the setup script:

```bash
cd /path/to/ai_governance_framework
python setup_regulations_rag.py
```

The setup script will install all required dependencies, including:
- faiss-cpu
- sentence-transformers
- numpy
- tqdm
- openai (for OpenAI-compatible API)

## Usage

To start the chat interface:

```bash
python regulation_chat.py
```

Optional arguments:
- `--model`: Specify a different sentence-transformers model (default: all-MiniLM-L6-v2)
- `--top_k`: Number of regulations to retrieve per query (default: 3)

Example:
```bash
python regulation_chat.py --model all-mpnet-base-v2 --top_k 5
```

## Integrating with Other Applications

You can import the `RegulationsRAG` class in your Python applications:

```python
from core.regulations_rag import RegulationsRAG

# Initialize the RAG system
rag = RegulationsRAG(model_name="all-MiniLM-L6-v2", top_k=3)

# Generate a response to a user query
response = rag.generate_response("What regulations apply to gender discrimination?")
print(response)
```

## System Architecture

The system consists of these main components:

1. **Regulations Database**: JSON file containing structured regulations data
2. **Embedding Layer**: Converts text to vector representations using Sentence Transformers
3. **FAISS Index**: Efficient vector search to find semantically similar regulations
4. **LLM Integration**: Uses Gemini via OpenAI-compatible API for response generation
5. **Chat Interface**: Simple command-line interface for user interaction

## Extending the System

To add new regulations:
1. Edit the regulations database JSON file
2. Re-initialize the RAG system to rebuild the index

## Troubleshooting

- **Missing dependencies**: Run `python setup_regulations_rag.py` to install required packages
- **API errors**: Ensure the Gemini API is properly configured in the environment variables
- **Slow performance**: Consider using a smaller embedding model or reducing the `top_k` parameter

## License

This project is part of the AI Governance Framework.

## Contact

For questions or support, please contact the AI Governance team.