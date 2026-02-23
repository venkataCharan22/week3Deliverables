# 🧠 Agentic RAG with Autonomous Retrieval

An advanced **Retrieval-Augmented Generation** system with autonomous decision-making, built with **LangGraph**, **ChromaDB**, and **Ollama**. The agent independently decides when to retrieve, grades document relevance, reformulates queries, and checks for hallucinations.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-v0.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange)

## How It Works

Unlike basic RAG (retrieve → generate), Agentic RAG makes **autonomous decisions** at each step:

```
┌─────────────┐
│   Question   │
└──────┬──────┘
       │
┌──────▼──────┐     ┌──────────────┐
│    Route     │────►│ Direct Answer│──► END
│   Question   │     └──────────────┘
└──────┬──────┘
       │ needs retrieval
┌──────▼──────┐
│  Retrieve    │◄────────────────────┐
│  Documents   │                     │
└──────┬──────┘                     │
       │                     ┌──────┴──────┐
┌──────▼──────┐              │ Reformulate  │
│    Grade     │──not ok────►│   Query      │
│  Documents   │              └─────────────┘
└──────┬──────┘
       │ relevant
┌──────▼──────┐
│   Generate   │
│   Answer     │
└──────┬──────┘
       │
┌──────▼──────┐
│ Hallucination│
│    Check     │──► END
└─────────────┘
```

## Key Features

| Feature | Description |
|---------|------------|
| 🔀 Query Routing | Agent decides: retrieve from docs OR answer directly |
| 📥 Smart Retrieval | Semantic search with similarity scoring via ChromaDB |
| ⚖️ Document Grading | LLM evaluates each document's relevance to the question |
| 🔄 Query Reformulation | Auto-rewrites queries when initial retrieval fails |
| ✨ Grounded Generation | Generates answers strictly from relevant documents |
| 🔍 Hallucination Check | Verifies answer is supported by source documents |

## Architecture

Built using **LangGraph's StateGraph** with 7 nodes and conditional routing:

- **State**: Custom `RAGState` with question, documents, scores, decision log
- **Route Question**: LLM classifies query → retrieval or direct answer
- **Retrieve**: ChromaDB similarity search with HuggingFace embeddings
- **Grade Documents**: LLM grades each doc as relevant/not relevant
- **Reformulate Query**: Rewrites query for better retrieval (max 3 attempts)
- **Generate**: Creates answer from graded, relevant documents
- **Hallucination Check**: Verifies answer is grounded in source docs

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running
ollama serve

# Pull a model
ollama pull llama3

# Run the app
streamlit run app.py
```

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF, TXT, or MD files (or load sample docs)
2. **Ask Questions**: Type questions about your documents in the chat
3. **Watch the Agent Think**: See the decision trace showing each node's reasoning
4. **Check Sources**: View relevance scores and source citations
5. **Explore**: Use the Explore tab to search the knowledge base directly

## Sample Documents Included

The `sample_docs/` folder contains three documents for testing:
- `artificial_intelligence.txt` — Overview of AI, ML, and LLMs
- `climate_change.txt` — Climate science and mitigation strategies
- `space_exploration.txt` — History and future of space exploration

## Project Structure

```
agentic-rag/
├── app.py                  # Streamlit UI
├── rag_agent.py            # LangGraph RAG agent (7-node graph)
├── document_processor.py   # Document loading and chunking
├── vector_store.py         # ChromaDB manager
├── requirements.txt        # Dependencies
├── README.md
├── .streamlit/
│   └── config.toml         # Theme configuration
└── sample_docs/
    ├── artificial_intelligence.txt
    ├── climate_change.txt
    └── space_exploration.txt
```
