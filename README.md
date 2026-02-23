# Week 3 Deliverables — AI Agents & LangGraph

Building intelligent AI agents using **LangGraph**, **LangChain**, and **Ollama** with local LLMs.

## Projects

### 1. [ReAct Agent with Tool Calling](./react-agent/)
A Reasoning + Acting agent that can use multiple tools (calculator, web search, Wikipedia, Python executor, text analyzer, datetime) to solve complex queries step by step. Visual reasoning chain shows each Thought → Action → Observation loop.

### 2. [Multi-Agent System](./multi-agent-system/)
A collaborative multi-agent system using the Supervisor pattern. Four specialized agents (Supervisor, Researcher, Writer, Reviewer) work together to complete writing tasks through iterative collaboration.

### 3. [Agentic RAG](./agentic-rag/)
An advanced Retrieval-Augmented Generation system with autonomous decision-making. The agent decides when to retrieve, grades document relevance, reformulates queries when needed, and checks for hallucinations — all through a LangGraph state machine.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| Local LLMs | Ollama (Llama 3, Mistral) |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Web UI | Streamlit |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Models pulled: `ollama pull llama3` and `ollama pull mistral`

## Quick Start

```bash
# Clone the repo
git clone https://github.com/venkataCharan22/week3Deliverables.git
cd week3Deliverables

# Pick a project and install deps
cd react-agent
pip install -r requirements.txt
streamlit run app.py
```

Each project runs independently on its own Streamlit port. See individual README files for details.
