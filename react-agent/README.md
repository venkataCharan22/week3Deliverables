# ⚡ ReAct Agent with Tool Calling

A **Reasoning + Acting** agent built with **LangGraph** and **Ollama** that can use multiple tools to solve complex queries. Watch the agent think, act, and observe in real-time through an interactive Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-v0.2-green)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange)

## How It Works

The ReAct (Reasoning + Acting) pattern allows the agent to:
1. **Think** about what it needs to do
2. **Act** by calling the appropriate tool
3. **Observe** the tool's output
4. **Repeat** or produce a final answer

```
User Query → Agent Thinks → Calls Tool → Observes Result → Thinks Again → Final Answer
```

## Available Tools

| Tool | Description |
|------|------------|
| 🧮 Calculator | Evaluate math expressions (arithmetic, trig, roots) |
| 🕐 DateTime | Get current date, time, and timezone info |
| 📊 Text Analyzer | Word count, readability stats, frequency analysis |
| 🔍 Web Search | Search the web via DuckDuckGo |
| 📚 Wikipedia | Search Wikipedia for encyclopedic information |
| 🐍 Python Executor | Run Python code snippets safely |

## Architecture

Built using **LangGraph's StateGraph** with the prebuilt tool-calling pattern:

```
┌──────────┐     ┌───────────┐
│  Agent   │────►│  Tools    │
│  (LLM)  │◄────│  (6 tools)│
└────┬─────┘     └───────────┘
     │
     ▼
   Answer
```

- **State**: `MessagesState` tracks the conversation
- **Agent Node**: LLM with tools bound via `bind_tools()`
- **Tool Node**: LangGraph's `ToolNode` for execution
- **Routing**: `tools_condition` for conditional edges

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running
ollama serve

# Pull a model (if not already done)
ollama pull llama3

# Run the app
streamlit run app.py
```

## Features

- Real-time reasoning chain visualization (Thought → Action → Observation)
- Toggle individual tools on/off
- Adjustable model and temperature
- Tool usage statistics dashboard
- Click-to-try example prompts
- Dark theme UI with custom styling

## Example Queries

- "What is sqrt(144) + 15 * 3?"
- "Search Wikipedia for quantum computing and summarize it"
- "Write Python code to generate the first 20 Fibonacci numbers"
- "Analyze this text: [paste any text]"
- "What's the current date and time?"

## Project Structure

```
react-agent/
├── app.py              # Streamlit UI
├── agent.py            # LangGraph agent + tool definitions
├── requirements.txt    # Dependencies
├── README.md
└── .streamlit/
    └── config.toml     # Theme configuration
```
