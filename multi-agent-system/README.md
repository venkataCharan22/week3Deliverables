# 🤝 Multi-Agent Collaboration System

A **multi-agent system** built with **LangGraph** where specialized AI agents collaborate to complete complex writing tasks. Uses the **Supervisor pattern** for agent orchestration.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-v0.2-green)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange)

## How It Works

Four specialized agents collaborate through a supervisor-managed workflow:

```
                    ┌────────────┐
              ┌────►│ Researcher │────┐
              │     └────────────┘    │
┌──────────┐  │     ┌────────────┐    │  ┌───────────┐
│Supervisor│──┼────►│   Writer   │────┼─►│ Finalizer │
└──────────┘  │     └────────────┘    │  └───────────┘
              │     ┌────────────┐    │
              └────►│  Reviewer  │────┘
                    └────────────┘
```

## Agent Team

| Agent | Role | Specialty |
|-------|------|-----------|
| 👔 Supervisor | Orchestrator | Decides task delegation and workflow routing |
| 🔬 Researcher | Information Gathering | Analyzes topics and provides key facts |
| ✍️ Writer | Content Creation | Drafts polished content from research |
| 🔍 Reviewer | Quality Assurance | Reviews drafts and suggests improvements |

## Workflow

1. **Supervisor** analyzes the task and delegates to **Researcher**
2. **Researcher** gathers information and key facts
3. **Supervisor** routes to **Writer** with research context
4. **Writer** creates a draft based on findings
5. **Supervisor** sends draft to **Reviewer**
6. **Reviewer** evaluates quality and provides feedback
7. If revisions needed → back to **Writer**; otherwise → **Finalize**

## Architecture

Built using **LangGraph's StateGraph** with conditional routing:

- **State**: Custom `AgentState` tracking task, outputs, history, and iteration count
- **Nodes**: Each agent is a node with a specialized system prompt
- **Edges**: Supervisor node uses conditional edges to route to the right agent
- **Termination**: Finalizer node or max iteration limit

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

## Features

- Visual workflow tracking showing agent execution order
- Real-time agent activity log with colored entries
- Configurable max iterations
- Task history with expandable details
- Agent team info cards in sidebar
- Example tasks to try
- Dark-themed UI with purple accent

## Example Tasks

- "Write a technical blog post about machine learning in healthcare"
- "Create a product description for a smart home device"
- "Explain quantum computing for a general audience"
- "Write an executive summary about renewable energy trends"

## Project Structure

```
multi-agent-system/
├── app.py              # Streamlit UI
├── agents.py           # Agent definitions + LangGraph workflow
├── requirements.txt    # Dependencies
├── README.md
└── .streamlit/
    └── config.toml     # Theme configuration
```
