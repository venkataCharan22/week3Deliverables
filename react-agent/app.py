"""
ReAct Agent Studio - Interactive AI Agent with Tool Calling
Built with LangGraph, Ollama, and Streamlit
"""

import streamlit as st
import json
import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import (
    create_react_agent,
    get_available_tools,
    check_ollama_connection,
    TOOL_REGISTRY,
)

# ═══════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="ReAct Agent Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1e1b4b 40%, #0f172a 100%);
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    .status-banner {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-size: 0.85rem;
    }
    .status-connected {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
    }
    .status-disconnected {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #fca5a5;
    }

    .thought-step {
        border-left: 3px solid #60a5fa;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        background: rgba(96, 165, 250, 0.08);
        border-radius: 0 10px 10px 0;
        color: #e2e8f0;
    }
    .action-step {
        border-left: 3px solid #fbbf24;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        background: rgba(251, 191, 36, 0.08);
        border-radius: 0 10px 10px 0;
        color: #e2e8f0;
    }
    .observation-step {
        border-left: 3px solid #34d399;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        background: rgba(52, 211, 153, 0.08);
        border-radius: 0 10px 10px 0;
        color: #e2e8f0;
    }
    .answer-step {
        border-left: 3px solid #a78bfa;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        background: rgba(167, 139, 250, 0.08);
        border-radius: 0 10px 10px 0;
        color: #e2e8f0;
    }

    .tool-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.7rem;
        margin: 0.3rem 0;
        transition: all 0.2s ease;
    }
    .tool-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.15);
    }

    .stats-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stats-number {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stats-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .example-btn {
        display: block;
        width: 100%;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        color: #c7d2fe;
        text-align: left;
        font-size: 0.82rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .example-btn:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .sidebar .sidebar-content {
        background: rgba(15, 23, 42, 0.95);
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1e1b4b, #0f172a);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "reasoning_traces" not in st.session_state:
    st.session_state.reasoning_traces = {}
if "tool_usage_stats" not in st.session_state:
    st.session_state.tool_usage_stats = {}
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ Agent Configuration")
    st.markdown("---")

    # Connection status
    connected, available_models = check_ollama_connection()
    if connected:
        st.markdown(
            '<div class="status-banner status-connected">● Connected to Ollama</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-banner status-disconnected">● Ollama not detected</div>',
            unsafe_allow_html=True,
        )
        st.warning("Start Ollama to use the agent: `ollama serve`")

    # Model selection
    model_options = available_models if available_models else ["llama3", "mistral"]
    # Clean model names (remove :latest suffix for display)
    clean_models = list(dict.fromkeys([m.replace(":latest", "") for m in model_options]))
    selected_model = st.selectbox("🤖 Model", clean_models, index=0)

    # Temperature
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.1, 0.05)

    st.markdown("---")
    st.markdown("### 🛠️ Tools")

    # Tool toggles
    enabled_tools = []
    tools = get_available_tools()
    for tool_name, tool_info in tools.items():
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.markdown(f"<span style='font-size:1.2rem'>{tool_info['icon']}</span>", unsafe_allow_html=True)
        with col2:
            if st.checkbox(tool_name, value=True, key=f"tool_{tool_name}"):
                enabled_tools.append(tool_name)

    st.markdown("---")

    # Example prompts
    st.markdown("### 💡 Try These")
    examples = [
        "What is sqrt(144) + 15 * 3?",
        "What's the current date and time?",
        "Search Wikipedia for quantum computing",
        "Analyze this text: The quick brown fox jumps over the lazy dog",
        "Write Python code to generate the first 20 Fibonacci numbers",
        "Search the web for latest AI news",
    ]

    for ex in examples:
        if st.button(f"→ {ex}", key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_example = ex

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.reasoning_traces = {}
        st.session_state.tool_usage_stats = {}
        st.session_state.total_queries = 0
        st.rerun()


# ═══════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>⚡ ReAct Agent Studio</h1>
    <p>Reasoning + Acting Agent with Tool Calling • Built with LangGraph & Ollama</p>
</div>
""", unsafe_allow_html=True)

# Stats dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{st.session_state.total_queries}</div>
        <div class="stats-label">Total Queries</div>
    </div>""", unsafe_allow_html=True)
with col2:
    total_tool_calls = sum(st.session_state.tool_usage_stats.values())
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{total_tool_calls}</div>
        <div class="stats-label">Tool Calls</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{len(enabled_tools)}</div>
        <div class="stats-label">Active Tools</div>
    </div>""", unsafe_allow_html=True)
with col4:
    most_used = max(st.session_state.tool_usage_stats, key=st.session_state.tool_usage_stats.get) if st.session_state.tool_usage_stats else "—"
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number" style="font-size:1rem">{most_used}</div>
        <div class="stats-label">Most Used Tool</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def extract_reasoning_trace(events):
    """Extract the reasoning trace from LangGraph stream events."""
    trace = []
    for event in events:
        for node_name, node_output in event.items():
            if node_name == "agent":
                msg = node_output["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        trace.append({
                            "type": "thought",
                            "content": f"I need to use **{tc['name']}** to help answer this.",
                        })
                        trace.append({
                            "type": "action",
                            "tool": tc["name"],
                            "input": json.dumps(tc["args"], indent=2) if isinstance(tc["args"], dict) else str(tc["args"]),
                        })
                elif hasattr(msg, "content") and msg.content:
                    trace.append({
                        "type": "answer",
                        "content": msg.content,
                    })
            elif node_name == "tools":
                msg = node_output["messages"][-1]
                if hasattr(msg, "content"):
                    trace.append({
                        "type": "observation",
                        "content": msg.content[:500],
                    })
    return trace


def display_reasoning_trace(trace):
    """Display the reasoning trace with styled components."""
    for i, step in enumerate(trace):
        if step["type"] == "thought":
            st.markdown(
                f'<div class="thought-step">💭 <strong>Thinking:</strong> {step["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif step["type"] == "action":
            tool_icon = TOOL_REGISTRY.get(step.get("tool", ""), {}).get("icon", "🔧")
            st.markdown(
                f'<div class="action-step">{tool_icon} <strong>Action:</strong> Using <code>{step["tool"]}</code><br>'
                f'<pre style="margin-top:0.5rem;color:#fbbf24;font-size:0.82rem">{step["input"]}</pre></div>',
                unsafe_allow_html=True,
            )
        elif step["type"] == "observation":
            content = step["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            st.markdown(
                f'<div class="observation-step">👁️ <strong>Observation:</strong><br>'
                f'<pre style="margin-top:0.5rem;color:#34d399;font-size:0.82rem;white-space:pre-wrap">{content}</pre></div>',
                unsafe_allow_html=True,
            )
        elif step["type"] == "answer":
            st.markdown(
                f'<div class="answer-step">✨ <strong>Final Answer:</strong><br><br>{step["content"]}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════
# CHAT INTERFACE
# ═══════════════════════════════════════════

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show reasoning trace if available
        if msg["role"] == "assistant" and str(i) in st.session_state.reasoning_traces:
            with st.expander("🔍 View Reasoning Chain", expanded=False):
                display_reasoning_trace(st.session_state.reasoning_traces[str(i)])

# Handle example prompts
if "pending_example" in st.session_state:
    prompt = st.session_state.pending_example
    del st.session_state.pending_example
else:
    prompt = st.chat_input("Ask me anything... I can calculate, search, analyze, and more!")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with agent
    with st.chat_message("assistant"):
        if not connected:
            st.error("Ollama is not running. Please start it with `ollama serve`")
        elif not enabled_tools:
            st.warning("Please enable at least one tool in the sidebar.")
        else:
            with st.status("🧠 Agent is thinking...", expanded=True) as status:
                try:
                    agent = create_react_agent(
                        model_name=selected_model,
                        temperature=temperature,
                        enabled_tools=enabled_tools,
                    )

                    # Collect all events
                    events = []
                    input_messages = [HumanMessage(content=prompt)]

                    for event in agent.stream({"messages": input_messages}):
                        events.append(event)
                        # Show progress
                        for node_name, node_output in event.items():
                            if node_name == "agent":
                                msg = node_output["messages"][-1]
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_name = tc["name"]
                                        st.write(f"🔧 Calling tool: **{tool_name}**")
                                        # Track tool usage
                                        st.session_state.tool_usage_stats[tool_name] = (
                                            st.session_state.tool_usage_stats.get(tool_name, 0) + 1
                                        )
                            elif node_name == "tools":
                                st.write("📥 Got tool response")

                    status.update(label="✅ Agent finished!", state="complete")

                    # Extract and display reasoning trace
                    trace = extract_reasoning_trace(events)

                    # Get final answer
                    final_answer = ""
                    for step in reversed(trace):
                        if step["type"] == "answer":
                            final_answer = step["content"]
                            break

                    if not final_answer:
                        # Fallback: get last AI message from events
                        for event in reversed(events):
                            for node_name, node_output in event.items():
                                if node_name == "agent":
                                    msg = node_output["messages"][-1]
                                    if hasattr(msg, "content") and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                                        final_answer = msg.content
                                        break
                            if final_answer:
                                break

                    if final_answer:
                        st.markdown(final_answer)
                    else:
                        st.info("The agent processed your request but didn't produce a text response.")

                    # Show reasoning trace
                    if trace and any(s["type"] in ("thought", "action", "observation") for s in trace):
                        with st.expander("🔍 View Reasoning Chain", expanded=True):
                            display_reasoning_trace(trace)

                    # Save to session state
                    msg_idx = str(len(st.session_state.messages))
                    st.session_state.reasoning_traces[msg_idx] = trace
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer or "Agent processed the request.",
                    })
                    st.session_state.total_queries += 1

                except Exception as e:
                    st.error(f"Agent error: {str(e)}")
                    st.info("💡 **Tip:** Make sure the selected model supports tool calling. Try `llama3` or `mistral`.")


# ═══════════════════════════════════════════
# TOOL USAGE BREAKDOWN (bottom section)
# ═══════════════════════════════════════════

if st.session_state.tool_usage_stats:
    st.markdown("---")
    st.markdown("### 📊 Tool Usage Breakdown")
    cols = st.columns(len(st.session_state.tool_usage_stats))
    for i, (tool_name, count) in enumerate(
        sorted(st.session_state.tool_usage_stats.items(), key=lambda x: x[1], reverse=True)
    ):
        info = TOOL_REGISTRY.get(tool_name, {})
        icon = info.get("icon", "🔧")
        with cols[i]:
            st.metric(
                label=f"{icon} {tool_name}",
                value=count,
                delta=f"{count} calls",
            )
