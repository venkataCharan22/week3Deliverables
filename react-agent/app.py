"""
ReAct Agent Studio - Interactive AI Agent with Tool Calling
Built with LangGraph, Ollama, and Streamlit
"""

import streamlit as st
import json

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

    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1e1b4b, #0f172a);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SESSION STATE
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

    model_options = available_models if available_models else ["llama3", "mistral"]
    clean_models = list(dict.fromkeys([m.replace(":latest", "") for m in model_options]))
    selected_model = st.selectbox("🤖 Model", clean_models, index=0)
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.1, 0.05)
    max_steps = st.slider("🔄 Max Reasoning Steps", 3, 15, 8)

    st.markdown("---")
    st.markdown("### 🛠️ Tools")

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
    st.markdown("### 💡 Try These")
    examples = [
        "What is sqrt(144) + 15 * 3?",
        "What's the current date and time?",
        "Search Wikipedia for quantum computing",
        "Analyze this text: The quick brown fox jumps over the lazy dog",
        "Write Python code to generate first 20 Fibonacci numbers",
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
    st.markdown(f"""<div class="stats-card">
        <div class="stats-number">{st.session_state.total_queries}</div>
        <div class="stats-label">Total Queries</div>
    </div>""", unsafe_allow_html=True)
with col2:
    total_tool_calls = sum(st.session_state.tool_usage_stats.values())
    st.markdown(f"""<div class="stats-card">
        <div class="stats-number">{total_tool_calls}</div>
        <div class="stats-label">Tool Calls</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stats-card">
        <div class="stats-number">{len(enabled_tools)}</div>
        <div class="stats-label">Active Tools</div>
    </div>""", unsafe_allow_html=True)
with col4:
    most_used = max(st.session_state.tool_usage_stats, key=st.session_state.tool_usage_stats.get) if st.session_state.tool_usage_stats else "—"
    st.markdown(f"""<div class="stats-card">
        <div class="stats-number" style="font-size:1rem">{most_used}</div>
        <div class="stats-label">Most Used Tool</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# HELPER: Display reasoning trace
# ═══════════════════════════════════════════

def display_reasoning_trace(trace):
    """Render the reasoning chain with styled components."""
    for step in trace:
        if step["type"] == "thought":
            st.markdown(
                f'<div class="thought-step">💭 <strong>Thinking:</strong> {step["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif step["type"] == "action":
            # Find the tool icon
            tool_icon = "🔧"
            for name, info in TOOL_REGISTRY.items():
                if name.lower() in step.get("tool", "").lower() or step.get("tool", "").lower() in name.lower():
                    tool_icon = info["icon"]
                    break
            st.markdown(
                f'<div class="action-step">{tool_icon} <strong>Action:</strong> {step["tool"]}<br>'
                f'<pre style="margin-top:0.5rem;color:#fbbf24;font-size:0.82rem">{step["input"]}</pre></div>',
                unsafe_allow_html=True,
            )
        elif step["type"] == "observation":
            content = step["content"]
            if len(content) > 400:
                content = content[:400] + "..."
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
        if msg["role"] == "assistant" and str(i) in st.session_state.reasoning_traces:
            with st.expander("🔍 View Reasoning Chain", expanded=False):
                display_reasoning_trace(st.session_state.reasoning_traces[str(i)])

# Handle examples
if "pending_example" in st.session_state:
    prompt = st.session_state.pending_example
    del st.session_state.pending_example
else:
    prompt = st.chat_input("Ask me anything... I can calculate, search, analyze, and more!")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not connected:
            st.error("Ollama is not running. Please start it with `ollama serve`")
        elif not enabled_tools:
            st.warning("Please enable at least one tool in the sidebar.")
        else:
            with st.status("🧠 Agent is reasoning...", expanded=True) as status:
                try:
                    agent = create_react_agent(
                        model_name=selected_model,
                        temperature=temperature,
                        enabled_tools=enabled_tools,
                    )

                    initial_state = {
                        "question": prompt,
                        "reasoning_steps": [],
                        "tool_calls_count": {},
                        "final_answer": "",
                        "iteration": 0,
                        "max_iterations": max_steps,
                        "scratchpad": "",
                    }

                    # Stream events
                    all_steps = []
                    final_answer = ""
                    tool_counts = {}

                    for event in agent.stream(initial_state, {"recursion_limit": max_steps + 5}):
                        for node_name, node_output in event.items():
                            steps = node_output.get("reasoning_steps", [])
                            all_steps.extend(steps)

                            for step in steps:
                                if step["type"] == "thought":
                                    st.write(f"💭 **Thinking:** {step['content'][:100]}...")
                                elif step["type"] == "action":
                                    st.write(f"🔧 **Using tool:** {step['tool']}")
                                elif step["type"] == "observation":
                                    st.write(f"👁️ **Got result** ({len(step['content'])} chars)")

                            if node_output.get("final_answer"):
                                final_answer = node_output["final_answer"]
                            if node_output.get("tool_calls_count"):
                                tool_counts = node_output["tool_calls_count"]

                    status.update(label="✅ Agent finished!", state="complete")

                    # Update global tool stats
                    for tool_name, count in tool_counts.items():
                        st.session_state.tool_usage_stats[tool_name] = (
                            st.session_state.tool_usage_stats.get(tool_name, 0) + count
                        )

                    # Display final answer
                    if final_answer:
                        st.markdown(final_answer)
                    else:
                        st.info("The agent processed your request but didn't produce a final answer.")

                    # Show reasoning trace
                    has_reasoning = any(s["type"] in ("thought", "action", "observation") for s in all_steps)
                    if has_reasoning:
                        with st.expander("🔍 View Full Reasoning Chain", expanded=True):
                            display_reasoning_trace(all_steps)

                    # Save to session
                    msg_idx = str(len(st.session_state.messages))
                    st.session_state.reasoning_traces[msg_idx] = all_steps
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer or "Agent processed the request.",
                    })
                    st.session_state.total_queries += 1

                except Exception as e:
                    st.error(f"Agent error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")


# ═══════════════════════════════════════════
# TOOL USAGE BREAKDOWN
# ═══════════════════════════════════════════

if st.session_state.tool_usage_stats:
    st.markdown("---")
    st.markdown("### 📊 Tool Usage Breakdown")
    cols = st.columns(min(len(st.session_state.tool_usage_stats), 6))
    for i, (tool_name, count) in enumerate(
        sorted(st.session_state.tool_usage_stats.items(), key=lambda x: x[1], reverse=True)
    ):
        # Find matching tool info
        icon = "🔧"
        for name, info in TOOL_REGISTRY.items():
            if name.lower() in tool_name.lower() or tool_name.lower() in name.lower():
                icon = info["icon"]
                break
        with cols[i % len(cols)]:
            st.metric(label=f"{icon} {tool_name}", value=count, delta=f"{count} calls")
