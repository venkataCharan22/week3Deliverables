"""
Multi-Agent Collaboration System - Interactive Agent Orchestration
Built with LangGraph, Ollama, and Streamlit
"""

import streamlit as st
import time
import json

from agents import (
    create_multi_agent_graph,
    check_ollama_connection,
    AGENT_PROFILES,
    AgentState,
)

# ═══════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1a0f2e 40%, #0f172a 100%);
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #c084fc, #818cf8, #c084fc);
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

    .agent-card {
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    .agent-card.active {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
    }
    .agent-card .agent-name {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.2rem;
    }
    .agent-card .agent-role {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    .workflow-step {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    .step-completed {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
    }
    .step-active {
        background: rgba(139, 92, 246, 0.2);
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #c4b5fd;
        animation: pulse-glow 2s infinite;
    }
    .step-pending {
        background: rgba(100, 116, 139, 0.1);
        border: 1px solid rgba(100, 116, 139, 0.2);
        color: #64748b;
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(139, 92, 246, 0.3); }
        50% { box-shadow: 0 0 15px rgba(139, 92, 246, 0.5); }
    }

    .log-entry {
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-radius: 10px;
        font-size: 0.85rem;
        border-left: 3px solid;
    }
    .log-supervisor {
        background: rgba(139, 92, 246, 0.08);
        border-left-color: #8b5cf6;
    }
    .log-researcher {
        background: rgba(59, 130, 246, 0.08);
        border-left-color: #3b82f6;
    }
    .log-writer {
        background: rgba(16, 185, 129, 0.08);
        border-left-color: #10b981;
    }
    .log-reviewer {
        background: rgba(245, 158, 11, 0.08);
        border-left-color: #f59e0b;
    }
    .log-finalizer {
        background: rgba(239, 68, 68, 0.08);
        border-left-color: #ef4444;
    }

    .output-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
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
        background: linear-gradient(135deg, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stats-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1a0f2e, #0f172a);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════

if "task_history" not in st.session_state:
    st.session_state.task_history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤝 System Configuration")
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
        st.warning("Start Ollama: `ollama serve`")

    # Model selection
    model_options = available_models if available_models else ["llama3", "mistral"]
    clean_models = list(dict.fromkeys([m.replace(":latest", "") for m in model_options]))
    selected_model = st.selectbox("🤖 Model", clean_models, index=0)

    # Max iterations
    max_iterations = st.slider("🔄 Max Iterations", 3, 10, 6)

    st.markdown("---")
    st.markdown("### 👥 Agent Team")

    # Agent info cards
    for agent_type, profile in AGENT_PROFILES.items():
        color = profile["color"]
        st.markdown(
            f'<div class="agent-card" style="background: {color}10; border-color: {color}40;">'
            f'<div class="agent-name">{profile["icon"]} {profile["name"]}</div>'
            f'<div class="agent-role">{profile["role"]}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 💡 Example Tasks")

    examples = [
        "Write a technical blog post about machine learning in healthcare",
        "Create a product description for a smart home device",
        "Write an executive summary about renewable energy trends",
        "Explain quantum computing for a general audience",
    ]

    for ex in examples:
        if st.button(f"→ {ex[:50]}...", key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_task = ex

    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.task_history = []
        st.session_state.current_result = None
        st.rerun()


# ═══════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🤝 Multi-Agent Collaboration</h1>
    <p>Supervisor + Researcher + Writer + Reviewer • Powered by LangGraph & Ollama</p>
</div>
""", unsafe_allow_html=True)

# Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{len(st.session_state.task_history)}</div>
        <div class="stats-label">Tasks Completed</div>
    </div>""", unsafe_allow_html=True)
with col2:
    total_steps = sum(len(t.get("agent_history", [])) for t in st.session_state.task_history)
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{total_steps}</div>
        <div class="stats-label">Total Agent Steps</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">4</div>
        <div class="stats-label">Active Agents</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TASK INPUT
# ═══════════════════════════════════════════

# Handle example tasks
task_value = ""
if "pending_task" in st.session_state:
    task_value = st.session_state.pending_task
    del st.session_state.pending_task

task_input = st.text_area(
    "📝 Enter a task for the agent team",
    value=task_value,
    height=80,
    placeholder="Describe a writing task... e.g., 'Write a blog post about the future of AI agents'",
)

col_run, col_space = st.columns([0.3, 0.7])
with col_run:
    run_button = st.button("🚀 Run Agent Team", type="primary", use_container_width=True)


# ═══════════════════════════════════════════
# AGENT EXECUTION
# ═══════════════════════════════════════════

if run_button and task_input.strip():
    if not connected:
        st.error("Ollama is not running. Please start it with `ollama serve`")
    else:
        st.markdown("---")

        # Workflow visualization placeholder
        workflow_container = st.container()

        # Agent log
        st.markdown("### 📋 Agent Activity Log")
        log_container = st.container()

        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            graph = create_multi_agent_graph(model_name=selected_model)

            initial_state = {
                "task": task_input.strip(),
                "current_agent": "",
                "research_output": "",
                "draft_output": "",
                "review_output": "",
                "final_output": "",
                "agent_history": [],
                "iteration": 0,
                "max_iterations": max_iterations,
                "status": "starting",
            }

            # Stream execution
            all_history = []
            step_count = 0
            agents_used = set()
            final_output = ""

            for event in graph.stream(initial_state, {"recursion_limit": 25}):
                for node_name, node_output in event.items():
                    step_count += 1
                    agents_used.add(node_name)

                    # Update progress
                    progress = min(step_count / (max_iterations * 2), 0.95)
                    progress_bar.progress(progress)

                    # Get agent profile
                    profile = AGENT_PROFILES.get(node_name, {
                        "icon": "⚙️", "name": node_name.title(), "color": "#64748b"
                    })

                    status_text.markdown(
                        f"**{profile['icon']} {profile['name']}** is working..."
                    )

                    # Capture final output as it's produced
                    if node_output.get("final_output"):
                        final_output = node_output["final_output"]
                    elif node_output.get("draft_output"):
                        final_output = node_output["draft_output"]

                    # Log entry
                    history_entries = node_output.get("agent_history", [])
                    all_history.extend(history_entries)

                    with log_container:
                        for entry in history_entries:
                            agent = entry.get("agent", node_name)
                            action = entry.get("action", "Processing...")
                            reason = entry.get("reason", "")
                            preview = entry.get("output_preview", "")

                            agent_profile = AGENT_PROFILES.get(agent, {"icon": "⚙️", "name": agent.title(), "color": "#64748b"})

                            log_html = (
                                f'<div class="log-entry log-{agent}">'
                                f'<strong>{agent_profile["icon"]} {agent_profile["name"]}</strong> — {action}'
                            )
                            if reason:
                                log_html += f'<br><span style="color:#94a3b8;font-size:0.8rem">💬 {reason}</span>'
                            if preview:
                                log_html += f'<br><span style="color:#64748b;font-size:0.78rem">📄 {preview[:150]}...</span>'
                            log_html += '</div>'

                            st.markdown(log_html, unsafe_allow_html=True)

            progress_bar.progress(1.0)
            status_text.markdown("**✅ All agents finished!**")

            if not final_output:
                final_output = "The agent team processed your task. Check the activity log for details."

            # Display workflow visualization
            with workflow_container:
                st.markdown("### 🔄 Workflow Visualization")
                agent_order = []
                for entry in all_history:
                    a = entry.get("agent", "")
                    if a and (not agent_order or agent_order[-1] != a):
                        agent_order.append(a)

                flow_html = '<div style="display:flex;flex-wrap:wrap;gap:0.3rem;align-items:center;">'
                for i, agent in enumerate(agent_order):
                    profile = AGENT_PROFILES.get(agent, {"icon": "⚙️", "name": agent.title(), "color": "#64748b"})
                    flow_html += (
                        f'<span class="workflow-step step-completed">'
                        f'{profile["icon"]} {profile["name"]}</span>'
                    )
                    if i < len(agent_order) - 1:
                        flow_html += '<span style="color:#4b5563">→</span>'
                flow_html += '</div>'
                st.markdown(flow_html, unsafe_allow_html=True)

            # Display final output
            st.markdown("---")
            st.markdown("### 📝 Final Output")
            st.markdown(
                f'<div class="output-card">{final_output}</div>',
                unsafe_allow_html=True,
            )

            # Save to history
            st.session_state.task_history.append({
                "task": task_input.strip(),
                "output": final_output,
                "agent_history": all_history,
                "agents_used": list(agents_used),
                "steps": step_count,
            })
            st.session_state.current_result = final_output

        except Exception as e:
            st.error(f"Execution error: {str(e)}")
            st.info("💡 Try a simpler task or different model.")
            progress_bar.progress(0)


# ═══════════════════════════════════════════
# TASK HISTORY
# ═══════════════════════════════════════════

if st.session_state.task_history:
    st.markdown("---")
    st.markdown("### 📜 Previous Tasks")
    for i, task in enumerate(reversed(st.session_state.task_history)):
        with st.expander(f"Task {len(st.session_state.task_history) - i}: {task['task'][:60]}...", expanded=False):
            st.markdown(f"**Agents Used:** {', '.join(task.get('agents_used', []))}")
            st.markdown(f"**Steps:** {task.get('steps', 0)}")
            st.markdown("**Output:**")
            st.markdown(task.get("output", "No output recorded."))
