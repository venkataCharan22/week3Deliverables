"""
Agentic RAG - Autonomous Retrieval-Augmented Generation
Built with LangGraph, Ollama, ChromaDB, and Streamlit
"""

import streamlit as st
import os
import json

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_agent import create_rag_agent, check_ollama_connection

# ═══════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #052e16 40%, #0f172a 100%);
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #34d399, #60a5fa, #34d399);
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

    .decision-node {
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 10px;
        font-size: 0.85rem;
        border-left: 3px solid;
        color: #e2e8f0;
    }

    .relevance-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.1);
        overflow: hidden;
        margin-top: 0.3rem;
    }
    .relevance-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.4rem 0;
    }
    .source-name {
        font-weight: 600;
        color: #34d399;
        font-size: 0.85rem;
    }
    .source-preview {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.3rem;
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
        background: linear-gradient(135deg, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stats-label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
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

    .flow-step {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.4rem 0.8rem;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.15rem;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #052e16, #0f172a);
    }

    .upload-zone {
        background: rgba(52, 211, 153, 0.05);
        border: 2px dashed rgba(52, 211, 153, 0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "decision_traces" not in st.session_state:
    st.session_state.decision_traces = {}
if "total_rag_queries" not in st.session_state:
    st.session_state.total_rag_queries = 0
if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 RAG Configuration")
    st.markdown("---")

    # Connection
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

    # Model
    model_options = available_models if available_models else ["llama3", "mistral"]
    clean_models = list(dict.fromkeys([m.replace(":latest", "") for m in model_options]))
    selected_model = st.selectbox("🤖 Model", clean_models, index=0)

    st.markdown("---")

    # Document upload
    st.markdown("### 📁 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDFs, text, or markdown files",
    )

    if uploaded_files:
        if st.button("📥 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    try:
                        chunks = st.session_state.doc_processor.process_uploaded_file(uploaded_file)
                        added = st.session_state.vector_store.add_documents(chunks)
                        total_chunks += added
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                if total_chunks > 0:
                    st.success(f"Added {total_chunks} chunks to knowledge base!")

    # Load sample docs
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.txt', '.md', '.pdf'))]
        if sample_files and st.button("📂 Load Sample Documents", use_container_width=True):
            with st.spinner("Loading sample docs..."):
                total = 0
                for sf in sample_files:
                    path = os.path.join(sample_dir, sf)
                    try:
                        chunks = st.session_state.doc_processor.process_file(path)
                        added = st.session_state.vector_store.add_documents(chunks)
                        total += added
                    except Exception as e:
                        st.error(f"Error: {e}")
                if total:
                    st.success(f"Loaded {total} chunks from samples!")

    # Knowledge base stats
    stats = st.session_state.vector_store.get_collection_stats()
    st.markdown("---")
    st.markdown(f"**📊 Chunks in DB:** {stats['total_chunks']}")
    if stats['sample_sources']:
        st.markdown("**📄 Sources:**")
        for src in stats['sample_sources']:
            st.markdown(f"  • {src}")

    if stats['total_chunks'] > 0:
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.vector_store.clear_collection()
            st.session_state.vector_store = VectorStoreManager()
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Try These")
    examples = [
        "What are the main topics in my documents?",
        "Summarize the key points from the uploaded files",
        "What specific details are mentioned about AI?",
        "Hello, how are you?",
    ]
    for ex in examples:
        if st.button(f"→ {ex}", key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_rag_example = ex

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.rag_messages = []
        st.session_state.decision_traces = {}
        st.session_state.total_rag_queries = 0
        st.rerun()


# ═══════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🧠 Agentic RAG</h1>
    <p>Autonomous Retrieval-Augmented Generation • Built with LangGraph, ChromaDB & Ollama</p>
</div>
""", unsafe_allow_html=True)

# Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{st.session_state.total_rag_queries}</div>
        <div class="stats-label">Queries</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{stats['total_chunks']}</div>
        <div class="stats-label">Doc Chunks</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">{len(stats.get('sample_sources', []))}</div>
        <div class="stats-label">Sources</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-number">7</div>
        <div class="stats-label">Agent Nodes</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Architecture diagram
with st.expander("🏗️ Agent Architecture (Click to view)", expanded=False):
    st.markdown("""
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
    │    Grade     │──not ok───►│   Query      │
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
    """)

# Tabs
tab_chat, tab_explore = st.tabs(["💬 Chat", "🔍 Explore Knowledge Base"])


# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def display_decision_trace(trace):
    """Display the agent's decision-making trace."""
    node_colors = {
        "Route Question": "#3b82f6",
        "Retrieve": "#8b5cf6",
        "Grade Documents": "#f59e0b",
        "Reformulate Query": "#06b6d4",
        "Generate": "#10b981",
        "Direct Answer": "#10b981",
        "Hallucination Check": "#ef4444",
    }
    node_icons = {
        "Route Question": "🔀",
        "Retrieve": "📥",
        "Grade Documents": "⚖️",
        "Reformulate Query": "🔄",
        "Generate": "✨",
        "Direct Answer": "💡",
        "Hallucination Check": "🔍",
    }

    # Flow visualization
    flow_html = '<div style="display:flex;flex-wrap:wrap;gap:0.2rem;align-items:center;margin-bottom:1rem;">'
    for i, step in enumerate(trace):
        node = step.get("node", "Unknown")
        color = step.get("color", node_colors.get(node, "#64748b"))
        icon = node_icons.get(node, "⚙️")
        flow_html += (
            f'<span class="flow-step" style="background:{color}20;border:1px solid {color}50;color:{color}">'
            f'{icon} {node}</span>'
        )
        if i < len(trace) - 1:
            flow_html += '<span style="color:#4b5563">→</span>'
    flow_html += '</div>'
    st.markdown(flow_html, unsafe_allow_html=True)

    # Detailed steps
    for step in trace:
        node = step.get("node", "Unknown")
        color = step.get("color", node_colors.get(node, "#64748b"))
        icon = node_icons.get(node, "⚙️")
        decision = step.get("decision", "")
        reasoning = step.get("reasoning", "")

        st.markdown(
            f'<div class="decision-node" style="border-left-color:{color};background:{color}10;">'
            f'<strong>{icon} {node}</strong><br>'
            f'<span style="color:#e2e8f0">{decision}</span>'
            f'{"<br><span style=color:#94a3b8;font-size:0.8rem>" + reasoning + "</span>" if reasoning else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )


def display_relevance_scores(scores):
    """Display relevance scores as visual bars."""
    if not scores:
        return

    st.markdown("**📊 Relevance Scores:**")
    for score_info in scores:
        score = score_info.get("score", 0)
        source = score_info.get("source", "Unknown")
        chunk = score_info.get("chunk", "?")
        preview = score_info.get("preview", "")

        # Color based on score
        if score >= 0.7:
            color = "#10b981"
        elif score >= 0.4:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        pct = min(score * 100, 100)

        st.markdown(f"""
        <div class="source-card">
            <div class="source-name">📄 {source} (chunk #{chunk})</div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.3rem">
                <div class="relevance-bar" style="flex:1;margin-right:0.5rem">
                    <div class="relevance-fill" style="width:{pct}%;background:{color}"></div>
                </div>
                <span style="color:{color};font-weight:600;font-size:0.85rem">{score:.1%}</span>
            </div>
            <div class="source-preview">{preview}</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# CHAT TAB
# ═══════════════════════════════════════════

with tab_chat:
    # Display messages
    for i, msg in enumerate(st.session_state.rag_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and str(i) in st.session_state.decision_traces:
                trace_data = st.session_state.decision_traces[str(i)]
                with st.expander("🧠 Agent Decision Trace", expanded=False):
                    display_decision_trace(trace_data.get("trace", []))
                if trace_data.get("scores"):
                    with st.expander("📊 Document Relevance", expanded=False):
                        display_relevance_scores(trace_data["scores"])

    # Handle examples
    if "pending_rag_example" in st.session_state:
        prompt = st.session_state.pending_rag_example
        del st.session_state.pending_rag_example
    else:
        prompt = st.chat_input("Ask questions about your documents or anything else...")

    if prompt:
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not connected:
                st.error("Ollama is not running. Please start it with `ollama serve`")
            else:
                with st.status("🧠 Agent is reasoning...", expanded=True) as status:
                    try:
                        agent = create_rag_agent(
                            model_name=selected_model,
                            vector_store=st.session_state.vector_store,
                        )

                        initial_state = {
                            "question": prompt,
                            "generation": "",
                            "documents": [],
                            "relevance_scores": [],
                            "decision_log": [],
                            "reformulated_question": "",
                            "retrieval_count": 0,
                            "max_retrievals": 3,
                            "route": "",
                        }

                        # Collect events
                        all_decisions = []
                        all_scores = []
                        final_generation = ""

                        for event in agent.stream(initial_state, {"recursion_limit": 15}):
                            for node_name, node_output in event.items():
                                st.write(f"🔄 **{node_name.replace('_', ' ').title()}**...")

                                if "decision_log" in node_output:
                                    all_decisions.extend(node_output["decision_log"])
                                if "relevance_scores" in node_output and node_output["relevance_scores"]:
                                    all_scores = node_output["relevance_scores"]
                                if "generation" in node_output and node_output["generation"]:
                                    final_generation = node_output["generation"]

                        status.update(label="✅ Agent finished!", state="complete")

                        # Display answer
                        if final_generation:
                            st.markdown(final_generation)
                        else:
                            st.info("The agent processed your query but didn't generate a text response.")

                        # Show decision trace
                        if all_decisions:
                            with st.expander("🧠 Agent Decision Trace", expanded=True):
                                display_decision_trace(all_decisions)

                        # Show relevance scores
                        if all_scores:
                            with st.expander("📊 Document Relevance Scores", expanded=False):
                                display_relevance_scores(all_scores)

                        # Save state
                        msg_idx = str(len(st.session_state.rag_messages))
                        st.session_state.decision_traces[msg_idx] = {
                            "trace": all_decisions,
                            "scores": all_scores,
                        }
                        st.session_state.rag_messages.append({
                            "role": "assistant",
                            "content": final_generation or "Query processed.",
                        })
                        st.session_state.total_rag_queries += 1

                    except Exception as e:
                        st.error(f"Agent error: {str(e)}")
                        st.info("💡 Make sure documents are uploaded and Ollama is running.")


# ═══════════════════════════════════════════
# EXPLORE TAB
# ═══════════════════════════════════════════

with tab_explore:
    st.markdown("### 🔍 Explore Your Knowledge Base")

    if not st.session_state.vector_store.has_documents():
        st.markdown("""
        <div class="upload-zone">
            <h3 style="color:#34d399">No documents yet</h3>
            <p style="color:#94a3b8">Upload PDF, TXT, or MD files using the sidebar to get started.<br>
            Or click "Load Sample Documents" to try with pre-loaded content.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        search_query = st.text_input("🔍 Search knowledge base", placeholder="Enter a search query...")
        if search_query:
            results = st.session_state.vector_store.similarity_search_with_scores(search_query, k=5)
            st.markdown(f"**Found {len(results)} results:**")
            for doc, score in results:
                similarity = max(0, 1 - score) if score <= 2 else 0
                color = "#10b981" if similarity >= 0.7 else "#f59e0b" if similarity >= 0.4 else "#ef4444"
                st.markdown(f"""
                <div class="source-card">
                    <div style="display:flex;justify-content:space-between">
                        <span class="source-name">📄 {doc.metadata.get('source', 'Unknown')}</span>
                        <span style="color:{color};font-weight:600">{similarity:.1%} match</span>
                    </div>
                    <div class="source-preview" style="margin-top:0.5rem">{doc.page_content[:300]}...</div>
                </div>""", unsafe_allow_html=True)

        # Collection overview
        st.markdown("---")
        st.markdown("### 📊 Collection Overview")
        cstats = st.session_state.vector_store.get_collection_stats()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Chunks", cstats["total_chunks"])
        with c2:
            st.metric("Sources", len(cstats.get("sample_sources", [])))
