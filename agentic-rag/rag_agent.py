"""
Agentic RAG with Autonomous Retrieval - Built with LangGraph + Ollama
Implements an intelligent RAG agent that autonomously decides when to retrieve,
grades document relevance, reformulates queries, and checks for hallucinations.
"""

import json
import re
from typing import TypedDict, List, Optional, Annotated
import operator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START


# ═══════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════

class RAGState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    relevance_scores: list
    decision_log: Annotated[list, operator.add]
    reformulated_question: str
    retrieval_count: int
    max_retrievals: int
    route: str


# ═══════════════════════════════════════════
# NODE FUNCTIONS
# ═══════════════════════════════════════════

def create_rag_agent(model_name="llama3", vector_store=None):
    """Create the Agentic RAG graph."""

    llm = ChatOllama(model=model_name, temperature=0.1)
    llm_creative = ChatOllama(model=model_name, temperature=0.7)

    # ── Route Question ──────────────────────
    def route_question(state: RAGState) -> dict:
        """Decide if we need to retrieve documents or can answer directly."""
        question = state["question"]
        q_lower = question.strip().lower().rstrip("!?.")

        has_docs = vector_store is not None and vector_store.has_documents()

        # Quick check for obvious greetings - skip LLM call for these
        greetings = {"hello", "hi", "hey", "howdy", "greetings", "good morning",
                     "good afternoon", "good evening", "how are you", "what's up",
                     "whats up", "sup"}
        if q_lower in greetings or (not has_docs and len(question.split()) <= 3):
            route = "direct"
            reasoning = f"Detected greeting or simple query: '{question}'"
        elif has_docs:
            # If we have documents, default to retrieval for any real question
            route = "retrieve"
            reasoning = f"Knowledge base has documents - retrieving for: '{question}'"
        else:
            # No documents - answer directly
            route = "direct"
            reasoning = "Knowledge base is empty - answering from general knowledge"

        return {
            "route": route,
            "decision_log": [{
                "node": "Route Question",
                "decision": f"Route → {'Retrieve from docs' if route == 'retrieve' else 'Direct answer'}",
                "reasoning": reasoning,
                "color": "#3b82f6",
            }],
        }

    # ── Retrieve Documents ──────────────────
    def retrieve(state: RAGState) -> dict:
        """Retrieve relevant documents from the vector store."""
        query = state.get("reformulated_question") or state["question"]
        retrieval_count = state.get("retrieval_count", 0) + 1

        if vector_store is None or not vector_store.has_documents():
            return {
                "documents": [],
                "relevance_scores": [],
                "retrieval_count": retrieval_count,
                "decision_log": [{
                    "node": "Retrieve",
                    "decision": "No documents in knowledge base",
                    "reasoning": "Vector store is empty. Upload documents first.",
                    "color": "#ef4444",
                }],
            }

        results_with_scores = vector_store.similarity_search_with_scores(query, k=4)

        documents = []
        scores = []
        for doc, score in results_with_scores:
            documents.append(doc)
            # ChromaDB returns distance (lower = better), convert to similarity
            similarity = max(0, 1 - score) if score <= 2 else 0
            scores.append({
                "source": doc.metadata.get("source", "Unknown"),
                "chunk": doc.metadata.get("chunk_index", "?"),
                "score": round(similarity, 3),
                "preview": doc.page_content[:100] + "...",
            })

        return {
            "documents": documents,
            "relevance_scores": scores,
            "retrieval_count": retrieval_count,
            "decision_log": [{
                "node": "Retrieve",
                "decision": f"Retrieved {len(documents)} documents (attempt #{retrieval_count})",
                "reasoning": f"Query: '{query}' | Top score: {scores[0]['score'] if scores else 'N/A'}",
                "color": "#8b5cf6",
            }],
        }

    # ── Grade Documents ─────────────────────
    def grade_documents(state: RAGState) -> dict:
        """Grade retrieved documents for relevance to the question."""
        question = state.get("reformulated_question") or state["question"]
        documents = state.get("documents", [])

        if not documents:
            return {
                "documents": [],
                "route": "reformulate",
                "decision_log": [{
                    "node": "Grade Documents",
                    "decision": "No documents to grade → Reformulate",
                    "reasoning": "No documents were retrieved",
                    "color": "#f59e0b",
                }],
            }

        # Use relevance scores from retrieval to pre-filter
        scores = state.get("relevance_scores", [])
        relevant_docs = []
        grading_details = []

        for i, doc in enumerate(documents):
            # Check similarity score first (if available) - accept docs with score > 0.3
            score = scores[i].get("score", 0) if i < len(scores) else 0

            if score >= 0.5:
                # High similarity - accept without LLM call
                relevant_docs.append(doc)
                grading_details.append(f"Doc {i+1}: Relevant ✓ (score: {score:.2f})")
            elif score >= 0.2:
                # Medium similarity - use LLM to verify
                prompt = f"""You are a relevance grader. Assess if the document is relevant to the question.

Question: {question}

Document content:
{doc.page_content[:500]}

Is this document relevant to answering the question? Respond with ONLY 'yes' or 'no'."""

                response = llm.invoke([HumanMessage(content=prompt)])
                is_relevant = "yes" in response.content.lower()

                if is_relevant:
                    relevant_docs.append(doc)
                grading_details.append(f"Doc {i+1}: {'Relevant ✓' if is_relevant else 'Not relevant ✗'} (score: {score:.2f})")
            else:
                grading_details.append(f"Doc {i+1}: Not relevant ✗ (score: {score:.2f})")

        # Decide next step based on relevant docs
        if len(relevant_docs) >= 1:
            route = "generate"
            decision = f"{len(relevant_docs)}/{len(documents)} docs relevant → Generate answer"
        else:
            route = "reformulate"
            decision = f"0/{len(documents)} docs relevant → Reformulate query"

        return {
            "documents": relevant_docs,
            "route": route,
            "decision_log": [{
                "node": "Grade Documents",
                "decision": decision,
                "reasoning": " | ".join(grading_details),
                "color": "#f59e0b",
            }],
        }

    # ── Reformulate Query ───────────────────
    def reformulate_query(state: RAGState) -> dict:
        """Reformulate the query for better retrieval results."""
        original = state["question"]
        previous = state.get("reformulated_question", "")

        prompt = f"""You are a query reformulation expert. The original search query didn't return relevant results.

Original question: {original}
{f"Previous reformulation (also failed): {previous}" if previous else ""}

Create a better search query that:
1. Uses different keywords and phrasing
2. Is more specific or broader as needed
3. Focuses on the core information need

Respond with ONLY the reformulated query, nothing else."""

        response = llm.invoke([HumanMessage(content=prompt)])
        new_query = response.content.strip().strip('"').strip("'")

        return {
            "reformulated_question": new_query,
            "decision_log": [{
                "node": "Reformulate Query",
                "decision": f"New query: '{new_query}'",
                "reasoning": f"Original: '{original}' → Reformulated for better retrieval",
                "color": "#06b6d4",
            }],
        }

    # ── Generate Answer ─────────────────────
    def generate(state: RAGState) -> dict:
        """Generate answer using relevant documents as context."""
        question = state["question"]
        documents = state.get("documents", [])

        if not documents:
            prompt = f"""Answer the following question based on your general knowledge.
Note: No relevant documents were found in the knowledge base.

Question: {question}

Provide a helpful answer and mention that no specific documents were found."""
        else:
            context = "\n\n---\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')} | "
                f"Chunk {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
                for doc in documents
            ])
            prompt = f"""Answer the question based on the following context from documents.
Use the information from the context to provide an accurate, detailed answer.
Always cite your sources by mentioning the document name.

Context:
{context}

Question: {question}

Answer:"""

        response = llm_creative.invoke([HumanMessage(content=prompt)])

        # Build source citations
        sources = []
        seen = set()
        for doc in documents:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)

        generation = response.content
        if sources:
            generation += f"\n\n📚 **Sources:** {', '.join(sources)}"

        return {
            "generation": generation,
            "decision_log": [{
                "node": "Generate",
                "decision": f"Generated answer using {len(documents)} documents",
                "reasoning": f"Sources: {', '.join(sources) if sources else 'General knowledge'}",
                "color": "#10b981",
            }],
        }

    # ── Direct Answer ───────────────────────
    def direct_answer(state: RAGState) -> dict:
        """Answer directly without retrieval."""
        question = state["question"]

        response = llm_creative.invoke([
            SystemMessage(content="You are a helpful assistant. Answer the question directly and concisely."),
            HumanMessage(content=question),
        ])

        return {
            "generation": response.content,
            "decision_log": [{
                "node": "Direct Answer",
                "decision": "Answered directly without document retrieval",
                "reasoning": "Question classified as general knowledge / greeting",
                "color": "#10b981",
            }],
        }

    # ── Check Hallucination ─────────────────
    def check_hallucination(state: RAGState) -> dict:
        """Check if the generated answer is grounded in the documents."""
        generation = state.get("generation", "")
        documents = state.get("documents", [])

        if not documents:
            return {
                "route": "done",
                "decision_log": [{
                    "node": "Hallucination Check",
                    "decision": "Skipped (no source documents)",
                    "reasoning": "No documents to verify against",
                    "color": "#64748b",
                }],
            }

        context = "\n".join([doc.page_content[:300] for doc in documents[:3]])

        prompt = f"""You are a fact-checker. Determine if the answer is grounded in the provided documents.

Documents:
{context}

Answer to check:
{generation[:500]}

Is this answer supported by the documents? Respond with ONLY 'grounded' or 'hallucination'."""

        response = llm.invoke([HumanMessage(content=prompt)])
        is_grounded = "grounded" in response.content.lower()

        return {
            "route": "done" if is_grounded else "regenerate",
            "decision_log": [{
                "node": "Hallucination Check",
                "decision": "✓ Answer is grounded" if is_grounded else "✗ Possible hallucination detected",
                "reasoning": response.content.strip()[:200],
                "color": "#10b981" if is_grounded else "#ef4444",
            }],
        }

    # ═══════════════════════════════════════
    # ROUTING FUNCTIONS
    # ═══════════════════════════════════════

    def route_after_initial(state: RAGState) -> str:
        if state.get("route") == "direct":
            return "direct_answer"
        return "retrieve"

    def route_after_grading(state: RAGState) -> str:
        route = state.get("route", "generate")
        retrieval_count = state.get("retrieval_count", 0)
        max_retrievals = state.get("max_retrievals", 3)

        if route == "reformulate" and retrieval_count < max_retrievals:
            return "reformulate_query"
        return "generate"

    def route_after_hallucination_check(state: RAGState) -> str:
        if state.get("route") == "regenerate":
            return "generate"
        return END

    # ═══════════════════════════════════════
    # BUILD GRAPH
    # ═══════════════════════════════════════

    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("route_question", route_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("reformulate_query", reformulate_query)
    graph.add_node("generate", generate)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("check_hallucination", check_hallucination)

    # Set entry point
    graph.set_entry_point("route_question")

    # Add edges
    graph.add_conditional_edges("route_question", route_after_initial, {
        "retrieve": "retrieve",
        "direct_answer": "direct_answer",
    })

    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges("grade_documents", route_after_grading, {
        "reformulate_query": "reformulate_query",
        "generate": "generate",
    })

    graph.add_edge("reformulate_query", "retrieve")
    graph.add_edge("generate", "check_hallucination")
    graph.add_edge("direct_answer", END)

    graph.add_conditional_edges("check_hallucination", route_after_hallucination_check, {
        "generate": "generate",
        END: END,
    })

    return graph.compile()


def check_ollama_connection():
    """Check if Ollama is running and return available models."""
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []
