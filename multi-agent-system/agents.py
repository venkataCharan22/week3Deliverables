"""
Multi-Agent System - Built with LangGraph + Ollama
Implements a Supervisor-Worker pattern where specialized agents
collaborate to complete complex tasks.
"""

import json
import re
import operator
from typing import Annotated, TypedDict, Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START


# ═══════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════

class AgentState(TypedDict):
    task: str
    current_agent: str
    research_output: str
    draft_output: str
    review_output: str
    final_output: str
    agent_history: Annotated[list, operator.add]
    iteration: int
    max_iterations: int
    status: str


# ═══════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════

AGENT_PROFILES = {
    "supervisor": {
        "name": "Supervisor",
        "icon": "👔",
        "color": "#8b5cf6",
        "role": "Orchestrates the workflow and delegates tasks",
        "system_prompt": """You are a Supervisor agent managing a team of specialists:
- Researcher: Gathers information and facts about topics
- Writer: Creates well-structured written content
- Reviewer: Reviews drafts and provides improvement feedback

Based on the current state, decide which agent should work next.
You MUST respond with EXACTLY this format:

NEXT: [researcher/writer/reviewer/FINISH]
REASON: [brief explanation of why]

Rules:
- Start with 'researcher' to gather information
- Then 'writer' to create a draft
- Then 'reviewer' to check quality
- If the review suggests major changes, send back to 'writer'
- When the output is satisfactory, choose 'FINISH'
- Maximum {max_iter} iterations total""",
    },
    "researcher": {
        "name": "Researcher",
        "icon": "🔬",
        "color": "#3b82f6",
        "role": "Gathers information and key facts",
        "system_prompt": """You are a Research Agent. Your job is to analyze the given task
and provide comprehensive research findings.

Your output should include:
1. Key facts and information relevant to the task
2. Important points that should be covered
3. Any relevant data, statistics, or examples
4. Suggested structure/outline for the content

Be thorough but concise. Focus on accuracy and relevance.
Format your research clearly with bullet points and sections.""",
    },
    "writer": {
        "name": "Writer",
        "icon": "✍️",
        "color": "#10b981",
        "role": "Creates polished written content",
        "system_prompt": """You are a Writer Agent. Your job is to create well-written,
engaging content based on the research provided.

Guidelines:
- Use clear, professional language
- Structure content with headings and paragraphs
- Include relevant examples and details from the research
- Make the content engaging and informative
- If review feedback is provided, incorporate those improvements

Produce polished, ready-to-read content.""",
    },
    "reviewer": {
        "name": "Reviewer",
        "icon": "🔍",
        "color": "#f59e0b",
        "role": "Reviews content and suggests improvements",
        "system_prompt": """You are a Reviewer Agent. Your job is to critically review
the written draft and provide constructive feedback.

Evaluate the draft on:
1. Accuracy - Are facts correct and well-supported?
2. Completeness - Are all important aspects covered?
3. Clarity - Is the writing clear and easy to understand?
4. Structure - Is the content well-organized?
5. Engagement - Is the content interesting to read?

Provide:
- An overall quality score (1-10)
- Specific strengths
- Areas for improvement with concrete suggestions
- A verdict: APPROVE (score >= 7) or REVISE (score < 7)""",
    },
}


def create_agent_chain(model_name, agent_type):
    """Create an LLM chain for a specific agent type."""
    profile = AGENT_PROFILES[agent_type]
    llm = ChatOllama(model=model_name, temperature=0.3 if agent_type == "supervisor" else 0.7)
    system_prompt = profile["system_prompt"]
    return llm, system_prompt


# ═══════════════════════════════════════════
# NODE FUNCTIONS
# ═══════════════════════════════════════════

def supervisor_node(state: AgentState, model_name: str = "llama3") -> dict:
    """Supervisor decides which agent should work next."""
    llm, system_prompt = create_agent_chain(model_name, "supervisor")
    system_prompt = system_prompt.format(max_iter=state.get("max_iterations", 6))

    context_parts = [f"TASK: {state['task']}"]
    if state.get("research_output"):
        context_parts.append(f"\nRESEARCH COMPLETED:\n{state['research_output'][:500]}...")
    if state.get("draft_output"):
        context_parts.append(f"\nDRAFT COMPLETED:\n{state['draft_output'][:500]}...")
    if state.get("review_output"):
        context_parts.append(f"\nREVIEW FEEDBACK:\n{state['review_output'][:500]}...")
    context_parts.append(f"\nITERATION: {state.get('iteration', 0)} / {state.get('max_iterations', 6)}")

    context = "\n".join(context_parts)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    response_text = response.content

    # Parse the supervisor's decision
    next_agent = "FINISH"
    reason = "Max iterations reached"

    next_match = re.search(r"NEXT:\s*(researcher|writer|reviewer|FINISH)", response_text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+?)(?:\n|$)", response_text)

    if next_match:
        next_agent = next_match.group(1).lower()
    if reason_match:
        reason = reason_match.group(1).strip()

    # Force FINISH if max iterations reached
    iteration = state.get("iteration", 0)
    if iteration >= state.get("max_iterations", 6):
        next_agent = "FINISH"
        reason = "Maximum iterations reached, finalizing output"

    return {
        "current_agent": next_agent,
        "iteration": iteration + 1,
        "agent_history": [{
            "agent": "supervisor",
            "action": f"Delegating to: {next_agent}",
            "reason": reason,
            "iteration": iteration + 1,
        }],
        "status": "routing",
    }


def researcher_node(state: AgentState, model_name: str = "llama3") -> dict:
    """Researcher gathers information about the task."""
    llm, system_prompt = create_agent_chain(model_name, "researcher")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research the following task thoroughly:\n\n{state['task']}"),
    ]

    response = llm.invoke(messages)

    return {
        "research_output": response.content,
        "current_agent": "supervisor",
        "agent_history": [{
            "agent": "researcher",
            "action": "Completed research",
            "output_preview": response.content[:200] + "...",
            "iteration": state.get("iteration", 0),
        }],
        "status": "research_complete",
    }


def writer_node(state: AgentState, model_name: str = "llama3") -> dict:
    """Writer creates content based on research and feedback."""
    llm, system_prompt = create_agent_chain(model_name, "writer")

    context = f"TASK: {state['task']}\n\n"
    if state.get("research_output"):
        context += f"RESEARCH:\n{state['research_output']}\n\n"
    if state.get("review_output"):
        context += f"REVIEW FEEDBACK (please address these points):\n{state['review_output']}\n\n"
    if state.get("draft_output"):
        context += f"PREVIOUS DRAFT (improve upon this):\n{state['draft_output']}\n\n"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)

    return {
        "draft_output": response.content,
        "current_agent": "supervisor",
        "agent_history": [{
            "agent": "writer",
            "action": "Created/revised draft",
            "output_preview": response.content[:200] + "...",
            "iteration": state.get("iteration", 0),
        }],
        "status": "draft_complete",
    }


def reviewer_node(state: AgentState, model_name: str = "llama3") -> dict:
    """Reviewer evaluates the draft and provides feedback."""
    llm, system_prompt = create_agent_chain(model_name, "reviewer")

    context = (
        f"ORIGINAL TASK: {state['task']}\n\n"
        f"DRAFT TO REVIEW:\n{state.get('draft_output', 'No draft available')}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)

    return {
        "review_output": response.content,
        "current_agent": "supervisor",
        "agent_history": [{
            "agent": "reviewer",
            "action": "Completed review",
            "output_preview": response.content[:200] + "...",
            "iteration": state.get("iteration", 0),
        }],
        "status": "review_complete",
    }


def finalizer_node(state: AgentState, **kwargs) -> dict:
    """Finalize the output - use the best available content."""
    final = state.get("draft_output", "")
    if not final:
        final = state.get("research_output", "No output was generated.")

    return {
        "final_output": final,
        "status": "complete",
        "agent_history": [{
            "agent": "finalizer",
            "action": "Finalized output",
            "iteration": state.get("iteration", 0),
        }],
    }


# ═══════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════

def route_from_supervisor(state: AgentState) -> str:
    """Route to the next agent based on supervisor's decision."""
    next_agent = state.get("current_agent", "FINISH").lower()
    if next_agent == "researcher":
        return "researcher"
    elif next_agent == "writer":
        return "writer"
    elif next_agent == "reviewer":
        return "reviewer"
    else:
        return "finalizer"


def create_multi_agent_graph(model_name="llama3"):
    """Create the multi-agent collaboration graph."""

    # Create model-bound node functions
    def sup_node(state):
        return supervisor_node(state, model_name)

    def res_node(state):
        return researcher_node(state, model_name)

    def wri_node(state):
        return writer_node(state, model_name)

    def rev_node(state):
        return reviewer_node(state, model_name)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", sup_node)
    graph.add_node("researcher", res_node)
    graph.add_node("writer", wri_node)
    graph.add_node("reviewer", rev_node)
    graph.add_node("finalizer", finalizer_node)

    # Set entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to the appropriate agent
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "finalizer": "finalizer",
        },
    )

    # All agents return to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")
    graph.add_edge("finalizer", END)

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
