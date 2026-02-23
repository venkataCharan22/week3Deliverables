"""
ReAct Agent with Tool Calling - Built with LangGraph + Ollama
Implements a Reasoning + Acting agent that can use multiple tools
to solve complex queries step by step.
"""

import math
import json
import re
import datetime
import io
import contextlib
from typing import Annotated

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition


# ═══════════════════════════════════════════
# TOOL DEFINITIONS
# ═══════════════════════════════════════════

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports arithmetic, powers, roots, trig functions.

    Args:
        expression: Math expression like '2 + 2', 'sqrt(16)', 'sin(3.14/2)', '2**10'
    """
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "log": math.log, "log10": math.log10,
        "log2": math.log2, "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "pi": math.pi, "e": math.e,
        "ceil": math.ceil, "floor": math.floor,
        "factorial": math.factorial, "gcd": math.gcd,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


@tool
def get_current_datetime(format: str = "full") -> str:
    """Get the current date and time.

    Args:
        format: One of 'full', 'date', 'time', 'timestamp', 'iso'
    """
    now = datetime.datetime.now()
    formats = {
        "date": f"Current date: {now.strftime('%Y-%m-%d (%A, %B %d, %Y)')}",
        "time": f"Current time: {now.strftime('%I:%M:%S %p')}",
        "timestamp": f"Unix timestamp: {int(now.timestamp())}",
        "iso": f"ISO format: {now.isoformat()}",
    }
    return formats.get(format, (
        f"Date: {now.strftime('%A, %B %d, %Y')}\n"
        f"Time: {now.strftime('%I:%M:%S %p')}\n"
        f"Timezone: Local\n"
        f"ISO: {now.isoformat()}"
    ))


@tool
def text_analyzer(text: str) -> str:
    """Analyze text and return detailed statistics including word count, readability, and frequency.

    Args:
        text: The text to analyze
    """
    words = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    word_freq = {}
    for w in words:
        cleaned = w.lower().strip(".,!?;:\"'()-")
        if cleaned:
            word_freq[cleaned] = word_freq.get(cleaned, 0) + 1

    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    avg_sentence_len = len(words) / max(len(sentences), 1)

    return json.dumps({
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "")),
        "words": len(words),
        "unique_words": len(set(w.lower() for w in words)),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length": round(avg_sentence_len, 2),
        "top_10_words": dict(top_words),
        "lexical_diversity": round(len(set(w.lower() for w in words)) / max(len(words), 1), 3),
    }, indent=2)


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo for current information.

    Args:
        query: The search query
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found for this query."
        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"[{i}] {r['title']}\n"
                f"    {r['body']}\n"
                f"    Source: {r['href']}"
            )
        return "\n\n".join(output)
    except ImportError:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for encyclopedic information on a topic.

    Args:
        query: The topic to search for
    """
    try:
        import wikipedia
        results = wikipedia.search(query, results=3)
        if not results:
            return "No Wikipedia articles found for this query."
        try:
            page = wikipedia.page(results[0], auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0])
        summary = page.summary[:2000]
        return (
            f"Title: {page.title}\n"
            f"URL: {page.url}\n\n"
            f"Summary:\n{summary}"
        )
    except ImportError:
        return "Error: wikipedia not installed. Run: pip install wikipedia"
    except Exception as e:
        return f"Wikipedia error: {str(e)}"


@tool
def python_executor(code: str) -> str:
    """Execute Python code safely and return the printed output.

    Args:
        code: Python code to execute (use print() to see output)
    """
    output_buffer = io.StringIO()
    restricted_globals = {
        "__builtins__": {
            "print": print, "range": range, "len": len, "int": int,
            "float": float, "str": str, "list": list, "dict": dict,
            "tuple": tuple, "set": set, "bool": bool, "type": type,
            "sorted": sorted, "enumerate": enumerate, "zip": zip,
            "map": map, "filter": filter, "sum": sum, "min": min,
            "max": max, "abs": abs, "round": round, "reversed": reversed,
            "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
        }
    }
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, restricted_globals, {})
        result = output_buffer.getvalue()
        return f"Output:\n{result}" if result else "Code executed successfully (no output). Use print() to see results."
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {str(e)}"


# ═══════════════════════════════════════════
# TOOL REGISTRY
# ═══════════════════════════════════════════

TOOL_REGISTRY = {
    "Calculator": {
        "func": calculator,
        "icon": "🧮",
        "description": "Math expressions & calculations",
        "color": "#3b82f6",
    },
    "DateTime": {
        "func": get_current_datetime,
        "icon": "🕐",
        "description": "Current date & time info",
        "color": "#8b5cf6",
    },
    "Text Analyzer": {
        "func": text_analyzer,
        "icon": "📊",
        "description": "Text statistics & analysis",
        "color": "#10b981",
    },
    "Web Search": {
        "func": web_search,
        "icon": "🔍",
        "description": "Search the web via DuckDuckGo",
        "color": "#f59e0b",
    },
    "Wikipedia": {
        "func": wikipedia_search,
        "icon": "📚",
        "description": "Search Wikipedia articles",
        "color": "#ef4444",
    },
    "Python Executor": {
        "func": python_executor,
        "icon": "🐍",
        "description": "Execute Python code snippets",
        "color": "#06b6d4",
    },
}

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools. When a user asks a question:
1. Think about whether you need to use a tool to answer
2. If yes, call the appropriate tool with the right arguments
3. Use the tool's output to formulate your final answer
4. If the question is simple and you know the answer, respond directly

Always be accurate and helpful. When using tools, explain what you're doing and why."""


def get_available_tools():
    """Return dict of tool name -> tool info."""
    return TOOL_REGISTRY


def create_react_agent(model_name="llama3", temperature=0.1, enabled_tools=None):
    """Create and compile a ReAct agent graph with specified tools and model."""
    if enabled_tools is None:
        enabled_tools = list(TOOL_REGISTRY.keys())

    tools = [TOOL_REGISTRY[name]["func"] for name in enabled_tools if name in TOOL_REGISTRY]

    if not tools:
        raise ValueError("At least one tool must be enabled")

    llm = ChatOllama(model=model_name, temperature=temperature)
    model_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

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
