"""
ReAct Agent with Tool Calling - Built with LangGraph + Ollama
Implements a Reasoning + Acting agent from scratch using structured prompting.
Works with ANY Ollama model (no native tool-calling required).
"""

import math
import json
import re
import datetime
import io
import contextlib

from typing import TypedDict, Annotated
import operator

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# ═══════════════════════════════════════════
# TOOL DEFINITIONS
# ═══════════════════════════════════════════

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
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


def get_current_datetime(format: str = "full") -> str:
    """Get the current date and time."""
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


def text_analyzer(text: str) -> str:
    """Analyze text and return detailed statistics."""
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


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"[{i}] {r['title']}\n    {r['body']}\n    Source: {r['href']}")
        return "\n\n".join(output)
    except ImportError:
        return "Error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {str(e)}"


def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information."""
    try:
        import wikipedia
        results = wikipedia.search(query, results=3)
        if not results:
            return "No Wikipedia articles found."
        try:
            page = wikipedia.page(results[0], auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0])
        summary = page.summary[:2000]
        return f"Title: {page.title}\nURL: {page.url}\n\nSummary:\n{summary}"
    except ImportError:
        return "Error: wikipedia not installed. Run: pip install wikipedia"
    except Exception as e:
        return f"Wikipedia error: {str(e)}"


def python_executor(code: str) -> str:
    """Execute Python code safely."""
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
        return f"Output:\n{result}" if result else "Code executed (no output). Use print()."
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {str(e)}"


# ═══════════════════════════════════════════
# TOOL REGISTRY
# ═══════════════════════════════════════════

TOOL_REGISTRY = {
    "Calculator": {
        "func": calculator,
        "icon": "🧮",
        "description": "Evaluate math expressions (e.g., 'sqrt(16) + 5', '2**10', 'sin(pi/2)')",
        "color": "#3b82f6",
    },
    "DateTime": {
        "func": get_current_datetime,
        "icon": "🕐",
        "description": "Get current date and time (format: 'full', 'date', 'time', 'timestamp')",
        "color": "#8b5cf6",
    },
    "Text Analyzer": {
        "func": text_analyzer,
        "icon": "📊",
        "description": "Analyze text for word count, sentence count, readability, frequency",
        "color": "#10b981",
    },
    "Web Search": {
        "func": web_search,
        "icon": "🔍",
        "description": "Search the web for current information using DuckDuckGo",
        "color": "#f59e0b",
    },
    "Wikipedia": {
        "func": wikipedia_search,
        "icon": "📚",
        "description": "Look up topics on Wikipedia for encyclopedic information",
        "color": "#ef4444",
    },
    "Python Executor": {
        "func": python_executor,
        "icon": "🐍",
        "description": "Execute Python code and return printed output",
        "color": "#06b6d4",
    },
}


# ═══════════════════════════════════════════
# REACT STATE & GRAPH (built from scratch)
# ═══════════════════════════════════════════

class ReActState(TypedDict):
    question: str
    reasoning_steps: Annotated[list, operator.add]
    tool_calls_count: dict
    final_answer: str
    iteration: int
    max_iterations: int
    scratchpad: str


def build_react_prompt(question, enabled_tools, scratchpad=""):
    """Build the structured ReAct prompt."""
    tool_descriptions = []
    for name in enabled_tools:
        if name in TOOL_REGISTRY:
            info = TOOL_REGISTRY[name]
            tool_descriptions.append(f"  - {name}: {info['description']}")
    tools_text = "\n".join(tool_descriptions)
    tool_names = ", ".join(enabled_tools)

    prompt = f"""You are a helpful AI assistant that reasons step-by-step and uses tools.

Available tools:
{tools_text}

You MUST respond in one of these two formats:

FORMAT 1 - When you need to use a tool:
Thought: [your reasoning about what to do]
Action: [tool name, must be one of: {tool_names}]
Action Input: [the input string for the tool]

FORMAT 2 - When you have enough info for a final answer:
Thought: [your final reasoning]
Final Answer: [your complete answer]

RULES:
- Always start with "Thought:"
- Only use one tool per step
- After seeing an Observation, continue with another Thought
- Give a Final Answer when you have enough information

Question: {question}
{scratchpad}"""
    return prompt


def clean_tool_input(tool_input: str) -> str:
    """Clean tool input from LLM - strip quotes, parenthetical comments, etc."""
    cleaned = tool_input.strip()
    # Remove surrounding quotes
    if (cleaned.startswith("'") and cleaned.endswith("'")) or \
       (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = cleaned[1:-1]
    # For calculator: remove parenthetical explanations like "(since sqrt...)"
    cleaned = re.sub(r"\s*\(since\s.+?\)\s*$", "", cleaned)
    cleaned = re.sub(r"\s*\(this\s.+?\)\s*$", "", cleaned)
    cleaned = re.sub(r"\s*\(we\s.+?\)\s*$", "", cleaned)
    cleaned = re.sub(r"\s*\(i\.e\.\s.+?\)\s*$", "", cleaned, flags=re.IGNORECASE)
    # Remove trailing quotes again after cleanup
    cleaned = cleaned.strip().strip("'\"")
    return cleaned


def execute_tool(tool_name: str, tool_input: str, enabled_tools: list) -> str:
    """Execute a tool by name with fuzzy matching and input cleaning."""
    matched = None
    for name in enabled_tools:
        if name.lower() == tool_name.lower() or \
           name.lower().replace(" ", "_") == tool_name.lower().replace(" ", "_") or \
           name.lower().replace(" ", "") == tool_name.lower().replace(" ", ""):
            matched = name
            break
    if not matched:
        for name in enabled_tools:
            if tool_name.lower() in name.lower() or name.lower() in tool_name.lower():
                matched = name
                break
    if not matched:
        return f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(enabled_tools)}"

    cleaned_input = clean_tool_input(tool_input)
    try:
        return TOOL_REGISTRY[matched]["func"](cleaned_input)
    except Exception as e:
        return f"Tool error: {str(e)}"


def reason_node(state: ReActState, llm=None, enabled_tools=None):
    """LLM reasoning node - decides next action or gives final answer."""
    prompt = build_react_prompt(
        state["question"],
        enabled_tools or list(TOOL_REGISTRY.keys()),
        state.get("scratchpad", ""),
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    steps = []
    new_scratchpad = state.get("scratchpad", "")

    # Parse Thought
    thought_match = re.search(
        r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", text, re.DOTALL
    )
    thought = thought_match.group(1).strip() if thought_match else ""
    if thought:
        steps.append({"type": "thought", "content": thought})
        new_scratchpad += f"\nThought: {thought}"

    # Check Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        answer = final_match.group(1).strip()
        steps.append({"type": "answer", "content": answer})
        return {
            "reasoning_steps": steps,
            "final_answer": answer,
            "scratchpad": new_scratchpad + f"\nFinal Answer: {answer}",
            "iteration": state.get("iteration", 0) + 1,
        }

    # Check Action
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)

    if action_match:
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip() if input_match else ""
        steps.append({"type": "action", "tool": action, "input": action_input})
        new_scratchpad += f"\nAction: {action}\nAction Input: {action_input}"

        # Execute tool
        tool_result = execute_tool(
            action, action_input, enabled_tools or list(TOOL_REGISTRY.keys())
        )
        steps.append({"type": "observation", "content": tool_result})
        new_scratchpad += f"\nObservation: {tool_result}\n"

        counts = dict(state.get("tool_calls_count", {}))
        counts[action] = counts.get(action, 0) + 1

        return {
            "reasoning_steps": steps,
            "final_answer": "",
            "scratchpad": new_scratchpad,
            "tool_calls_count": counts,
            "iteration": state.get("iteration", 0) + 1,
        }

    # Fallback: treat whole response as final answer
    steps.append({"type": "answer", "content": text})
    return {
        "reasoning_steps": steps,
        "final_answer": text,
        "scratchpad": new_scratchpad,
        "iteration": state.get("iteration", 0) + 1,
    }


def should_continue(state: ReActState) -> str:
    """Decide whether to continue the reasoning loop."""
    if state.get("final_answer"):
        return "end"
    if state.get("iteration", 0) >= state.get("max_iterations", 10):
        return "end"
    return "continue"


def create_react_agent(model_name="llama3", temperature=0.1, enabled_tools=None):
    """Create and compile the ReAct agent LangGraph."""
    if enabled_tools is None:
        enabled_tools = list(TOOL_REGISTRY.keys())

    llm = ChatOllama(model=model_name, temperature=temperature)

    def react_step(state):
        return reason_node(state, llm=llm, enabled_tools=enabled_tools)

    graph = StateGraph(ReActState)
    graph.add_node("reason", react_step)
    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {"continue": "reason", "end": END},
    )
    return graph.compile()


def get_available_tools():
    """Return the tool registry."""
    return TOOL_REGISTRY


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
