import os
import time
import json
import math
import logging
import datetime
import urllib.request
import urllib.parse
import re
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from huggingface_hub import login

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition


# ==========================================
# Logging configuration
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================
# Load environment variables & Observability
# ==========================================

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
os.environ["HF_TOKEN"] = hf_token

try:
    login(token=hf_token)
except Exception as login_err:
    logger.warning(f"HuggingFace login warning: {login_err}")

# Log Observability / LangSmith configuration status
tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
project_name = os.getenv("LANGCHAIN_PROJECT", "default")

if tracing_enabled:
    logger.info(f"📊 LangSmith Observability ENABLED (Project: '{project_name}')")
else:
    logger.info("ℹ️ LangSmith Observability DISABLED")


# ==========================================
# Tools Definition
# ==========================================

@tool
def calculate(expression: str) -> str:
    """Useful for doing mathematical calculations and evaluating expressions like '25 * 4 + 10' or 'sqrt(144)'."""
    allowed_names = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "exp": math.exp, "pow": pow, "pi": math.pi, "e": math.e,
        "abs": abs, "round": round
    }
    try:
        clean_expr = expression.strip().replace("^", "**")
        result = eval(clean_expr, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Returns the current date and time. Useful when asked what time or date it is."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return f"Current UTC Date & Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for summaries or background knowledge about people, places, concepts, or events."""
    try:
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "utf8": 1
        })
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LangGraphChatbotBot/1.0 (Educational Assistant)"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return f"No Wikipedia results found for '{query}'."

            clean_results = []
            for item in search_results[:3]:
                title = item.get("title", "")
                snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                clean_results.append(f"Title: {title}\nSnippet: {snippet}")

            return "\n\n".join(clean_results)
    except Exception as e:
        return f"Wikipedia search error: {e}"


tools = [calculate, get_current_time, search_wikipedia]


# ==========================================
# HuggingFace Model with Tools Bound
# ==========================================

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=hf_token
)

base_model = ChatHuggingFace(llm=llm)
model = base_model.bind_tools(tools)


# ==========================================
# State
# ==========================================

class State(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ==========================================
# Node
# ==========================================

def call_model(state: State):
    """
    Call the LLM with tool definitions bound. The LLM can respond with text or a tool call.
    """
    start_time = time.perf_counter()
    try:
        messages = state["messages"]
        logger.info(f"[Model Node] Invoking model with {len(messages)} message(s)...")

        response = model.invoke(messages)

        elapsed = time.perf_counter() - start_time
        tool_calls = getattr(response, "tool_calls", [])
        if tool_calls:
            logger.info(f"[Model Node] Success in {elapsed:.2f}s | Requested tool calls: {[tc.get('name') for tc in tool_calls]}")
        else:
            logger.info(f"[Model Node] Success in {elapsed:.2f}s | Direct text response ({len(str(response.content))} chars)")

        return {
            "messages": [response]
        }

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "messages": [
                AIMessage(content="I couldn't process your request due to invalid input. Please try again.")
            ]
        }
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return {
            "messages": [
                AIMessage(content="I'm having trouble connecting to the AI service. Please check your internet connection and try again.")
            ]
        }
    except Exception as e:
        logger.exception(f"Unexpected error in call_model: {e}")
        return {
            "messages": [
                AIMessage(content="I encountered an unexpected error. Please try again later.")
            ]
        }


# ==========================================
# Build ReAct Tool-Calling Graph
# ==========================================

tool_node = ToolNode(tools)

graph_builder = StateGraph(State)

# Add Nodes
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", tool_node)

# Flow: START -> agent
graph_builder.add_edge(START, "agent")

# If agent outputs tool_calls, route to "tools". Otherwise route to END.
graph_builder.add_conditional_edges(
    "agent",
    tools_condition
)

# After tool execution, route back to agent to synthesize the final answer.
graph_builder.add_edge("tools", "agent")


# ==========================================
# Memory Checkpointer
# ==========================================

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)